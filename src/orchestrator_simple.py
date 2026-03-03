
from __future__ import annotations

from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import json
from json import JSONDecodeError
import os
import base64
import re
from pathlib import Path
import random
import hashlib

import httpx
from pydantic import ValidationError

from src.agents import Agent
from src.toolbox import Toolbox
from src.apicallhandler import OpenRouterClient, TRACE_HOOK
from src.router import ModelRouter
from src.pydantic_schemas import (
    Setting,
    StoryOutlineFull,
    OutlineBeat,
    SceneContract,
    SceneScript,
    SceneChoice,
    SceneChoiceOption,
    CharacterAppearance,
    LocationDescription,
    UserRequest,
    CharacterImage,
    LocationImage,
    BranchSpec,
    BranchingInfo,
    PlotPreferences,
    StoryState,
    SceneLine,
    LocationAffordance,
)
from src.logger import app_logger
from src.utils.artifacts import ArtifactStore
from src.utils.names import NameCanonicalizer
from src.prompts import (
    char_prompt,
    loc_prompt,
    setting_prompt,
    outline_prompt,
    scene_plan_prompt,
    writer_prompt_simple,
    user_request_prompt,
    branch_planner_prompt,
    location_affordance_prompt,
)


def _sha1(text: str) -> str:
    return hashlib.sha1((text or "").encode("utf-8")).hexdigest()


class VNOrchestratorSimple:
    """
    No RAG (no embeddings, no retrieval, no rag_context agent)
    No critic (no critique passes, no microplanner, no scene editor, no contract patch critic)
    No multi-candidate scene generation (always 1 candidate per attempt)
    """

    def __init__(self, client: OpenRouterClient, router: ModelRouter):
        self.client = client
        self.router = router

    @staticmethod
    def _dedupe_preserve_order(items: List[str]) -> List[str]:
        out: List[str] = []
        seen = set()
        for x in items or []:
            s = str(x or "").strip()
            if not s:
                continue
            k = s.casefold()
            if k in seen:
                continue
            seen.add(k)
            out.append(s)
        return out

    def _max_locations_for_length(self, story_length: str) -> int:
        sl = (story_length or "medium").strip().lower()
        key = f"MAX_LOCATIONS_{sl.upper()}"
        env = os.getenv(key) or os.getenv("MAX_LOCATIONS")
        if env:
            try:
                v = int(env)
                return max(4, min(v, 40))
            except Exception:
                pass
        return {"short": 8, "medium": 12, "long": 16}.get(sl, 12)

    def _limit_locations(
        self,
        *,
        candidates: List[str],
        story_length: str,
        artifact_store: Optional[ArtifactStore] = None,
        reason: str = "auto_loc_list",
    ) -> List[str]:
        max_locs = self._max_locations_for_length(story_length)
        uniq = self._dedupe_preserve_order([str(x) for x in (candidates or [])])
        if len(uniq) <= max_locs:
            return uniq

        limited = uniq[:max_locs]

        if artifact_store is not None:
            artifact_store.event(
                "loc_list.limited",
                {
                    "reason": reason,
                    "max_locations": max_locs,
                    "before": len(uniq),
                    "after": len(limited),
                    "dropped": uniq[max_locs : max_locs + 80],
                },
            )
            artifact_store.save(
                "checkpoints/loc_list_limited_preview.json",
                {"max_locations": max_locs, "limited": limited, "original": uniq[:200]},
            )

        return limited


    async def _parse_json_with_repair(
        self,
        raw: str,
        model_name: str,
        operation_name: str,
        schema_hint: str,
        artifact_store: Optional[ArtifactStore] = None,
    ) -> Dict[str, Any]:
        raw_strip = (raw or "").strip()

        try:
            return json.loads(raw_strip)
        except JSONDecodeError:
            pass

        start = raw_strip.find("{")
        end = raw_strip.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = raw_strip[start : end + 1]
            try:
                return json.loads(candidate)
            except JSONDecodeError:
                app_logger.warning(f"{operation_name}: parse failed, invoking repair LLM...")
                if artifact_store is not None:
                    artifact_store.event(
                        "parse.repair_called",
                        {"operation_name": operation_name, "model": model_name, "raw_preview": raw_strip[:800]},
                    )

        repair_system = (
            "Ты помощник, который ЧИНИТ НЕВАЛИДНЫЙ JSON.\n"
            "Верни СТРОГО ОДИН ВАЛИДНЫЙ JSON-ОБЪЕКТ по schema_hint.\n"
            "Никакого Markdown, только JSON."
        )
        repair_payload = {"broken_json": raw_strip, "schema_hint": schema_hint}

        resp = await self.client.generate_completion(
            model=model_name,
            temperature=0.0,
            system_prompt=repair_system,
            prompt=json.dumps(repair_payload, ensure_ascii=False),
            response_format={"type": "json_object"},
            operation_name=f"{operation_name}_json_repair",
        )

        fixed = (resp["choices"][0]["message"]["content"] or "").strip()
        try:
            return json.loads(fixed)
        except JSONDecodeError:
            start = fixed.find("{")
            end = fixed.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(fixed[start : end + 1])
                except JSONDecodeError:
                    pass
        return {}


    @staticmethod
    def _build_loc_desc_map(loc_list: List[str], loc_description: Any) -> Dict[str, str]:
        descs: List[str] = []
        if isinstance(loc_description, dict):
            descs = loc_description.get("descriptions") or []
        else:
            descs = getattr(loc_description, "descriptions", None) or []
        out: Dict[str, str] = {}
        for name, desc in zip(loc_list or [], descs or []):
            out[str(name)] = str(desc or "")
        for name in loc_list or []:
            out.setdefault(str(name), "")
        return out

    @staticmethod
    def _extract_location_aff(loc_affordances: Dict[str, Dict[str, Any]], loc: str) -> Dict[str, Any]:
        if not loc_affordances:
            return {"kind": "mixed", "enterable": True, "scale": "area", "notes": ""}
        v = loc_affordances.get(loc)
        if isinstance(v, dict):
            return {
                "kind": str(v.get("kind") or "mixed"),
                "enterable": bool(v.get("enterable", True)),
                "scale": str(v.get("scale") or "area"),
                "notes": str(v.get("notes") or ""),
            }
        return {"kind": "mixed", "enterable": True, "scale": "area", "notes": ""}

    async def _infer_location_affordances(
        self,
        setting: Setting,
        loc_list: List[str],
        loc_canons: Dict[str, str],
        artifact_store: Optional[ArtifactStore] = None,
    ) -> Dict[str, Dict[str, Any]]:
        app_logger.info("Inferring location affordances (enterable/scale/kind)...")
        model_name = self.router.get_model_for_agent("outline_agent")

        payload = {
            "setting": setting.dict(),
            "loc_list": loc_list,
            "loc_canons": {k: (v[:1200] if isinstance(v, str) else "") for k, v in (loc_canons or {}).items()},
        }

        resp = await self.client.generate_completion(
            model=model_name,
            temperature=0.2,
            system_prompt=location_affordance_prompt,
            prompt=json.dumps(payload, ensure_ascii=False),
            response_format={"type": "json_object"},
            operation_name="location_affordances",
        )

        raw = (resp["choices"][0]["message"]["content"] or "").strip()
        schema_hint = '{"locations":[{"location":"...","kind":"outdoor","enterable":false,"scale":"object","notes":"..."}]}'
        data = await self._parse_json_with_repair(
            raw,
            model_name,
            "location_affordances_parse",
            schema_hint,
            artifact_store=artifact_store,
        )

        items = data.get("locations") or []
        out: Dict[str, Dict[str, Any]] = {}
        if isinstance(items, list):
            for it in items:
                if not isinstance(it, dict):
                    continue
                loc = str(it.get("location") or "").strip()
                if not loc:
                    continue
                if loc not in loc_list:
                    continue
                try:
                    aff = LocationAffordance(
                        location=loc,
                        kind=str(it.get("kind") or "mixed"),
                        enterable=bool(it.get("enterable", True)),
                        scale=str(it.get("scale") or "area"),
                        notes=str(it.get("notes") or ""),
                    )
                    out[loc] = aff.dict()
                except Exception:
                    out[loc] = {
                        "kind": str(it.get("kind") or "mixed"),
                        "enterable": bool(it.get("enterable", True)),
                        "scale": str(it.get("scale") or "area"),
                        "notes": str(it.get("notes") or ""),
                    }

        for loc in loc_list:
            out.setdefault(loc, {"kind": "mixed", "enterable": True, "scale": "area", "notes": ""})

        if artifact_store is not None:
            artifact_store.checkpoint("07b_location_affordances", {"affordances": out})
        return out

    async def _normalize_user_request(
        self,
        raw_prompt: str,
        explicit_length: Optional[str] = None,
        explicit_tone: Optional[str] = None,
        explicit_artstyle: Optional[str] = None,
        explicit_max_branches: Optional[int] = None,
        artifact_store: Optional[ArtifactStore] = None,
    ) -> UserRequest:
        app_logger.info("Normalizing user prompt into UserRequest...")
        model_name = self.router.get_model_for_agent("setting_agent")

        payload = {"user_prompt": raw_prompt}

        resp = await self.client.generate_completion(
            model=model_name,
            temperature=0.2,
            system_prompt=user_request_prompt,
            prompt=json.dumps(payload, ensure_ascii=False),
            response_format={"type": "json_object"},
            operation_name="user_request_normalizer",
        )

        content = (resp["choices"][0]["message"]["content"] or "").strip()
        data = json.loads(content) if content else {}
        data["user_prompt"] = raw_prompt

        ur = UserRequest(**data)
        if explicit_length:
            ur.story_length = explicit_length
        if explicit_tone:
            ur.tone = explicit_tone
        if explicit_artstyle:
            ur.general_artstyle = explicit_artstyle
        if explicit_max_branches is not None:
            ur.max_branches = explicit_max_branches

        if artifact_store is not None:
            artifact_store.checkpoint("01_user_request_normalized", ur.dict())
        return ur

    async def _generate_setting(
        self,
        user_prompt: str,
        setting_override: str | None = None,
        time_choice: Optional[str] = None,
        genre_choice: Optional[str] = None,
        artifact_store: Optional[ArtifactStore] = None,
    ) -> Setting:
        app_logger.info("Generating setting...")
        model_name = self.router.get_model_for_agent("setting_agent")

        payload: Dict[str, Any] = {
            "user_prompt": user_prompt,
            "setting_override": setting_override or "",
            "time_choice": time_choice,
            "genre_choice": genre_choice,
        }

        resp = await self.client.generate_completion(
            model=model_name,
            temperature=0.4,
            system_prompt=setting_prompt,
            prompt=json.dumps(payload, ensure_ascii=False),
            response_format={"type": "json_object"},
            operation_name="setting_from_prompt",
        )

        data = json.loads(resp["choices"][0]["message"]["content"])
        setting = Setting(**data)

        if time_choice:
            setting.time_period = time_choice
        if genre_choice:
            setting.genre = genre_choice

        if artifact_store is not None:
            artifact_store.checkpoint("02_setting", setting.dict())
        return setting

    async def _generate_outline(
        self,
        user_prompt: str,
        story_length: str,
        setting: Setting,
        plot_prefs: Optional[PlotPreferences] = None,
        plot_freeform: Optional[str] = None,
        artifact_store: Optional[ArtifactStore] = None,
    ) -> StoryOutlineFull:
        app_logger.info("Generating outline...")
        model_name = self.router.get_model_for_agent("outline_agent")

        payload: Dict[str, Any] = {
            "user_prompt": user_prompt,
            "story_length": story_length,
            "setting": setting.dict(),
            "plot_prefs": plot_prefs.dict() if plot_prefs else None,
            "plot_freeform": plot_freeform,
        }

        schema_hint = (
            '{"theory":"three_act","beats":[{"id":"beat_01","act":1,"order":1,'
            '"title":"...","summary":"...","tension_level":"low","purpose":"setup"}]}'
        )

        last_resp: Optional[Dict[str, Any]] = None
        for attempt in range(1, 4):
            resp = await self.client.generate_completion(
                model=model_name,
                temperature=0.4 if attempt == 1 else 0.2,
                system_prompt=outline_prompt,
                prompt=json.dumps(payload, ensure_ascii=False),
                response_format={"type": "json_object"},
                operation_name=f"outline_from_setting_a{attempt}",
            )
            last_resp = resp

            msg = (((resp or {}).get("choices") or [{}])[0].get("message") or {})
            content = (msg.get("content") or "")
            content_stripped = content.strip() if isinstance(content, str) else ""

            if artifact_store is not None:
                artifact_store.save(
                    f"raw/outline_resp_a{attempt}.json",
                    {"resp": resp, "content_preview": content_stripped[:2000]},
                )

            if not content_stripped:
                app_logger.warning(f"outline_from_setting: empty content on attempt {attempt}, retrying...")
                continue

            data = await self._parse_json_with_repair(
                raw=content_stripped,
                model_name=model_name,
                operation_name=f"outline_from_setting_parse_a{attempt}",
                schema_hint=schema_hint,
                artifact_store=artifact_store,
            )

            if not isinstance(data, dict) or not data.get("beats"):
                app_logger.warning(f"outline_from_setting: parsed JSON missing beats on attempt {attempt}, retrying...")
                continue

            outline = StoryOutlineFull(**data)
            outline.beats = sorted(outline.beats, key=lambda b: b.order)

            if artifact_store is not None:
                artifact_store.checkpoint("03_outline", outline.dict())
            return outline

        if artifact_store is not None:
            artifact_store.save("raw/outline_failed_last_resp.json", {"resp": last_resp})
        raise ValueError("Failed to generate outline: model returned empty/invalid JSON multiple times")

    async def _plan_branches(
        self,
        outline: StoryOutlineFull,
        max_branches: int,
        tone: Optional[str],
        preferred_ending_types: Optional[List[str]] = None,
        artifact_store: Optional[ArtifactStore] = None,
    ) -> BranchingInfo:
        max_branches = max(1, min(int(max_branches or 1), 5))
        if max_branches <= 1:
            main_spec = BranchSpec(
                id="main",
                from_beat_id=None,
                from_scene_id=None,
                kind="route",
                title="Основной маршрут",
                description="Каноничная история с хорошей, логичной развязкой.",
                ending_tone="good",
                is_canonical=True,
            )
            bi = BranchingInfo(max_branches=1, branches=[main_spec])
            if artifact_store is not None:
                artifact_store.checkpoint("05_branching", bi.dict())
            return bi

        model_name = self.router.get_model_for_agent("outline_agent")
        payload: Dict[str, Any] = {
            "beats": [b.dict() for b in outline.beats],
            "max_branches": max_branches,
            "tone": tone or "balanced",
            "preferred_ending_types": preferred_ending_types or [],
        }

        resp = await self.client.generate_completion(
            model=model_name,
            temperature=0.3,
            system_prompt=branch_planner_prompt,
            prompt=json.dumps(payload, ensure_ascii=False),
            response_format={"type": "json_object"},
            operation_name="branch_planner",
        )

        raw = (resp["choices"][0]["message"]["content"] or "").strip()
        schema_hint = '{"branches":[{"from_beat_id":"beat_06","title":"...","description":"...","ending_tone":"neutral"}]}'
        data = await self._parse_json_with_repair(
            raw,
            model_name,
            "branch_planner_parse",
            schema_hint,
            artifact_store=artifact_store,
        )

        branch_specs_raw = data.get("branches") or []
        if not isinstance(branch_specs_raw, list):
            branch_specs_raw = []

        branches: List[BranchSpec] = [
            BranchSpec(
                id="main",
                from_beat_id=None,
                from_scene_id=None,
                kind="route",
                title="Основной маршрут",
                description="Каноничная история с хорошей, логичной развязкой.",
                ending_tone="good",
                is_canonical=True,
            )
        ]

        for idx, br in enumerate(branch_specs_raw[: max(0, max_branches - 1)], start=1):
            if not isinstance(br, dict):
                continue
            from_beat_id = br.get("from_beat_id")
            if not from_beat_id:
                continue
            branches.append(
                BranchSpec(
                    id=f"branch_{idx:02d}",
                    from_beat_id=str(from_beat_id),
                    from_scene_id=None,
                    kind="ending",
                    title=str(br.get("title") or f"Ветка {idx}"),
                    description=str(br.get("description") or ""),
                    ending_tone=str(br.get("ending_tone") or "neutral"),
                    is_canonical=False,
                )
            )

        bi = BranchingInfo(max_branches=max_branches, branches=branches)
        if artifact_store is not None:
            artifact_store.checkpoint("05_branching", bi.dict())
        return bi


    async def _ensure_char_list(
        self,
        user_prompt: str,
        outline: StoryOutlineFull,
        char_list: List[str] | None,
    ) -> List[str]:
        if char_list:
            return char_list

        app_logger.info("No char_list provided. Generating character names...")
        model_name = self.router.get_model_for_agent("char_agent")

        system_prompt = (
            "Ты придумываешь список имён персонажей для визуальной новеллы.\n"
            'Выведи строгий JSON-объект вида:\n{"characters": ["Имя1", "Имя2", "..."]}\n'
            "Не используй Markdown, только JSON."
        )

        payload = {"user_prompt": user_prompt, "beats": [b.dict() for b in outline.beats]}

        resp = await self.client.generate_completion(
            model=model_name,
            temperature=0.6,
            system_prompt=system_prompt,
            prompt=json.dumps(payload, ensure_ascii=False),
            response_format={"type": "json_object"},
            operation_name="char_name_list",
        )

        data = json.loads(resp["choices"][0]["message"]["content"])
        chars = data.get("characters", [])
        if not isinstance(chars, list) or not chars:
            raise ValueError("Failed to generate character list")
        return [str(x) for x in chars]

    async def _ensure_loc_list(
        self,
        user_prompt: str,
        outline: StoryOutlineFull,
        loc_list: List[str] | None,
    ) -> List[str]:
        if loc_list:
            return loc_list

        app_logger.info("No loc_list provided. Generating location names...")
        model_name = self.router.get_model_for_agent("loc_agent")

        system_prompt = (
            "Ты придумываешь список локаций для визуальной новеллы.\n"
            'Выведи строгий JSON-объект вида:\n{"locations": ["Локация1", "Локация2", "..."]}\n'
            "Не используй Markdown, только JSON."
        )

        payload = {"user_prompt": user_prompt, "beats": [b.dict() for b in outline.beats]}

        resp = await self.client.generate_completion(
            model=model_name,
            temperature=0.6,
            system_prompt=system_prompt,
            prompt=json.dumps(payload, ensure_ascii=False),
            response_format={"type": "json_object"},
            operation_name="loc_name_list",
        )

        data = json.loads(resp["choices"][0]["message"]["content"])
        locs = data.get("locations", [])
        if not isinstance(locs, list) or not locs:
            raise ValueError("Failed to generate location list")
        return [str(x) for x in locs]

    async def _run_char_agent(
        self,
        char_list: List[str],
        setting: Setting,
        hints: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        app_logger.info("Running char_agent...")
        char_toolbox = Toolbox().add_tool_schema("schema_char_graph").add_tool_schema("schema_char_appearance")
        char_supports = self.router.supports_function_calling("char_agent")

        char_agent = Agent(
            api_key=self.client.api_key,
            model=self.router.get_model_for_agent("char_agent"),
            temperature=0.1,
            toolbox=char_toolbox,
            freedom="required",
            reasoning="high",
            supports_tools=char_supports,
        ).set_role(char_prompt)

        char_agent.set_input(char_list=char_list, setting=setting.setting, hints=hints or {})
        return await char_agent.run(self.client)

    async def _run_loc_agent(
        self,
        loc_list: List[str],
        setting: Setting,
    ) -> Dict[str, Any]:
        app_logger.info("Running loc_agent...")
        loc_toolbox = Toolbox().add_tool_schema("schema_loc_description").add_tool_schema("schema_loc_graph")
        loc_supports = self.router.supports_function_calling("loc_agent")

        loc_agent = Agent(
            api_key=self.client.api_key,
            model=self.router.get_model_for_agent("loc_agent"),
            temperature=0.1,
            toolbox=loc_toolbox,
            freedom="required",
            reasoning="high",
            supports_tools=loc_supports,
        ).set_role(loc_prompt)

        loc_agent.set_input(loc_list=loc_list, setting=setting.setting)
        return await loc_agent.run(self.client)


    def _beats_for_main_route(self, outline: StoryOutlineFull) -> StoryOutlineFull:
        beats_sorted = sorted(outline.beats, key=lambda b: b.order)
        first_res_idx: Optional[int] = None
        for idx, b in enumerate(beats_sorted):
            p = (b.purpose or "").lower()
            if any(key in p for key in ("resolution", "ending", "epilogue")):
                first_res_idx = idx
                break
        used = beats_sorted if first_res_idx is None else beats_sorted[: first_res_idx + 1]
        return StoryOutlineFull(theory=outline.theory, beats=used)

    async def _generate_scene_contracts_main(
        self,
        outline: StoryOutlineFull,
        char_list: List[str],
        loc_list: List[str],
        story_length: str,
        artifact_store: Optional[ArtifactStore] = None,
    ) -> List[SceneContract]:
        app_logger.info("Generating scene contracts (main)...")
        model_name = self.router.get_model_for_agent("outline_agent")
        mc_name = char_list[0] if char_list else None
        outline_main = self._beats_for_main_route(outline)

        payload = {
            "outline": outline_main.dict(),
            "char_list": char_list,
            "loc_list": loc_list,
            "story_length": story_length,
            "mc_name": mc_name,
        }

        resp = await self.client.generate_completion(
            model=model_name,
            temperature=0.4,
            system_prompt=scene_plan_prompt,
            prompt=json.dumps(payload, ensure_ascii=False),
            response_format={"type": "json_object"},
            operation_name="scene_contracts_main",
        )

        raw = (resp["choices"][0]["message"]["content"] or "").strip()
        schema_hint = '{"scenes":[{"beat_id":"beat_01","location":"...","pov_character":"...","present_characters":["..."],"summary":"..."}]}'
        data = await self._parse_json_with_repair(
            raw,
            model_name,
            "scene_contracts_main_parse",
            schema_hint,
            artifact_store=artifact_store,
        )
        raw_scenes = data.get("scenes") or []

        beats_sorted = sorted(outline_main.beats, key=lambda b: b.order)
        if not isinstance(raw_scenes, list) or not raw_scenes:
            pov = mc_name or (char_list[0] if char_list else "Protagonist")
            default_loc = loc_list[0] if loc_list else "Default Location"
            contracts = [
                SceneContract(
                    id=f"scene_{i+1:03d}",
                    beat_id=b.id,
                    location=default_loc,
                    pov_character=pov,
                    present_characters=[pov],
                    summary=b.summary,
                    branch_id="main",
                    branch_order=i + 1,
                )
                for i, b in enumerate(beats_sorted)
            ]
            if artifact_store is not None:
                artifact_store.checkpoint("08_scene_contracts_main", [c.dict() for c in contracts])
            return contracts

        contracts: List[SceneContract] = []
        for idx, s in enumerate(raw_scenes):
            if not isinstance(s, dict):
                continue
            beat_id = s.get("beat_id") or (beats_sorted[idx].id if idx < len(beats_sorted) else beats_sorted[-1].id)
            location = s.get("location") or (loc_list[0] if loc_list else "Default Location")
            pov_character = s.get("pov_character") or (mc_name or (char_list[0] if char_list else "Protagonist"))
            present = s.get("present_characters")
            if not isinstance(present, list) or not present:
                present = [pov_character]
            summary = str(s.get("summary") or "")
            contracts.append(
                SceneContract(
                    id=f"scene_{idx + 1:03d}",
                    beat_id=str(beat_id),
                    location=str(location),
                    pov_character=str(pov_character),
                    present_characters=[str(x) for x in present],
                    summary=summary,
                    branch_id="main",
                    branch_order=idx + 1,
                )
            )

        if artifact_store is not None:
            artifact_store.checkpoint("08_scene_contracts_main", [c.dict() for c in contracts])
        return contracts

    async def _generate_scene_contracts_for_branch(
        self,
        outline: StoryOutlineFull,
        char_list: List[str],
        loc_list: List[str],
        story_length: str,
        branch: BranchSpec,
    ) -> List[SceneContract]:
        if not branch.from_beat_id:
            return []

        beats_sorted = sorted(outline.beats, key=lambda b: b.order)
        order_map = {b.id: b.order for b in beats_sorted}
        if branch.from_beat_id not in order_map:
            return []

        split_order = order_map[branch.from_beat_id]
        tail_beats = [b for b in beats_sorted if b.order > split_order]
        if not tail_beats:
            return []

        model_name = self.router.get_model_for_agent("outline_agent")
        mc_name = char_list[0] if char_list else None

        outline_tail = StoryOutlineFull(theory=outline.theory, beats=tail_beats)
        payload = {
            "outline": outline_tail.dict(),
            "char_list": char_list,
            "loc_list": loc_list,
            "story_length": story_length,
            "mc_name": mc_name,
            "branch_context": {
                "branch_id": branch.id,
                "from_beat_id": branch.from_beat_id,
                "title": branch.title,
                "description": branch.description,
                "ending_tone": branch.ending_tone,
            },
        }

        resp = await self.client.generate_completion(
            model=model_name,
            temperature=0.4,
            system_prompt=scene_plan_prompt,
            prompt=json.dumps(payload, ensure_ascii=False),
            response_format={"type": "json_object"},
            operation_name=f"scene_contracts_{branch.id}",
        )

        raw = (resp["choices"][0]["message"]["content"] or "").strip()
        schema_hint = '{"scenes":[{"beat_id":"beat_01","location":"...","pov_character":"...","present_characters":["..."],"summary":"..."}]}'
        data = await self._parse_json_with_repair(raw, model_name, f"scene_contracts_{branch.id}_parse", schema_hint)
        raw_scenes = data.get("scenes") or []
        if not isinstance(raw_scenes, list) or not raw_scenes:
            return []

        contracts: List[SceneContract] = []
        for idx, s in enumerate(raw_scenes):
            if not isinstance(s, dict):
                continue
            beat_id = s.get("beat_id") or (tail_beats[idx].id if idx < len(tail_beats) else tail_beats[-1].id)
            location = s.get("location") or (loc_list[0] if loc_list else "Unknown Location")
            pov_character = s.get("pov_character") or mc_name or (char_list[0] if char_list else "Protagonist")
            present = s.get("present_characters")
            if not isinstance(present, list) or not present:
                present = [pov_character]
            summary = str(s.get("summary") or "")
            contracts.append(
                SceneContract(
                    id=f"{branch.id}_scene_{idx + 1:03d}",
                    beat_id=str(beat_id),
                    location=str(location),
                    pov_character=str(pov_character),
                    present_characters=[str(x) for x in present],
                    summary=summary,
                    branch_id=branch.id,
                    branch_order=idx + 1,
                )
            )
        return contracts


    def _min_lines(self, story_length: str) -> int:
        key = f"MIN_SCENE_LINES_{(story_length or 'medium').upper()}"
        env = os.getenv(key)
        if env:
            try:
                return int(env)
            except Exception:
                pass
        return {"short": 30, "medium": 45, "long": 70}.get(story_length or "medium", 45)

    def _writer_max_tokens(self, story_length: str) -> int:
        env = os.getenv("WRITER_MAX_TOKENS")
        if env:
            try:
                return int(env)
            except Exception:
                pass
        return {"short": 6000, "medium": 9000, "long": 14000}.get(story_length or "medium", 9000)

    @staticmethod
    def _extract_last_lines(script: SceneScript, n: int = 3) -> List[str]:
        out: List[str] = []
        if not script or not script.lines:
            return out
        for line in script.lines[-n:]:
            speaker_prefix = f"{line.speaker}: " if line.speaker else ""
            out.append(f"[{line.type}] {speaker_prefix}{line.text}")
        return out

    @staticmethod
    def _is_transition_required(prev_location: Optional[str], cur_location: Optional[str]) -> bool:
        if not prev_location or not cur_location:
            return False
        return str(prev_location) != str(cur_location)

    def _build_scene_context_simple(
        self,
        setting: Setting,
        outline: StoryOutlineFull,
        scene_contract: SceneContract,
        char_appearance_map: Dict[str, str],
        previous_summaries: List[str],
        previous_last_lines: Optional[List[str]],
        loc_canons: Dict[str, str],
        loc_affordances: Dict[str, Dict[str, Any]],
        prev_location: Optional[str] = None,
    ) -> str:
        beat_map = {b.id: b for b in outline.beats}
        beat: Optional[OutlineBeat] = beat_map.get(scene_contract.beat_id)

        lines: List[str] = []
        lines.append("=== WORLD / SETTING ===")
        if setting.setting:
            lines.append(setting.setting)
        if setting.world_rules:
            lines.append("\nWORLD RULES:")
            lines.append(setting.world_rules)

        if beat is not None:
            lines.append("\n=== CURRENT BEAT (TEXT) ===")
            if beat.title:
                lines.append(f"Title: {beat.title}")
            if beat.summary:
                lines.append(beat.summary)

        lines.append("\n=== CURRENT SCENE PLAN ===")
        lines.append(f"Location: {scene_contract.location}")
        lines.append(f"POV character: {scene_contract.pov_character}")
        lines.append(f"Present characters: {', '.join(scene_contract.present_characters)}")
        if scene_contract.summary:
            lines.append(f"Planned scene summary: {scene_contract.summary}")

        transition_required = self._is_transition_required(prev_location, scene_contract.location)
        if transition_required:
            lines.append("\n=== TRANSITION REQUIRED ===")
            lines.append(f"Previous location: {prev_location}")
            lines.append("Include travel/arrival glue early so it doesn't look like teleportation.")

        canon = (loc_canons or {}).get(scene_contract.location, "") or ""
        aff = self._extract_location_aff(loc_affordances, scene_contract.location)
        lines.append("\n=== CURRENT LOCATION CANON (BG) ===")
        lines.append(f"[{scene_contract.location}] {canon}".strip())
        lines.append("\n=== LOCATION AFFORDANCES ===")
        lines.append(json.dumps(aff, ensure_ascii=False))

        lines.append("\n=== CHARACTER NOTES (PRESENT IN SCENE) ===")
        for name in scene_contract.present_characters:
            desc = char_appearance_map.get(name, "")
            lines.append(f"{name}: {desc or '(no detailed description provided)'}")

        if previous_summaries:
            lines.append("\n=== RECENT SCENE SUMMARIES ===")
            for s in previous_summaries[-8:]:
                text = s
                if ":" in text:
                    text = text.split(":", 1)[1].strip()
                lines.append(f"- {text}")

        if previous_last_lines:
            lines.append("\n=== IMMEDIATE CONTEXT (PREVIOUS SCENE END) ===")
            lines.append("Continue smoothly from these exact lines:")
            for l in previous_last_lines:
                lines.append(f"> {l}")

        return "\n".join(lines)

    def _canon_script_inplace(
        self,
        script: SceneScript,
        contract: SceneContract,
        canon: NameCanonicalizer,
    ) -> None:
        pov = contract.pov_character
        present = contract.present_characters or []
        pov_fb = pov if pov else (canon.characters[0] if canon.characters else None)
        present_fb = present[0] if present else pov_fb

        for line in script.lines or []:
            if line.type == "narration":
                line.speaker = None
                continue

            if line.type == "thought":
                rr = canon.canonicalize_character(line.speaker or pov, fallback=pov_fb)
                line.speaker = rr.output
                continue

            rr0 = canon.canonicalize_character(line.speaker, fallback=None)
            if rr0.output is not None:
                line.speaker = rr0.output
                continue

            rr2 = canon.canonicalize_character(line.speaker, fallback=present_fb)
            line.speaker = rr2.output

    async def _write_single_scene_simple(
        self,
        contract: SceneContract,
        setting: Setting,
        outline: StoryOutlineFull,
        char_appearance_map: Dict[str, str],
        previous_summaries: List[str],
        previous_last_lines: Optional[List[str]],
        story_length: str,
        prev_location: Optional[str],
        loc_canons: Dict[str, str],
        loc_affordances: Dict[str, Dict[str, Any]],
        name_canon: Optional[NameCanonicalizer] = None,
        artifact_store: Optional[ArtifactStore] = None,
    ) -> SceneScript:
        writer_model = self.router.get_model_for_agent("writer_agent")
        min_lines = self._min_lines(story_length)
        transition_required = self._is_transition_required(prev_location, contract.location)

        base_context_text = self._build_scene_context_simple(
            setting=setting,
            outline=outline,
            scene_contract=contract,
            char_appearance_map=char_appearance_map,
            previous_summaries=previous_summaries,
            previous_last_lines=previous_last_lines,
            loc_canons=loc_canons,
            loc_affordances=loc_affordances,
            prev_location=prev_location,
        )

        payload: Dict[str, Any] = {
            "base_context_text": base_context_text,
            "scene_contract": contract.dict(),
            "story_length": story_length,
            "min_lines": min_lines,
            "prev_location": prev_location,
            "transition_required": transition_required,
            "location_canon": (loc_canons.get(contract.location) or ""),
            "location_affordances": self._extract_location_aff(loc_affordances, contract.location),
        }

        if artifact_store is not None:
            artifact_store.save(
                f"context_simple/{contract.branch_id}/{contract.id}.json",
                {"scene_contract": contract.dict(), "payload": payload},
            )

        resp = await self.client.generate_completion(
            model=writer_model,
            temperature=0.7,
            system_prompt=writer_prompt_simple,
            prompt=json.dumps(payload, ensure_ascii=False),
            response_format={"type": "json_object"},
            operation_name=f"write_scene_simple_{contract.id}",
            max_tokens=self._writer_max_tokens(story_length),
        )

        raw = (resp["choices"][0]["message"]["content"] or "").strip()
        schema_hint_scene = '{ "scene_id":"scene_001", "lines":[{"type":"dialogue","speaker":"Имя","text":"..."}], "summary":"..." }'
        data = await self._parse_json_with_repair(
            raw=raw,
            model_name=writer_model,
            operation_name=f"write_scene_simple_{contract.id}_parse",
            schema_hint=schema_hint_scene,
            artifact_store=artifact_store,
        )

        try:
            script = SceneScript(**data)
        except ValidationError:
            fb_text = "Техническая сцена-заглушка: baseline-пайплайн не смог сформировать валидный SceneScript."
            script = SceneScript(
                scene_id=contract.id,
                branch_id=contract.branch_id,
                branch_order=contract.branch_order,
                lines=[SceneLine(type="narration", speaker=None, text=fb_text)],
                summary=contract.summary or fb_text,
            )

        script.scene_id = contract.id
        script.branch_id = contract.branch_id
        script.branch_order = contract.branch_order

        for line in script.lines:
            if line.type == "narration":
                line.speaker = None
            if line.type == "thought" and not line.speaker:
                line.speaker = contract.pov_character

        if name_canon is not None:
            self._canon_script_inplace(script, contract, name_canon)

        return script

    async def _write_scenes_simple(
        self,
        setting: Setting,
        outline: StoryOutlineFull,
        scene_contracts: List[SceneContract],
        char_list: List[str],
        char_appearance: Dict[str, Any] | CharacterAppearance | None,
        story_length: str,
        initial_previous_summaries: Optional[List[str]] = None,
        initial_previous_last_lines: Optional[List[str]] = None,
        initial_prev_location: Optional[str] = None,
        loc_canons: Optional[Dict[str, str]] = None,
        loc_affordances: Optional[Dict[str, Dict[str, Any]]] = None,
        artifact_store: Optional[ArtifactStore] = None,
        artifact_prefix: str = "main",
        name_canon: Optional[NameCanonicalizer] = None,
    ) -> Dict[str, SceneScript]:
        app_logger.info(f"[Simple] Writing {len(scene_contracts)} scenes... story_length={story_length}")

        loc_canons = loc_canons or {}
        loc_affordances = loc_affordances or {}

        if isinstance(char_appearance, CharacterAppearance):
            char_appearance_model = char_appearance
        elif isinstance(char_appearance, dict) and "descriptions" in char_appearance:
            char_appearance_model = CharacterAppearance(**char_appearance)
        else:
            char_appearance_model = CharacterAppearance(descriptions=[""] * len(char_list))

        char_appearance_map: Dict[str, str] = {}
        for name, desc in zip(char_list, char_appearance_model.descriptions):
            char_appearance_map[name] = desc

        scene_scripts: Dict[str, SceneScript] = {}
        previous_summaries: List[str] = list(initial_previous_summaries or [])
        last_scene_lines_buffer: List[str] = list(initial_previous_last_lines or [])
        prev_location: Optional[str] = initial_prev_location

        for contract in scene_contracts:
            if artifact_store is not None:
                artifact_store.event(
                    "scene.start.simple",
                    {
                        "scene_id": contract.id,
                        "branch_id": contract.branch_id,
                        "order": contract.branch_order,
                        "summary_plan": contract.summary,
                        "location": contract.location,
                        "prev_location": prev_location,
                    },
                )

            script = await self._write_single_scene_simple(
                contract=contract,
                setting=setting,
                outline=outline,
                char_appearance_map=char_appearance_map,
                previous_summaries=previous_summaries,
                previous_last_lines=last_scene_lines_buffer,
                story_length=story_length,
                prev_location=prev_location,
                loc_canons=loc_canons,
                loc_affordances=loc_affordances,
                name_canon=name_canon,
                artifact_store=artifact_store,
            )

            scene_scripts[contract.id] = script
            previous_summaries.append(f"{contract.id}: {script.summary}")

            last_scene_lines_buffer = self._extract_last_lines(script, n=3)
            prev_location = contract.location

            app_logger.info(f"[Simple] Scene {contract.id} written, lines={len(script.lines)}")

            if artifact_store is not None:
                artifact_store.save(f"scenes_simple/{artifact_prefix}/{contract.id}.json", script.dict())
                artifact_store.event("scene.done.simple", {"scene_id": contract.id, "branch_id": contract.branch_id, "lines": len(script.lines)})

        return scene_scripts


    def _inject_branch_choices(
        self,
        main_contracts: List[SceneContract],
        main_scripts: Dict[str, SceneScript],
        branch_contracts: List[SceneContract],
        branching: BranchingInfo,
    ) -> None:
        beat_to_main_indices: Dict[str, int] = {}
        for idx, sc in enumerate(main_contracts):
            beat_to_main_indices[sc.beat_id] = idx

        branch_first_scene: Dict[str, str] = {}
        for bc in branch_contracts:
            if bc.branch_order == 1:
                branch_first_scene[bc.branch_id] = bc.id

        branch_ids = {c.branch_id for c in branch_contracts if c.branch_id != "main"}
        num_endings = 1 + len(branch_ids)
        total_fake_options = num_endings * random.randint(1, 2)
        fake_options_remaining = total_fake_options

        main_choice_text = "Следовать уже намеченному пути."
        branch_choice_variants = [
            "Рискнуть и изменить ход событий.",
            "Выбрать неожиданный путь.",
            "Сделать шаг в неизвестность.",
            "Пойти наперекор очевидному решению.",
        ]
        fake_choice_variants = [
            "Промолчать и просто наблюдать за происходящим.",
            "Сменить тему и уйти от прямого ответа.",
            "Сделать паузу, не принимая окончательного решения.",
        ]

        divergence_to_branches: Dict[str, List[BranchSpec]] = {}
        for br in branching.branches:
            if br.id == "main" or not br.from_beat_id:
                continue
            if br.from_beat_id not in beat_to_main_indices:
                continue
            split_idx = beat_to_main_indices[br.from_beat_id]
            divergence_scene_id = main_contracts[split_idx].id
            br.from_scene_id = divergence_scene_id
            divergence_to_branches.setdefault(divergence_scene_id, []).append(br)

        for divergence_scene_id, branches_at_scene in divergence_to_branches.items():
            script = main_scripts.get(divergence_scene_id)
            if not script:
                continue

            idx = next((i for i, sc in enumerate(main_contracts) if sc.id == divergence_scene_id), None)
            if idx is None:
                continue

            if idx + 1 >= len(main_contracts):
                continue

            next_main_scene_id = main_contracts[idx + 1].id

            choice_id = f"choice_{len(script.choices) + 1:02d}"
            appears_after_line = max(0, len(script.lines) - 1)

            options: List[SceneChoiceOption] = [
                SceneChoiceOption(
                    id="opt_main",
                    text=main_choice_text,
                    leads_to_scene_id=next_main_scene_id,
                    leads_to_branch_id="main",
                    is_fake=False,
                )
            ]

            for br in branches_at_scene:
                first_branch_scene_id = branch_first_scene.get(br.id)
                if not first_branch_scene_id:
                    continue
                options.append(
                    SceneChoiceOption(
                        id=f"opt_{br.id}",
                        text=random.choice(branch_choice_variants),
                        leads_to_scene_id=first_branch_scene_id,
                        leads_to_branch_id=br.id,
                        is_fake=False,
                    )
                )

            if fake_options_remaining > 0:
                fake_here = 1 if random.random() < 0.5 else 0
                for _ in range(fake_here):
                    options.append(
                        SceneChoiceOption(
                            id=f"opt_fake_{len(options) + 1}",
                            text=random.choice(fake_choice_variants),
                            leads_to_scene_id=next_main_scene_id,
                            leads_to_branch_id=None,
                            is_fake=True,
                        )
                    )
                fake_options_remaining -= fake_here

            if len(options) <= 1:
                continue

            script.choices.append(SceneChoice(id=choice_id, appears_after_line=appears_after_line, options=options))


    def _ensure_dir(self, p: Path) -> None:
        p.mkdir(parents=True, exist_ok=True)

    async def _save_image_any_url(self, url: str, out_path: Path) -> None:
        if url.startswith("data:"):
            header, b64 = url.split(",", 1)
            out_path.write_bytes(base64.b64decode(b64))
            return
        async with httpx.AsyncClient(timeout=120, trust_env=False) as c:
            r = await c.get(url)
            r.raise_for_status()
            out_path.write_bytes(r.content)

    async def _generate_char_images(self, generation_id: str, char_list: List[str], char_appearance: Any) -> List[CharacterImage]:
        out_dir = Path(os.getenv("OUTPUT_DIR", "output")) / generation_id / "images" / "characters"
        self._ensure_dir(out_dir)

        descriptions: List[str] = []
        if isinstance(char_appearance, dict):
            descriptions = char_appearance.get("descriptions") or []
        elif isinstance(char_appearance, CharacterAppearance):
            descriptions = char_appearance.descriptions
        if not descriptions:
            descriptions = [""] * len(char_list)

        images: List[CharacterImage] = []
        for idx, name in enumerate(char_list, start=1):
            desc = descriptions[idx - 1] if idx - 1 < len(descriptions) else ""
            prompt = f"Character sprite: {name}\n{desc}".strip()

            img_obj = await self.client.generate_image(
                prompt=prompt,
                aspect_ratio="1:1",
                operation_name=f"char_image_{generation_id}_{idx:03d}",
            )
            url = (img_obj.get("image_url") or {}).get("url")
            if not url:
                continue

            safe = re.sub(r"[^0-9A-Za-zА-Яа-яЁё_-]+", "_", name)[:40]
            file_path = out_dir / f"char_{idx:03d}_{safe}.png"
            await self._save_image_any_url(url, file_path)

            images.append(CharacterImage(id=f"char_{idx:03d}", character=name, path=str(file_path), aspect_ratio="1:1"))
        return images

    async def _generate_loc_images(self, generation_id: str, loc_list: List[str], loc_description: Any) -> List[LocationImage]:
        out_dir = Path(os.getenv("OUTPUT_DIR", "output")) / generation_id / "images" / "locations"
        self._ensure_dir(out_dir)

        descriptions: List[str] = []
        if isinstance(loc_description, dict):
            descriptions = loc_description.get("descriptions") or []
        elif isinstance(loc_description, LocationDescription):
            descriptions = loc_description.descriptions
        if not descriptions:
            descriptions = [""] * len(loc_list)

        images: List[LocationImage] = []
        for idx, name in enumerate(loc_list, start=1):
            desc = descriptions[idx - 1] if idx - 1 < len(descriptions) else ""
            prompt = f"Background location: {name}\n{desc}".strip()

            img_obj = await self.client.generate_image(
                prompt=prompt,
                aspect_ratio="16:9",
                operation_name=f"loc_image_{generation_id}_{idx:03d}",
            )
            url = (img_obj.get("image_url") or {}).get("url")
            if not url:
                continue

            safe = re.sub(r"[^0-9A-Za-zА-Яа-яЁё_-]+", "_", name)[:40]
            file_path = out_dir / f"bg_{idx:03d}_{safe}.png"
            await self._save_image_any_url(url, file_path)

            images.append(LocationImage(id=f"bg_{idx:03d}", location=name, path=str(file_path), aspect_ratio="16:9"))
        return images


    async def generate_vn(
        self,
        user_prompt: str,
        story_length: str = "medium",
        char_list: list | None = None,
        loc_list: list | None = None,
        setting: str | None = "",
        max_branches: int | None = None,
        tone: str | None = None,
        artstyle: str | None = None,
        generate_images: bool = True,
        time_choice: Optional[str] = None,
        genre_choice: Optional[str] = None,
        tone_choice_ru: Optional[str] = None,
        mc_name: Optional[str] = None,
        mc_description: Optional[str] = None,
        extra_character_names: Optional[List[str]] = None,
        plot_prefs: Optional[PlotPreferences] = None,
        plot_freeform: Optional[str] = None,
        graphic_style_ru: Optional[str] = None,
    ) -> Dict[str, Any]:
        generation_id = f"vn_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        app_logger.info(f"[Simple] Starting generation {generation_id}")

        seed = int(hashlib.sha256(generation_id.encode("utf-8")).hexdigest()[:8], 16)
        random.seed(seed)

        output_root = Path(os.getenv("OUTPUT_DIR", "output")) / generation_id
        store = ArtifactStore(output_root)

        token = TRACE_HOOK.set(lambda evt: store.event(evt.get("type", "trace"), evt))

        try:
            def map_tone_ru_to_en(val: Optional[str]) -> Optional[str]:
                if not val:
                    return None
                m = {"веселый": "light", "грустный": "dark"}
                return m.get(val.strip().lower())

            def map_graphic_style(val: Optional[str]) -> Optional[str]:
                if not val:
                    return None
                m = {"аниме": "anime", "реализм": "realistic", "рисованная графика": "illustrative"}
                return m.get(val.strip().lower())

            explicit_tone = map_tone_ru_to_en(tone_choice_ru) or tone
            explicit_artstyle = map_graphic_style(graphic_style_ru) or artstyle

            store.checkpoint(
                "00_start_request",
                {
                    "generation_id": generation_id,
                    "seed": seed,
                    "user_prompt": user_prompt,
                    "story_length": story_length,
                    "max_branches": max_branches,
                    "generate_images": generate_images,
                    "pipeline_variant": "worse",
                    "disabled": {
                        "rag": True,
                        "critic": True,
                        "multi_candidate": True,
                        "microplanner": True,
                        "scene_editor": True,
                        "contract_location_critic": True,
                    },
                },
            )

            user_request = await self._normalize_user_request(
                raw_prompt=user_prompt,
                explicit_length=story_length,
                explicit_tone=explicit_tone,
                explicit_artstyle=explicit_artstyle,
                explicit_max_branches=max_branches,
                artifact_store=store,
            )

            normalized_length = user_request.story_length or "medium"
            branches_max = user_request.max_branches or 1

            setting_obj = await self._generate_setting(
                user_prompt=user_request.user_prompt,
                setting_override=setting if setting else None,
                time_choice=time_choice,
                genre_choice=genre_choice,
                artifact_store=store,
            )

            outline_obj = await self._generate_outline(
                user_prompt=user_request.user_prompt,
                story_length=normalized_length,
                setting=setting_obj,
                plot_prefs=plot_prefs,
                plot_freeform=plot_freeform,
                artifact_store=store,
            )

            preferred_endings = plot_prefs.ending_types if plot_prefs and plot_prefs.ending_types else None
            branching_info = await self._plan_branches(
                outline=outline_obj,
                max_branches=branches_max,
                tone=user_request.tone,
                preferred_ending_types=preferred_endings,
                artifact_store=store,
            )

            seed_char_list: Optional[List[str]] = None
            if mc_name:
                seed_char_list = [mc_name]
                if extra_character_names:
                    seed_char_list.extend(extra_character_names)

            effective_char_list = char_list or seed_char_list
            char_list_final = await self._ensure_char_list(user_request.user_prompt, outline_obj, effective_char_list)
            if mc_name and mc_name in char_list_final:
                char_list_final = [mc_name] + [n for n in char_list_final if n != mc_name]

            loc_list_final = await self._ensure_loc_list(user_request.user_prompt, outline_obj, loc_list)

            limit_user = str(os.getenv("LIMIT_USER_LOC_LIST", "false")).strip().lower() in ("1", "true", "yes", "on")
            is_user_loc_list = loc_list is not None
            if (not is_user_loc_list) or limit_user:
                loc_list_final = self._limit_locations(
                    candidates=loc_list_final,
                    story_length=normalized_length,
                    artifact_store=store,
                    reason="user_loc_list" if is_user_loc_list else "auto_loc_list",
                )

            name_canon = NameCanonicalizer(characters=char_list_final, locations=loc_list_final)

            hints: Dict[str, str] = {}
            if mc_name and mc_description:
                hints[mc_name] = mc_description

            char_result = await self._run_char_agent(char_list_final, setting_obj, hints=hints or None)
            char_results_map = char_result.get("results", {}) or {}
            char_graph = char_results_map.get("char_graph")
            char_appearance = char_results_map.get("char_appearance")

            if hasattr(char_graph, "model_dump"):
                char_graph = char_graph.model_dump()
            if hasattr(char_appearance, "model_dump"):
                char_appearance = char_appearance.model_dump()

            store.checkpoint(
                "06_characters",
                {"char_list": char_list_final, "char_graph": char_graph, "char_appearance": char_appearance},
            )

            loc_result = await self._run_loc_agent(loc_list_final, setting_obj)
            loc_results_map = loc_result.get("results", {}) or {}
            loc_graph = loc_results_map.get("loc_graph")
            loc_description = loc_results_map.get("loc_description")

            if hasattr(loc_graph, "model_dump"):
                loc_graph = loc_graph.model_dump()
            if hasattr(loc_description, "model_dump"):
                loc_description = loc_description.model_dump()

            store.checkpoint(
                "07_locations",
                {"loc_list": loc_list_final, "loc_graph": loc_graph, "loc_description": loc_description},
            )

            loc_canons = self._build_loc_desc_map(loc_list_final, loc_description)
            loc_affordances = await self._infer_location_affordances(setting_obj, loc_list_final, loc_canons, artifact_store=store)
            store.checkpoint("07c_location_kb", {"loc_canons": loc_canons, "loc_affordances": loc_affordances})

            #images
            char_images: List[CharacterImage] = []
            loc_images: List[LocationImage] = []
            if generate_images:
                try:
                    char_images = await self._generate_char_images(generation_id, char_list_final, char_appearance)
                except Exception as e:
                    app_logger.error(f"Char image generation failed: {e}", exc_info=True)
                try:
                    loc_images = await self._generate_loc_images(generation_id, loc_list_final, loc_description)
                except Exception as e:
                    app_logger.error(f"Loc image generation failed: {e}", exc_info=True)
            else:
                app_logger.info("[Simple] Image generation disabled.")

            main_contracts = await self._generate_scene_contracts_main(
                outline_obj, char_list_final, loc_list_final, normalized_length, artifact_store=store
            )


            main_scripts = await self._write_scenes_simple(
                setting=setting_obj,
                outline=outline_obj,
                scene_contracts=main_contracts,
                char_list=char_list_final,
                char_appearance=char_appearance,
                story_length=normalized_length,
                initial_previous_summaries=None,
                initial_previous_last_lines=None,
                initial_prev_location=None,
                loc_canons=loc_canons,
                loc_affordances=loc_affordances,
                artifact_store=store,
                artifact_prefix="main",
                name_canon=name_canon,
            )


            branch_contracts: List[SceneContract] = []
            branch_scripts: Dict[str, SceneScript] = {}

            if len(branching_info.branches) > 1:
                beat_to_main_scene: Dict[str, str] = {}
                for sc in main_contracts:
                    beat_to_main_scene[sc.beat_id] = sc.id

                for br in branching_info.branches:
                    if br.id == "main" or not br.from_beat_id:
                        continue

                    br_contracts = await self._generate_scene_contracts_for_branch(
                        outline_obj, char_list_final, loc_list_final, normalized_length, br
                    )
                    if not br_contracts:
                        continue

                    branch_contracts.extend(br_contracts)

                    divergence_scene_id = beat_to_main_scene.get(br.from_beat_id)
                    initial_prev: List[str] = []
                    initial_last_lines: List[str] = []
                    initial_prev_location: Optional[str] = None

                    if divergence_scene_id:
                        for sc in main_contracts:
                            initial_prev.append(f"{sc.id}: {main_scripts[sc.id].summary}")
                            if sc.id == divergence_scene_id:
                                initial_last_lines = self._extract_last_lines(main_scripts[divergence_scene_id], n=3)
                                initial_prev_location = sc.location
                                break

                    br_scripts = await self._write_scenes_simple(
                        setting=setting_obj,
                        outline=outline_obj,
                        scene_contracts=br_contracts,
                        char_list=char_list_final,
                        char_appearance=char_appearance,
                        story_length=normalized_length,
                        initial_previous_summaries=initial_prev,
                        initial_previous_last_lines=initial_last_lines,
                        initial_prev_location=initial_prev_location,
                        loc_canons=loc_canons,
                        loc_affordances=loc_affordances,
                        artifact_store=store,
                        artifact_prefix=br.id,
                        name_canon=name_canon,
                    )

                    branch_scripts.update(br_scripts)

                self._inject_branch_choices(main_contracts, main_scripts, branch_contracts, branching_info)

            all_contracts = main_contracts + branch_contracts
            all_scripts: Dict[str, SceneScript] = {}
            all_scripts.update(main_scripts)
            all_scripts.update(branch_scripts)

            story_state_main = StoryState()
            story_state_by_branch: Dict[str, Any] = {"main": story_state_main.dict()}

            result: Dict[str, Any] = {
                "generation_id": generation_id,
                "status": "completed",
                "user_prompt": user_request.user_prompt,
                "story_length": normalized_length,
                "user_request": user_request.dict(),
                "setting": setting_obj.dict(),
                "outline": outline_obj.dict(),
                "char_list": char_list_final,
                "char_graph": char_graph,
                "char_appearance": char_appearance,
                "loc_list": loc_list_final,
                "loc_graph": loc_graph,
                "loc_description": loc_description,
                "loc_canons": loc_canons,
                "loc_affordances": loc_affordances,
                "char_images": [img.dict() for img in char_images],
                "loc_images": [img.dict() for img in loc_images],
                "branching": branching_info.dict(),
                "scene_contracts": [sc.dict() for sc in all_contracts],
                "scenes": {sid: script.dict() for sid, script in all_scripts.items()},
                "story_state_main": story_state_main.dict(),
                "story_state_by_branch": story_state_by_branch,
            }

            store.save("final.json", result)
            store.checkpoint(
                "99_final_summary",
                {
                    "generation_id": generation_id,
                    "status": "completed",
                    "scenes_total": len(all_scripts),
                    "tokens_used": self.client.total_tokens_used,
                    "api_calls": self.client.call_count,
                    "pipeline_variant": "worse",
                },
            )

            return result

        finally:
            TRACE_HOOK.reset(token)