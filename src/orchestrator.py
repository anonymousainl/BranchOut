from __future__ import annotations

from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import json
from json import JSONDecodeError
import os
import math
import base64
import re
from pathlib import Path
import random
from collections import Counter, defaultdict, deque
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
    writer_prompt,
    rag_context_prompt,
    critic_prompt,
    user_request_prompt,
    character_critic_prompt,
    branch_planner_prompt,
    plot_thread_extractor_prompt,
    scene_microplanner_prompt,
    scene_editor_prompt,
    location_affordance_prompt,
    contract_location_critic_prompt,
)


def _sha1(text: str) -> str:
    return hashlib.sha1((text or "").encode("utf-8")).hexdigest()


class RAGIndex:
    def __init__(self, client: OpenRouterClient, embed_model: Optional[str] = None):
        self.client = client
        self.embed_model = embed_model
        self.items: List[Dict[str, Any]] = []
        self._df: Dict[str, int] = defaultdict(int)
        self._doc_len_sum: int = 0

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        if not text:
            return []
        toks = re.findall(r"[0-9A-Za-zА-Яа-яЁё_]+", text.lower())
        return [t for t in toks if len(t) > 1]

    @staticmethod
    def _cosine(a: List[float], b: List[float]) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = 0.0
        na = 0.0
        nb = 0.0
        for x, y in zip(a, b):
            dot += x * y
            na += x * x
            nb += y * y
        if na == 0.0 or nb == 0.0:
            return 0.0
        return float(dot / (math.sqrt(na) * math.sqrt(nb)))

    def _bm25_score(self, q_tokens: List[str], item: Dict[str, Any], k1: float = 1.2, b: float = 0.75) -> float:
        N = len(self.items)
        if N <= 0:
            return 0.0
        avgdl = (self._doc_len_sum / N) if N else 1.0
        if avgdl <= 0.0:
            avgdl = 1.0

        tf: Counter = item.get("tf") or Counter()
        dl: int = len(item.get("tokens") or [])
        if dl <= 0:
            dl = 1

        score = 0.0
        for t in q_tokens:
            df = self._df.get(t, 0)
            idf = math.log((N - df + 0.5) / (df + 0.5) + 1.0)
            f = tf.get(t, 0)
            if f <= 0:
                continue
            denom = f + k1 * (1.0 - b + b * (dl / avgdl))
            score += idf * (f * (k1 + 1.0) / denom)
        return float(score)

    def _remove_stats(self, tokens: List[str]) -> None:
        uniq = set(tokens)
        for t in uniq:
            self._df[t] = max(0, int(self._df.get(t, 0)) - 1)
        self._doc_len_sum = max(0, self._doc_len_sum - len(tokens))

    def _add_stats(self, tokens: List[str]) -> None:
        uniq = set(tokens)
        for t in uniq:
            self._df[t] += 1
        self._doc_len_sum += len(tokens)

    async def upsert_item(self, item_id: str, kind: str, text: str) -> None:
        if not text or not text.strip():
            return

        for idx, it in enumerate(list(self.items)):
            if it.get("id") == item_id and it.get("kind") == kind:
                self._remove_stats(it.get("tokens") or [])
                self.items.pop(idx)
                break

        tokens = self._tokenize(text)
        tf = Counter(tokens)

        embedding: List[float] = []
        if self.embed_model:
            embedding = await self.client.generate_embedding(
                text=text,
                model=self.embed_model,
                operation_name=f"embed_{kind}_{item_id}",
            )

        self._add_stats(tokens)
        self.items.append(
            {
                "id": item_id,
                "kind": kind,
                "text": text,
                "embedding": embedding,
                "tokens": tokens,
                "tf": tf,
            }
        )

    async def add_item(self, item_id: str, kind: str, text: str) -> None:
        await self.upsert_item(item_id, kind, text)

    def import_items(self, items: List[Dict[str, Any]], kind_filter: Optional[str] = None) -> None:
        for it in items:
            if not isinstance(it, dict):
                continue
            if kind_filter and it.get("kind") != kind_filter:
                continue
            item_id = it.get("id")
            kind = it.get("kind")
            text = it.get("text")
            if not item_id or not kind or not text:
                continue

            if any(x.get("id") == item_id and x.get("kind") == kind for x in self.items):
                continue

            tokens = it.get("tokens") or self._tokenize(text)
            tf = it.get("tf") or Counter(tokens)
            emb = it.get("embedding") or []

            self._add_stats(tokens)
            self.items.append(
                {
                    "id": item_id,
                    "kind": kind,
                    "text": text,
                    "embedding": emb,
                    "tokens": tokens,
                    "tf": tf,
                }
            )

    async def query(
        self,
        query_text: str,
        top_k: int = 5,
        kinds: Optional[List[str]] = None,
        alpha: float = 0.7,
        q_tokens: Optional[List[str]] = None,
        q_emb: Optional[List[float]] = None,
    ) -> List[Dict[str, Any]]:
        if not self.items or not (query_text or "").strip():
            return []

        q_tokens = q_tokens or self._tokenize(query_text)

        lex: List[Tuple[float, Dict[str, Any]]] = []
        for item in self.items:
            if kinds and item.get("kind") not in kinds:
                continue
            bm_raw = self._bm25_score(q_tokens, item)
            lex.append((bm_raw, item))

        max_bm = max((s for s, _ in lex), default=0.0)
        if max_bm <= 0.0:
            max_bm = 1.0

        use_emb = bool(self.embed_model)
        if use_emb and q_emb is None:
            q_emb = await self.client.generate_embedding(
                text=query_text,
                model=self.embed_model,
                operation_name="embed_query",
            )

        scored: List[Tuple[float, Dict[str, Any], float, float, float]] = []
        for bm_raw, item in lex:
            bm_norm = float(bm_raw / max_bm)
            cos = 0.0
            if use_emb and q_emb is not None:
                cos = self._cosine(q_emb, item.get("embedding") or [])
            score = alpha * cos + (1.0 - alpha) * bm_norm
            scored.append((score, item, cos, bm_raw, bm_norm))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:top_k]
        return [
            {
                "id": it["id"],
                "kind": it["kind"],
                "text": it["text"],
                "score": float(score),
                "embedding_score": float(cos),
                "lexical_score_raw": float(bm_raw),
                "lexical_score_norm": float(bm_norm),
            }
            for score, it, cos, bm_raw, bm_norm in top
        ]


class RAGBundle:
    def __init__(
        self,
        client: OpenRouterClient,
        embed_model: Optional[str],
        world_index: Optional[RAGIndex] = None,
        char_index: Optional[RAGIndex] = None,
        thread_index: Optional[RAGIndex] = None,
    ):
        self.client = client
        self.embed_model = embed_model
        self.story = RAGIndex(client, embed_model)
        self.world = world_index or RAGIndex(client, embed_model)
        self.characters = char_index or RAGIndex(client, embed_model)
        self.threads = thread_index or RAGIndex(client, embed_model)


class VNOrchestrator:
    def __init__(self, client: OpenRouterClient, router: ModelRouter):
        self.client = client
        self.router = router
    @staticmethod
    def _strict_name_canon_enabled() -> bool:
        val = str(os.getenv("STRICT_NAME_CANON", "true")).strip().lower()
        return val in ("1", "true", "yes", "on")

    @staticmethod
    def _strict_location_gate_enabled() -> bool:
        val = str(os.getenv("STRICT_LOCATION_GATE", "true")).strip().lower()
        return val in ("1", "true", "yes", "on")

    @staticmethod
    def _unwrap_last(v: Any) -> Any:
        return v[-1] if isinstance(v, list) and v else v

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
        user_prompt: str,
        setting: Setting,
        outline: StoryOutlineFull,
        story_length: str,
        artifact_store: Optional[ArtifactStore] = None,
        reason: str = "auto_loc_list",
    ) -> List[str]:
        max_locs = self._max_locations_for_length(story_length)
        uniq = self._dedupe_preserve_order([str(x) for x in (candidates or [])])
        if len(uniq) <= max_locs:
            return uniq

        corpus_parts: List[str] = []
        corpus_parts.append(user_prompt or "")
        corpus_parts.append(setting.setting or "")
        corpus_parts.append(setting.world_rules or "")
        for b in (outline.beats or []):
            corpus_parts.append(b.title or "")
            corpus_parts.append(b.summary or "")
        corpus = "\n".join(corpus_parts).casefold()

        scored: List[Tuple[int, int, str]] = []
        for i, name in enumerate(uniq):
            n = name.casefold()
            score = 0
            if n:
                score += corpus.count(n)
                toks = re.findall(r"[0-9A-Za-zА-Яа-яЁё]+", n)
                for t in toks:
                    if len(t) >= 4:
                        score += 1 if t in corpus else 0
            scored.append((score, -i, name))

        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
        selected = {name for _, _, name in scored[:max_locs]}
        limited = [x for x in uniq if x in selected]

        if artifact_store is not None:
            artifact_store.event(
                "loc_list.limited",
                {
                    "reason": reason,
                    "max_locations": max_locs,
                    "before": len(uniq),
                    "after": len(limited),
                    "dropped": [x for x in uniq if x not in selected][:80],
                },
            )
            artifact_store.save(
                "checkpoints/loc_list_limited_preview.json",
                {"max_locations": max_locs, "limited": limited, "original": uniq[:200]},
            )

        return limited

    def _normalize_outline_order(self, outline: StoryOutlineFull,
                                 artifact_store: Optional[ArtifactStore] = None) -> StoryOutlineFull:
        beats = list(outline.beats or [])
        if not beats:
            return outline

        orders = [b.order for b in beats]
        strictly_increasing = all(orders[i] < orders[i + 1] for i in range(len(orders) - 1))
        unique = len(set(orders)) == len(orders)

        if strictly_increasing and unique:
            return outline
        beats_sorted = sorted(beats, key=lambda b: (b.act, b.order, b.id))
        for idx, b in enumerate(beats_sorted, start=1):
            b.order = idx

        outline.beats = beats_sorted

        if artifact_store is not None:
            artifact_store.event(
                "outline.order_repaired",
                {
                    "reason": "non_unique_or_non_monotonic_order",
                    "original_orders_preview": orders[:60],
                    "new_orders_preview": [b.order for b in beats_sorted[:60]],
                },
            )

        return outline

    def _canon_contract_inplace(
        self,
        contract: SceneContract,
        canon: NameCanonicalizer,
        *,
        store: Optional[ArtifactStore] = None,
    ) -> None:
        loc_fb = canon.locations[0] if canon.locations else None
        rloc = canon.canonicalize_location(contract.location, fallback=loc_fb)
        if rloc.output:
            contract.location = rloc.output
        if store is not None and rloc.status in ("unknown", "fallback"):
            store.event(
                "name.location_canon",
                {"scene_id": contract.id, "input": rloc.input, "output": rloc.output, "status": rloc.status, "detail": rloc.detail},
            )

        pov_fb = canon.characters[0] if canon.characters else None
        rpov = canon.canonicalize_character(contract.pov_character, fallback=pov_fb)
        if rpov.output:
            contract.pov_character = rpov.output
        if store is not None and rpov.status in ("unknown", "fallback"):
            store.event(
                "name.pov_canon",
                {"scene_id": contract.id, "input": rpov.input, "output": rpov.output, "status": rpov.status, "detail": rpov.detail},
            )

        fixed: List[str] = []
        seen = set()
        for x in contract.present_characters or []:
            rr = canon.canonicalize_character(x, fallback=None)
            if rr.output is None:
                if store is not None:
                    store.event(
                        "name.present_drop_unknown",
                        {"scene_id": contract.id, "input": rr.input, "status": rr.status, "detail": rr.detail},
                    )
                continue
            if rr.output in seen:
                continue
            seen.add(rr.output)
            fixed.append(rr.output)

        if contract.pov_character and contract.pov_character not in seen:
            fixed.insert(0, contract.pov_character)

        contract.present_characters = fixed

    def _canon_script_inplace(
        self,
        script: SceneScript,
        contract: SceneContract,
        canon: NameCanonicalizer,
        *,
        store: Optional[ArtifactStore] = None,
    ) -> bool:
        had_unmatched = False

        pov = contract.pov_character
        present = contract.present_characters or []
        pov_fb = pov if pov else (canon.characters[0] if canon.characters else None)
        present_fb = present[0] if present else pov_fb

        for i, line in enumerate(script.lines or []):
            if line.type == "narration":
                line.speaker = None
                continue

            if line.type == "thought":
                rr0 = canon.canonicalize_character(line.speaker or pov, fallback=None)
                if rr0.output is None:
                    had_unmatched = True
                rr = canon.canonicalize_character(line.speaker or pov, fallback=pov_fb)
                line.speaker = rr.output
                if store is not None and rr0.output is None:
                    store.event(
                        "name.thought_speaker_unmatched",
                        {"scene_id": script.scene_id, "line": i, "input": rr0.input, "output": rr.output, "status": rr.status, "detail": rr.detail},
                    )
                continue

            rr0 = canon.canonicalize_character(line.speaker, fallback=None)
            if rr0.output is not None:
                line.speaker = rr0.output
                continue

            had_unmatched = True
            rr2 = canon.canonicalize_character(line.speaker, fallback=present_fb)
            line.speaker = rr2.output
            if store is not None:
                store.event(
                    "name.dialogue_speaker_unmatched_fallback",
                    {"scene_id": script.scene_id, "line": i, "input": rr0.input, "output": rr2.output, "status": rr2.status, "detail": rr2.detail},
                )

        return had_unmatched


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

    async def _patch_scene_contracts_with_location_critic(
        self,
        scene_contracts: List[SceneContract],
        loc_list: List[str],
        loc_canons: Dict[str, str],
        loc_affordances: Dict[str, Dict[str, Any]],
        loc_graph: Optional[Any],
        artifact_store: Optional[ArtifactStore] = None,
        artifact_name: str = "contracts_patch",
    ) -> List[SceneContract]:
        if not scene_contracts:
            return scene_contracts

        app_logger.info("Running contract_location_critic (patch contracts for location consistency)...")
        model_name = self.router.get_model_for_agent("outline_agent")

        payload = {
            "loc_list": loc_list,
            "loc_canons": {k: (v[:1200] if isinstance(v, str) else "") for k, v in (loc_canons or {}).items()},
            "loc_affordances": loc_affordances,
            "loc_graph": loc_graph,
            "scene_contracts": [c.dict() for c in scene_contracts],
        }

        resp = await self.client.generate_completion(
            model=model_name,
            temperature=0.2,
            system_prompt=contract_location_critic_prompt,
            prompt=json.dumps(payload, ensure_ascii=False),
            response_format={"type": "json_object"},
            operation_name=f"contract_location_critic_{artifact_name}",
        )

        raw = (resp["choices"][0]["message"]["content"] or "").strip()
        schema_hint = '{"patches":[{"scene_id":"scene_010","new_location":null,"new_summary":null,"reason":"..."}]}'
        data = await self._parse_json_with_repair(
            raw,
            model_name,
            f"contract_location_critic_{artifact_name}_parse",
            schema_hint,
            artifact_store=artifact_store,
        )
        patches = data.get("patches") or []
        if artifact_store is not None:
            artifact_store.save(f"contracts/{artifact_name}.json", {"patches": patches})

        if not isinstance(patches, list) or not patches:
            return scene_contracts

        by_id: Dict[str, SceneContract] = {c.id: c for c in scene_contracts}
        applied: List[Dict[str, Any]] = []

        for p in patches[:200]:
            if not isinstance(p, dict):
                continue
            sid = str(p.get("scene_id") or "")
            if not sid or sid not in by_id:
                continue
            c = by_id[sid]
            new_loc = p.get("new_location")
            new_sum = p.get("new_summary")
            changed = False

            if isinstance(new_loc, str) and new_loc.strip() and new_loc.strip() in loc_list and new_loc.strip() != c.location:
                c.location = new_loc.strip()
                changed = True
            if isinstance(new_sum, str) and new_sum.strip() and new_sum.strip() != c.summary:
                c.summary = new_sum.strip()
                changed = True

            if changed:
                applied.append({"scene_id": sid, "new_location": c.location, "new_summary": c.summary, "reason": p.get("reason")})

        if artifact_store is not None and applied:
            artifact_store.save(f"contracts/{artifact_name}_applied.json", {"applied": applied})

        return list(by_id.values())


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

    def _ensure_beat_coverage_contracts(
            self,
            *,
            beats: List[OutlineBeat],
            contracts: List[SceneContract],
            branch_id: str,
            id_prefix: str,
            default_loc: str,
            default_pov: str,
    ) -> List[SceneContract]:
        beats_in_order = [b.id for b in sorted(beats, key=lambda x: x.order)]
        by_beat: Dict[str, List[SceneContract]] = {}
        for c in contracts:
            by_beat.setdefault(c.beat_id, []).append(c)

        out: List[SceneContract] = []
        for bid in beats_in_order:
            existing = by_beat.get(bid) or []
            if existing:
                out.extend(sorted(existing, key=lambda x: x.branch_order))
            else:
                beat_obj = next((b for b in beats if b.id == bid), None)
                out.append(
                    SceneContract(
                        id="__TMP__",
                        beat_id=bid,
                        location=default_loc,
                        pov_character=default_pov,
                        present_characters=[default_pov],
                        summary=(beat_obj.summary if beat_obj else ""),
                        branch_id=branch_id,
                        branch_order=0,
                    )
                )

        for i, c in enumerate(out, start=1):
            c.branch_order = i
            c.id = f"{id_prefix}{i:03d}"

        return out

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
            outline = self._normalize_outline_order(outline, artifact_store=artifact_store)
            outline.beats = sorted(outline.beats, key=lambda b: b.order)

            if artifact_store is not None:
                artifact_store.checkpoint("03_outline", outline.dict())
            return outline

        if artifact_store is not None:
            artifact_store.save("raw/outline_failed_last_resp.json", {"resp": last_resp})
        raise ValueError("Failed to generate outline: model returned empty/invalid JSON multiple times")

    async def _extract_plot_threads(
        self,
        user_prompt: str,
        setting: Setting,
        outline: StoryOutlineFull,
        artifact_store: Optional[ArtifactStore] = None,
    ) -> List[Dict[str, Any]]:
        app_logger.info("Extracting plot threads...")
        model_name = self.router.get_model_for_agent("plot_thread_agent")

        payload = {
            "user_prompt": user_prompt,
            "setting": setting.dict(),
            "beats": [b.dict() for b in outline.beats],
        }

        resp = await self.client.generate_completion(
            model=model_name,
            temperature=0.2,
            system_prompt=plot_thread_extractor_prompt,
            prompt=json.dumps(payload, ensure_ascii=False),
            response_format={"type": "json_object"},
            operation_name="plot_thread_extractor",
        )

        raw = (resp["choices"][0]["message"]["content"] or "").strip()
        schema_hint = '{"threads":[{"id":"thread_01","title":"...","description":"...","status":"open","anchors":["beat_01"]}]}'
        data = await self._parse_json_with_repair(
            raw,
            model_name,
            "plot_thread_extractor_parse",
            schema_hint,
            artifact_store=artifact_store,
        )

        threads = data.get("threads") or []
        if not isinstance(threads, list):
            return []

        norm: List[Dict[str, Any]] = []
        for i, t in enumerate(threads[:30], start=1):
            if not isinstance(t, dict):
                continue
            tid = str(t.get("id") or f"thread_{i:02d}")
            norm.append(
                {
                    "id": tid,
                    "title": str(t.get("title") or tid),
                    "description": str(t.get("description") or ""),
                    "status": str(t.get("status") or "open"),
                    "anchors": t.get("anchors") if isinstance(t.get("anchors"), list) else [],
                }
            )

        if artifact_store is not None:
            artifact_store.checkpoint("04_threads", {"threads": norm})
        return norm

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
            main_route_beat_ids = [
                b.id for b in self._trim_beats_to_single_ending(sorted(outline.beats, key=lambda b: b.order))
            ]
            bi = BranchingInfo(
                max_branches=1,
                main_route_beat_ids=main_route_beat_ids,
                branches=[main_spec],
            )
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
        main_route_beat_ids = [
            b.id for b in self._trim_beats_to_single_ending(sorted(outline.beats, key=lambda b: b.order))
        ]
        main_route_set = set(main_route_beat_ids[:-1])

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
            if not from_beat_id or str(from_beat_id) not in main_route_set:
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

        bi = BranchingInfo(
            max_branches=max_branches,
            main_route_beat_ids=main_route_beat_ids,
            branches=branches,
        )
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
        
    def _trim_beats_to_single_ending(self, beats: List[OutlineBeat]) -> List[OutlineBeat]:
        beats_sorted = sorted(beats or [], key=lambda b: b.order)
        if not beats_sorted:
            return beats_sorted
    
        terminal = {"resolution", "epilogue", "финал", "развязка", "эпилог"}
        major_reopen = {
            "turning_point", "rising_action", "conflict", "midpoint", "crisis", "climax",
            "поворот", "конфликт", "кризис", "кульминация",
        }
    
        end_idx, seen_terminal = None, False
        for idx, b in enumerate(beats_sorted):
            p = str(b.purpose or "").strip().casefold()
            if p in terminal:
                seen_terminal = True
                end_idx = idx
            elif seen_terminal and p in major_reopen:
                break
            elif seen_terminal:
                end_idx = idx
    
        return beats_sorted if end_idx is None else beats_sorted[: end_idx + 1]
        
    def _beats_for_main_route(
        self,
        outline: StoryOutlineFull,
        branching: Optional[BranchingInfo] = None,
    ) -> StoryOutlineFull:
        beats_sorted = sorted(outline.beats, key=lambda b: b.order)
        if not beats_sorted:
            return StoryOutlineFull(theory=outline.theory, beats=[])
    
        main_route_beat_ids = getattr(branching, "main_route_beat_ids", None) if branching is not None else None
        if main_route_beat_ids:
            route_set = {str(x).strip() for x in main_route_beat_ids if str(x).strip()}
            used = [b for b in beats_sorted if b.id in route_set]
            if used:
                return StoryOutlineFull(theory=outline.theory, beats=used)
    
        used = self._trim_beats_to_single_ending(beats_sorted)
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
        outline_main = outline
        beats_sorted = sorted(outline_main.beats, key=lambda b: b.order)
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
            default_loc = loc_list[0] if loc_list else "Default Location"
            default_pov = mc_name or (char_list[0] if char_list else "Protagonist")

            contracts = self._ensure_beat_coverage_contracts(
                beats=beats_sorted,
                contracts=contracts,
                branch_id="main",
                id_prefix="scene_",
                default_loc=default_loc,
                default_pov=default_pov,
            )

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

        default_loc = loc_list[0] if loc_list else "Default Location"
        default_pov = mc_name or (char_list[0] if char_list else "Protagonist")

        contracts = self._ensure_beat_coverage_contracts(
            beats=beats_sorted,
            contracts=contracts,
            branch_id="main",
            id_prefix="scene_",
            default_loc=default_loc,
            default_pov=default_pov,
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
    ) -> Tuple[StoryOutlineFull, List[SceneContract]]:
        if not branch.from_beat_id:
            return StoryOutlineFull(theory=outline.theory, beats=[]), []
    
        beats_sorted = sorted(outline.beats, key=lambda b: b.order)
        order_map = {b.id: b.order for b in beats_sorted}
        if branch.from_beat_id not in order_map:
            return StoryOutlineFull(theory=outline.theory, beats=[]), []
    
        split_order = order_map[branch.from_beat_id]
        tail_beats = [b for b in beats_sorted if b.order > split_order]
        tail_beats = self._trim_beats_to_single_ending(tail_beats)
        if not tail_beats:
            return StoryOutlineFull(theory=outline.theory, beats=[]), []
    
        br_outline = StoryOutlineFull(theory=outline.theory, beats=tail_beats)
    
        model_name = self.router.get_model_for_agent("outline_agent")
        mc_name = char_list[0] if char_list else None
    
        payload = {
            "outline": br_outline.dict(),
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
            return br_outline, []
    
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
    
        default_loc = loc_list[0] if loc_list else "Unknown Location"
        default_pov = mc_name or (char_list[0] if char_list else "Protagonist")
    
        contracts = self._ensure_beat_coverage_contracts(
            beats=tail_beats,
            contracts=contracts,
            branch_id=branch.id,
            id_prefix=f"{branch.id}_scene_",
            default_loc=default_loc,
            default_pov=default_pov,
        )
    
        return br_outline, contracts

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
    def _neighbors_from_loc_graph(loc_graph: Any, loc: str, limit: int = 12) -> List[str]:
        if not loc_graph or not loc:
            return []
        edges = loc_graph.get("edges") if isinstance(loc_graph, dict) else getattr(loc_graph, "edges", None)
        if not isinstance(edges, list):
            return []
        neigh: List[str] = []
        for e in edges:
            if isinstance(e, dict):
                src = e.get("source")
                tgt = e.get("target")
                bidir = bool(e.get("bidirectional", False))
            else:
                src = getattr(e, "source", None)
                tgt = getattr(e, "target", None)
                bidir = bool(getattr(e, "bidirectional", False))
            if src == loc and tgt:
                neigh.append(str(tgt))
            if tgt == loc and src and bidir:
                neigh.append(str(src))
            if len(neigh) >= limit:
                break
        seen = set()
        uniq: List[str] = []
        for x in neigh:
            if x in seen:
                continue
            seen.add(x)
            uniq.append(x)
        return uniq

    @staticmethod
    def _indoor_markers_found(text: str) -> bool:
        if not text:
            return False
        t = text.lower()
        markers = [
            "вош", "внутр", "двер", "стен", "потол", "комнат", "коридор", "бункер",
            "кабинет", "лестниц", "туннел", "помещени", "забаррикад", "закрыл", "запер"
        ]
        return any(m in t for m in markers)

    @staticmethod
    def _has_travel_glue(lines: List[SceneLine], max_lines: int = 8) -> bool:
        if not lines:
            return False
        head = " ".join((ln.text or "").lower() for ln in lines[:max_lines] if ln and ln.text)
        patterns = [
            "спустя", "через", "по дороге", "путь", "мы шли", "мы брели", "мы добира",
            "добрались", "когда пришли", "на подходе", "на подступах", "мы вышли", "мы поднял", "мы спустил"
        ]
        return any(p in head for p in patterns)

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

    @staticmethod
    def _is_transition_required(prev_location: Optional[str], cur_location: Optional[str]) -> bool:
        if not prev_location or not cur_location:
            return False
        return str(prev_location) != str(cur_location)

    def _build_scene_context(
        self,
        setting: Setting,
        outline: StoryOutlineFull,
        scene_contract: SceneContract,
        char_appearance_map: Dict[str, str],
        previous_summaries: List[str],
        previous_last_lines: Optional[List[str]],
        story_state: StoryState,
        loc_graph: Optional[Any],
        loc_canons: Dict[str, str],
        loc_affordances: Dict[str, Dict[str, Any]],
        prev_location: Optional[str] = None,
    ) -> str:
        beat_map = {b.id: b for b in outline.beats}
        beat: Optional[OutlineBeat] = beat_map.get(scene_contract.beat_id)

        prev_beat: Optional[OutlineBeat] = None
        if beat is not None:
            beats_sorted = sorted(outline.beats, key=lambda b: b.order)
            for idx, b_ in enumerate(beats_sorted):
                if b_.id == beat.id and idx > 0:
                    prev_beat = beats_sorted[idx - 1]
                    break

        lines: List[str] = []
        lines.append("=== WORLD / SETTING ===")
        if setting.setting:
            lines.append(setting.setting)
        if setting.world_rules:
            lines.append("\nWORLD RULES:")
            lines.append(setting.world_rules)

        if prev_beat is not None:
            lines.append("\n=== PREVIOUS BEAT (TEXT) ===")
            if prev_beat.title:
                lines.append(f"Title: {prev_beat.title}")
            if prev_beat.summary:
                lines.append(prev_beat.summary)

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
            lines.append("You MUST include travel/arrival glue in the first lines. No teleportation.")

        if story_state.characters:
            lines.append("\n=== STORY STATE (CHAR LOCATIONS / MOODS) ===")
            for name in scene_contract.present_characters:
                st = story_state.characters.get(name) or {}
                if isinstance(st, dict) and (st.get("location") or st.get("mood")):
                    lines.append(f"- {name}: location={st.get('location')}, mood={st.get('mood')}")

        if loc_graph is not None:
            neigh = self._neighbors_from_loc_graph(loc_graph, scene_contract.location, limit=12)
            if neigh:
                lines.append("\n=== LOCATION GRAPH HINT ===")
                lines.append(f"Adjacent plausible locations from '{scene_contract.location}': {', '.join(neigh)}")

        canon = (loc_canons or {}).get(scene_contract.location, "") or ""
        aff = self._extract_location_aff(loc_affordances, scene_contract.location)
        lines.append("\n=== CURRENT LOCATION CANON (BG) ===")
        lines.append(f"[{scene_contract.location}] {canon}".strip())
        lines.append("\n=== LOCATION AFFORDANCES (HARD CONSTRAINTS) ===")
        lines.append(json.dumps(aff, ensure_ascii=False))

        if not aff.get("enterable", True) or str(aff.get("scale")) == "object":
            lines.append("IMPORTANT: This location is NOT a walkable interior. Scene must be outside; no rooms/walls/ceilings/entering inside.")

        lines.append("\n=== CHARACTER NOTES (PRESENT IN SCENE) ===")
        for name in scene_contract.present_characters:
            desc = char_appearance_map.get(name, "")
            lines.append(f"{name}: {desc or '(no detailed description provided)'}")

        if previous_summaries:
            lines.append("\n=== RECENT SCENE SUMMARIES ===")
            for s in previous_summaries[-10:]:
                text = s
                if ":" in text:
                    text = text.split(":", 1)[1].strip()
                lines.append(f"- {text}")

        if previous_last_lines:
            lines.append("\n=== IMMEDIATE CONTEXT (PREVIOUS SCENE END) ===")
            lines.append("The previous scene ended with these exact lines. Continue smoothly:")
            for l in previous_last_lines:
                lines.append(f"> {l}")

        return "\n".join(lines)

    async def _build_advanced_rag_context(
        self,
        setting: Setting,
        outline: StoryOutlineFull,
        scene_contract: SceneContract,
        char_appearance_map: Dict[str, str],
        previous_summaries: List[str],
        base_context_text: str,
        rag_bundle: RAGBundle,
        story_state: StoryState,
        artifact_store: Optional[ArtifactStore] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        model_name = self.router.get_model_for_agent("outline_agent")

        beat_map = {b.id: b for b in outline.beats}
        beat: Optional[OutlineBeat] = beat_map.get(scene_contract.beat_id)

        query_parts: List[str] = []
        if scene_contract.summary:
            query_parts.append(scene_contract.summary)
        if beat and beat.summary:
            query_parts.append(beat.summary)
        query_text = "\n\n".join(query_parts).strip()

        q_tokens = RAGIndex._tokenize(query_text)
        q_emb: Optional[List[float]] = None
        if rag_bundle.embed_model and query_text:
            q_emb = await self.client.generate_embedding(
                text=query_text,
                model=rag_bundle.embed_model,
                operation_name="embed_scene_query_once",
            )

        story_items = await rag_bundle.story.query(query_text=query_text, top_k=6, q_tokens=q_tokens, q_emb=q_emb)
        world_items = await rag_bundle.world.query(query_text=query_text, top_k=6, q_tokens=q_tokens, q_emb=q_emb)
        char_items = await rag_bundle.characters.query(query_text=query_text, top_k=6, q_tokens=q_tokens, q_emb=q_emb)
        thread_items = await rag_bundle.threads.query(query_text=query_text, top_k=6, q_tokens=q_tokens, q_emb=q_emb)

        retrieved_items = {"story": story_items, "world": world_items, "characters": char_items, "threads": thread_items}

        payload = {
            "setting": setting.dict(),
            "current_beat": beat.dict() if beat else None,
            "scene_contract": scene_contract.dict(),
            "char_appearance_map": char_appearance_map,
            "previous_summaries": previous_summaries[-10:],
            "base_context_text": base_context_text,
            "retrieved_items": retrieved_items,
            "story_state": story_state.dict(),
        }

        resp = await self.client.generate_completion(
            model=model_name,
            temperature=0.2,
            system_prompt=rag_context_prompt,
            prompt=json.dumps(payload, ensure_ascii=False),
            response_format={"type": "json_object"},
            operation_name=f"rag_context_{scene_contract.id}",
        )

        raw = (resp["choices"][0]["message"]["content"] or "").strip()
        schema_hint = '{"global_facts":"...","current_beat_facts":"...","character_facts":{},"recent_events":"...","open_threads":[]}'
        ctx = await self._parse_json_with_repair(
            raw,
            model_name,
            f"rag_context_{scene_contract.id}_parse",
            schema_hint,
            artifact_store=artifact_store,
        )
        return ctx, retrieved_items

    async def _plan_scene_microplan(
        self,
        setting: Setting,
        outline: StoryOutlineFull,
        scene_contract: SceneContract,
        story_state: StoryState,
        retrieved_items: Dict[str, Any],
        branch_context: Optional[Dict[str, Any]],
        artifact_store: Optional[ArtifactStore] = None,
    ) -> Dict[str, Any]:
        model_name = self.router.get_model_for_agent("scene_microplanner_agent")
        beat_map = {b.id: b for b in outline.beats}
        beat: Optional[OutlineBeat] = beat_map.get(scene_contract.beat_id)

        payload = {
            "setting": setting.dict(),
            "current_beat": beat.dict() if beat else None,
            "scene_contract": scene_contract.dict(),
            "story_state": story_state.dict(),
            "retrieved_items": retrieved_items,
            "branch_context": branch_context,
        }

        resp = await self.client.generate_completion(
            model=model_name,
            temperature=0.2,
            system_prompt=scene_microplanner_prompt,
            prompt=json.dumps(payload, ensure_ascii=False),
            response_format={"type": "json_object"},
            operation_name=f"scene_microplan_{scene_contract.id}",
        )

        raw = (resp["choices"][0]["message"]["content"] or "").strip()
        schema_hint = '{"microbeats":["..."],"must_hold_true":["..."],"must_touch_threads":["thread_01"],"required_mentions":[],"forbidden":[]}'
        return await self._parse_json_with_repair(
            raw,
            model_name,
            f"scene_microplan_{scene_contract.id}_parse",
            schema_hint,
            artifact_store=artifact_store,
        )

    async def _critique_scene(
        self,
        knowledge_context: Dict[str, Any],
        scene_script: SceneScript,
        scene_contract: SceneContract,
        story_state: StoryState,
        branch_context: Optional[Dict[str, Any]],
        loc_graph: Optional[Any],
        char_graph: Optional[Any],
        microplan: Optional[Dict[str, Any]],
        story_length: str,
        prev_location: Optional[str],
        transition_required: bool,
        location_canon: str,
        location_affordances: Dict[str, Any],
        loc_list: List[str],
        artifact_store: Optional[ArtifactStore] = None,
    ) -> Dict[str, Any]:
        model_name = self.router.get_model_for_agent("critic_agent")
        payload = {
            "knowledge_context": knowledge_context,
            "scene_contract": scene_contract.dict(),
            "scene_script": scene_script.dict(),
            "story_state": story_state.dict(),
            "branch_context": branch_context,
            "loc_graph": loc_graph,
            "char_graph": char_graph,
            "microplan": microplan,
            "story_length": story_length,
            "prev_location": prev_location,
            "transition_required": transition_required,
            "location_canon": location_canon,
            "location_affordances": location_affordances,
            "loc_list": loc_list,
        }

        resp = await self.client.generate_completion(
            model=model_name,
            temperature=0.1,
            system_prompt=critic_prompt,
            prompt=json.dumps(payload, ensure_ascii=False),
            response_format={"type": "json_object"},
            operation_name=f"critique_{scene_script.scene_id}",
        )

        raw = (resp["choices"][0]["message"]["content"] or "").strip()
        schema_hint = '{"ok":true,"issues":[],"must_regenerate":false,"state_updates":{"world":{},"characters":{},"plot_threads":{}}}'
        return await self._parse_json_with_repair(
            raw,
            model_name,
            f"critique_{scene_script.scene_id}_parse",
            schema_hint,
            artifact_store=artifact_store,
        )

    def _apply_state_updates(self, story_state: StoryState, state_updates: Dict[str, Any]) -> None:
        if not state_updates:
            return

        world_updates = state_updates.get("world") or {}
        if isinstance(world_updates, dict):
            for k, v in world_updates.items():
                story_state.world[k] = v

        char_updates = state_updates.get("characters") or {}
        if isinstance(char_updates, dict):
            for name, delta in char_updates.items():
                base = story_state.characters.get(name, {})
                if isinstance(base, dict) and isinstance(delta, dict):
                    base.update(delta)
                    story_state.characters[name] = base
                elif isinstance(delta, dict):
                    story_state.characters[name] = delta

        thread_updates = state_updates.get("plot_threads") or {}
        if isinstance(thread_updates, dict):
            for tid, status in thread_updates.items():
                story_state.plot_threads[str(tid)] = str(status)

    def _soft_update_character_locations(self, story_state: StoryState, contract: SceneContract) -> None:
        for name in contract.present_characters:
            st = story_state.characters.get(name)
            if not isinstance(st, dict):
                st = {}
            if not st.get("location"):
                st["location"] = contract.location
            story_state.characters[name] = st

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

    async def _edit_scene(
        self,
        setting: Setting,
        scene_contract: SceneContract,
        story_state: StoryState,
        microplan: Dict[str, Any],
        critic_issues: List[str],
        scene_script: SceneScript,
        location_canon: str,
        location_affordances: Dict[str, Any],
        prev_location: Optional[str],
        transition_required: bool,
        target_min_lines: Optional[int] = None,
        artifact_store: Optional[ArtifactStore] = None,
    ) -> Optional[SceneScript]:
        model_name = self.router.get_model_for_agent("scene_editor_agent")

        payload = {
            "setting": setting.dict(),
            "scene_contract": scene_contract.dict(),
            "story_state": story_state.dict(),
            "microplan": microplan,
            "critic_issues": critic_issues,
            "scene_script": scene_script.dict(),
            "target_min_lines": target_min_lines,
            "location_canon": location_canon,
            "location_affordances": location_affordances,
            "prev_location": prev_location,
            "transition_required": transition_required,
        }

        resp = await self.client.generate_completion(
            model=model_name,
            temperature=0.2,
            system_prompt=scene_editor_prompt,
            prompt=json.dumps(payload, ensure_ascii=False),
            response_format={"type": "json_object"},
            operation_name=f"edit_scene_{scene_contract.id}",
            max_tokens=8000,
        )

        raw = (resp["choices"][0]["message"]["content"] or "").strip()
        schema_hint = '{ "scene_id":"...", "lines":[{"type":"narration","speaker":null,"text":"..."}], "summary":"..." }'
        data = await self._parse_json_with_repair(
            raw,
            model_name,
            f"edit_scene_{scene_contract.id}_parse",
            schema_hint,
            artifact_store=artifact_store,
        )
        if not isinstance(data, dict) or not data:
            return None

        try:
            edited = SceneScript(**data)
        except Exception:
            return None

        edited.scene_id = scene_contract.id
        edited.branch_id = scene_contract.branch_id
        edited.branch_order = scene_contract.branch_order
        return edited

    async def _write_single_scene_with_retries(
        self,
        contract: SceneContract,
        setting: Setting,
        outline: StoryOutlineFull,
        char_appearance_map: Dict[str, str],
        rag_bundle: RAGBundle,
        story_state: StoryState,
        previous_summaries: List[str],
        story_length: str,
        branching: Optional[BranchingInfo] = None,
        previous_last_lines: Optional[List[str]] = None,
        loc_graph: Optional[Any] = None,
        char_graph: Optional[Any] = None,
        max_retries: int = 3,
        artifact_store: Optional[ArtifactStore] = None,
        prev_location: Optional[str] = None,
        loc_canons: Optional[Dict[str, str]] = None,
        loc_affordances: Optional[Dict[str, Dict[str, Any]]] = None,
        loc_list: Optional[List[str]] = None,
        name_canon: Optional[NameCanonicalizer] = None,
        strict_names: Optional[bool] = None,
        strict_locations: Optional[bool] = None,
    ) -> SceneScript:
        writer_model = self.router.get_model_for_agent("writer_agent")
        min_lines = self._min_lines(story_length)

        loc_canons = loc_canons or {}
        loc_affordances = loc_affordances or {}
        loc_list = loc_list or []

        if strict_names is None:
            strict_names = self._strict_name_canon_enabled()
        if strict_locations is None:
            strict_locations = self._strict_location_gate_enabled()

        try:
            num_candidates = int(os.getenv("SCENE_NUM_CANDIDATES", "2"))
        except Exception:
            num_candidates = 2
        num_candidates = max(1, min(num_candidates, 4))

        branch_spec: Optional[BranchSpec] = None
        if branching:
            branch_spec = next((b for b in branching.branches if b.id == contract.branch_id), None)

        branch_context_payload: Optional[Dict[str, Any]] = None
        if branch_spec:
            branch_context_payload = {
                "id": branch_spec.id,
                "title": branch_spec.title,
                "description": branch_spec.description,
                "ending_tone": branch_spec.ending_tone,
                "kind": branch_spec.kind,
                "is_canonical": branch_spec.is_canonical,
            }

        schema_hint_scene = '{ "scene_id":"scene_001", "lines":[{"type":"dialogue","speaker":"Имя","text":"..."}], "summary":"..." }'

        location_patch_budget = 1

        for _ctx_rebuild in range(0, 2):
            transition_required = self._is_transition_required(prev_location, contract.location)
            location_canon = (loc_canons.get(contract.location) or "").strip()
            location_aff = self._extract_location_aff(loc_affordances, contract.location)

            base_context_text = self._build_scene_context(
                setting=setting,
                outline=outline,
                scene_contract=contract,
                char_appearance_map=char_appearance_map,
                previous_summaries=previous_summaries,
                previous_last_lines=previous_last_lines,
                story_state=story_state,
                loc_graph=loc_graph,
                loc_canons=loc_canons,
                loc_affordances=loc_affordances,
                prev_location=prev_location,
            )

            rag_context, retrieved_items = await self._build_advanced_rag_context(
                setting=setting,
                outline=outline,
                scene_contract=contract,
                char_appearance_map=char_appearance_map,
                previous_summaries=previous_summaries,
                base_context_text=base_context_text,
                rag_bundle=rag_bundle,
                story_state=story_state,
                artifact_store=artifact_store,
            )

            microplan = await self._plan_scene_microplan(
                setting=setting,
                outline=outline,
                scene_contract=contract,
                story_state=story_state,
                retrieved_items=retrieved_items,
                branch_context=branch_context_payload,
                artifact_store=artifact_store,
            )

            combined_context = {
                "base_context_text": base_context_text,
                "rag_context": rag_context,
                "microplan": microplan,
                "location_canon": location_canon,
                "location_affordances": location_aff,
            }

            if artifact_store is not None:
                artifact_store.save(
                    f"context/{contract.branch_id}/{contract.id}.json",
                    {
                        "scene_contract": contract.dict(),
                        "combined_context": combined_context,
                        "min_lines": min_lines,
                        "prev_location": prev_location,
                        "transition_required": transition_required,
                        "strict_names": bool(strict_names),
                        "strict_locations": bool(strict_locations),
                    },
                )

            last_reason = ""
            last_details = ""
            last_short_script: Optional[SceneScript] = None

            for attempt in range(1, max_retries + 1):
                regen_info: Optional[Dict[str, Any]] = None
                if attempt > 1:
                    regen_info = {"attempt": attempt, "reason": last_reason or "unknown", "details": last_details[:900]}

                candidates: List[SceneScript] = []
                cand_unmatched_names: Dict[int, bool] = {}

                for cidx in range(1, num_candidates + 1):
                    payload: Dict[str, Any] = {
                        "context": combined_context,
                        "scene_contract": contract.dict(),
                        "story_length": story_length,
                        "min_lines": min_lines,
                        "prev_location": prev_location,
                        "transition_required": transition_required,
                    }
                    if branch_context_payload is not None:
                        payload["branch_context"] = branch_context_payload
                    if regen_info is not None:
                        payload["regen_info"] = {**regen_info, "candidate_idx": cidx}
                    if last_short_script is not None:
                        payload["previous_scene_script"] = last_short_script.dict()

                    resp = await self.client.generate_completion(
                        model=writer_model,
                        temperature=0.7 if attempt == 1 else 0.6,
                        system_prompt=writer_prompt,
                        prompt=json.dumps(payload, ensure_ascii=False),
                        response_format={"type": "json_object"},
                        operation_name=f"write_scene_{contract.id}_a{attempt}_c{cidx}",
                        max_tokens=self._writer_max_tokens(story_length),
                    )

                    raw = (resp["choices"][0]["message"]["content"] or "").strip()
                    data = await self._parse_json_with_repair(
                        raw=raw,
                        model_name=writer_model,
                        operation_name=f"write_scene_{contract.id}_parse",
                        schema_hint=schema_hint_scene,
                        artifact_store=artifact_store,
                    )
                    if not isinstance(data, dict) or not data:
                        continue

                    try:
                        script = SceneScript(**data)
                    except ValidationError:
                        continue

                    script.scene_id = contract.id
                    script.branch_id = contract.branch_id
                    script.branch_order = contract.branch_order

                    for line in script.lines:
                        if line.type == "narration":
                            line.speaker = None
                        if line.type == "thought" and not line.speaker:
                            line.speaker = contract.pov_character

                    had_unmatched = False
                    if name_canon is not None:
                        had_unmatched = self._canon_script_inplace(script, contract, name_canon, store=artifact_store)
                    cand_unmatched_names[id(script)] = had_unmatched

                    candidates.append(script)

                if not candidates:
                    last_reason = "no_valid_candidates"
                    last_details = f"attempt={attempt}: all candidates invalid (json/validation)"
                    continue

                scored: List[Tuple[float, SceneScript, Dict[str, Any], bool]] = []

                for script in candidates:
                    cr = await self._critique_scene(
                        knowledge_context=rag_context,
                        scene_script=script,
                        scene_contract=contract,
                        story_state=story_state,
                        branch_context=branch_context_payload,
                        loc_graph=loc_graph,
                        char_graph=char_graph,
                        microplan=microplan,
                        story_length=story_length,
                        prev_location=prev_location,
                        transition_required=transition_required,
                        location_canon=location_canon,
                        location_affordances=location_aff,
                        loc_list=loc_list,
                        artifact_store=artifact_store,
                    )

                    unmatched_names = bool(cand_unmatched_names.get(id(script), False))
                    must_regen_llm = bool(cr.get("must_regenerate", False))

                    issues = cr.get("issues") or []
                    issues_list = issues if isinstance(issues, list) else [str(issues)]
                    issues_count = len(issues_list)

                    hard_issues: List[str] = []
                    must_regen_rules = False

                    if strict_names and unmatched_names:
                        must_regen_rules = True
                        hard_issues.append("Non-canonical speaker name detected")

                    missing_travel = transition_required and (not self._has_travel_glue(script.lines, max_lines=8))
                    if missing_travel:
                        hard_issues.append("Missing travel glue at scene start while location changed (no teleportation).")
                        if strict_locations:
                            must_regen_rules = True

                    violates_aff = False
                    if (not location_aff.get("enterable", True)) or (str(location_aff.get("scale")) == "object"):
                        head_text = " ".join((ln.text or "") for ln in script.lines[:30])
                        if self._indoor_markers_found(head_text):
                            violates_aff = True
                            hard_issues.append("Location is not enterable / scale=object but text implies interior (walls/rooms/entering inside).")
                            if strict_locations:
                                must_regen_rules = True

                    loc_check = cr.get("location_check") or {}
                    if isinstance(loc_check, dict) and bool(loc_check.get("mismatch", False)):
                        rec = str(loc_check.get("recommended_action") or "edit_text")
                        sugg = loc_check.get("suggested_location")
                        if strict_locations and rec != "change_location":
                            must_regen_rules = True
                            hard_issues.append(f"Critic reports location mismatch (recommended_action={rec}); strict location gate requires regeneration.")
                        if strict_locations and rec == "change_location" and (not (isinstance(sugg, str) and sugg in loc_list)):
                            must_regen_rules = True
                            hard_issues.append("Critic suggests changing location but suggested_location is missing/invalid; strict location gate requires regeneration.")

                    must_regen_effective = must_regen_llm or must_regen_rules

                    score = 100.0
                    if must_regen_effective:
                        score -= 60.0
                    score -= 6.0 * issues_count
                    score -= 18.0 * len(hard_issues)

                    score += min(20.0, len(script.lines) / 10.0)
                    if len(script.lines) < min_lines:
                        score -= min(30.0, float(min_lines - len(script.lines)) * 0.6)

                    if hard_issues:
                        cr["_hard_issues"] = hard_issues
                    cr["_must_regen_rules"] = must_regen_rules
                    cr["_must_regenerate_effective"] = must_regen_effective

                    scored.append((score, script, cr, must_regen_effective))

                scored.sort(key=lambda x: x[0], reverse=True)
                best_score, best_script, best_critic, best_must_regen = scored[0]

                if artifact_store is not None:
                    artifact_store.save(
                        f"critics/{contract.branch_id}/{contract.id}_a{attempt}.json",
                        {"attempt": attempt, "best_score": best_score, "best_script_lines": len(best_script.lines), "best_critic": best_critic},
                    )

                loc_check_best = best_critic.get("location_check") or {}
                if isinstance(loc_check_best, dict) and bool(loc_check_best.get("mismatch", False)):
                    rec = str(loc_check_best.get("recommended_action") or "edit_text")
                    sugg = loc_check_best.get("suggested_location")
                    if rec == "change_location" and location_patch_budget > 0 and isinstance(sugg, str) and sugg in loc_list:
                        if artifact_store is not None:
                            artifact_store.event(
                                "contract.location_patched",
                                {"scene_id": contract.id, "old_location": contract.location, "new_location": sugg, "reason": loc_check_best.get("details")},
                            )
                        contract.location = sugg
                        location_patch_budget -= 1
                        break

                if best_must_regen and attempt < max_retries:
                    last_reason = "critic_or_rules_feedback"
                    llm_issues = best_critic.get("issues") or []
                    llm_issues_list = llm_issues if isinstance(llm_issues, list) else [str(llm_issues)]
                    hard_issues = best_critic.get("_hard_issues") or []
                    hard_issues_list = hard_issues if isinstance(hard_issues, list) else [str(hard_issues)]
                    last_details = "; ".join((llm_issues_list + hard_issues_list))[:900]
                    continue

                final_script = best_script
                final_critic = best_critic


                if len(final_script.lines) < min_lines:
                    issues_list = final_critic.get("issues") or []
                    if not isinstance(issues_list, list):
                        issues_list = [str(issues_list)]
                    edited = await self._edit_scene(
                        setting=setting,
                        scene_contract=contract,
                        story_state=story_state,
                        microplan=microplan,
                        critic_issues=issues_list,
                        scene_script=final_script,
                        location_canon=location_canon,
                        location_affordances=location_aff,
                        prev_location=prev_location,
                        transition_required=transition_required,
                        target_min_lines=min_lines,
                        artifact_store=artifact_store,
                    )
                    if edited is not None and len(edited.lines) >= len(final_script.lines):
                        if name_canon is not None:
                            self._canon_script_inplace(edited, contract, name_canon, store=artifact_store)
                        final_script = edited

                hard_issues2 = final_critic.get("_hard_issues") or []
                if hard_issues2:
                    edited = await self._edit_scene(
                        setting=setting,
                        scene_contract=contract,
                        story_state=story_state,
                        microplan=microplan,
                        critic_issues=list(hard_issues2) if isinstance(hard_issues2, list) else [str(hard_issues2)],
                        scene_script=final_script,
                        location_canon=location_canon,
                        location_affordances=location_aff,
                        prev_location=prev_location,
                        transition_required=transition_required,
                        target_min_lines=min_lines if len(final_script.lines) < min_lines else None,
                        artifact_store=artifact_store,
                    )
                    if edited is not None:
                        if name_canon is not None:
                            self._canon_script_inplace(edited, contract, name_canon, store=artifact_store)
                        final_script = edited

                if len(final_script.lines) < min_lines and attempt < max_retries:
                    last_reason = "too_short"
                    last_details = f"lines={len(final_script.lines)} < min_lines={min_lines}"
                    last_short_script = final_script
                    continue

                cr2 = await self._critique_scene(
                    knowledge_context=rag_context,
                    scene_script=final_script,
                    scene_contract=contract,
                    story_state=story_state,
                    branch_context=branch_context_payload,
                    loc_graph=loc_graph,
                    char_graph=char_graph,
                    microplan=microplan,
                    story_length=story_length,
                    prev_location=prev_location,
                    transition_required=transition_required,
                    location_canon=location_canon,
                    location_affordances=location_aff,
                    loc_list=loc_list,
                    artifact_store=artifact_store,
                )

                self._apply_state_updates(story_state, cr2.get("state_updates") or {})
                self._soft_update_character_locations(story_state, contract)
                return final_script

            else:
                pass

            #ff we broke attempt loop due to location patch rebuild context and retry outer ctx loop
            continue

        # fallback
        from src.pydantic_schemas import SceneLine as SceneLineModel, SceneScript as SceneScriptModel

        fb_text = "Техническая сцена-заглушка: генерация не смогла стабильно сформировать сцену."
        if artifact_store is not None:
            artifact_store.event("scene.fallback", {"scene_id": contract.id, "branch_id": contract.branch_id, "reason": "exhausted_retries"})

        return SceneScriptModel(
            scene_id=contract.id,
            branch_id=contract.branch_id,
            branch_order=contract.branch_order,
            lines=[SceneLineModel(type="narration", speaker=None, text=fb_text)],
            summary=contract.summary or fb_text,
        )

    async def _write_scenes(
        self,
        setting: Setting,
        outline: StoryOutlineFull,
        scene_contracts: List[SceneContract],
        char_list: List[str],
        char_appearance: Dict[str, Any] | CharacterAppearance | None,
        rag_bundle: RAGBundle,
        story_state: StoryState,
        story_length: str,
        branching: Optional[BranchingInfo] = None,
        initial_previous_summaries: Optional[List[str]] = None,
        initial_previous_last_lines: Optional[List[str]] = None,
        initial_prev_location: Optional[str] = None,
        state_snapshots: Optional[Dict[str, StoryState]] = None,
        loc_graph: Optional[Any] = None,
        char_graph: Optional[Any] = None,
        loc_canons: Optional[Dict[str, str]] = None,
        loc_affordances: Optional[Dict[str, Dict[str, Any]]] = None,
        loc_list: Optional[List[str]] = None,
        artifact_store: Optional[ArtifactStore] = None,
        artifact_prefix: str = "main",
        name_canon: Optional[NameCanonicalizer] = None,
        strict_names: Optional[bool] = None,
        strict_locations: Optional[bool] = None,
    ) -> Dict[str, SceneScript]:
        app_logger.info(f"Writing {len(scene_contracts)} scenes... story_length={story_length}")

        loc_canons = loc_canons or {}
        loc_affordances = loc_affordances or {}
        loc_list = loc_list or []

        if strict_names is None:
            strict_names = self._strict_name_canon_enabled()
        if strict_locations is None:
            strict_locations = self._strict_location_gate_enabled()

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
        previous_summaries: List[str] = []

        if initial_previous_summaries:
            for entry in initial_previous_summaries:
                previous_summaries.append(entry)
                await rag_bundle.story.add_item(_sha1(entry), "scene", entry)

        last_scene_lines_buffer: List[str] = list(initial_previous_last_lines or [])
        prev_location: Optional[str] = initial_prev_location

        for contract in scene_contracts:
            if name_canon is not None:
                self._canon_contract_inplace(contract, name_canon, store=artifact_store)

            if artifact_store is not None:
                artifact_store.event(
                    "scene.start",
                    {"scene_id": contract.id, "branch_id": contract.branch_id, "order": contract.branch_order, "summary_plan": contract.summary, "location": contract.location, "prev_location": prev_location},
                )

            script = await self._write_single_scene_with_retries(
                contract=contract,
                setting=setting,
                outline=outline,
                char_appearance_map=char_appearance_map,
                rag_bundle=rag_bundle,
                story_state=story_state,
                previous_summaries=previous_summaries,
                story_length=story_length,
                branching=branching,
                previous_last_lines=last_scene_lines_buffer,
                loc_graph=loc_graph,
                char_graph=char_graph,
                max_retries=3,
                artifact_store=artifact_store,
                prev_location=prev_location,
                loc_canons=loc_canons,
                loc_affordances=loc_affordances,
                loc_list=loc_list,
                name_canon=name_canon,
                strict_names=strict_names,
                strict_locations=strict_locations,
            )

            scene_scripts[contract.id] = script
            previous_summaries.append(f"{contract.id}: {script.summary}")
            await rag_bundle.story.upsert_item(contract.id, "scene", script.summary)

            if state_snapshots is not None:
                state_snapshots[contract.id] = StoryState(**story_state.dict())

            last_scene_lines_buffer = self._extract_last_lines(script, n=3)
            prev_location = contract.location

            app_logger.info(f"Scene {contract.id} written, lines={len(script.lines)}")

            if artifact_store is not None:
                artifact_store.save(f"scenes/{artifact_prefix}/{contract.id}.json", script.dict())
                artifact_store.save(f"state/{artifact_prefix}/{contract.id}.json", story_state.dict())
                artifact_store.event("scene.done", {"scene_id": contract.id, "branch_id": contract.branch_id, "lines": len(script.lines)})

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
        app_logger.info(f"Starting generation {generation_id}")

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
                    "strict_name_canon": self._strict_name_canon_enabled(),
                    "strict_location_gate": self._strict_location_gate_enabled(),
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

            embed_model = os.getenv("EMBED_MODEL_NAME", "text-embedding-3-small")
            rag_main = RAGBundle(self.client, embed_model)
            story_state_main = StoryState()

            setting_obj = await self._generate_setting(
                user_prompt=user_request.user_prompt,
                setting_override=setting if setting else None,
                time_choice=time_choice,
                genre_choice=genre_choice,
                artifact_store=store,
            )

            world_text = setting_obj.setting or ""
            if setting_obj.world_rules:
                world_text += "\nWorld rules: " + setting_obj.world_rules
            await rag_main.world.upsert_item("setting_core", "world_lore", world_text)

            outline_obj = await self._generate_outline(
                user_prompt=user_request.user_prompt,
                story_length=normalized_length,
                setting=setting_obj,
                plot_prefs=plot_prefs,
                plot_freeform=plot_freeform,
                artifact_store=store,
            )

            try:
                threads = await self._extract_plot_threads(user_request.user_prompt, setting_obj, outline_obj, artifact_store=store)
                for t in threads:
                    story_state_main.plot_threads[t["id"]] = t.get("status") or "open"
                    await rag_main.threads.upsert_item(
                        f"{t['id']}::core",
                        "thread",
                        f"{t.get('title','')}\n{t.get('description','')}\nanchors: {', '.join(t.get('anchors') or [])}",
                    )
            except Exception as e:
                app_logger.error(f"Thread extraction failed: {e}", exc_info=True)

            preferred_endings = plot_prefs.ending_types if plot_prefs and plot_prefs.ending_types else None
            branching_info = await self._plan_branches(
                outline=outline_obj,
                max_branches=branches_max,
                tone=user_request.tone,
                preferred_ending_types=preferred_endings,
                artifact_store=store,
            )
            outline_main_obj = self._beats_for_main_route(outline_obj, branching_info)
            for beat in outline_main_obj.beats:
                await rag_main.story.upsert_item(beat.id, "beat", beat.summary or "")

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
                    user_prompt=user_request.user_prompt,
                    setting=setting_obj,
                    outline=outline_obj,
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
            char_graph = self._unwrap_last(char_results_map.get("char_graph"))
            char_appearance = self._unwrap_last(char_results_map.get("char_appearance"))

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
            loc_graph = self._unwrap_last(loc_results_map.get("loc_graph"))
            loc_description = self._unwrap_last(loc_results_map.get("loc_description"))

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
                app_logger.info("Image generation disabled.")

            main_contracts = await self._generate_scene_contracts_main(
                outline_main_obj, char_list_final, loc_list_final, normalized_length, artifact_store=store
            )

            main_contracts = await self._patch_scene_contracts_with_location_critic(
                scene_contracts=main_contracts,
                loc_list=loc_list_final,
                loc_canons=loc_canons,
                loc_affordances=loc_affordances,
                loc_graph=loc_graph,
                artifact_store=store,
                artifact_name="main",
            )

            for c in main_contracts:
                self._canon_contract_inplace(c, name_canon, store=store)

            state_snapshots_main: Dict[str, StoryState] = {}
            main_scripts = await self._write_scenes(
                setting=setting_obj,
                outline=outline_main_obj,
                scene_contracts=main_contracts,
                char_list=char_list_final,
                char_appearance=char_appearance,
                rag_bundle=rag_main,
                story_state=story_state_main,
                story_length=normalized_length,
                branching=branching_info,
                initial_previous_summaries=None,
                initial_previous_last_lines=None,
                initial_prev_location=None,
                state_snapshots=state_snapshots_main,
                loc_graph=loc_graph,
                char_graph=char_graph,
                loc_canons=loc_canons,
                loc_affordances=loc_affordances,
                loc_list=loc_list_final,
                artifact_store=store,
                artifact_prefix="main",
                name_canon=name_canon,
                strict_names=self._strict_name_canon_enabled(),
                strict_locations=self._strict_location_gate_enabled(),
            )

            branch_contracts: List[SceneContract] = []
            branch_scripts: Dict[str, SceneScript] = {}
            branch_states: Dict[str, StoryState] = {}

            if len(branching_info.branches) > 1:
                beat_to_main_scene: Dict[str, str] = {}
                for sc in main_contracts:
                    beat_to_main_scene[sc.beat_id] = sc.id


                for br in branching_info.branches:
                    if br.id == "main" or not br.from_beat_id:
                        continue

                    br_outline, br_contracts = await self._generate_scene_contracts_for_branch(
                        outline_obj, char_list_final, loc_list_final, normalized_length, br
                    )
                    if not br_contracts:
                        continue

                    br_contracts = await self._patch_scene_contracts_with_location_critic(
                        scene_contracts=br_contracts,
                        loc_list=loc_list_final,
                        loc_canons=loc_canons,
                        loc_affordances=loc_affordances,
                        loc_graph=loc_graph,
                        artifact_store=store,
                        artifact_name=br.id,
                    )

                    for c in br_contracts:
                        self._canon_contract_inplace(c, name_canon, store=store)

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

                    rag_branch = RAGBundle(
                        self.client,
                        embed_model,
                        world_index=rag_main.world,
                        char_index=rag_main.characters,
                        thread_index=rag_main.threads,
                    )
                    for beat in br_outline.beats:
                        await rag_branch.story.upsert_item(
                            f"{br.id}::{beat.id}",
                            "beat",
                            beat.summary or "",
                        )


                    if divergence_scene_id and divergence_scene_id in state_snapshots_main:
                        branch_state = StoryState(**state_snapshots_main[divergence_scene_id].dict())
                    else:
                        store.event(
                            "branch.state_snapshot_missing",
                            {"branch_id": br.id, "from_beat_id": br.from_beat_id, "divergence_scene_id": divergence_scene_id},
                        )
                        branch_state = StoryState()

                    br_scripts = await self._write_scenes(
                        setting=setting_obj,
                        outline=br_outline,
                        scene_contracts=br_contracts,
                        char_list=char_list_final,
                        char_appearance=char_appearance,
                        rag_bundle=rag_branch,
                        story_state=branch_state,
                        story_length=normalized_length,
                        branching=branching_info,
                        initial_previous_summaries=initial_prev,
                        initial_previous_last_lines=initial_last_lines,
                        initial_prev_location=initial_prev_location,
                        state_snapshots=None,
                        loc_graph=loc_graph,
                        char_graph=char_graph,
                        loc_canons=loc_canons,
                        loc_affordances=loc_affordances,
                        loc_list=loc_list_final,
                        artifact_store=store,
                        artifact_prefix=br.id,
                        name_canon=name_canon,
                        strict_names=self._strict_name_canon_enabled(),
                        strict_locations=self._strict_location_gate_enabled(),
                    )

                    branch_scripts.update(br_scripts)
                    branch_states[br.id] = branch_state

                self._inject_branch_choices(main_contracts, main_scripts, branch_contracts, branching_info)

            all_contracts = main_contracts + branch_contracts
            all_scripts: Dict[str, SceneScript] = {}
            all_scripts.update(main_scripts)
            all_scripts.update(branch_scripts)

            story_state_by_branch: Dict[str, Any] = {"main": story_state_main.dict()}
            for bid, st in branch_states.items():
                story_state_by_branch[bid] = st.dict()

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
                },
            )

            return result

        finally:
            TRACE_HOOK.reset(token)
