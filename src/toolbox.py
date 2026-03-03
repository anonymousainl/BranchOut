import json
from json import JSONDecodeError
import uuid
from typing import Dict, Any, List, Optional

from src.pydantic_schemas import (
    CharacterGraph,
    CharacterAppearance,
    LocationDescription,
    LocationGraph,
    StoryGraph,
)
from src.logger import app_logger
from src.apicallhandler import OpenRouterClient


class Toolbox:
    def __init__(self):
        self.model: Optional[str] = None
        self.agents: Dict[str, Any] = {}
        self.input_dict: Dict[str, Any] = {}
        self._enabled_schemas: List[str] = []

    def add_tool_schema(self, schema_attr: str) -> "Toolbox":
        if not hasattr(self, schema_attr):
            raise AttributeError(f"Tool schema '{schema_attr}' not found on Toolbox")
        if schema_attr not in self._enabled_schemas:
            self._enabled_schemas.append(schema_attr)
        return self

    def get_tools_for_openrouter(self) -> List[Dict[str, Any]]:
        return [getattr(self, schema_attr) for schema_attr in self._enabled_schemas]

    def extract_tool_calls(self, content: str, tools: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
        if not content:
            return []

        def try_parse_json(text: str) -> Optional[Dict[str, Any]]:
            try:
                return json.loads(text)
            except Exception:
                return None

        parsed: Optional[Any] = None
        if "```json" in content:
            json_start = content.find("```json") + 7
            json_end = content.find("```", json_start)
            if json_end != -1:
                json_content = content[json_start:json_end].strip()
                parsed = try_parse_json(json_content)

        if parsed is None:
            parsed = try_parse_json(content.strip())

        if parsed is None:
            return []

        allowed_names = set()
        if tools:
            for t in tools:
                fn = t.get("function", {})
                if fn and "name" in fn:
                    allowed_names.add(fn["name"])

        calls_norm: List[Dict[str, Any]] = []
        if isinstance(parsed, dict) and "tool_calls" in parsed and isinstance(parsed["tool_calls"], list):
            for c in parsed["tool_calls"]:
                name = c.get("name") or c.get("function", {}).get("name")
                args = c.get("arguments") or c.get("function", {}).get("arguments") or {}
                if allowed_names and name not in allowed_names:
                    continue
                calls_norm.append({"name": name, "arguments": args})
        elif isinstance(parsed, dict) and "name" in parsed and "arguments" in parsed:
            if not allowed_names or parsed["name"] in allowed_names:
                calls_norm.append({"name": parsed["name"], "arguments": parsed.get("arguments", {})})
        elif isinstance(parsed, dict) and len(parsed.keys()) == 1:
            name = next(iter(parsed.keys()))
            if not allowed_names or name in allowed_names:
                calls_norm.append({"name": name, "arguments": parsed[name]})

        tool_calls: List[Dict[str, Any]] = []
        for c in calls_norm:
            try:
                args_obj = c["arguments"]
                if not isinstance(args_obj, str):
                    args_str = json.dumps(args_obj, ensure_ascii=False)
                else:
                    args_str = args_obj
                tool_calls.append(
                    {
                        "id": f"shim_{uuid.uuid4().hex[:8]}",
                        "type": "function",
                        "function": {"name": c["name"], "arguments": args_str},
                    }
                )
            except Exception as e:
                app_logger.warning(f"Failed to normalize tool call {c}: {e}")

        return tool_calls

    @staticmethod
    def json_parser(agent_results: Dict[str, Any], input_data: Dict[str, Any]) -> Dict[str, Any]:
        def find_results_block(obj):
            if not isinstance(obj, dict):
                return None
            if "results" in obj:
                inner = obj["results"]
                if isinstance(inner, dict) and "call_agent" in inner:
                    return find_results_block(inner["call_agent"])
                return inner
            return None

        extracted = find_results_block(agent_results)

        if isinstance(extracted, dict):
            for key, value in extracted.items():
                if key not in input_data or (value not in [None, {}, [], ""]):
                    input_data[key] = value

        return input_data


    async def char_graph(self, client: OpenRouterClient, char_list: List[str]) -> CharacterGraph:
        prompt = """Task: Generate a JSON graph representing character relationships.

Relationship States (ONLY use these):
- "friends"
- "enemies"
- "love interest"

Output must be valid JSON with:
1. "nodes": Array of {id: string, label: string}
2. "edges": Array of {source: string, target: string, label: relationship, directed: true}

Rules:
- Include ALL characters from the list
- Relationships can be one-way
- Return ONLY the JSON object
"""

        response = await client.generate_completion(
            prompt=json.dumps({"char_list": char_list}, ensure_ascii=False),
            model=self.model or "google/gemini-2.5-flash",
            temperature=0.1,
            system_prompt=prompt,
            operation_name="char_graph",
            response_format={"type": "json_object"},
        )

        raw = (response["choices"][0]["message"]["content"] or "").strip()
        if "```json" in raw:
            json_start = raw.find("```json") + 7
            json_end = raw.find("```", json_start)
            if json_end != -1:
                raw = raw[json_start:json_end].strip()

        try:
            data = json.loads(raw)
        except JSONDecodeError:
            start = raw.find("{")
            end = raw.rfind("}")
            if start != -1 and end != -1 and end > start:
                data = json.loads(raw[start : end + 1])
            else:
                raise

        return CharacterGraph(**data)

    async def char_appearance(
        self,
        client: OpenRouterClient,
        char_list: List[str],
        setting: str,
        hints: Optional[Dict[str, str]] = None,
    ) -> CharacterAppearance:

        prompt = """Task: Generate detailed visual descriptions for AI image generation.

Output: JSON object with "descriptions": array of description strings, one per character, IN THE SAME ORDER as input list.

Each description must:
- Be a single dense paragraph.
- Include: pose, facial features, physique, attire, signature traits, emotional expression.
- Be consistent with the global setting.
- If a hint is provided for this character (by exact name match), you MUST:
  - incorporate and refine it,
  - not contradict the hint,
  - expand it into a full visual description.
- End with the phrase:
  "Rendered in visual novel sprite style with clean lines, high detail, no blur, high resolution. Solid color background."
- Explicitly state that the background is a simple solid color, without objects or patterns.

Return ONLY JSON.
"""

        input_data = {"characters": char_list, "setting": setting, "hints": hints or {}}

        response = await client.generate_completion(
            prompt=json.dumps(input_data, ensure_ascii=False),
            model=self.model or "google/gemini-2.5-flash",
            temperature=0.1,
            system_prompt=prompt,
            operation_name="char_appearance",
            response_format={"type": "json_object"},
        )

        raw = (response["choices"][0]["message"]["content"] or "").strip()
        if "```json" in raw:
            raw = raw.split("```json", 1)[1].split("```", 1)[0].strip()

        try:
            data = json.loads(raw)
        except Exception:
            descs = [raw]
            while len(descs) < len(char_list):
                descs.append("")
            return CharacterAppearance(descriptions=descs[: len(char_list)])

        descs = data.get("descriptions")
        if not isinstance(descs, list):
            descs = [raw]

        if len(descs) < len(char_list):
            descs = list(descs) + [""] * (len(char_list) - len(descs))
        if len(descs) > len(char_list):
            descs = descs[: len(char_list)]

        return CharacterAppearance(descriptions=[str(x) for x in descs])

    @property
    def schema_char_graph(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "char_graph",
                "description": "Creates character relationship graph",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "char_list": {
                            "type": "array",
                            "description": "List of character names",
                            "items": {"type": "string"},
                        }
                    },
                    "required": ["char_list"],
                },
            },
        }

    @property
    def schema_char_appearance(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "char_appearance",
                "description": "Generate character visual descriptions",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "char_list": {"type": "array", "items": {"type": "string"}},
                        "setting": {"type": "string"},
                        "hints": {
                            "type": "object",
                            "additionalProperties": {"type": "string"},
                            "description": "Optional per-character hints: { 'Имя': 'краткое описание' }",
                        },
                    },
                    "required": ["char_list", "setting"],
                },
            },
        }


    async def loc_description(self, client: OpenRouterClient, loc_list: List[str], setting: str) -> LocationDescription:
        prompt = """Task: Generate detailed visual location descriptions for AI image generation.

Output: JSON object:
{"descriptions": ["...", "..."]} one per location in SAME ORDER.

Each description must:
- Start with camera perspective and composition
- Include: architecture, layout, atmosphere, lighting, textures
- Include distinctive features
- End with: "Rendered in environmental concept art style with high detail, clear lighting, atmospheric perspective, and high resolution."
- NO characters, text, frames, or UI elements

Return ONLY JSON.
"""

        input_data = {"locations": loc_list, "setting": setting}

        response = await client.generate_completion(
            prompt=json.dumps(input_data, ensure_ascii=False),
            model=self.model or "google/gemini-2.5-flash",
            temperature=0.1,
            system_prompt=prompt,
            operation_name="loc_description",
            response_format={"type": "json_object"},
        )

        raw = (response["choices"][0]["message"]["content"] or "").strip()
        if "```json" in raw:
            raw = raw.split("```json", 1)[1].split("```", 1)[0].strip()

        try:
            data = json.loads(raw)
        except Exception:
            descs = [raw]
            while len(descs) < len(loc_list):
                descs.append("")
            return LocationDescription(descriptions=descs[: len(loc_list)])

        descs = data.get("descriptions")
        if not isinstance(descs, list):
            descs = [raw]

        if len(descs) < len(loc_list):
            descs = list(descs) + [""] * (len(loc_list) - len(descs))
        if len(descs) > len(loc_list):
            descs = descs[: len(loc_list)]

        return LocationDescription(descriptions=[str(x) for x in descs])

    async def loc_graph(self, client: OpenRouterClient, location_list: List[str]) -> LocationGraph:
        base_prompt = """Task: Generate JSON graph of location transitions.

Transition Types (ONLY use these):
- "direct_path" (door, corridor, road)
- "transport" (vehicle/special transport)
- "teleport" (instant transition)
- "conditional" (requires conditions)
- "secret" (hidden passage)

Output format (STRICT):
{
  "nodes": [
    {
      "id": "string - EXACTLY the location name",
      "label": "string - display name (usually same as id)",
      "type": "indoor" | "outdoor" | "dungeon" | "city" | "wilderness" | "special"
    }
  ],
  "edges": [
    {
      "source": "string - id of source location",
      "target": "string - id of target location",
      "label": "direct_path" | "transport" | "teleport" | "conditional" | "secret",
      "directed": true,
      "bidirectional": false or true
    }
  ]
}

Rules:
- Include ALL locations from location_list at least once in nodes.
- Use ONLY the types above, EXACT spelling.
- Do NOT wrap response in any additional fields like "graph" or "data".
- Return ONLY ONE JSON object with fields "nodes" and "edges".
Return ONLY JSON.
"""

        max_attempts = 3
        error_hint = ""
        allowed_node_types = {"indoor", "outdoor", "dungeon", "city", "wilderness", "special"}

        for attempt in range(1, max_attempts + 1):
            prompt = base_prompt + (f"\nPrevious attempt error: {error_hint}\n" if error_hint else "")

            response = await client.generate_completion(
                prompt=json.dumps({"location_list": location_list}, ensure_ascii=False),
                model=self.model or "google/gemini-2.5-flash",
                temperature=0.1,
                system_prompt=prompt,
                operation_name=f"loc_graph_attempt_{attempt}",
                response_format={"type": "json_object"},
            )

            raw = (response["choices"][0]["message"]["content"] or "").strip()
            if "```json" in raw:
                raw = raw.split("```json", 1)[1].split("```", 1)[0].strip()

            try:
                data = json.loads(raw)
            except JSONDecodeError as e:
                error_hint = f"Not valid JSON: {e}"
                if attempt == max_attempts:
                    raise
                continue

            if not (isinstance(data, dict) and "nodes" in data and "edges" in data):
                error_hint = "Must return object with keys nodes and edges"
                if attempt == max_attempts:
                    raise ValueError("loc_graph: unexpected JSON schema")
                continue

            for node in data.get("nodes", []):
                if isinstance(node, dict):
                    t = node.get("type")
                    if t not in allowed_node_types:
                        node["type"] = "special"

            try:
                return LocationGraph(**data)
            except Exception as ve:
                error_hint = f"Validation failed: {ve}"
                if attempt == max_attempts:
                    raise

        raise RuntimeError("loc_graph: unexpected fall-through")

    @property
    def schema_loc_description(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "loc_description",
                "description": "Generate location visual descriptions",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "loc_list": {"type": "array", "items": {"type": "string"}},
                        "setting": {"type": "string"},
                    },
                    "required": ["loc_list", "setting"],
                },
            },
        }

    @property
    def schema_loc_graph(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "loc_graph",
                "description": "Generate location connection graph",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location_list": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["location_list"],
                },
            },
        }



    async def story_graph(
        self,
        client: OpenRouterClient,
        char_graph: Dict,
        loc_graph: Dict,
        ellipsis: str,
        setting: str,
    ) -> StoryGraph:
        prompt = """Task: Generate narrative story graph.

Node Types:
- "character_group"
- "key_interaction"
- "character_state"
- "plot_point"

Edge Types:
- "action"
- "decision"
- "revelation"
- "conflict"
- "transformation"
- "external_event"

Return ONLY JSON with nodes/edges.
"""

        context = {"char_graph": char_graph, "loc_graph": loc_graph, "ellipsis": ellipsis, "setting": setting}

        response = await client.generate_completion(
            prompt=json.dumps(context, ensure_ascii=False),
            model=self.model or "google/gemini-2.5-flash",
            temperature=0.3,
            system_prompt=prompt,
            operation_name="story_graph",
            response_format={"type": "json_object"},
        )

        raw = (response["choices"][0]["message"]["content"] or "").strip()
        if "```json" in raw:
            raw = raw.split("```json", 1)[1].split("```", 1)[0].strip()

        try:
            data = json.loads(raw)
        except JSONDecodeError:
            start = raw.find("{")
            end = raw.rfind("}")
            if start != -1 and end != -1 and end > start:
                data = json.loads(raw[start : end + 1])
            else:
                raise

        return StoryGraph(**data)

    @property
    def schema_story_graph(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "story_graph",
                "description": "Generate narrative story graph",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "char_graph": {"type": "object"},
                        "loc_graph": {"type": "object"},
                        "ellipsis": {"type": "string"},
                        "setting": {"type": "string"},
                    },
                    "required": ["char_graph", "loc_graph", "ellipsis", "setting"],
                },
            },
        }



    async def call_agent(
        self,
        agent_name: str,
        input_data: Dict[str, Any],
        client: OpenRouterClient,
    ) -> Dict[str, Any]:
        if agent_name not in self.agents:
            error_msg = f"Agent '{agent_name}' not found. Available: {list(self.agents.keys())}"
            return {"error": error_msg}

        agent = self.agents[agent_name]

        allowed_keys: List[str] = []
        for tool_schema in agent.toolbox.get_tools_for_openrouter():
            params = tool_schema.get("function", {}).get("parameters", {}) or {}
            props = params.get("properties", {}) or {}
            allowed_keys.extend(list(props.keys()))
        allowed_keys = list(set(allowed_keys))

        base_inputs = {k: v for k, v in self.input_dict.items() if k in allowed_keys}
        call_inputs = {k: v for k, v in (input_data or {}).items() if k in allowed_keys}
        merged_inputs = {**base_inputs, **call_inputs}
        agent.inputs = merged_inputs

        result = await agent.run(client)

        try:
            if "results" in result and result["results"]:
                Toolbox.json_parser(result, self.input_dict)
        except Exception as e:
            app_logger.error(f"json_parser failed: {e}")

        return result

    @property
    def schema_call_agent(self) -> Dict[str, Any]:
        available_agents = list(self.agents.keys()) if self.agents else []
        return {
            "type": "function",
            "function": {
                "name": "call_agent",
                "description": f"Call specialized agent. Available: {', '.join(available_agents)}",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "agent_name": {"type": "string"},
                        "input_data": {"type": "object"},
                    },
                    "required": ["agent_name", "input_data"],
                },
            },
        }