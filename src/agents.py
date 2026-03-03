from typing import Optional, Dict, Any, List, Tuple
import json
from src.logger import app_logger
from src.apicallhandler import OpenRouterClient


class Agent:
    def __init__(
            self,
            api_key: str,
            model: str,
            temperature: float,
            toolbox: 'Toolbox',
            freedom: str = "auto",
            reasoning: str = "minimal",
            supports_tools: Optional[bool] = None,
    ):
        self.api_key = api_key
        self.client: Optional[OpenRouterClient] = None
        self.model = model
        self.temperature = temperature
        self.toolbox = toolbox
        self.toolbox.model = model
        self.freedom = freedom
        self.reasoning = reasoning
        self.supports_tools = supports_tools
        self.role: Optional[str] = None
        self.inputs: Dict[str, Any] = {}

    def set_role(self, system_prompt: str) -> 'Agent':
        self.role = system_prompt
        return self

    def set_input(self, **kwargs: Any) -> 'Agent':
        self.inputs.update(kwargs)
        return self

    def _normalize_tool_choice(self) -> Any:
        if isinstance(self.freedom, dict):
            return self.freedom
        val = (self.freedom or "auto").lower()
        if val in ("auto", "none", "required"):
            return val
        return "auto"

    async def _tool_loop(
        self,
        client: OpenRouterClient,
        base_messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        max_iters: int = 2
    ) -> Tuple[Dict[str, Any], Dict[str, Any], List[Dict[str, Any]]]:
        messages = list(base_messages)
        collected_results: Dict[str, Any] = {}
        tool_choice = self._normalize_tool_choice()

        for i in range(max_iters):
            app_logger.info(f"[{self.__class__.__name__}] LLM turn {i + 1}")
            resp = await client.generate_completion(
                messages=messages,
                model=self.model,
                temperature=self.temperature,
                tools=tools,
                tool_choice=tool_choice,
                operation_name=f"{self.__class__.__name__}_tool_loop",
                supports_tools=self.supports_tools,
            )

            msg = resp["choices"][0]["message"]
            tool_calls = msg.get("tool_calls") or []

            if not tool_calls:
                fallback_calls = self.toolbox.extract_tool_calls(msg.get("content") or "", tools=tools)
                if fallback_calls:
                    tool_calls = fallback_calls

            if tool_calls:
                messages.append({
                    "role": "assistant",
                    "content": msg.get("content") or "",
                    "tool_calls": tool_calls
                })

                for tc in tool_calls:
                    func = tc["function"]["name"]
                    raw_args = tc["function"].get("arguments", "{}")
                    try:
                        args = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
                    except Exception:
                        args = {}

                    if hasattr(self.toolbox, func):
                        app_logger.info(f"Executing tool: {func}")
                        tool_fn = getattr(self.toolbox, func)
                        result_obj = await tool_fn(client, **args)

                        if hasattr(result_obj, "model_dump"):
                            result_payload = result_obj.model_dump()
                        elif hasattr(result_obj, "dict"):
                            result_payload = result_obj.dict()
                        else:
                            result_payload = result_obj

                        if func in collected_results:
                            prev = collected_results[func]
                            if isinstance(prev, list):
                                prev.append(result_payload)
                                collected_results[func] = prev
                            else:
                                collected_results[func] = [prev, result_payload]
                        else:
                            collected_results[func] = result_payload

                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc.get("id") or f"shim_{func}",
                            "content": json.dumps(result_payload, ensure_ascii=False)
                        })
                    else:
                        app_logger.warning(f"Тул '{func}' не найден. Игнорируем")
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc.get("id") or f"shim_{func}",
                            "content": json.dumps({"error": f"!!! Неизвестный тул {func}"}, ensure_ascii=False)
                        })
                continue

            messages.append({"role": "assistant", "content": msg.get("content", "")})
            return resp, collected_results, messages

        app_logger.warning(f"[{self.__class__.__name__}] Достигли лимита использования тулов")
        return resp, collected_results, messages

    async def run(self, client: OpenRouterClient) -> Dict[str, Any]:
        if not self.role:
            raise ValueError("Роль агента не установлена set_role()")

        tools = self.toolbox.get_tools_for_openrouter()
        base_messages = [
            {"role": "system", "content": self.role},
            {"role": "user", "content": json.dumps(self.inputs, ensure_ascii=False)}
        ]
        resp, collected_results, _ = await self._tool_loop(client, base_messages, tools)
        final_output = {
            "results": collected_results,
            "direct_response": resp["choices"][0]["message"].get("content")
        }
        return final_output


class MasterAgent(Agent):
    """
    Master agent with OPTIONAL evaluator/critic step (can be disabled).
    """

    def __init__(
            self,
            api,
            model,
            temperature,
            agents,
            toolbox,
            input_dict,
            freedom="auto",
            reasoning="minimal",
            enable_critic: bool = False,
    ):
        super().__init__(api, model, temperature, toolbox, freedom, reasoning)

        if not isinstance(input_dict, dict):
            raise ValueError("input_dict must be a dictionary")

        self.toolbox.agents = agents
        self.toolbox.input_dict = input_dict

        self.inputs = input_dict
        self.mock_inputs = None
        self.enable_critic = bool(enable_critic)

    @staticmethod
    async def make_mock_input(input_dict: dict) -> dict:
        def is_filled(value):
            if value is None:
                return False
            if isinstance(value, str) and value.strip() == "":
                return False
            if isinstance(value, (list, dict)) and len(value) == 0:
                return False
            return True

        mock = {}
        for key, value in input_dict.items():
            mock[key] = is_filled(value)
        return mock

    async def decide_next_agent(self, mock_context, client) -> str:
        prompt = f"""You are a master agent of a multi agent system.

Here is the current context:
{json.dumps(mock_context, ensure_ascii=False)}

Available agents: {list(self.toolbox.agents.keys())}.

Your goal: choose the next agent needed to fill ALL fields
required for a visual novel template.

Respond ONLY:
- the agent name (e.g. "char_agent"), OR
- "finished" if all fields are filled.
"""

        resp = await client.generate_completion(
            messages=[{"role": "system", "content": prompt}],
            model=self.model,
            temperature=0.3,
            operation_name="decide_next_agent_async",
        )

        text = resp["choices"][0]["message"]["content"]
        return (text or "").strip().lower()

    async def evaluate_agent_result(self, agent_name: str, result_payload: Any, client) -> Any:
        prompt_system = """
You are masteragent. Your task is to evaluate the response of an agent. Criteria: consistent, meaningful, detailed.
Response format: JSON only
{"OK": true} or {"OK": false}
No other messages.
""".strip()

        prompt_user = f"""
Agent result: '{agent_name}':
{json.dumps(result_payload, ensure_ascii=False)}
""".strip()

        resp = await client.generate_completion(
            messages=[
                {"role": "system", "content": prompt_system},
                {"role": "user", "content": prompt_user}
            ],
            model=self.model,
            temperature=0.0,
            operation_name=f"evaluate_{agent_name}_result",
            response_format={"type": "json_object"},
        )

        decision_json = json.loads(resp["choices"][0]["message"]["content"])
        return bool(decision_json.get("OK", False))

    async def run(self, client, max_steps=15):
        step = 0
        while step < max_steps:
            mock_inputs = await self.make_mock_input(self.inputs)
            next_agent = await self.decide_next_agent(mock_inputs, client)

            if not next_agent:
                app_logger.warning("[MasterAgent] No next agent. Stopping.")
                break

            if next_agent.lower() in {"finished", "done", "stop", "end"}:
                app_logger.info("[MasterAgent] Finished.")
                break

            app_logger.info(f"[MasterAgent] Step {step + 1}: Calling agent: {next_agent}")

            result = await self.toolbox.call_agent(
                next_agent,
                self.inputs,
                client=client
            )

            if not self.enable_critic:
                checker = True
            else:
                checker = await self.evaluate_agent_result(next_agent, result, client)

            if checker is True:
                self.toolbox.input_dict = self.toolbox.json_parser(result, self.toolbox.input_dict)
                self.inputs = self.toolbox.input_dict
                app_logger.info(f"[MasterAgent] Updated keys: {list(self.inputs.keys())}")
                step += 1
            else:
                app_logger.info(f"[MasterAgent] Agent {next_agent} rejected by evaluator")

        return self.inputs