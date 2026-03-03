import os
from typing import Dict, Optional
from dataclasses import dataclass
from src.logger import app_logger


@dataclass
class ModelConfig:
    name: str
    display_name: str
    context_window: int
    cost_per_1k_tokens: float
    best_for: list[str]
    supports_function_calling: bool = True


class ModelRouter:
    MODELS = {
        "fast": ModelConfig(
            name="openai/gpt-oss-120b",
            display_name="Gemini 2.5 Flash Lite",
            context_window=1_000_000,
            cost_per_1k_tokens=0.0,
            best_for=["quick_tasks", "simple_generation", "tool_calling"],
            supports_function_calling=True,
        ),
        "smart": ModelConfig(
            name="openai/gpt-oss-120b",
            display_name="Gemini 2.5 Flash",
            context_window=200_000,
            cost_per_1k_tokens=0.003,
            best_for=["complex_reasoning", "creative_writing", "critique"],
            supports_function_calling=True,
        ),
        "balanced": ModelConfig(
            name="openai/gpt-oss-120b",
            display_name="Gemini 2.5 Flash",
            context_window=32_000,
            cost_per_1k_tokens=0.0,
            best_for=["balanced_tasks", "reasoning", "medium_complexity"],
            supports_function_calling=True,
        ),
        "reasoning": ModelConfig(
            name="google/gemini-2.5-flash",
            display_name="Gemini 2.5 Flash",
            context_window=64_000,
            cost_per_1k_tokens=0.0008,
            best_for=["deep_reasoning", "complex_logic", "planning"],
            supports_function_calling=False,
        ),
        "creative": ModelConfig(
            name="openai/gpt-oss-120b",
            display_name="Gemini 2.5 Flash",
            context_window=200_000,
            cost_per_1k_tokens=0.003,
            best_for=["creative_writing", "storytelling", "character_development"],
            supports_function_calling=True,
        ),
    }

    DEFAULT_ROUTING = {
        "master": "reasoning",
        "char_agent": "creative",
        "loc_agent": "balanced",
        "story_agent": "smart",
        "setting_agent": "balanced",
        "outline_agent": "smart",
        "writer_agent": "creative",
        "critic_agent": "smart",
        "plot_thread_agent": "smart",
        "scene_microplanner_agent": "smart",
        "scene_editor_agent": "smart",
        "setting": "balanced",
        "outline": "smart",
        "critic": "smart",
        "writer": "creative",
        "renpy_agent": "fast"}

    def __init__(self, custom_routing: Optional[Dict[str, str]] = None):
        self.routing = self.DEFAULT_ROUTING.copy()
        if custom_routing:
            self.routing.update(custom_routing)

        for key in list(self.MODELS.keys()):
            env_name = os.getenv(f"LLM_MODEL_{key.upper()}_NAME")
            if env_name:
                self.MODELS[key].name = env_name
            env_fc = os.getenv(f"LLM_MODEL_{key.upper()}_FC")
            if env_fc is not None:
                self.MODELS[key].supports_function_calling = str(env_fc).lower() in ("1", "true", "yes")

        app_logger.info(f"ModelRouter initialized with routing: {self.routing}")

    def get_model_for_agent(self, agent_name: str) -> str:
        model_type = self.routing.get(agent_name.lower(), "balanced")
        model_config = self.MODELS.get(model_type)

        if not model_config:
            app_logger.warning(f"Unknown model type {model_type}, falling back to balanced")
            model_config = self.MODELS["balanced"]

        app_logger.info(f"Routing {agent_name} to {model_config.display_name} ({model_config.name})")
        return model_config.name

    def get_model_config(self, model_type: str) -> ModelConfig:
        return self.MODELS.get(model_type, self.MODELS["balanced"])

    def update_routing(self, agent_name: str, model_type: str):
        if model_type not in self.MODELS:
            raise ValueError(f"Unknown model type: {model_type}. Available: {list(self.MODELS.keys())}")

        self.routing[agent_name.lower()] = model_type
        app_logger.info(f"Updated routing: {agent_name} -> {model_type}")

    def get_all_models(self) -> Dict[str, ModelConfig]:
        return self.MODELS.copy()

    def supports_function_calling(self, agent_name: str) -> bool:
        model_type = self.routing.get(agent_name.lower(), "balanced")
        model_config = self.MODELS.get(model_type)
        return model_config.supports_function_calling if model_config else False

    def estimate_cost(self, agent_name: str, tokens_used: int) -> float:
        model_type = self.routing.get(agent_name.lower(), "balanced")
        model_config = self.MODELS.get(model_type, self.MODELS["balanced"])
        cost = (tokens_used / 1000) * model_config.cost_per_1k_tokens
        return round(cost, 6)