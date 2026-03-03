import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv

from src.logger import setup_logging, app_logger
from src.apicallhandler import OpenRouterClient
from src.orchestrator import VNOrchestrator
from src.orchestrator_simple import VNOrchestratorSimple
from src.router import ModelRouter
from src.utils.fileio import save_json


def _pick_orchestrator(client: OpenRouterClient, router: ModelRouter):
    variant = str(os.getenv("PIPELINE_VARIANT", "full")).strip().lower()
    if variant in ("worse", "simple", "baseline"):
        app_logger.info("PIPELINE_VARIANT=worse -> using VNOrchestratorSimple")
        return VNOrchestratorSimple(client, router)
    app_logger.info("PIPELINE_VARIANT=full -> using VNOrchestrator")
    return VNOrchestrator(client, router)


async def test_pipeline():
    load_dotenv()
    setup_logging("DEBUG")
    app_logger.info("Running generation...")

    api_key = os.getenv("API_KEY") or os.getenv("LLM_API_KEY")
    base_url = os.getenv("LLM_BASE_URL", "https://openrouter.ai/api/v1")
    if "openrouter.ai" in base_url and not api_key:
        app_logger.error("Ключ не найден")

    client = OpenRouterClient(api_key=api_key, base_url=base_url)
    router = ModelRouter()
    orchestrator = _pick_orchestrator(client, router)

    try:
        user_prompt = """Enter prompt here
""".strip()

        mc_name = "Имя ГГ"
        mc_description = (
            "описание главного персонажа"
        )

        extra_character_names = [
            "Пример"
        ]

        plot_freeform = """
""".strip()

        result = await orchestrator.generate_vn(
            user_prompt=user_prompt,
            story_length="medium",
            char_list=None,
            loc_list=None,
            setting="",
            max_branches=4,
            tone="dark",
            artstyle="anime",
            generate_images=False,
            time_choice="будущее",
            genre_choice="антиутопия",
            tone_choice_ru="напряжённый",
            mc_name=mc_name,
            mc_description=mc_description,
            extra_character_names=extra_character_names,
            plot_prefs=None,
            plot_freeform=plot_freeform,
            graphic_style_ru="аниме",
        )

        app_logger.info("Генерация завершена")
        app_logger.info(f"Generation ID: {result.get('generation_id')}")
        app_logger.info(f"Status: {result.get('status')}")
        app_logger.info(f"Total API calls: {client.call_count}")
        app_logger.info(f"Total tokens used: {client.total_tokens_used}")

        output_dir = os.getenv("OUTPUT_DIR", "output")
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        out_path = save_json(result, output_dir=output_dir, name=f"{result['generation_id']}.json")
        app_logger.info(f"Result saved to: {out_path}")

        return result

    finally:
        await client.close()
        app_logger.info("Long test complete")



if __name__ == "__main__":
    asyncio.run(test_pipeline())