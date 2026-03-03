import os
import httpx
import json
import time
import hashlib
from collections import OrderedDict
from contextvars import ContextVar
from datetime import datetime
from typing import Dict, Optional, Any, List, Union, Callable

from src.utils.decorators import infinite_retry_with_backoff
from src.logger import api_logger


TRACE_HOOK: ContextVar[Optional[Callable[[Dict[str, Any]], None]]] = ContextVar("TRACE_HOOK", default=None)


def _maybe_trace(event: Dict[str, Any]) -> None:
    hook = TRACE_HOOK.get()
    if hook is None:
        return
    try:
        hook(event)
    except Exception:
        pass


def _sha(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


class OpenRouterClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 300.0,
        extra_headers: Optional[Dict[str, str]] = None,
    ):
        self.api_key = api_key or os.getenv("LLM_API_KEY") or os.getenv("API_KEY")
        self.base_url = (base_url or os.getenv("LLM_BASE_URL", "https://openrouter.ai/api/v1")).rstrip("/")

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if extra_headers:
            headers.update(extra_headers)

        self.client = httpx.AsyncClient(
            headers=headers,
            timeout=timeout,
            trust_env=False)

        self.call_count = 0
        self.total_tokens_used = 0
        self.supports_functions = str(os.getenv("LLM_SUPPORTS_FUNCTIONS", "true")).lower() in ("1", "true", "yes")

        try:
            self._embed_cache_max = int(os.getenv("EMBED_CACHE_MAX", "2048"))
        except Exception:
            self._embed_cache_max = 2048
        self._embed_cache: "OrderedDict[str, List[float]]" = OrderedDict()

    def _embed_cache_get(self, key: str) -> Optional[List[float]]:
        if not key:
            return None
        v = self._embed_cache.get(key)
        if v is None:
            return None
        self._embed_cache.move_to_end(key, last=True)
        return v

    def _embed_cache_put(self, key: str, vec: List[float]) -> None:
        if not key:
            return
        self._embed_cache[key] = vec
        self._embed_cache.move_to_end(key, last=True)
        while len(self._embed_cache) > self._embed_cache_max:
            self._embed_cache.popitem(last=False)

    @infinite_retry_with_backoff(max_wait=120, max_retries=5)
    async def generate_completion(
        self,
        *,
        model: str = "google/gemini-2.5-flash",
        temperature: float = 0.7,
        max_tokens: int = 4000,
        messages: Optional[List[Dict[str, Any]]] = None,
        prompt: Optional[str] = None,
        system_prompt: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        operation_name: str = "unknown",
        response_format: Optional[Dict[str, str]] = None,
        supports_tools: Optional[bool] = None,
    ) -> Dict[str, Any]:
        self.call_count += 1
        call_id = f"{operation_name}_{self.call_count}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        t0 = time.perf_counter()

        if messages is None:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            if prompt is not None:
                messages.append({"role": "user", "content": prompt})

        api_logger.info(f"=== API CALL {call_id} START ===")
        api_logger.info(f"Operation: {operation_name}")
        api_logger.info(f"Model: {model}")

        store_full = str(os.getenv("TRACE_STORE_FULL_PROMPTS", "false")).lower() in ("1", "true", "yes")
        last_user = ""
        if messages:
            for m in reversed(messages):
                if m.get("role") == "user" and m.get("content"):
                    last_user = str(m.get("content"))
                    break

        _maybe_trace({
            "type": "llm.request",
            "call_id": call_id,
            "operation_name": operation_name,
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages_count": len(messages),
            "last_user_preview": last_user[:300],
            "last_user_sha256": _sha(last_user),
            "messages": messages if store_full else None,
        })

        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        use_tools = self.supports_functions if supports_tools is None else bool(supports_tools)
        if use_tools:
            if tools:
                payload["tools"] = tools
            if tool_choice is not None:
                payload["tool_choice"] = tool_choice
        if response_format:
            payload["response_format"] = response_format

        response = await self.client.post(f"{self.base_url}/chat/completions", json=payload)
        response.raise_for_status()
        result = response.json()

        if "choices" not in result or not result["choices"]:
            api_logger.error(f"Ошибка апи: {result}")
            raise ValueError("API returned response without choices array")

        content = result["choices"][0].get("message", {}).get("content", "")
        usage = result.get("usage", {}) or {}
        tokens_used = usage.get("total_tokens")
        if tokens_used is None:
            tokens_used = (usage.get("prompt_tokens", 0) or 0) + (usage.get("completion_tokens", 0) or 0)

        self.total_tokens_used += int(tokens_used or 0)

        dt = time.perf_counter() - t0
        api_logger.info(f"Response: {len(content)} chars, {tokens_used} tokens")
        api_logger.info(f"Cumulative tokens: {self.total_tokens_used}")
        api_logger.info(f"=== API CALL {call_id} END ===\n")

        _maybe_trace({
            "type": "llm.response",
            "call_id": call_id,
            "operation_name": operation_name,
            "model": model,
            "latency_s": dt,
            "tokens_used": int(tokens_used or 0),
            "content_preview": (content or "")[:500],
            "content_sha256": _sha(content or ""),
            "raw": result if store_full else None,
        })

        return result

    @infinite_retry_with_backoff(max_wait=120, max_retries=5)
    async def generate_embedding(
        self,
        text: str,
        model: Optional[str] = None,
        operation_name: str = "embedding",
    ) -> List[float]:
        embed_model = model or os.getenv("EMBED_MODEL_NAME", "text-embedding-3-small")
        key = f"{embed_model}:{_sha(text)}"

        cached = self._embed_cache_get(key)
        if cached is not None:
            _maybe_trace({
                "type": "embed.cache_hit",
                "operation_name": operation_name,
                "model": embed_model,
                "text_sha256": _sha(text),
                "len": len(cached),
            })
            return cached

        call_id = f"{operation_name}_{self.call_count + 1}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        t0 = time.perf_counter()

        api_logger.info(f"=== EMBEDDING CALL {call_id} START ===")
        api_logger.info(f"Operation: {operation_name}")
        api_logger.info(f"Embed model: {embed_model}")

        store_full = str(os.getenv("TRACE_STORE_FULL_PROMPTS", "false")).lower() in ("1", "true", "yes")
        _maybe_trace({
            "type": "embed.request",
            "call_id": call_id,
            "operation_name": operation_name,
            "model": embed_model,
            "text_preview": (text or "")[:300],
            "text_sha256": _sha(text),
            "text": text if store_full else None,
        })

        payload = {
            "model": embed_model,
            "input": text,
        }

        response = await self.client.post(f"{self.base_url}/embeddings", json=payload)
        response.raise_for_status()
        data = response.json()

        if "data" not in data or not data["data"]:
            api_logger.error(f"Ошибка эмбеддинга: {data}")
            raise ValueError("Embedding API returned no data")

        embedding = data["data"][0].get("embedding")
        if not isinstance(embedding, list):
            raise ValueError("Embedding format unexpected")

        api_logger.info(f"Embedding length: {len(embedding)}")
        api_logger.info(f"=== EMBEDDING CALL {call_id} END ===\n")

        dt = time.perf_counter() - t0
        _maybe_trace({
            "type": "embed.response",
            "call_id": call_id,
            "operation_name": operation_name,
            "model": embed_model,
            "latency_s": dt,
            "len": len(embedding),
        })

        self._embed_cache_put(key, embedding)
        return embedding  # type: ignore[return-value]

    @infinite_retry_with_backoff(max_wait=120, max_retries=5)
    async def generate_image(
        self,
        prompt: str,
        model: Optional[str] = None,
        aspect_ratio: str = "1:1",
        operation_name: str = "image_generation",
    ) -> Dict[str, Any]:
        image_model = model or os.getenv("IMAGE_MODEL_NAME", "google/gemini-2.5-flash-image")
        call_id = f"{operation_name}_{self.call_count + 1}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        t0 = time.perf_counter()

        api_logger.info(f"=== IMAGE CALL {call_id} START ===")
        api_logger.info(f"Operation: {operation_name}")
        api_logger.info(f"Image model: {image_model}")

        store_full = str(os.getenv("TRACE_STORE_FULL_PROMPTS", "false")).lower() in ("1", "true", "yes")
        _maybe_trace({
            "type": "image.request",
            "call_id": call_id,
            "operation_name": operation_name,
            "model": image_model,
            "aspect_ratio": aspect_ratio,
            "prompt_preview": (prompt or "")[:300],
            "prompt_sha256": _sha(prompt or ""),
            "prompt": prompt if store_full else None,
        })

        payload = {
            "model": image_model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            "modalities": ["image", "text"],
            "image_config": {
                "aspect_ratio": aspect_ratio,
            },
            "stream": False,
        }

        response = await self.client.post(f"{self.base_url}/chat/completions", json=payload)
        response.raise_for_status()
        data = response.json()

        if "choices" not in data or not data["choices"]:
            api_logger.error(f"Ошибка генерации изображения: {data}")
            raise ValueError("Image API returned no choices")

        message = data["choices"][0].get("message", {}) or {}
        images = message.get("images") or []

        if not images:
            api_logger.error(f"Ответ без images: {data}")
            raise ValueError("Image API returned no images")

        img_obj = images[0]
        api_logger.info(f"Image object keys: {list(img_obj.keys())}")
        api_logger.info(f"=== IMAGE CALL {call_id} END ===\n")

        dt = time.perf_counter() - t0
        _maybe_trace({
            "type": "image.response",
            "call_id": call_id,
            "operation_name": operation_name,
            "model": image_model,
            "latency_s": dt,
            "has_image_url": bool((img_obj.get("image_url") or {}).get("url")),
        })

        return img_obj

    async def close(self):
        await self.client.aclose()