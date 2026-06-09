"""
Derivative Notice:
- Inspired by: AOrchestra `base/engine/async_llm.py` (Apache-2.0)
- Modified by: SciOrch contributors for OpenAI-compatible multimodal Reasoning calls.
"""
from __future__ import annotations

import base64
import io
import os
from pathlib import Path
import os
from dataclasses import dataclass
import re
from typing import Any, Optional
from urllib.parse import urlparse


@dataclass
class LLMCallResult:
    text: str
    model: str
    input_tokens: int
    output_tokens: int
    cost: float
    reasoning: str = ""
    raw: Any = None


class ModelPricing:
    """USD price per 1K tokens."""

    PRICES = {
        # OpenAI
        "gpt-4o": {"input": 0.0025, "output": 0.01},
        "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
        "gpt-4.1": {"input": 0.002, "output": 0.008},
        "gpt-4.1-mini": {"input": 0.0004, "output": 0.0016},
        "gpt-4.1-nano": {"input": 0.0001, "output": 0.0004},
        "o3": {"input": 0.002, "output": 0.008},
        "o3-mini": {"input": 0.0011, "output": 0.0044},
        "o4-mini": {"input": 0.0011, "output": 0.0044},
        "gpt-5": {"input": 0.00125, "output": 0.01},
        "gpt-5-pro": {"input": 0.015, "output": 0.12},
        "gpt-5.2": {"input": 0.00175, "output": 0.014},
        "gpt-5.2-pro": {"input": 0.021, "output": 0.168},
        "gpt-5.4": {"input": 0.0025, "output": 0.015},
        "gpt-5.4-pro": {"input": 0.03, "output": 0.18},
        "gpt-5.4-mini": {"input": 0.00075, "output": 0.0045},
        "gpt-5.4-nano": {"input": 0.0002, "output": 0.00125},
        "gpt-5-mini": {"input": 0.00025, "output": 0.002},
        "gpt-5-nano": {"input": 0.00005, "output": 0.0004},
        # Anthropic
        "claude-sonnet-4-20250514": {"input": 0.003, "output": 0.015},
        "claude-4-sonnet": {"input": 0.003, "output": 0.015},
        "claude-4-sonnet-20250514": {"input": 0.003, "output": 0.015},
        "claude-4-5-sonnet": {"input": 0.003, "output": 0.015},
        "claude-sonnet-4-5": {"input": 0.003, "output": 0.015},
        "claude-sonnet-4-5-20250929": {"input": 0.003, "output": 0.015},
        "claude-4-5-haiku": {"input": 0.00088, "output": 0.0044},
        "claude-haiku-4-5-20251001": {"input": 0.00088, "output": 0.0044},
        # Google Gemini
        "gemini-2.5-pro": {"input": 0.00125, "output": 0.01},
        "gemini-2.5-flash": {"input": 0.0003, "output": 0.00252},
        "gemini-2.5-flash-image": {"input": 0.0003, "output": 0.03},
        "gemini-3-pro-preview": {"input": 0.002, "output": 0.004},
        "gemini-3.1-pro-preview": {"input": 0.003, "output": 0.012},
        "gemini-3-flash-preview": {"input": 0.0005, "output": 0.003},
        # DeepSeek
        "deepseek/deepseek-chat-v3.1": {"input": 0.00025, "output": 0.001},
        "deepseek-chat": {"input": 0.00025, "output": 0.001},
        "deepseek-v3": {"input": 0.00025, "output": 0.001},
        "deepseek-v3.1": {"input": 0.00025, "output": 0.001},
        "deepseek-v3.2": {"input": 0.00025, "output": 0.001},
        "deepseek-r1": {"input": 0.00055, "output": 0.00219},
        # Other gateways used in AOrchestra
        "moonshotai/kimi-k2": {"input": 0.000296, "output": 0.001185},
        "z-ai/glm-4.5": {"input": 0.00033, "output": 0.00132},
        "x-ai/grok-4-fast": {"input": 0.0002, "output": 0.0005},
    }

    @classmethod
    def resolve_pricing(cls, model_name: str) -> Optional[dict[str, float]]:
        if model_name in cls.PRICES:
            return cls.PRICES[model_name]
        for known, price in cls.PRICES.items():
            if known in model_name:
                return price
        return None

    @classmethod
    def get_price(cls, model_name: str, token_type: str) -> float:
        pricing = cls.resolve_pricing(model_name)
        if pricing is not None:
            return pricing[token_type]
        return 0.0


class OpenAICompatibleClient:
    """Async client for OpenAI-compatible chat/responses endpoints."""

    THINK_BLOCK_PATTERN = re.compile(r"<think>\s*([\s\S]*?)\s*</think>", re.IGNORECASE)
    DATA_URL_PATTERN = re.compile(r"data:image/[^;]+;base64,[A-Za-z0-9+/=\n\r]+")
    MAX_ERROR_MESSAGE_CHARS = 800

    @staticmethod
    def _should_bypass_env_proxies(base_url: Optional[str]) -> bool:
        if not base_url:
            return False
        hostname = (urlparse(str(base_url)).hostname or "").strip().lower()
        return hostname in {"127.0.0.1", "localhost", "::1"}

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key_env: str = "OPENAI_API_KEY",
        timeout_s: int = 300,
        default_temperature: Optional[float] = None,
        enable_thinking: Optional[bool] = None,
        allow_responses_fallback: Optional[bool] = None,
    ) -> None:
        try:
            from openai import AsyncOpenAI
        except Exception as exc:
            raise RuntimeError("openai package is required: pip install openai") from exc

        # Priority:
        # 1) explicit constructor args
        # 2) configured env key name
        # 3) common fallback env names for gateway setups
        resolved_key = (
            api_key
            or os.getenv(api_key_env)
            or os.getenv("OPENAI_API_KEY")
            or os.getenv("api_key")
            or os.getenv("API_KEY")
        )
        if not resolved_key:
            raise ValueError(f"API key is required via {api_key_env} or constructor argument")

        resolved_base_url = (
            base_url
            or os.getenv("OPENAI_BASE_URL")
            or os.getenv("base_url")
            or os.getenv("BASE_URL")
        )

        client_kwargs: dict[str, Any] = {
            "api_key": resolved_key,
            "base_url": resolved_base_url,
            "timeout": timeout_s,
        }
        if self._should_bypass_env_proxies(resolved_base_url):
            try:
                import httpx
            except Exception as exc:
                raise RuntimeError("httpx package is required: pip install httpx") from exc
            client_kwargs["http_client"] = httpx.AsyncClient(
                trust_env=False,
                timeout=timeout_s,
                follow_redirects=True,
            )

        try:
            self._client = AsyncOpenAI(**client_kwargs)
        except ImportError as exc:
            # If a SOCKS proxy is exported in env but socksio is unavailable,
            # fall back to a client that ignores env proxies instead of failing
            # at construction time.
            if "socksio" not in str(exc).lower() or "http_client" in client_kwargs:
                raise
            try:
                import httpx
            except Exception as httpx_exc:
                raise RuntimeError("httpx package is required: pip install httpx") from httpx_exc
            fallback_kwargs = dict(client_kwargs)
            fallback_kwargs["http_client"] = httpx.AsyncClient(
                trust_env=False,
                timeout=timeout_s,
                follow_redirects=True,
            )
            self._client = AsyncOpenAI(**fallback_kwargs)
        self._default_temperature = default_temperature
        self._enable_thinking = enable_thinking
        if allow_responses_fallback is None:
            self._allow_responses_fallback = not self._should_bypass_env_proxies(resolved_base_url)
        else:
            self._allow_responses_fallback = allow_responses_fallback

    @staticmethod
    def _image_to_data_url(image: Any) -> str:
        """Convert a PIL image or file path to base64 data URL."""
        if isinstance(image, (str, os.PathLike)):
            img_path = Path(image)
            if not img_path.exists():
                raise FileNotFoundError(f"Image not found: {img_path}")
            with open(img_path, "rb") as f:
                raw = f.read()
            suffix = img_path.suffix.lower().lstrip(".")
            if suffix == "jpg":
                suffix = "jpeg"
            if suffix not in ("png", "jpeg", "gif", "webp"):
                suffix = "png"
            encoded = base64.b64encode(raw).decode("utf-8")
            return f"data:image/{suffix};base64,{encoded}"
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{encoded}"

    @classmethod
    def _split_thinking_text(cls, text: str) -> tuple[str, str]:
        if not text:
            return "", ""
        reasoning_chunks = [match.strip() for match in cls.THINK_BLOCK_PATTERN.findall(text) if match.strip()]
        cleaned = cls.THINK_BLOCK_PATTERN.sub("", text).strip()
        reasoning = "\n".join(chunk for chunk in reasoning_chunks if chunk)
        return cleaned, reasoning

    @classmethod
    def _sanitize_error_text(cls, text: str, *, max_chars: Optional[int] = None) -> str:
        if not text:
            return ""
        sanitized = cls.DATA_URL_PATTERN.sub("<image-data-url omitted>", str(text))
        limit = cls.MAX_ERROR_MESSAGE_CHARS if max_chars is None else max_chars
        if len(sanitized) > limit:
            remaining = len(sanitized) - limit
            sanitized = f"{sanitized[:limit]}... [truncated {remaining} chars]"
        return sanitized

    @classmethod
    def _format_exception(cls, exc: Exception) -> str:
        return f"{type(exc).__name__}: {cls._sanitize_error_text(str(exc))}"

    @classmethod
    def _content_block_to_text(cls, block: Any) -> str:
        if block is None:
            return ""
        if isinstance(block, str):
            return block
        if isinstance(block, dict):
            block_type = str(block.get("type") or "").strip().lower()
            if block_type in {"reasoning", "reasoning_content"}:
                return ""
            for key in ("text", "content", "value"):
                value = block.get(key)
                text = cls._content_to_text(value)
                if text:
                    return text
            return ""
        block_type = str(getattr(block, "type", "") or "").strip().lower()
        if block_type in {"reasoning", "reasoning_content"}:
            return ""
        for attr in ("text", "content", "value"):
            if hasattr(block, attr):
                text = cls._content_to_text(getattr(block, attr))
                if text:
                    return text
        return ""

    @classmethod
    def _content_to_text(cls, content: Any) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts = [cls._content_block_to_text(item) for item in content]
            return "\n".join(part for part in parts if part).strip()
        if isinstance(content, dict):
            return cls._content_block_to_text(content).strip()
        return str(content).strip()

    @classmethod
    def _extract_chat_text(cls, response: Any) -> str:
        try:
            message = response.choices[0].message
        except Exception:
            return ""
        content = getattr(message, "content", None)
        if content is None and isinstance(message, dict):
            content = message.get("content")
        text = cls._content_to_text(content)
        cleaned, _ = cls._split_thinking_text(text)
        return cleaned

    @classmethod
    def _reasoning_to_text(cls, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, list):
            parts = [cls._reasoning_to_text(item) for item in value]
            return "\n".join(part for part in parts if part)
        if isinstance(value, dict):
            for key in ("reasoning", "reasoning_content", "text", "content"):
                text = cls._reasoning_to_text(value.get(key))
                if text:
                    return text
            return ""
        for attr in ("reasoning", "reasoning_content", "text", "content"):
            if hasattr(value, attr):
                text = cls._reasoning_to_text(getattr(value, attr))
                if text:
                    return text
        return ""

    @classmethod
    def _extract_chat_reasoning(cls, response: Any) -> str:
        try:
            message = response.choices[0].message
        except Exception:
            return ""
        for attr in ("reasoning", "reasoning_content"):
            if hasattr(message, attr):
                text = cls._reasoning_to_text(getattr(message, attr))
                if text:
                    return text
        if isinstance(message, dict):
            for key in ("reasoning", "reasoning_content"):
                text = cls._reasoning_to_text(message.get(key))
                if text:
                    return text
        content = getattr(message, "content", None)
        if content is None and isinstance(message, dict):
            content = message.get("content")
        _, inline_reasoning = cls._split_thinking_text(cls._content_to_text(content))
        return inline_reasoning

    @staticmethod
    def _extract_responses_text(response: Any) -> str:
        if hasattr(response, "output_text") and response.output_text:
            return str(response.output_text)
        return ""

    @classmethod
    def _extract_responses_reasoning(cls, response: Any) -> str:
        output = getattr(response, "output", None)
        if not output:
            return ""
        chunks: list[str] = []
        for item in output:
            content = getattr(item, "content", None)
            if content is None and isinstance(item, dict):
                content = item.get("content")
            if not content:
                continue
            for block in content:
                block_type = getattr(block, "type", None)
                if block_type is None and isinstance(block, dict):
                    block_type = block.get("type")
                if block_type not in {"reasoning", "reasoning_content"}:
                    continue
                text = cls._reasoning_to_text(block)
                if text:
                    chunks.append(text)
        return "\n".join(chunk for chunk in chunks if chunk)

    @staticmethod
    def _chat_usage(response: Any) -> tuple[int, int]:
        usage = getattr(response, "usage", None)
        if usage is None:
            return 0, 0
        return int(getattr(usage, "prompt_tokens", 0) or 0), int(getattr(usage, "completion_tokens", 0) or 0)

    @staticmethod
    def _responses_usage(response: Any) -> tuple[int, int]:
        usage = getattr(response, "usage", None)
        if usage is None:
            return 0, 0
        in_tokens = int(getattr(usage, "input_tokens", 0) or 0)
        out_tokens = int(getattr(usage, "output_tokens", 0) or 0)
        return in_tokens, out_tokens

    @staticmethod
    def _estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
        input_cost = (input_tokens / 1000.0) * ModelPricing.get_price(model, "input")
        output_cost = (output_tokens / 1000.0) * ModelPricing.get_price(model, "output")
        return input_cost + output_cost

    def _build_extra_body(self) -> dict[str, Any] | None:
        if self._enable_thinking is None:
            return None
        return {
            "chat_template_kwargs": {
                "enable_thinking": self._enable_thinking,
            }
        }

    async def ask(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        images: Optional[list[Any]] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
    ) -> LLMCallResult:
        """Call model with optional multimodal input."""
        resolved_temperature = temperature if temperature is not None else self._default_temperature
        content_blocks: list[dict[str, Any]] = []
        if images:
            for image in images:
                content_blocks.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": self._image_to_data_url(image), "detail": "auto"},
                    }
                )
        content_blocks.append({"type": "text", "text": user_prompt})

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content_blocks if images else user_prompt},
        ]

        # Prefer chat.completions first.
        chat_exc: Exception | None = None
        try:
            kwargs: dict[str, Any] = {
                "model": model,
                "messages": messages,
            }
            if max_tokens is not None:
                kwargs["max_tokens"] = max_tokens
            if resolved_temperature is not None:
                kwargs["temperature"] = resolved_temperature
            extra_body = self._build_extra_body()
            if repetition_penalty is not None:
                extra_body = extra_body or {}
                extra_body["repetition_penalty"] = repetition_penalty
            if extra_body is not None:
                kwargs["extra_body"] = extra_body

            response = await self._client.chat.completions.create(**kwargs)
            text = self._extract_chat_text(response)
            reasoning = self._extract_chat_reasoning(response)
            input_tokens, output_tokens = self._chat_usage(response)
            cost = self._estimate_cost(model, input_tokens, output_tokens)
            return LLMCallResult(
                text=text,
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost=cost,
                reasoning=reasoning,
                raw=response,
            )
        except Exception as exc:
            chat_exc = exc

        if chat_exc is not None and not self._allow_responses_fallback:
            raise RuntimeError(
                "chat.completions call failed and responses fallback is disabled for this endpoint. "
                f"chat_error={self._format_exception(chat_exc)}"
            ) from chat_exc

        # Fallback to responses API.
        input_blocks: list[dict[str, Any]] = []
        if images:
            for image in images:
                input_blocks.append(
                    {
                        "type": "input_image",
                        "image_url": self._image_to_data_url(image),
                    }
                )
        input_blocks.append({"type": "input_text", "text": user_prompt})

        try:
            kwargs: dict[str, Any] = {
                "model": model,
                "input": [
                    {"role": "system", "content": [{"type": "input_text", "text": system_prompt}]},
                    {"role": "user", "content": input_blocks},
                ],
                "max_output_tokens": max_tokens,
                "temperature": resolved_temperature,
            }
            extra_body = self._build_extra_body()
            if extra_body is not None:
                kwargs["extra_body"] = extra_body
            response = await self._client.responses.create(**kwargs)
            text = self._extract_responses_text(response)
            reasoning = self._extract_responses_reasoning(response)
            input_tokens, output_tokens = self._responses_usage(response)
            cost = self._estimate_cost(model, input_tokens, output_tokens)
            return LLMCallResult(
                text=text,
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost=cost,
                reasoning=reasoning,
                raw=response,
            )
        except Exception as responses_exc:
            if chat_exc is None:
                raise RuntimeError(
                    "responses API call failed. "
                    f"responses_error={self._format_exception(responses_exc)}"
                ) from responses_exc
            raise RuntimeError(
                "Both chat.completions and responses API calls failed. "
                f"chat_error={self._format_exception(chat_exc)}; "
                f"responses_error={self._format_exception(responses_exc)}"
            ) from responses_exc
