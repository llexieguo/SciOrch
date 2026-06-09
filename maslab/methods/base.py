from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sciorch.llm.openai_compatible import LLMCallResult


@dataclass
class PromptMASConfig:
    model: str
    max_tokens: int | None = 2048
    temperature: float | None = 0.7
    use_images: bool = True
    tolerate_truncated_images: bool = True


class PromptMASBaseline:
    system_prompt = (
        "You are a careful scientific reasoning agent. Solve the multiple-choice task. "
        "Always state the final answer exactly as \\boxed{LETTER}."
    )

    def __init__(self, client: Any, config: PromptMASConfig) -> None:
        self.client = client
        self.config = config

    async def _ask(self, prompt: str, images: list[Any] | None = None) -> LLMCallResult:
        return await self.client.ask(
            model=self.config.model,
            system_prompt=self.system_prompt,
            user_prompt=prompt,
            images=images if self.config.use_images else None,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
        )

    @staticmethod
    def _render_history(history: list[dict[str, str]]) -> str:
        rendered = []
        for message in history:
            role = message["role"].upper()
            rendered.append(f"{role}:\n{message['content']}")
        return "\n\n".join(rendered)

    @staticmethod
    def _aggregate_metrics(calls: list[dict[str, Any]]) -> dict[str, Any]:
        input_tokens = sum(int(call.get("input_tokens") or 0) for call in calls)
        output_tokens = sum(int(call.get("output_tokens") or 0) for call in calls)
        cost = sum(float(call.get("cost") or 0.0) for call in calls)
        return {
            "num_llm_calls": len(calls),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "cost": cost,
        }
