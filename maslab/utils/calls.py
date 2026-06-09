from __future__ import annotations

from typing import Any

from sciorch.llm.openai_compatible import LLMCallResult


def call_to_json(result: LLMCallResult, *, role: str, label: str) -> dict[str, Any]:
    return {
        "role": role,
        "label": label,
        "model": result.model,
        "text": result.text,
        "reasoning": result.reasoning,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "cost": result.cost,
    }
