from __future__ import annotations

from sciorch.llm.openai_compatible import ModelPricing


KNOWN_MODEL_STRENGTH: dict[str, int] = {
    # Frontier / strongest
    "o3": 5,
    "gpt-5": 5,
    "gpt-5-pro": 5,
    "gpt-5.2": 5,
    "gpt-5.2-pro": 5,
    "gpt-5.4": 5,
    "gpt-5.4-pro": 5,
    "claude-sonnet-4-5": 5,
    "claude-sonnet-4-5-20250929": 5,
    "gemini-3-pro-preview": 5,
    "deepseek-r1": 5,
    # Strong
    "gpt-5.4-mini": 4,
    "gpt-4o": 5,
    "gpt-4.1": 4,
    "gemini-2.5-pro": 4,
    "claude-sonnet-4-20250514": 4,
    "claude-4-sonnet": 4,
    "claude-4-sonnet-20250514": 4,
    "claude-4-5-sonnet": 4,
    # Balanced
    "gpt-5-mini": 3,
    # Economy
    "gpt-5.4-nano": 2,
    "gpt-5-nano": 2,
    "gpt-4.1-mini": 2,
    "gpt-4.1-nano": 2,
    "gpt-4o-mini": 2,
    "o3-mini": 2,
    "o4-mini": 2,
    "gemini-2.5-flash": 2,
    "gemini-2.5-flash-image": 2,
    "gemini-3-flash-preview": 2,
    "claude-4-5-haiku": 2,
    "claude-haiku-4-5-20251001": 2,
}


def supports_image_inputs(model_name: str) -> bool:
    normalized = model_name.strip().lower()
    if normalized in {
        "o3-mini",
        "o3-mini-2025-01-31",
    }:
        return False
    return True


def model_strength_score(model_name: str) -> int:
    normalized = model_name.strip().lower()
    if normalized in KNOWN_MODEL_STRENGTH:
        return KNOWN_MODEL_STRENGTH[normalized]

    for known_name, score in sorted(KNOWN_MODEL_STRENGTH.items(), key=lambda item: len(item[0]), reverse=True):
        if known_name in normalized:
            return score

    if any(token in normalized for token in ("mini", "flash", "haiku", "fast")):
        return 2
    if any(token in normalized for token in ("sonnet", "pro", "gpt-4", "gpt-5")):
        return 4
    return 3


def model_strength_tier(model_name: str) -> str:
    score = model_strength_score(model_name)
    if score >= 5:
        return "frontier"
    if score >= 4:
        return "strong"
    if score >= 3:
        return "balanced"
    return "economy"


def model_relative_cost_tier(model_name: str) -> str:
    pricing = ModelPricing.resolve_pricing(model_name)
    if pricing is None:
        return "unknown"

    input_price = float(pricing["input"])
    if input_price <= 0.0004:
        return "low"
    if input_price <= 0.0015:
        return "medium"
    return "high"


def model_selection_note(model_name: str) -> str:
    tier = model_strength_tier(model_name)
    if tier == "frontier":
        return "Best for hard reasoning, subtle option distinctions, and final cross-checks."
    if tier == "strong":
        return "Good default for serious scientific or multimodal verification."
    if tier == "balanced":
        return "Use for moderate checks when the task is already narrowed down."
    return "Use only for narrow extraction or lightweight verification after the main claim is stable."
