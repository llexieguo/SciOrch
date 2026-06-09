"""Compatibility exports for prompt-based MAS baselines.

The implementation lives in maslab.methods.prompt_mas.
"""

from maslab.methods.prompt_mas import (
    LLMDebateBaseline,
    PromptMASBaseline,
    PromptMASConfig,
    SelfConsistencyBaseline,
    format_query,
    load_combined_dataset,
    load_sample_images,
    load_sample_images_with_warnings,
    normalize_answer_letter,
)

__all__ = [
    "LLMDebateBaseline",
    "PromptMASBaseline",
    "PromptMASConfig",
    "SelfConsistencyBaseline",
    "format_query",
    "load_combined_dataset",
    "load_sample_images",
    "load_sample_images_with_warnings",
    "normalize_answer_letter",
]
