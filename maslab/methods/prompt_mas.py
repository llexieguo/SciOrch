"""Compatibility exports for prompt-based MAS baselines.

Inspired by MASLab's prompt-based MAS methods:
https://github.com/MASWorks/MASLab

The implementation is split across MASLab-style modules:

- `maslab.methods.base`
- `maslab.methods.llm_debate.llm_debate_main`
- `maslab.methods.self_consistency.self_consistency_main`
- `maslab.datasets.loader`
- `maslab.utils.formatting`
- `maslab.utils.images`
"""

from maslab.datasets.loader import load_combined_dataset
from maslab.methods.base import PromptMASBaseline, PromptMASConfig
from maslab.methods.llm_debate.llm_debate_main import LLMDebateBaseline
from maslab.methods.self_consistency.self_consistency_main import SelfConsistencyBaseline
from maslab.utils.formatting import (
    extract_answer_letter,
    format_options,
    format_query,
    normalize_answer_letter,
    strip_option_prefix,
)
from maslab.utils.images import (
    ImageLoadResult,
    load_sample_images,
    load_sample_images_with_warnings,
    resolve_image_path,
)

__all__ = [
    "ImageLoadResult",
    "LLMDebateBaseline",
    "PromptMASBaseline",
    "PromptMASConfig",
    "SelfConsistencyBaseline",
    "extract_answer_letter",
    "format_options",
    "format_query",
    "load_combined_dataset",
    "load_sample_images",
    "load_sample_images_with_warnings",
    "normalize_answer_letter",
    "resolve_image_path",
    "strip_option_prefix",
]
