"""Self-consistency baseline for SciOrch.

References:
- MASLab: https://github.com/MASWorks/MASLab
- CoMAS MASLab-style layout: https://github.com/xxyQwQ/CoMAS/tree/main/maslab

This module implements a MASLab-style self-consistency baseline: generate
multiple independent solutions, then aggregate them into one final boxed answer.
The implementation is adapted to SciOrch's multimodal benchmark format,
OpenAI-compatible API client, JSONL logging, image warning tracking, and
per-source metrics.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

from sciorch.llm.openai_compatible import LLMCallResult

from maslab.methods.base import PromptMASBaseline, PromptMASConfig
from maslab.utils.calls import call_to_json
from maslab.utils.formatting import extract_answer_letter, format_query, normalize_answer_letter
from maslab.utils.images import ImageLoadResult, load_sample_images_with_warnings


class SelfConsistencyBaseline(PromptMASBaseline):
    """MASLab-style self-consistency: independent samples followed by final aggregation."""

    def __init__(
        self,
        client: Any,
        config: PromptMASConfig,
        *,
        parallel_num: int = 5,
    ) -> None:
        super().__init__(client, config)
        if parallel_num <= 0:
            raise ValueError("parallel_num must be positive")
        self.parallel_num = parallel_num

    async def run_sample(
        self,
        sample: dict[str, Any],
        *,
        image_root: Path | None = None,
    ) -> dict[str, Any]:
        start = time.perf_counter()
        query = format_query(sample)
        image_result = (
            load_sample_images_with_warnings(
                sample,
                image_root,
                tolerate_truncated=self.config.tolerate_truncated_images,
            )
            if self.config.use_images
            else ImageLoadResult(images=[], warnings=[])
        )
        images = image_result.images

        async def run_candidate(index: int) -> tuple[int, LLMCallResult]:
            prompt = (
                f"{query}\n\n"
                f"This is independent solution attempt {index + 1}. "
                "Reason carefully and state the final answer exactly as \\boxed{LETTER}."
            )
            return index, await self._ask(prompt, images=images)

        candidate_pairs = await asyncio.gather(*(run_candidate(i) for i in range(self.parallel_num)))
        candidate_pairs.sort(key=lambda item: item[0])
        candidate_results = [result for _, result in candidate_pairs]
        calls = [
            call_to_json(result, role="candidate", label=f"solution_{index + 1}")
            for index, result in enumerate(candidate_results)
        ]

        aggregate_prompt = f"[Task]:\n{query}\n\n"
        for index, result in enumerate(candidate_results):
            aggregate_prompt += f"[Solution {index + 1}]:\n{result.text}\n\n"
        aggregate_prompt += (
            "Given the task and all the above solutions, reason over them carefully and provide "
            "a final answer exactly as \\boxed{LETTER}."
        )
        aggregate_result = await self._ask(aggregate_prompt, images=images)
        calls.append(call_to_json(aggregate_result, role="aggregator", label="final_aggregate"))

        options_len = len(sample.get("options") or [])
        predicted_letter, parse_error = extract_answer_letter(aggregate_result.text, options_len)
        gold_letter = normalize_answer_letter(sample.get("answer"), options_len)
        metrics = self._aggregate_metrics(calls)
        metrics["latency_seconds"] = time.perf_counter() - start
        metrics["mca"] = 1.0 if predicted_letter is not None and predicted_letter == gold_letter else 0.0

        return {
            "id": sample.get("id"),
            "task_id": sample.get("id"),
            "method": "self_consistency",
            "model": self.config.model,
            "source": sample.get("source"),
            "subject": sample.get("subject"),
            "question": sample.get("question"),
            "options": sample.get("options"),
            "gold_answer_raw": sample.get("answer"),
            "gold_answer_letter": gold_letter,
            "response": aggregate_result.text,
            "final_boxed_letter": predicted_letter,
            "parse_error": parse_error,
            "mca": metrics["mca"],
            "candidate_answers": [result.text for result in candidate_results],
            "image_load_warnings": image_result.warnings,
            "image_load_warning_count": len(image_result.warnings),
            "calls": calls,
            "metrics": metrics,
            "config": {
                "parallel_num": self.parallel_num,
                "use_images": self.config.use_images,
                "tolerate_truncated_images": self.config.tolerate_truncated_images,
            },
        }


__all__ = ["SelfConsistencyBaseline"]
