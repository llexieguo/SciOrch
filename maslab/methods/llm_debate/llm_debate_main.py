"""LLM Debate baseline for SciOrch.

References:
- MASLab: https://github.com/MASWorks/MASLab
- CoMAS MASLab-style layout: https://github.com/xxyQwQ/CoMAS/tree/main/maslab
- Original LLM Debate implementation: https://github.com/composable-models/llm_multiagent_debate

This module reimplements the MASLab-style LLM Debate flow for SciOrch's
combined multimodal benchmark format, OpenAI-compatible API client, JSONL
logging, image warning tracking, and per-source metrics. It is not a direct
copy of MASLab or CoMAS code.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from maslab.methods.base import PromptMASBaseline, PromptMASConfig
from maslab.utils.calls import call_to_json
from maslab.utils.formatting import extract_answer_letter, format_query, normalize_answer_letter
from maslab.utils.images import ImageLoadResult, load_sample_images_with_warnings


class LLMDebateBaseline(PromptMASBaseline):
    """MASLab-style LLM debate: independent agents, revision rounds, final aggregation."""

    def __init__(
        self,
        client: Any,
        config: PromptMASConfig,
        *,
        agents_num: int = 3,
        rounds_num: int = 2,
    ) -> None:
        super().__init__(client, config)
        if agents_num <= 0:
            raise ValueError("agents_num must be positive")
        if rounds_num <= 0:
            raise ValueError("rounds_num must be positive")
        self.agents_num = agents_num
        self.rounds_num = rounds_num

    def construct_revision_message(
        self,
        other_agent_histories: list[list[dict[str, str]]],
        question: str,
        assistant_index: int,
    ) -> str:
        if not other_agent_histories:
            return (
                "Can you verify that your answer is correct? Please reiterate your answer, "
                "making sure to state your answer at the end as \\boxed{LETTER}."
            )

        parts = ["These are the recent/updated opinions from other agents:"]
        for history in other_agent_histories:
            if assistant_index >= len(history):
                continue
            agent_response = history[assistant_index]["content"]
            parts.append(f"One agent response:\n```text\n{agent_response}\n```")

        parts.append(
            "Use these opinions carefully as additional advice. Provide an updated answer. "
            "Make sure to state your answer at the end as \\boxed{LETTER}."
        )
        parts.append(f"The original problem is:\n{question}")
        return "\n\n".join(parts)

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
        histories = [[{"role": "user", "content": query}] for _ in range(self.agents_num)]
        calls: list[dict[str, Any]] = []

        for round_index in range(self.rounds_num):
            for agent_index, history in enumerate(histories):
                if round_index != 0:
                    other_histories = histories[:agent_index] + histories[agent_index + 1 :]
                    message = self.construct_revision_message(
                        other_histories,
                        query,
                        2 * round_index - 1,
                    )
                    history.append({"role": "user", "content": message})

                result = await self._ask(self._render_history(history), images=images)
                history.append({"role": "assistant", "content": result.text})
                calls.append(
                    call_to_json(
                        result,
                        role="agent",
                        label=f"agent_{agent_index + 1}_round_{round_index + 1}",
                    )
                )

        answers = [history[-1]["content"] for history in histories]
        aggregate_prompt = f"Task:\n{query}\n\n"
        for index, answer in enumerate(answers):
            aggregate_prompt += f"Solution {index + 1}:\n{answer}\n\n"
        aggregate_prompt += (
            "Given all the above solutions, reason over them carefully and provide a final answer "
            "exactly as \\boxed{LETTER}."
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
            "method": "llm_debate",
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
            "agent_answers": answers,
            "image_load_warnings": image_result.warnings,
            "image_load_warning_count": len(image_result.warnings),
            "calls": calls,
            "metrics": metrics,
            "config": {
                "agents_num": self.agents_num,
                "rounds_num": self.rounds_num,
                "use_images": self.config.use_images,
                "tolerate_truncated_images": self.config.tolerate_truncated_images,
            },
        }


__all__ = ["LLMDebateBaseline"]
