"""
Derivative Notice:
- Inspired by: SGI-Bench experimental reasoning scorer module (MIT)
- Modified by: SciOrch contributors for integrated Reasoning submission scoring.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from sciorch.core.parsing import extract_boxed_letter_from_payload, parse_json_fragment
from sciorch.types import ReasoningSample


@dataclass
class ReasoningScore:
    mca: float
    rv: float
    judge_raw: str
    judge_cost: float
    judge_input_tokens: int
    judge_output_tokens: int
    system_prompt: str
    user_prompt: str


def _gold_letter(answer_index: int) -> str:
    return chr(ord("A") + answer_index)


def compute_mca(sample: ReasoningSample, final_answer_text: str) -> float:
    """Compute MCA locally from boxed option letter, without judge model."""
    parsed = parse_json_fragment(final_answer_text)
    predicted_letter, _ = extract_boxed_letter_from_payload(final_answer_text, parsed)
    return 1.0 if predicted_letter and predicted_letter.upper() == _gold_letter(sample.answer_index) else 0.0


def _build_rv_prompt(sample: ReasoningSample, prediction: str) -> str:
    options = "\n".join(f"{chr(ord('A') + idx)}. {option}" for idx, option in enumerate(sample.options))
    reference_steps = "\n".join(f"{idx + 1}. {step}" for idx, step in enumerate(sample.steps))

    return f"""
You are a strict evaluator assessing the validity of the model prediction's reasoning process.
Score reasoning validity on a scale from 0 to 10.

Question:
{sample.question}

Options:
{options}

Reference Reasoning:
{reference_steps}

Model Prediction:
{prediction}

Rules:
1) Evaluate logical coherence and alignment with reference reasoning.
2) Penalize contradictions, missing critical steps, and irrelevant reasoning.
3) Output ONLY one integer from 0 to 10.
""".strip()


async def score_reasoning_sample(
    sample: ReasoningSample,
    final_answer_text: str,
    judge_client: any,
    judge_model: str,
) -> ReasoningScore:
    mca = compute_mca(sample, final_answer_text)

    rv_prompt = _build_rv_prompt(sample, final_answer_text)
    system_prompt = "You evaluate scientific reasoning quality."
    judge_response = await judge_client.ask(
        model=judge_model,
        system_prompt=system_prompt,
        user_prompt=rv_prompt,
        images=sample.images,
    )

    raw = judge_response.text.strip()
    match = re.search(r"(10|[0-9])", raw)
    if match:
        rv_score = float(match.group(1)) / 10.0
    else:
        rv_score = 0.0

    return ReasoningScore(
        mca=mca,
        rv=max(0.0, min(1.0, rv_score)),
        judge_raw=raw,
        judge_cost=judge_response.cost,
        judge_input_tokens=judge_response.input_tokens,
        judge_output_tokens=judge_response.output_tokens,
        system_prompt=system_prompt,
        user_prompt=rv_prompt,
    )
