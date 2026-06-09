from __future__ import annotations

from sciorch.types import AttemptRecord

DEFAULT_REFLECTION = "Re-check key evidence in image/question alignment before final answer."
TASK_TYPE_GUIDANCE = {
    "CALCULATION": "Focus on extracting exact values, units, and the required computation.",
    "SCI_REASONING": "Focus on the claim, mechanism, or comparison best supported by the evidence.",
    "VISION_REASONING": "Focus on direct visual evidence such as labels, regions, axes, counts, and spatial relations.",
}


def build_sub_prompt(
    question: str,
    options: list[str],
    instruction: str,
    prior_attempts: list[AttemptRecord],
    task_type: str | None = None,
) -> str:
    reflection_instruction = instruction.strip() if instruction and instruction.strip() else DEFAULT_REFLECTION
    task_type_text = str(task_type or "").strip().upper()
    task_type_block = ""
    if task_type_text in TASK_TYPE_GUIDANCE:
        task_type_block = f"\nTask type hint: {task_type_text}\n- {TASK_TYPE_GUIDANCE[task_type_text]}\n"
    del options
    del prior_attempts

    return f"""
    You are a specialized SubAgent for multimodal scientific reasoning.

MainAgent is asking you to resolve one focused uncertainty.

Focused question from MainAgent:
{reflection_instruction}
{task_type_block}

Rules:
- Answer only the focused question above.
- Do not give the whole-problem final decision.
- Use the original question and images only as context.
- If the focused question requires visual evidence, examine the provided images directly and cite specific features, values, or regions you observe.
- Keep the answer concise and grounded in the task context.
- Give a local confidence for your answer to the focused question.
- Do not assume MainAgent's current direction is correct; reason independently from the evidence.
- Use a conservative confidence: low when evidence is partial/conflicting, high only when the evidence is direct and clear.
- `answer` must directly answer MainAgent's focused question.
- `confidence` must be a number in [0, 1].
- Do not output markdown fences.

Original Question:
{question}

Output JSON only:
{{
  "answer": "direct answer to MainAgent's focused question",
  "evidence": "brief evidence-grounded explanation",
  "confidence": 0.00
}}
    """.strip()
