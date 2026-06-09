from __future__ import annotations

from dataclasses import dataclass
import json

from sciorch.core.parsing import extract_unique_boxed_letter
from sciorch.types import AttemptRecord, MainAction, SubmitResult


@dataclass
class SubmitTool:
    def __call__(self, action: MainAction, attempts: list[AttemptRecord], reason: str = "") -> SubmitResult:
        final_reason = reason or action.submit_reason or "Submitted orchestra final answer."
        final_boxed_letter = action.final_boxed_letter
        final_answer = action.final_answer.strip() if action.final_answer else ""
        if not final_boxed_letter and final_answer:
            final_boxed_letter, _ = extract_unique_boxed_letter(final_answer)
        if not final_answer and final_boxed_letter:
            final_answer = f"\\boxed{{{final_boxed_letter}}}"

        payload = {
            "reasoning": action.reasoning,
            "boxed_letter": final_boxed_letter,
            "final_answer": final_answer,
            "submit_reason": final_reason,
        }
        final_answer_text = json.dumps(payload, ensure_ascii=False)
        return SubmitResult(
            final_answer_text=final_answer_text,
            final_boxed_letter=final_boxed_letter,
            done=True,
            reason=final_reason,
            step_count=len(attempts),
        )
