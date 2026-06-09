"""
Derivative Notice:
- Inspired by: AOrchestra `base/agent/memory.py` (Apache-2.0)
- Modified by: SciOrch contributors for Reasoning orchestration state tracking.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from sciorch.types import AttemptRecord


@dataclass
class MainMemory:
    """In-memory state for one Reasoning sample orchestration."""

    attempts: list[AttemptRecord] = field(default_factory=list)

    def add_attempt(self, attempt: AttemptRecord) -> None:
        self.attempts.append(attempt)

    def latest_attempt(self) -> AttemptRecord | None:
        if not self.attempts:
            return None
        return self.attempts[-1]

    def successful_attempts(self) -> list[AttemptRecord]:
        return [attempt for attempt in self.attempts if attempt.delegate_result.parse_ok]

    def has_informative_attempts(self) -> bool:
        return bool(self.successful_attempts())

    def informative_attempt_count(self) -> int:
        return len(self.successful_attempts())

    def best_confidence(self) -> float | None:
        best = self.best_attempt()
        if best is None or not best.delegate_result.parse_ok:
            return None
        return best.delegate_result.confidence

    def used_models(self, informative_only: bool = False) -> list[str]:
        attempts = self.successful_attempts() if informative_only else self.attempts
        return [attempt.model for attempt in attempts if attempt.model]

    def distinct_model_count(self, informative_only: bool = False) -> int:
        return len(dict.fromkeys(self.used_models(informative_only=informative_only)))

    def best_attempt(self) -> AttemptRecord | None:
        """Pick highest-confidence parseable attempt; fallback to latest."""
        if not self.attempts:
            return None

        parseable = self.successful_attempts()
        if not parseable:
            return self.attempts[-1]

        return max(
            parseable,
            key=lambda item: item.delegate_result.confidence
            if item.delegate_result.confidence is not None
            else -1.0,
        )

    @staticmethod
    def _shorten(text: str | None, limit: int = 180) -> str:
        del limit
        if not text:
            return "-"
        collapsed = " ".join(text.split())
        return collapsed

    def as_brief_text(self) -> str:
        if not self.attempts:
            return "No steps yet."

        recent_attempts = self.successful_attempts()[-4:] or self.attempts[-2:]
        lines: list[str] = []
        for attempt in recent_attempts:
            delegate = attempt.delegate_result
            lines.append(
                "Step {step_index} | model={model} | question={question} | answer={answer}".format(
                    step_index=attempt.attempt_index,
                    model=attempt.model or "unknown",
                    question=self._shorten(attempt.instruction, limit=95),
                    answer=self._shorten(delegate.answer, limit=110),
                )
            )
            lines.append(
                "Evidence: {evidence}".format(
                    evidence=self._shorten(delegate.reasoning_summary, limit=220),
                )
            )
        return "\n".join(lines)

    def delegate_context_summary(self, next_instruction: str | None = None, max_points: int = 4) -> str:
        del next_instruction
        informative_attempts = self.successful_attempts()
        if not informative_attempts:
            return "No prior steps."

        lines: list[str] = []
        recent_attempts = informative_attempts[-max_points:]
        for attempt in recent_attempts:
            delegate = attempt.delegate_result
            lines.append(
                "Step {step_index}: asked={question}, answer={answer}, confidence={confidence}".format(
                    step_index=attempt.attempt_index,
                    question=self._shorten(attempt.instruction, limit=95),
                    answer=self._shorten(delegate.answer, limit=110),
                    confidence=delegate.confidence,
                )
            )
            lines.append(
                "Evidence: {evidence}".format(
                    evidence=self._shorten(delegate.reasoning_summary, limit=220),
                )
            )
        return "\n".join(lines)
