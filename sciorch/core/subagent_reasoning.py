from __future__ import annotations

from dataclasses import dataclass

from sciorch.core.parsing import extract_confidence, parse_json_fragment
from sciorch.prompts.sub_reasoning import build_sub_prompt
from sciorch.types import DelegateRequest, DelegateResult


@dataclass
class SubAgentReasoning:
    llm: any

    @staticmethod
    def _extract_answer_text(raw_text: str, parsed: dict | None) -> str:
        if parsed:
            for key in ("answer", "response", "finding", "conclusion"):
                value = parsed.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
            return ""
        stripped = raw_text.strip()
        if not stripped:
            return ""
        return stripped.splitlines()[0].strip()

    async def run(self, request: DelegateRequest) -> DelegateResult:
        system_prompt = "You are a rigorous multimodal scientific reasoning assistant."
        prompt = build_sub_prompt(
            question=request.question,
            options=request.options,
            instruction=request.instruction,
            task_type=request.task_type,
            prior_attempts=request.prior_attempts,
        )

        result = await self.llm.ask(
            model=request.model,
            system_prompt=system_prompt,
            user_prompt=prompt,
            images=request.images,
        )

        parsed = parse_json_fragment(result.text)
        confidence = extract_confidence(result.text, parsed)
        answer = self._extract_answer_text(result.text, parsed)
        reasoning = ""
        if parsed:
            for key in ("evidence", "reasoning"):
                value = parsed.get(key)
                if isinstance(value, str) and value.strip():
                    reasoning = value.strip()
                    break
        if not reasoning:
            reasoning = answer or result.text.strip()

        parse_ok = bool(answer) and confidence is not None
        errors: list[str] = []
        if not answer:
            errors.append("Missing focused answer")
        if confidence is None:
            errors.append("Missing confidence")

        return DelegateResult(
            raw_answer_text=result.text,
            answer=answer or None,
            confidence=confidence,
            reasoning_summary=reasoning,
            thinking=result.reasoning,
            parse_ok=parse_ok,
            error="; ".join(errors) if errors else None,
            cost=result.cost,
            system_prompt=system_prompt,
            user_prompt=prompt,
            parsed_payload=parsed or {},
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
        )
