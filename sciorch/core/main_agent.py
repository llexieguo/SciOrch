"""
Derivative Notice:
- Inspired by: AOrchestra `aorchestra/main_agent.py` (Apache-2.0)
- Modified by: SciOrch contributors for Reasoning delegate/submit decisioning.
"""
from __future__ import annotations

from sciorch.core.agent_base import AgentBase
from sciorch.core.memory import MainMemory
from sciorch.core.parsing import extract_boxed_letter_from_payload, parse_json_fragment
from sciorch.llm.model_capabilities import model_strength_score
from sciorch.prompts.main_reasoning import build_main_prompt, build_submit_only_prompt
from sciorch.types import MainAction, ReasoningSample


class MainAgent(AgentBase):
    def __init__(
        self,
        llm: any,
        main_model: str,
        sub_models: list[str],
        max_steps: int,
        use_images: bool = True,
        main_max_tokens: int | None = None,
        repetition_penalty: float | None = None,
    ) -> None:
        self.llm = llm
        self.main_model = main_model
        self.sub_models = sub_models
        self.use_images = use_images
        self.max_steps = max_steps
        self.main_max_tokens = main_max_tokens
        self.repetition_penalty = repetition_penalty

    async def step(
        self,
        sample: ReasoningSample,
        memory: MainMemory,
        step_index: int,
        force_submit: bool = False,
    ) -> MainAction:
        system_prompt = "You are a strict orchestration controller. Output JSON only."
        step_history_text = memory.as_brief_text()
        if force_submit:
            prompt = build_submit_only_prompt(
                sample=sample,
                step_history_text=step_history_text,
                step_index=step_index,
                max_steps=self.max_steps,
            )
        else:
            prompt = build_main_prompt(
                sample=sample,
                step_history_text=step_history_text,
                step_index=step_index,
                max_steps=self.max_steps,
                sub_models=self.sub_models,
                force_submit=False,
            )

        total_cost = 0.0
        total_input_tokens = 0
        total_output_tokens = 0
        raw_texts: list[str] = []
        reasoning_texts: list[str] = []

        for retry_index in range(2):
            response = await self.llm.ask(
                model=self.main_model,
                system_prompt=system_prompt,
                user_prompt=prompt,
                images=sample.images if self.use_images else None,
                max_tokens=self.main_max_tokens,
                repetition_penalty=self.repetition_penalty,
            )
            total_cost += response.cost
            total_input_tokens += response.input_tokens
            total_output_tokens += response.output_tokens
            raw_texts.append(response.text)
            if response.reasoning:
                reasoning_texts.append(response.reasoning)

            action = self._parse_action(response.text)
            action.thinking = "\n\n[retry]\n".join(reasoning_texts)
            action.system_prompt = system_prompt
            action.user_prompt = prompt
            action.raw_response = "\n\n[retry_response]\n".join(raw_texts)
            action.cost = total_cost
            action.input_tokens = total_input_tokens
            action.output_tokens = total_output_tokens

            if action.final_boxed_letter and not action.final_answer:
                action.final_answer = f"\\boxed{{{action.final_boxed_letter}}}"

            should_retry_same_prompt = False

            # Safety guardrails.
            if action.action == "submit":
                if action.final_boxed_letter is None:
                    if retry_index == 0 and not force_submit:
                        should_retry_same_prompt = True
                    else:
                        if force_submit:
                            return MainAction(
                                action="submit",
                                reasoning="Forced final-turn submit after unparsable final answer.",
                                thinking=action.thinking,
                                submit_reason="Reached final allowed step but final answer parsing failed.",
                                final_answer=action.final_answer,
                                final_boxed_letter=None,
                                cost=total_cost,
                                raw_response=action.raw_response,
                                input_tokens=total_input_tokens,
                                output_tokens=total_output_tokens,
                            )
                        return self._fallback_delegate_action(
                            reason="Submit rejected by guardrail; final answer was not parseable.",
                            response=action,
                            memory=memory,
                            step_index=step_index,
                            instruction="Ask one focused verification question that resolves the missing final answer.",
                            prefer_strong_model=True,
                        )
                elif force_submit:
                    return action
                elif not memory.has_informative_attempts():
                    if retry_index == 0:
                        should_retry_same_prompt = True
                    else:
                        return self._fallback_delegate_action(
                            reason="Submit rejected by guardrail; no informative delegate findings yet.",
                            response=action,
                            memory=memory,
                            step_index=step_index,
                            instruction="Ask one focused verification question before making the final choice.",
                            prefer_strong_model=True,
                        )
                else:
                    return action

            elif action.action == "delegate_task":
                if force_submit:
                    return MainAction(
                        action="submit",
                        reasoning="Forced final-turn submit after model requested another delegation.",
                        thinking=action.thinking,
                        submit_reason="Reached final allowed step; delegation was no longer allowed.",
                        final_answer=action.final_answer,
                        final_boxed_letter=action.final_boxed_letter,
                        cost=total_cost,
                        raw_response=action.raw_response,
                        input_tokens=total_input_tokens,
                        output_tokens=total_output_tokens,
                    )
                action.model = self._resolve_delegate_model(
                    requested_model=action.model,
                    memory=memory,
                    step_index=step_index,
                )
                if not action.instruction:
                    repaired_action = await self._repair_missing_delegate_instruction(
                        sample=sample,
                        response=action,
                        memory=memory,
                        step_index=step_index,
                        system_prompt=system_prompt,
                        original_user_prompt=prompt,
                    )
                    if repaired_action is not None:
                        action = repaired_action
                        if action.action != "delegate_task":
                            return action
                        action.model = self._resolve_delegate_model(
                            requested_model=action.model,
                            memory=memory,
                            step_index=step_index,
                        )
                    else:
                        return self._fallback_delegate_action(
                            reason="Delegate rejected by guardrail; missing focused question in instruction.",
                            response=action,
                            memory=memory,
                            step_index=step_index,
                            instruction="Ask one focused verification question with a concise local-confidence answer.",
                            prefer_strong_model=True,
                        )
                return action

            else:
                if retry_index == 0 and not force_submit:
                    should_retry_same_prompt = True
                else:
                    if force_submit:
                        return MainAction(
                            action="submit",
                            reasoning="Forced final-turn submit after malformed MainAgent response.",
                            thinking=action.thinking,
                            submit_reason="Reached final allowed step but response format was malformed.",
                            final_answer=action.final_answer,
                            final_boxed_letter=action.final_boxed_letter,
                            cost=total_cost,
                            raw_response=action.raw_response,
                            input_tokens=total_input_tokens,
                            output_tokens=total_output_tokens,
                        )
                    return MainAction(
                        action="submit",
                        reasoning="Malformed response twice; skipping this step.",
                        thinking=action.thinking,
                        submit_reason="Repeated malformed response from main model; giving up.",
                        final_answer=action.final_answer,
                        final_boxed_letter=action.final_boxed_letter,
                        cost=total_cost,
                        raw_response=action.raw_response,
                        input_tokens=total_input_tokens,
                        output_tokens=total_output_tokens,
                    )

            if not should_retry_same_prompt:
                return action

        return MainAction(
            action="submit",
            reasoning="Forced final-turn submit after repeated MainAgent guardrail failures."
            if force_submit
            else "Repeated MainAgent guardrail failures.",
            thinking="\n\n[retry]\n".join(reasoning_texts),
            submit_reason="Reached final allowed step after repeated failures."
            if force_submit
            else None,
            cost=total_cost,
            raw_response="\n\n[retry_response]\n".join(raw_texts),
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
        )

    def _fallback_delegate_action(
        self,
        reason: str,
        response: any,
        memory: MainMemory,
        step_index: int,
        instruction: str,
        prefer_strong_model: bool = False,
    ) -> MainAction:
        response_reasoning = getattr(response, "reasoning", None) or getattr(response, "thinking", None)
        response_text = getattr(response, "text", None) or getattr(response, "raw_response", "") or ""
        return MainAction(
            action="delegate_task",
            reasoning=reason,
            thinking=response_reasoning,
            model=self._select_sub_model(
                memory=memory,
                step_index=step_index,
                prefer_strong=prefer_strong_model,
            ),
            instruction=instruction,
            cost=response.cost,
            raw_response=response_text,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
        )

    async def _repair_missing_delegate_instruction(
        self,
        *,
        sample: ReasoningSample,
        response: any,
        memory: MainMemory,
        step_index: int,
        system_prompt: str,
        original_user_prompt: str,
    ) -> MainAction | None:
        response_text = getattr(response, "text", None) or getattr(response, "raw_response", "") or ""
        response_cost = float(getattr(response, "cost", 0.0) or 0.0)
        response_input_tokens = int(getattr(response, "input_tokens", 0) or 0)
        response_output_tokens = int(getattr(response, "output_tokens", 0) or 0)
        repair_prompt = (
            "Your previous response chose action=delegate_task but did not provide a usable focused question in "
            "`instruction`.\n"
            "Revise your answer now.\n"
            "- If you still want to delegate, output action=delegate_task with one concrete, answerable focused question.\n"
            "- If you no longer need delegation, output action=submit.\n"
            "- Do not leave `instruction` empty when using delegate_task.\n"
            "Output JSON only.\n\n"
            f"Question:\n{sample.question}\n\n"
            f"Prior delegate steps:\n{memory.as_brief_text()}\n\n"
            f"Previous response:\n{response_text}"
        )
        retry = await self.llm.ask(
            model=self.main_model,
            system_prompt=system_prompt,
            user_prompt=repair_prompt,
            images=sample.images if self.use_images else None,
            max_tokens=self.main_max_tokens,
        )
        repaired = self._parse_action(retry.text)
        repaired.thinking = retry.reasoning
        repaired.system_prompt = system_prompt
        repaired.user_prompt = original_user_prompt + "\n\n[repair]\n" + repair_prompt
        repaired.raw_response = response_text + "\n\n[repair_response]\n" + retry.text
        repaired.cost = response_cost + retry.cost
        repaired.input_tokens = response_input_tokens + retry.input_tokens
        repaired.output_tokens = response_output_tokens + retry.output_tokens
        if repaired.action == "delegate_task" and repaired.instruction:
            return repaired
        if repaired.action == "submit":
            if repaired.final_boxed_letter and not repaired.final_answer:
                repaired.final_answer = f"\\boxed{{{repaired.final_boxed_letter}}}"
            return repaired
        return None

    async def run(self, *args, **kwargs):
        raise NotImplementedError("Main loop is orchestrated by ReasoningRunner")

    def _resolve_delegate_model(
        self,
        *,
        requested_model: str | None,
        memory: MainMemory,
        step_index: int,
    ) -> str:
        if requested_model not in self.sub_models:
            return self._select_sub_model(memory=memory, step_index=step_index, prefer_strong=True)
        return requested_model

    def _select_sub_model(
        self,
        *,
        memory: MainMemory,
        step_index: int,
        prefer_strong: bool = False,
        avoid_models: set[str] | None = None,
    ) -> str:
        avoid_models = avoid_models or set()
        candidates = [model for model in self.sub_models if model not in avoid_models]
        if not candidates:
            candidates = list(self.sub_models)

        if prefer_strong or step_index <= min(2, self.max_steps - 1):
            used_models = set(memory.used_models())
            unused_candidates = [model for model in candidates if model not in used_models]
            ranked_pool = unused_candidates or candidates
            return sorted(
                ranked_pool,
                key=lambda model_name: (-model_strength_score(model_name), model_name),
            )[0]

        attempt_offset = len(memory.attempts)
        return candidates[attempt_offset % len(candidates)]

    @staticmethod
    def _parse_action(raw_text: str) -> MainAction:
        payload = parse_json_fragment(raw_text)
        if payload:
            action = str(payload.get("action", "invalid")).strip()
            reasoning = str(payload.get("reasoning", "")).strip()
            model = payload.get("model")
            instruction = payload.get("instruction")
            task_type = payload.get("task_type")
            submit_reason = payload.get("submit_reason")
            final_answer = payload.get("final_answer")
        else:
            action = "invalid"
            reasoning = "Failed to parse JSON"
            model = None
            instruction = None
            task_type = None
            submit_reason = None
            final_answer = None

        boxed_letter, _ = extract_boxed_letter_from_payload(raw_text, payload)
        final_answer_text = str(final_answer).strip() if isinstance(final_answer, str) else None
        if not final_answer_text and boxed_letter is not None:
            final_answer_text = f"\\boxed{{{boxed_letter}}}"

        return MainAction(
            action=action,
            reasoning=reasoning,
            model=str(model) if model is not None else None,
            instruction=str(instruction) if instruction is not None else None,
            task_type=str(task_type) if task_type is not None else None,
            submit_reason=str(submit_reason) if submit_reason is not None else None,
            final_answer=final_answer_text,
            final_boxed_letter=boxed_letter,
            parsed_payload=payload or {},
        )
