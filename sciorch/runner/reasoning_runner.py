from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from sciorch.adapters.reasoning.dataset import load_reasoning_samples
from sciorch.adapters.reasoning.io import dump_json, export_reasoning_compatible_logs
from sciorch.adapters.reasoning.scorer import compute_mca, score_reasoning_sample
from sciorch.config import OrchestratorConfig
from sciorch.core.main_agent import MainAgent
from sciorch.core.memory import MainMemory
from sciorch.core.tools.delegate import DelegateTaskTool
from sciorch.core.tools.submit import SubmitTool
from sciorch.core.subagent_reasoning import SubAgentReasoning
from sciorch.llm.model_capabilities import supports_image_inputs
from sciorch.llm.openai_compatible import OpenAICompatibleClient
from sciorch.types import AttemptRecord, DelegateRequest, MainAction, RunSummary, ReasoningRunRecord, ReasoningSample

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - optional dependency fallback
    tqdm = None


class ReasoningRunner:
    def __init__(
        self,
        config: OrchestratorConfig,
        main_client: Any | None = None,
        sub_client: Any | None = None,
        judge_client: Any | None = None,
        dataset_loader: Callable[[OrchestratorConfig], list[ReasoningSample]] = load_reasoning_samples,
    ) -> None:
        self.config = config
        self.dataset_loader = dataset_loader

        if main_client is None:
            if config.main_model_endpoint == "local":
                main_client = OpenAICompatibleClient(
                    api_key=config.main_local_api_key or os.getenv(config.main_local_api_key_env) or "EMPTY",
                    base_url=config.main_local_base_url,
                    api_key_env=config.main_local_api_key_env,
                    default_temperature=config.main_local_temperature,
                    enable_thinking=config.main_enable_thinking,
                )
            else:
                main_client = OpenAICompatibleClient(
                    base_url=config.openai_base_url,
                    api_key_env=config.openai_api_key_env,
                    enable_thinking=config.main_enable_thinking,
                )
        if sub_client is None:
            sub_client = OpenAICompatibleClient(
                api_key=config.openai_api_key,
                base_url=config.openai_base_url,
                api_key_env=config.openai_api_key_env,
                enable_thinking=config.sub_enable_thinking,
            )
        if judge_client is None:
            judge_client = OpenAICompatibleClient(
                api_key=config.openai_api_key,
                base_url=config.openai_base_url,
                api_key_env=config.openai_api_key_env,
                enable_thinking=config.judge_enable_thinking,
            )

        self.main_client = main_client
        self.sub_client = sub_client
        self.judge_client = judge_client

    @staticmethod
    def _classify_exception(error: Exception) -> tuple[str, str]:
        message = f"{type(error).__name__}: {error}"
        lowered = message.lower()
        if "ratelimiterror" in lowered or "rate limit" in lowered or "rate_limit_exceeded" in lowered:
            return "rate_limit_error", message
        if "apiconnectionerror" in lowered or "connection error" in lowered:
            return "api_connection_error", message
        if "authenticationerror" in lowered or "invalid api key" in lowered:
            return "authentication_error", message
        if "permissiondeniederror" in lowered or "permission denied" in lowered:
            return "permission_error", message
        if "timeout" in lowered:
            return "timeout_error", message
        if "internalservererror" in lowered or "servererror" in lowered or "503" in lowered:
            return "server_error", message
        if "badrequesterror" in lowered or "invalid_request_error" in lowered:
            return "invalid_request_error", message
        return type(error).__name__, message

    async def run(self, resume: bool = False) -> tuple[list[ReasoningRunRecord], RunSummary]:
        run_start = time.perf_counter()
        samples = self.dataset_loader(self.config)
        if not samples:
            raise RuntimeError("No Reasoning samples found after filtering")

        run_output_dir = self._build_run_output_dir(resume=resume)
        run_output_dir.mkdir(parents=True, exist_ok=True)

        # Resume: load existing trajectory results and skip those samples
        resumed_records: list[ReasoningRunRecord] = []
        done_task_ids: set[str] = set()
        if resume:
            traj_dir = run_output_dir / "trajectories"
            if traj_dir.exists():
                import json as _json
                for traj_file in traj_dir.rglob("*.json"):
                    try:
                        data = _json.loads(traj_file.read_text(encoding="utf-8"))
                        tid = data.get("task_id", "")
                        if tid:
                            done_task_ids.add(tid)
                            resumed_records.append(self._dict_to_record(data))
                    except Exception:
                        pass
                print(f"Resume: found {len(done_task_ids)} completed trajectories, skipping them")
        else:
            self._prepare_run_output_dir(run_output_dir)

        remaining_samples = [s for s in samples if s.task_id not in done_task_ids]
        semaphore = asyncio.Semaphore(max(1, self.config.max_concurrency))

        async def run_limited(sample: ReasoningSample) -> ReasoningRunRecord:
            async with semaphore:
                sample_start = time.perf_counter()
                try:
                    return await self._run_single_sample(sample)
                except Exception:
                    try:
                        return await self._run_single_sample(sample)
                    except Exception as error:
                        # Infrastructure failure (e.g. endpoint/network errors) after a
                        # retry. Record it as failed so it is excluded from the accuracy
                        # denominator rather than silently counted as correct.
                        return self._build_failed_record(
                            sample=sample,
                            error=error,
                            latency_seconds=time.perf_counter() - sample_start,
                        )

        progress = (
            tqdm(total=len(samples), desc="Running", unit="sample", initial=len(done_task_ids), dynamic_ncols=True)
            if tqdm is not None
            else None
        )

        tasks = [asyncio.create_task(run_limited(sample)) for sample in remaining_samples]
        order_map = {sample.task_id: idx for idx, sample in enumerate(samples)}
        records: list[ReasoningRunRecord] = list(resumed_records)

        completed_count = len(done_task_ids)
        scored_count = sum(1 for r in resumed_records if r.metadata.get("status") != "failed")
        correct_count = sum(1 for r in resumed_records if r.mca > 0)

        try:
            for task in asyncio.as_completed(tasks):
                record = await task
                records.append(record)
                self._checkpoint_record(record, output_dir=run_output_dir)
                completed_count += 1
                # Infra-failed samples are excluded from the accuracy denominator.
                if record.metadata.get("status") != "failed":
                    scored_count += 1
                    if record.mca > 0:
                        correct_count += 1
                if progress is not None:
                    acc = correct_count / scored_count if scored_count else 0.0
                    progress.set_postfix_str(f"acc={acc:.1%} ({correct_count}/{scored_count})")
                    progress.update(1)
        finally:
            if progress is not None:
                progress.close()

        records.sort(key=lambda item: order_map.get(item.task_id, 10**9))

        run_wall_time_seconds = time.perf_counter() - run_start
        summary = self._build_summary(
            records,
            run_wall_time_seconds=run_wall_time_seconds,
            output_dir=run_output_dir,
        )
        self._save_outputs(records, summary, output_dir=run_output_dir)
        return records, summary

    def _build_failed_record(
        self,
        sample: ReasoningSample,
        error: Exception,
        latency_seconds: float,
    ) -> ReasoningRunRecord:
        error_type, error_message = self._classify_exception(error)
        return ReasoningRunRecord(
            task_id=sample.task_id,
            discipline=sample.discipline,
            question=sample.question,
            options=sample.options,
            gold_answer_index=sample.answer_index,
            gold_answer_letter=chr(ord("A") + sample.answer_index),
            reference_steps=sample.steps,
            final_answer_text="",
            final_boxed_letter=None,
            mca=0.0,
            rv=0.0,
            total_cost=0.0,
            main_tokens=0,
            sub_tokens=0,
            total_tokens=0,
            latency_seconds=latency_seconds,
            models_used=[],
            model_usage={},
            attempts=[],
            metadata={
                "status": "failed",
                "error_type": error_type,
                "error": error_message,
                "exception_type": type(error).__name__,
            },
        )

    def _build_skipped_record(
        self,
        sample: ReasoningSample,
        latency_seconds: float,
    ) -> ReasoningRunRecord:
        return ReasoningRunRecord(
            task_id=sample.task_id,
            discipline=sample.discipline,
            question=sample.question,
            options=sample.options,
            gold_answer_index=sample.answer_index,
            gold_answer_letter=chr(ord("A") + sample.answer_index),
            reference_steps=sample.steps,
            final_answer_text="",
            final_boxed_letter=None,
            mca=0.0,
            rv=0.0,
            total_cost=0.0,
            latency_seconds=latency_seconds,
            metadata={"status": "skipped"},
        )

    @staticmethod
    def _sanitize_model_name(model_name: str) -> str:
        sanitized = re.sub(r"[^A-Za-z0-9._-]+", "-", model_name).strip("-._")
        return sanitized or "model"


    @staticmethod
    def _dict_to_record(data: dict) -> "ReasoningRunRecord":
        """Reconstruct a ReasoningRunRecord from a saved trajectory dict."""
        score = data.get("score", {})
        metrics = data.get("metrics", {})
        task = data.get("task", {})
        reference = data.get("reference", {})
        return ReasoningRunRecord(
            task_id=data.get("task_id", ""),
            discipline=task.get("discipline", ""),
            question=task.get("question", ""),
            options=task.get("options", []),
            gold_answer_index=reference.get("answer_index", 0),
            gold_answer_letter=reference.get("gold_answer_letter", ""),
            reference_steps=reference.get("reference_steps", []),
            final_answer_text=data.get("final_decision", {}).get("answer", "") if data.get("final_decision") else "",
            final_boxed_letter=data.get("final_decision", {}).get("boxed_letter") if data.get("final_decision") else None,
            mca=score.get("mca", 0.0),
            rv=score.get("rv", 0.0),
            total_cost=metrics.get("total_cost", 0.0),
            main_tokens=metrics.get("main_tokens", 0),
            sub_tokens=metrics.get("sub_tokens", 0),
            total_tokens=metrics.get("total_tokens", 0),
            latency_seconds=metrics.get("latency_seconds", 0.0),
            models_used=data.get("models_used", []),
            model_usage=data.get("model_usage", {}),
            decision_steps=data.get("steps", []),
        )

    def _build_run_output_dir(self, resume: bool = False) -> Path:
        base_output_dir = self.config.output_dir
        model_name = self._sanitize_model_name(self.config.main_model)
        candidate = base_output_dir / model_name
        if not candidate.exists():
            return candidate
        if resume:
            latest = candidate
            idx = 1
            while True:
                suffixed = base_output_dir / f"{model_name}_{idx}"
                if suffixed.exists():
                    latest = suffixed
                    idx += 1
                else:
                    break
            return latest
        idx = 1
        while True:
            suffixed = base_output_dir / f"{model_name}_{idx}"
            if not suffixed.exists():
                return suffixed
            idx += 1

    @staticmethod
    def _prepare_run_output_dir(output_dir: Path) -> None:
        for filename in (
            "predictions.json",
            "scored.json",
            "summary.json",
            "scored.checkpoint.jsonl",
            "raw_calls.jsonl",
        ):
            target = output_dir / filename
            if target.exists():
                target.unlink()

        trajectory_dir = output_dir / "trajectories"
        if trajectory_dir.exists():
            shutil.rmtree(trajectory_dir)

        samples_dir = output_dir / "samples"
        if samples_dir.exists():
            shutil.rmtree(samples_dir)

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _build_raw_call(
        *,
        call_id: str,
        task_id: str,
        step_index: int | None,
        actor: str,
        model: str | None,
        system_prompt: str | None,
        user_prompt: str | None,
        raw_text: str | None,
        thinking: str | None,
        parsed: dict[str, Any] | None,
        input_tokens: int,
        output_tokens: int,
        cost: float,
        error: str | None = None,
    ) -> dict[str, Any]:
        return {
            "call_id": call_id,
            "timestamp": ReasoningRunner._now(),
            "task_id": task_id,
            "step_index": step_index,
            "actor": actor,
            "model": model,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "raw_text": raw_text,
            "thinking": thinking or "",
            "parsed": parsed or {},
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost": cost,
            "error": error,
        }

    @staticmethod
    def _build_delegate_step(
        *,
        step_index: int,
        action: MainAction,
        request: DelegateRequest,
        delegate_result: Any,
        raw_call_ids: list[str],
    ) -> dict[str, Any]:
        return {
            "step_index": step_index,
            "action": "delegate",
            "orchestra_reasoning": action.reasoning,
            "focus_question": request.instruction,
            "delegate_model": request.model,
            "delegate_answer": delegate_result.answer,
            "delegate_evidence": delegate_result.reasoning_summary,
            "delegate_confidence": delegate_result.confidence,
            "status": "ok" if delegate_result.parse_ok else "error",
            "error": delegate_result.error,
            "raw_call_ids": raw_call_ids,
        }

    @staticmethod
    def _build_submit_step(
        *,
        step_index: int,
        action: MainAction | None,
        boxed_letter: str | None,
        submit_reason: str,
        raw_call_ids: list[str],
        source: str,
    ) -> dict[str, Any]:
        return {
            "step_index": step_index,
            "action": "submit",
            "source": source,
            "orchestra_reasoning": action.reasoning if action is not None else "",
            "boxed_letter": boxed_letter,
            "submit_reason": submit_reason,
            "raw_call_ids": raw_call_ids,
        }

    def _record_to_view_json(self, record: ReasoningRunRecord) -> dict[str, Any]:
        return {
            "task_id": record.task_id,
            "status": record.metadata.get("status", "completed"),
            "task": {
                "discipline": record.discipline,
                "question": record.question,
                "options": record.options,
            },
            "reference": {
                "gold_answer_index": record.gold_answer_index,
                "gold_answer_letter": record.gold_answer_letter,
                "steps": record.reference_steps,
            },
            "final_decision": {
                "boxed_letter": record.final_boxed_letter,
                "submit_reason": record.metadata.get("submit_reason", ""),
            },
            "score": {
                "mca": record.mca,
                "rv": record.rv,
                "scoring_enabled": record.metadata.get("scoring_enabled", True),
            },
            "metrics": {
                "total_cost": record.total_cost,
                "main_tokens": record.main_tokens,
                "sub_tokens": record.sub_tokens,
                "total_tokens": record.total_tokens,
                "latency_seconds": record.latency_seconds,
                "step_count": len(record.decision_steps),
            },
            "models_used": record.models_used,
            "model_usage": record.model_usage,
            "steps": record.decision_steps,
            "errors": {
                "error_type": record.metadata.get("error_type"),
                "error": record.metadata.get("error"),
            },
        }

    async def _run_single_sample(self, sample: ReasoningSample) -> ReasoningRunRecord:
        sample_start = time.perf_counter()
        delegate_models = self._eligible_sub_models(sample)
        main_agent = MainAgent(
            llm=self.main_client,
            main_model=self.config.main_model,
            sub_models=delegate_models,
            use_images=self.config.main_use_images,
            max_steps=self.config.max_steps,
            main_max_tokens=self.config.main_max_tokens,
            repetition_penalty=self.config.main_repetition_penalty,
        )
        submit_tool = SubmitTool()
        memory = MainMemory()

        total_cost = 0.0
        main_tokens = 0
        sub_tokens = 0
        submit_result = None
        submit_action = None
        submit_reason = ""
        model_usage: dict[str, int] = {}
        model_calls: list[str] = []
        decision_steps: list[dict[str, Any]] = []
        raw_calls: list[dict[str, Any]] = []
        raw_call_counter = 0

        def _record_model_call(model_name: str | None) -> None:
            if not model_name:
                return
            model_usage[model_name] = model_usage.get(model_name, 0) + 1
            model_calls.append(model_name)

        def _next_call_id() -> str:
            nonlocal raw_call_counter
            raw_call_counter += 1
            return f"call_{raw_call_counter:04d}"

        for step_idx in range(1, self.config.max_steps + 1):
            force_submit = step_idx == self.config.max_steps and bool(memory.attempts)
            action = await main_agent.step(
                sample=sample,
                memory=memory,
                step_index=step_idx,
                force_submit=force_submit,
            )
            if action.raw_response is not None:
                _record_model_call(self.config.main_model)
            total_cost += action.cost
            main_tokens += int(action.input_tokens) + int(action.output_tokens)
            main_call_id = _next_call_id()
            raw_calls.append(
                self._build_raw_call(
                    call_id=main_call_id,
                    task_id=sample.task_id,
                    step_index=step_idx,
                    actor="main",
                    model=self.config.main_model,
                    system_prompt=action.system_prompt,
                    user_prompt=action.user_prompt,
                    raw_text=action.raw_response,
                    thinking=action.thinking,
                    parsed=action.parsed_payload,
                    input_tokens=int(action.input_tokens),
                    output_tokens=int(action.output_tokens),
                    cost=action.cost,
                )
            )

            if action.action == "delegate_task":
                requested_model = action.model or delegate_models[0]
                api_model = requested_model
                request = DelegateRequest(
                    task_id=sample.task_id,
                    question=sample.question,
                    options=sample.options,
                    images=sample.images,
                    model=api_model,
                    instruction=action.instruction or "",
                    task_type=action.task_type,
                    prior_attempts=list(memory.attempts),
                )
                _record_model_call(requested_model)
                delegate_tool = DelegateTaskTool(
                    sub_agent=SubAgentReasoning(llm=self._delegate_client_for_model(api_model))
                )
                delegate_result = await delegate_tool(request)
                total_cost += delegate_result.cost
                sub_tokens += int(delegate_result.input_tokens) + int(delegate_result.output_tokens)
                delegate_call_id = _next_call_id()
                raw_calls.append(
                    self._build_raw_call(
                        call_id=delegate_call_id,
                        task_id=sample.task_id,
                        step_index=step_idx,
                        actor="delegate",
                        model=request.model,
                        system_prompt=delegate_result.system_prompt,
                        user_prompt=delegate_result.user_prompt,
                        raw_text=delegate_result.raw_answer_text,
                        thinking=delegate_result.thinking,
                        parsed={
                            **delegate_result.parsed_payload,
                            "answer": delegate_result.answer,
                            "evidence": delegate_result.reasoning_summary,
                            "confidence": delegate_result.confidence,
                            "parse_ok": delegate_result.parse_ok,
                        },
                        input_tokens=int(delegate_result.input_tokens),
                        output_tokens=int(delegate_result.output_tokens),
                        cost=delegate_result.cost,
                        error=delegate_result.error,
                    )
                )
                memory.add_attempt(
                    AttemptRecord(
                        attempt_index=step_idx,
                        model=requested_model,
                        instruction=request.instruction,
                        delegate_result=delegate_result,
                        main_reasoning=action.reasoning,
                    )
                )
                decision_steps.append(
                    self._build_delegate_step(
                        step_index=step_idx,
                        action=action,
                        request=request,
                        delegate_result=delegate_result,
                        raw_call_ids=[main_call_id, delegate_call_id],
                    )
                )
                continue

            if action.action == "submit":
                submit_action = action
                submit_reason = action.submit_reason or action.reasoning
                submit_result = submit_tool(action, memory.attempts, reason=submit_reason)
                decision_steps.append(
                    self._build_submit_step(
                        step_index=step_idx,
                        action=action,
                        boxed_letter=submit_result.final_boxed_letter,
                        submit_reason=submit_reason,
                        raw_call_ids=[main_call_id],
                        source="orchestra",
                    )
                )
                break

        if submit_result is None:
            submit_action = submit_action or action
            submit_reason = "Fallback submit after max steps."
            submit_result = submit_tool(
                submit_action,
                memory.attempts,
                reason=submit_reason,
            )
            decision_steps.append(
                self._build_submit_step(
                    step_index=len(decision_steps) + 1,
                    action=submit_action,
                    boxed_letter=submit_result.final_boxed_letter,
                    submit_reason=submit_reason,
                    raw_call_ids=[],
                    source="system_fallback",
                )
            )

        if self.config.enable_scoring:
            _record_model_call(self.config.judge_model)
            score = await score_reasoning_sample(
                sample=sample,
                final_answer_text=submit_result.final_answer_text,
                judge_client=self.judge_client,
                judge_model=self.config.judge_model,
            )
            mca = score.mca
            rv = score.rv
            judge_raw = score.judge_raw
            judge_cost = score.judge_cost
            judge_input_tokens = score.judge_input_tokens
            judge_output_tokens = score.judge_output_tokens
            total_cost += judge_cost
            judge_call_id = _next_call_id()
            raw_calls.append(
                self._build_raw_call(
                    call_id=judge_call_id,
                    task_id=sample.task_id,
                    step_index=None,
                    actor="judge",
                    model=self.config.judge_model,
                    system_prompt=score.system_prompt,
                    user_prompt=score.user_prompt,
                    raw_text=judge_raw,
                    thinking="",
                    parsed={"rv": rv},
                    input_tokens=judge_input_tokens,
                    output_tokens=judge_output_tokens,
                    cost=judge_cost,
                )
            )
        else:
            mca = compute_mca(sample, submit_result.final_answer_text)
            rv = 0.0
            judge_raw = "scoring_skipped(enable_scoring=false)"
            judge_cost = 0.0
            judge_input_tokens = 0
            judge_output_tokens = 0

        models_used = list(dict.fromkeys(model_calls))
        latency_seconds = time.perf_counter() - sample_start

        return ReasoningRunRecord(
            task_id=sample.task_id,
            discipline=sample.discipline,
            question=sample.question,
            options=sample.options,
            gold_answer_index=sample.answer_index,
            gold_answer_letter=chr(ord("A") + sample.answer_index),
            reference_steps=sample.steps,
            final_answer_text=submit_result.final_answer_text,
            final_boxed_letter=submit_result.final_boxed_letter,
            mca=mca,
            rv=rv,
            total_cost=total_cost,
            main_tokens=main_tokens,
            sub_tokens=sub_tokens,
            total_tokens=main_tokens + sub_tokens,
            latency_seconds=latency_seconds,
            models_used=models_used,
            model_usage=model_usage,
            attempts=memory.attempts,
            decision_steps=decision_steps,
            raw_calls=raw_calls,
            metadata={
                "submit_reason": submit_reason,
                "judge_raw": judge_raw,
                "judge_cost": judge_cost,
                "scoring_enabled": self.config.enable_scoring,
                "step_count": len(decision_steps),
            },
        )

    def _eligible_sub_models(self, sample: ReasoningSample) -> list[str]:
        configured_models = list(self.config.sub_models)

        if not sample.images:
            return configured_models
        eligible = [model for model in configured_models if supports_image_inputs(model)]
        if eligible:
            return eligible
        raise RuntimeError("No configured sub-models support image inputs for this sample.")

    def _delegate_client_for_model(self, model_name: str) -> Any:
        return self.sub_client

    def _build_summary(
        self,
        records: list[ReasoningRunRecord],
        run_wall_time_seconds: float = 0.0,
        output_dir: Path | None = None,
    ) -> RunSummary:
        total_samples = len(records)
        # Accuracy metrics exclude infra-failed samples from the denominator.
        scored_records = [record for record in records if record.metadata.get("status") != "failed"]
        scored_samples = len(scored_records)
        avg_mca = sum(record.mca for record in scored_records) / scored_samples if scored_samples else 0.0
        avg_rv = sum(record.rv for record in scored_records) / scored_samples if scored_samples else 0.0
        avg_steps = sum(len(record.decision_steps) for record in scored_records) / scored_samples if scored_samples else 0.0
        total_cost = sum(record.total_cost for record in records)
        total_main_tokens = sum(record.main_tokens for record in records)
        total_sub_tokens = sum(record.sub_tokens for record in records)
        total_tokens = sum(record.total_tokens for record in records)
        total_latency_seconds = sum(record.latency_seconds for record in records)
        avg_latency_seconds = total_latency_seconds / total_samples
        model_usage: dict[str, int] = {}
        model_calls: list[str] = []
        for record in records:
            for model_name, call_count in record.model_usage.items():
                model_usage[model_name] = model_usage.get(model_name, 0) + int(call_count)
            model_calls.extend(record.models_used)
        models_used = list(dict.fromkeys(model_calls))

        return RunSummary(
            total_samples=total_samples,
            avg_mca=avg_mca,
            avg_rv=avg_rv,
            avg_steps=avg_steps,
            total_cost=total_cost,
            total_main_tokens=total_main_tokens,
            total_sub_tokens=total_sub_tokens,
            total_tokens=total_tokens,
            total_latency_seconds=total_latency_seconds,
            avg_latency_seconds=avg_latency_seconds,
            run_wall_time_seconds=run_wall_time_seconds,
            models_used=models_used,
            model_usage=model_usage,
            sample_method=self.config.sample_method,
            sample_seed=self.config.sample_seed,
            sampled_task_ids=[record.task_id for record in records],
            output_dir=str(output_dir or self.config.output_dir),
        )

    def _record_to_json(self, record: ReasoningRunRecord) -> dict[str, Any]:
        return {
            "idx": record.task_id,
            "discipline": record.discipline,
            "question": record.question,
            "options": record.options,
            "answer": record.gold_answer_index,
            "gold_answer_letter": record.gold_answer_letter,
            "reference_steps": record.reference_steps,
            "model_answer": record.final_answer_text,
            "MCA": record.mca,
            "RV": record.rv,
            "total_cost": record.total_cost,
            "main_tokens": record.main_tokens,
            "sub_tokens": record.sub_tokens,
            "total_tokens": record.total_tokens,
            "latency_seconds": record.latency_seconds,
            "models_used": record.models_used,
            "model_usage": record.model_usage,
            "steps": record.decision_steps,
            **record.metadata,
        }

    def _checkpoint_record(self, record: ReasoningRunRecord, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        if record.metadata.get("status") == "skipped":
            record_json = self._record_to_json(record)
            checkpoint_file = output_dir / "scored.checkpoint.jsonl"
            with checkpoint_file.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record_json, ensure_ascii=False))
                f.write("\n")
            return
        trajectory_dir = output_dir / "trajectories"
        trajectory_dir.mkdir(parents=True, exist_ok=True)
        samples_dir = output_dir / "samples" / record.task_id
        samples_dir.mkdir(parents=True, exist_ok=True)

        record_json = self._record_to_json(record)
        view_json = self._record_to_view_json(record)
        dump_json(trajectory_dir / f"{record.task_id}.json", view_json)
        dump_json(samples_dir / "view.json", view_json)

        sample_calls_file = samples_dir / "calls.jsonl"
        with sample_calls_file.open("w", encoding="utf-8") as f:
            for item in record.raw_calls:
                f.write(json.dumps(item, ensure_ascii=False))
                f.write("\n")

        checkpoint_file = output_dir / "scored.checkpoint.jsonl"
        with checkpoint_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record_json, ensure_ascii=False))
            f.write("\n")

        raw_calls_file = output_dir / "raw_calls.jsonl"
        with raw_calls_file.open("a", encoding="utf-8") as f:
            for item in record.raw_calls:
                f.write(json.dumps(item, ensure_ascii=False))
                f.write("\n")

    def _save_outputs(
        self,
        records: list[ReasoningRunRecord],
        summary: RunSummary,
        output_dir: Path | None = None,
    ) -> None:
        output_dir = output_dir or self.config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        records_json = [self._record_to_json(record) for record in records]
        records_view_json = [self._record_to_view_json(record) for record in records]

        predictions = [
            {
                "idx": item["idx"],
                "discipline": item["discipline"],
                "question": item["question"],
                "options": item["options"],
                "model_answer": item["model_answer"],
            }
            for item in records_json
        ]

        dump_json(output_dir / "predictions.json", predictions)
        dump_json(output_dir / "scored.json", records_view_json)
        dump_json(output_dir / "summary.json", asdict(summary))

        trajectory_dir = output_dir / "trajectories"
        trajectory_dir.mkdir(parents=True, exist_ok=True)
        for view_item, record in zip(records_view_json, records, strict=True):
            dump_json(trajectory_dir / f"{record.task_id}.json", view_item)
            sample_dir = output_dir / "samples" / record.task_id
            sample_dir.mkdir(parents=True, exist_ok=True)
            dump_json(sample_dir / "view.json", view_item)
            with (sample_dir / "calls.jsonl").open("w", encoding="utf-8") as f:
                for raw_call in record.raw_calls:
                    f.write(json.dumps(raw_call, ensure_ascii=False))
                    f.write("\n")

        raw_calls_file = output_dir / "raw_calls.jsonl"
        with raw_calls_file.open("w", encoding="utf-8") as f:
            for record in records:
                for raw_call in record.raw_calls:
                    f.write(json.dumps(raw_call, ensure_ascii=False))
                    f.write("\n")

        export_reasoning_compatible_logs(
            output_dir=output_dir,
            model_name=self.config.main_model,
            discipline_repr=self.config.discipline_repr(),
            records=records_json,
        )
