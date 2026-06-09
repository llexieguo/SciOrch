from __future__ import annotations

import asyncio
import itertools
import json
import math
import os
import random
import re
import shutil
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sciorch.adapters.reasoning.dataset import load_reasoning_samples
from sciorch.adapters.reasoning.io import dump_json
from sciorch.adapters.reasoning.scorer import compute_mca
from sciorch.config import OrchestratorConfig
from sciorch.core.main_agent import MainAgent
from sciorch.core.memory import MainMemory
from sciorch.core.subagent_reasoning import SubAgentReasoning
from sciorch.core.tools.submit import SubmitTool
from sciorch.llm.model_capabilities import supports_image_inputs
from sciorch.llm.openai_compatible import OpenAICompatibleClient
from sciorch.types import AttemptRecord, DelegateRequest, DelegateResult, MainAction, ReasoningSample

from mcts.config import MCTSConfig
from mcts.instruction_similarity import MiniLMInstructionSimilarity
from mcts.prompts import build_node_instruction

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - optional dependency fallback
    tqdm = None


def _normalize_question_for_pool(text: str) -> str:
    text = str(text or "").lower()
    text = text.replace("<image><image>", " ")
    text = text.replace("<image> <image>", " ")
    text = text.replace("<image>", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


@dataclass
class SearchNode:
    node_id: str
    depth: int
    round_index: int
    parent_id: str | None
    instruction: str
    model_pool: list[str]
    action: str
    main_model: str
    chosen_model: str | None = None
    orchestra_reasoning: str = ""
    focus_question: str | None = None
    delegate_answer: str | None = None
    delegate_evidence: str = ""
    delegate_confidence: float | None = None
    delegate_parse_ok: bool | None = None
    submit_reason: str | None = None
    final_answer_text: str = ""
    boxed_letter: str | None = None
    cost: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    raw_call_ids: list[str] = field(default_factory=list)
    raw_calls: list[dict[str, Any]] = field(default_factory=list)
    is_terminal: bool = False
    status: str = "completed"
    is_correct: bool = False
    error: str | None = None
    children: list["SearchNode"] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "depth": self.depth,
            "round_index": self.round_index,
            "parent_id": self.parent_id,
            "instruction": self.instruction,
            "action": self.action,
            "main_model": self.main_model,
            "chosen_model": self.chosen_model,
            "model_pool": self.model_pool,
            "orchestra_reasoning": self.orchestra_reasoning,
            "focus_question": self.focus_question,
            "delegate_answer": self.delegate_answer,
            "delegate_evidence": self.delegate_evidence,
            "delegate_confidence": self.delegate_confidence,
            "delegate_parse_ok": self.delegate_parse_ok,
            "submit_reason": self.submit_reason,
            "final_answer_text": self.final_answer_text,
            "boxed_letter": self.boxed_letter,
            "cost": self.cost,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "raw_call_ids": self.raw_call_ids,
            "is_terminal": self.is_terminal,
            "status": self.status,
            "is_correct": self.is_correct,
            "error": self.error,
            "children_ids": [child.node_id for child in self.children],
        }

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> "SearchNode":
        return cls(
            node_id=str(payload["node_id"]),
            depth=int(payload.get("depth", 0)),
            round_index=int(payload.get("round_index", 0)),
            parent_id=payload.get("parent_id"),
            instruction=str(payload.get("instruction", "")),
            model_pool=[str(item) for item in payload.get("model_pool", [])],
            action=str(payload.get("action", "error")),
            main_model=str(payload.get("main_model", "")),
            chosen_model=payload.get("chosen_model"),
            orchestra_reasoning=str(payload.get("orchestra_reasoning", "")),
            focus_question=payload.get("focus_question"),
            delegate_answer=payload.get("delegate_answer"),
            delegate_evidence=str(payload.get("delegate_evidence", "")),
            delegate_confidence=(
                float(payload["delegate_confidence"])
                if payload.get("delegate_confidence") is not None
                else None
            ),
            delegate_parse_ok=payload.get("delegate_parse_ok"),
            submit_reason=payload.get("submit_reason"),
            final_answer_text=str(payload.get("final_answer_text", "")),
            boxed_letter=payload.get("boxed_letter"),
            cost=float(payload.get("cost", 0.0)),
            input_tokens=int(payload.get("input_tokens", 0)),
            output_tokens=int(payload.get("output_tokens", 0)),
            raw_call_ids=[str(item) for item in payload.get("raw_call_ids", [])],
            is_terminal=bool(payload.get("is_terminal", False)),
            status=str(payload.get("status", "completed")),
            is_correct=bool(payload.get("is_correct", False)),
            error=payload.get("error"),
        )


@dataclass
class SampledOrchestraCandidate:
    candidate_id: str
    prompt_index: int
    sample_index: int
    node_instruction: str
    model_pool: list[str]
    action: MainAction | None = None
    raw_call_id: str | None = None
    raw_call: dict[str, Any] | None = None
    cost: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    error: str | None = None

    @property
    def delegate_instruction(self) -> str:
        if self.action is None or self.action.action != "delegate_task":
            return ""
        instruction = self.action.instruction
        return instruction.strip() if isinstance(instruction, str) else ""


@dataclass
class ResumedTreeState:
    all_nodes: dict[str, SearchNode]
    frontier: list[SearchNode]
    final_leaves: list[SearchNode]
    failed_terminal_nodes: list[SearchNode]
    sample_raw_calls: list[dict[str, Any]]
    budget_spent: float
    total_tokens: int
    raw_call_counter_value: int
    node_counter: int
    rounds: list[dict[str, Any]]


class ChildModelPool(list[str]):
    def __init__(self, models: list[str] | None = None) -> None:
        super().__init__(models or [])


class MCTSReasoningRunner:
    def __init__(self, config: MCTSConfig) -> None:
        self.config = config
        self.orchestra_client = self._build_client(
            endpoint=config.orchestra_endpoint,
            enable_thinking=config.orchestra_enable_thinking,
            local_base_url=config.orchestra_local_base_url,
            local_api_key=config.orchestra_local_api_key,
            local_api_key_env=config.orchestra_local_api_key_env,
            local_temperature=config.orchestra_local_temperature,
            remote_base_url=config.orchestra_openai_base_url,
            remote_base_url_env=config.orchestra_openai_base_url_env,
            remote_api_key_env=config.orchestra_openai_api_key_env,
        )
        self.delegate_client = self._build_client(
            endpoint=config.delegate_endpoint,
            enable_thinking=config.delegate_enable_thinking,
            local_base_url=config.delegate_local_base_url,
            local_api_key=config.delegate_local_api_key,
            local_api_key_env=config.delegate_local_api_key_env,
            local_temperature=None,
            remote_base_url=config.delegate_openai_base_url,
            remote_base_url_env=config.delegate_openai_base_url_env,
            remote_api_key_env=config.delegate_openai_api_key_env,
        )
        self.submit_tool = SubmitTool()
        self._disabled_delegate_models: set[str] = set()
        self._progress = None
        self._total_cost: float = 0.0
        self._last_task_cost: dict[str, float] = {}
        self._correct_models_by_question: dict[str, list[str]] = (
            self._load_correct_models(self.config.correct_model_pool_dir)
            if self.config.correct_model_pool_dir is not None
            else {}
        )
        self._instruction_similarity: MiniLMInstructionSimilarity | None = None

    @staticmethod
    def _build_client(
        *,
        endpoint: str,
        enable_thinking: bool | None,
        local_base_url: str | None,
        local_api_key: str | None,
        local_api_key_env: str,
        local_temperature: float | None,
        remote_base_url: str | None,
        remote_base_url_env: str,
        remote_api_key_env: str,
    ) -> OpenAICompatibleClient:
        if endpoint == "local":
            return OpenAICompatibleClient(
                api_key=local_api_key or os.getenv(local_api_key_env) or "EMPTY",
                base_url=local_base_url,
                api_key_env=local_api_key_env,
                default_temperature=local_temperature,
                enable_thinking=enable_thinking,
                timeout_s=600,
            )
        return OpenAICompatibleClient(
            api_key=os.getenv(remote_api_key_env),
            base_url=remote_base_url or os.getenv(remote_base_url_env),
            api_key_env=remote_api_key_env,
            enable_thinking=enable_thinking,
        )

    @staticmethod
    def _next_call_id(task_id: str, raw_call_counter: dict[str, int]) -> str:
        raw_call_counter["value"] += 1
        return f"{task_id}_call_{raw_call_counter['value']:04d}"

    @staticmethod
    def _build_raw_call(
        *,
        call_id: str,
        task_id: str,
        node_id: str,
        round_index: int,
        depth: int,
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
            "timestamp": MCTSReasoningRunner._now(),
            "task_id": task_id,
            "node_id": node_id,
            "round_index": round_index,
            "depth": depth,
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

    async def run(self) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        run_dir = self._build_run_dir()
        run_dir.mkdir(parents=True, exist_ok=True)
        self._write_json(run_dir / "config.snapshot.json", self.config.to_json())

        samples = self._select_samples(self._load_dataset_samples())
        selected_task_ids = [sample.task_id for sample in samples]

        global_events = run_dir
        self._write_json(
            run_dir / "selected_tasks.json",
            {
                "dataset_name": self.config.dataset_name,
                "dataset_split": self.config.dataset_split,
                "discipline": self.config.discipline,
                "sample_count": len(samples),
                "sample_seed": self.config.sample_seed,
                "selected_task_ids": selected_task_ids,
                "selected_samples": [
                    {
                        "task_id": sample.task_id,
                        "discipline": sample.discipline,
                    }
                    for sample in samples
                ],
            },
        )

        existing_total_cost, existing_last_task_cost, initial_leaf_progress = self._load_existing_progress_state(
            run_dir=run_dir,
            selected_task_ids=selected_task_ids,
        )
        self._total_cost = existing_total_cost
        self._last_task_cost = existing_last_task_cost

        progress = None
        if self.config.show_progress and tqdm is not None:
            progress = tqdm(
                total=(
                    len(samples) * self.config.target_leaf_trajectories
                    if self.config.target_leaf_trajectories is not None
                    else None
                ),
                initial=initial_leaf_progress,
                desc="MCTS",
                unit="leaf",
                dynamic_ncols=True,
            )
            progress.set_postfix_str(f"total=${self._total_cost:.3f}")
        self._progress = progress

        semaphore = asyncio.Semaphore(max(1, self.config.max_concurrency))

        async def run_limited(sample: ReasoningSample) -> dict[str, Any]:
            async with semaphore:
                return await self._run_sample(
                    run_dir=run_dir,
                    sample=sample,
                    global_events=global_events,
                )

        try:
            tasks = [asyncio.create_task(run_limited(sample)) for sample in samples]
            bundles = await asyncio.gather(*tasks)
        finally:
            if progress is not None:
                progress.close()
                self._progress = None

        bundles.sort(key=lambda item: item["summary"]["task_id"])
        records = self._collect_sample_results(run_dir)
        all_raw_calls = self._collect_sample_raw_calls(run_dir)

        summary = self._build_summary(
            records=records,
            run_dir=run_dir,
        )
        self._write_json(run_dir / "scored.json", records)
        self._write_json(run_dir / "summary.json", summary)
        with (run_dir / "raw_calls.jsonl").open("w", encoding="utf-8") as f:
            for raw_call in all_raw_calls:
                f.write(json.dumps(raw_call, ensure_ascii=False))
                f.write("\n")
        return records, summary

    def _build_run_dir(self) -> Path:
        model_dir_name = self._sanitize_model_dir_name(self.config.orchestra_model)
        return (self.config.output_dir / model_dir_name).resolve()

    def _load_dataset_samples(self) -> list[ReasoningSample]:
        config = OrchestratorConfig(
            main_model="mcts-loader",
            sub_models=["mcts-loader"],
            output_dir=self.config.output_dir,
            discipline=self.config.discipline,
            dataset_split=self.config.dataset_split,
            dataset_name=self.config.dataset_name,
            exclude_task_ids=self.config.exclude_task_ids,
            exclude_task_ids_path=self.config.exclude_task_ids_path,
        )
        return load_reasoning_samples(config)

    def _select_samples(self, samples: list[ReasoningSample]) -> list[ReasoningSample]:
        if self.config.task_ids:
            sample_by_id = {sample.task_id: sample for sample in samples}
            missing = [task_id for task_id in self.config.task_ids if task_id not in sample_by_id]
            if missing:
                raise KeyError(f"Missing samples in dataset for task ids: {missing}")
            return [sample_by_id[task_id] for task_id in self.config.task_ids]

        if len(samples) < self.config.sample_count:
            raise ValueError(
                f"Only found {len(samples)} samples after dataset filtering, but sample_count={self.config.sample_count}"
            )
        rng = random.Random(self.config.sample_seed)
        picks = rng.sample(samples, self.config.sample_count)
        picks.sort(key=lambda sample: sample.task_id)
        return picks

    def _collect_sample_results(self, run_dir: Path) -> list[dict[str, Any]]:
        samples_dir = run_dir / "samples"
        if not samples_dir.is_dir():
            return []
        records: list[dict[str, Any]] = []
        for sample_dir in sorted(path for path in samples_dir.iterdir() if path.is_dir()):
            result_path = sample_dir / "result.json"
            if not result_path.exists():
                continue
            payload = self._read_json(result_path)
            if isinstance(payload, dict):
                records.append(payload)
        records.sort(key=lambda item: str(item.get("task_id", "")))
        return records

    def _collect_sample_raw_calls(self, run_dir: Path) -> list[dict[str, Any]]:
        samples_dir = run_dir / "samples"
        if not samples_dir.is_dir():
            return []
        raw_calls: list[dict[str, Any]] = []
        for sample_dir in sorted(path for path in samples_dir.iterdir() if path.is_dir()):
            calls_path = sample_dir / "calls.jsonl"
            if not calls_path.exists():
                continue
            raw_calls.extend(self._read_jsonl(calls_path))
        raw_calls.sort(
            key=lambda item: (
                str(item.get("task_id", "")),
                str(item.get("timestamp", "")),
                str(item.get("call_id", "")),
            )
        )
        return raw_calls

    def _load_existing_progress_state(
        self,
        *,
        run_dir: Path,
        selected_task_ids: list[str],
    ) -> tuple[float, dict[str, float], int]:
        samples_dir = run_dir / "samples"
        if not self.config.resume or not samples_dir.is_dir():
            return 0.0, {}, 0

        selected = set(selected_task_ids)
        total_cost = 0.0
        last_task_cost: dict[str, float] = {}
        initial_leaf_progress = 0

        for sample_dir in sorted(path for path in samples_dir.iterdir() if path.is_dir() and path.name in selected):
            result_path = sample_dir / "result.json"
            latest_path = sample_dir / "latest.json"
            payload: dict[str, Any] | None = None
            if result_path.exists():
                loaded = self._read_json(result_path)
                payload = loaded if isinstance(loaded, dict) else None
            elif latest_path.exists():
                loaded = self._read_json(latest_path)
                payload = loaded if isinstance(loaded, dict) else None

            if payload is None:
                continue

            task_id = str(payload.get("task_id", sample_dir.name))
            cost = float(payload.get("total_cost", payload.get("budget_spent", 0.0)) or 0.0)
            total_cost += cost
            last_task_cost[task_id] = cost
            initial_leaf_progress += int(payload.get("final_leaf_count", 0) or 0)

        return total_cost, last_task_cost, initial_leaf_progress

    async def _run_sample(
        self,
        *,
        run_dir: Path,
        sample: ReasoningSample,
        global_events: Path,
    ) -> dict[str, Any]:
        sample_dir = run_dir / "samples" / sample.task_id
        resume_state: ResumedTreeState | None = None
        if sample_dir.exists():
            result_path = sample_dir / "result.json"
            latest_path = sample_dir / "latest.json"
            if self.config.resume and result_path.exists():
                payload = self._read_json(result_path)
                return {
                    "summary": payload if isinstance(payload, dict) else {"task_id": sample.task_id},
                    "raw_calls": [],
                }
            if self.config.resume and latest_path.exists():
                resume_state = self._load_resume_state(sample_dir)
                if resume_state is None:
                    raise RuntimeError(
                        f"Found partial sample state for {sample.task_id}, but existing snapshot format is not resumable: {latest_path}"
                    )
            else:
                shutil.rmtree(sample_dir)
        sample_dir.mkdir(parents=True, exist_ok=True)
        sample_start = time.perf_counter()

        self._append_events(
            sample_dir=sample_dir,
            global_events=global_events,
            event={
                "event": "sample_started",
                "timestamp": self._now(),
                "task_id": sample.task_id,
            },
        )

        rng = random.Random(self._stable_seed(sample.task_id, self.config.tree_seed))
        tree = await self._run_tree(
            sample=sample,
            sample_dir=sample_dir,
            global_events=global_events,
            rng=rng,
            resume_state=resume_state,
        )

        latency_seconds = time.perf_counter() - sample_start
        result = {
            "task_id": sample.task_id,
            "discipline": sample.discipline,
            "status": "completed",
            "question": sample.question,
            "options": sample.options,
            "gold_answer_letter": chr(ord("A") + sample.answer_index),
            "orchestra_model": self.config.orchestra_model,
            "success": tree["success"],
            "any_correct_leaf": tree["any_correct_leaf"],
            "best_leaf_correct": tree["best_leaf_correct"],
            "majority_correct": tree["majority_correct"],
            "correct_leaf_count": tree["correct_leaf_count"],
            "final_leaf_count": tree["final_leaf_count"],
            "open_leaf_count": tree["open_leaf_count"],
            "target_leaf_trajectories": self.config.target_leaf_trajectories,
            "branching_factor": self.config.branching_factor,
            "leaf_expand_ratio": self.config.leaf_expand_ratio,
            "frontier_limit": self.config.frontier_limit,
            "sibling_pool_strategy": self.config.sibling_pool_strategy,
            "path_max_steps": self.config.node_max_steps,
            "budget_limit": self.config.tree_budget_usd,
            "budget_spent": tree["budget_spent"],
            "budget_exhausted": tree["budget_exhausted"],
            "stop_reason": tree["stop_reason"],
            "expansion_rounds_ran": len(tree["rounds"]),
            "best_leaf_node_id": tree["best_leaf_node_id"],
            "best_leaf_boxed_letter": tree["best_leaf_boxed_letter"],
            "best_leaf_latest_delegate_confidence": tree["best_leaf_latest_delegate_confidence"],
            "majority_boxed_letter": tree["majority_boxed_letter"],
            "latency_seconds": latency_seconds,
            "total_cost": tree["cost"],
            "total_tokens": tree["total_tokens"],
            "total_model_calls": tree["model_calls"],
            "failed_terminal_count": tree["failed_terminal_count"],
        }
        view = {
            "task_id": sample.task_id,
            "status": "completed",
            "task": {
                "discipline": sample.discipline,
                "question": sample.question,
                "options": sample.options,
            },
            "reference": {
                "gold_answer_letter": chr(ord("A") + sample.answer_index),
                "steps": sample.steps,
            },
            "config": {
                "orchestra_model": self.config.orchestra_model,
                "branching_factor": self.config.branching_factor,
                "leaf_expand_ratio": self.config.leaf_expand_ratio,
                "frontier_limit": self.config.frontier_limit,
                "sibling_pool_strategy": self.config.sibling_pool_strategy,
                "target_leaf_trajectories": self.config.target_leaf_trajectories,
                "path_max_steps": self.config.node_max_steps,
                "budget_limit": self.config.tree_budget_usd,
            },
            "final_summary": {
                "success": tree["success"],
                "any_correct_leaf": tree["any_correct_leaf"],
                "best_leaf_correct": tree["best_leaf_correct"],
                "majority_correct": tree["majority_correct"],
                "stop_reason": tree["stop_reason"],
                "best_leaf_node_id": tree["best_leaf_node_id"],
                "best_leaf_boxed_letter": tree["best_leaf_boxed_letter"],
                "best_leaf_latest_delegate_confidence": tree["best_leaf_latest_delegate_confidence"],
                "majority_boxed_letter": tree["majority_boxed_letter"],
            },
            "metrics": {
                "budget_spent": tree["budget_spent"],
                "budget_exhausted": tree["budget_exhausted"],
                "total_cost": tree["cost"],
                "total_tokens": tree["total_tokens"],
                "total_model_calls": tree["model_calls"],
                "latency_seconds": latency_seconds,
                "expansion_rounds_ran": len(tree["rounds"]),
                "final_leaf_count": tree["final_leaf_count"],
                "open_leaf_count": tree["open_leaf_count"],
                "correct_leaf_count": tree["correct_leaf_count"],
                "failed_terminal_count": tree["failed_terminal_count"],
                "raw_call_count": max(len(tree["raw_calls"]), tree["model_calls"]),
                "node_count": len(tree["nodes"]),
            },
            "rounds": tree["rounds"],
            "trajectories": tree["trajectories"],
            "open_frontier_node_ids": tree["open_frontier_node_ids"],
            "final_leaf_node_ids": tree["final_leaf_node_ids"],
            "failed_terminal_node_ids": tree["failed_terminal_node_ids"],
        }
        self._write_json(sample_dir / "view.json", view)
        self._write_json(sample_dir / "result.json", result)
        self._write_jsonl(sample_dir / "calls.jsonl", tree["raw_calls"])
        self._write_jsonl(sample_dir / "nodes.jsonl", tree["nodes"])
        self._append_events(
            sample_dir=sample_dir,
            global_events=global_events,
            event={
                "event": "sample_completed",
                "timestamp": self._now(),
                "task_id": sample.task_id,
                "success": result["success"],
                "correct_leaf_count": result["correct_leaf_count"],
                "final_leaf_count": result["final_leaf_count"],
                "stop_reason": result["stop_reason"],
                "total_cost": result["total_cost"],
            },
        )
        return {
            "summary": result,
            "raw_calls": tree["raw_calls"],
        }

    async def _run_tree(
        self,
        *,
        sample: ReasoningSample,
        sample_dir: Path,
        global_events: Path,
        rng: random.Random,
        resume_state: ResumedTreeState | None = None,
    ) -> dict[str, Any]:
        if resume_state is None:
            root = SearchNode(
                node_id="root",
                depth=0,
                round_index=0,
                parent_id=None,
                instruction="root",
                model_pool=[],
                action="root",
                main_model=self.config.orchestra_model,
                is_terminal=False,
            )
            all_nodes: dict[str, SearchNode] = {root.node_id: root}
            final_leaves: list[SearchNode] = []
            failed_terminal_nodes: list[SearchNode] = []
            sample_raw_calls: list[dict[str, Any]] = []
            raw_call_counter: dict[str, int] = {"value": 0}
            budget_spent = 0.0
            total_tokens_spent = 0
            model_calls = 0
            node_counter = 0
            start_nodes: list[SearchNode] = [root]
        else:
            all_nodes = resume_state.all_nodes
            final_leaves = resume_state.final_leaves
            failed_terminal_nodes = resume_state.failed_terminal_nodes
            sample_raw_calls = resume_state.sample_raw_calls
            raw_call_counter = {"value": resume_state.raw_call_counter_value}
            budget_spent = resume_state.budget_spent
            total_tokens_spent = resume_state.total_tokens
            model_calls = max(len(sample_raw_calls), raw_call_counter["value"])
            node_counter = resume_state.node_counter
            start_nodes = resume_state.frontier

        stop_event = asyncio.Event()
        if self.config.target_leaf_trajectories is not None and len(final_leaves) >= self.config.target_leaf_trajectories:
            stop_event.set()

        async def expand_node(node: SearchNode) -> None:
            nonlocal budget_spent, total_tokens_spent, model_calls, node_counter

            if stop_event.is_set() or not self._is_expandable_node(node):
                return

            budget_snapshot = budget_spent
            if budget_snapshot >= self.config.tree_budget_usd:
                return

            # Per-node deterministic rng, scoped by task_id so different tasks
            # do not reuse the same node-level model pool sequence.
            node_rng = random.Random(
                self._stable_seed(f"{sample.task_id}:{node.node_id}", self.config.tree_seed)
            )
            child_model_pools = self._build_child_model_pools(
                node_rng,
                child_count=self.config.branching_factor,
                requires_image_inputs=self.config.main_use_images and bool(sample.images),
                sample=sample,
            )
            path = self._reconstruct_path(node, all_nodes)
            force_submit = (
                budget_snapshot >= self.config.tree_budget_usd
                or self._should_force_submit(len(final_leaves))
            )

            # Pre-allocate node IDs synchronously before any await
            first_node_id = node_counter + 1
            node_counter += len(child_model_pools)

            if self.config.orchestra_samples_per_prompt > 1:
                children, discarded_raw_calls = await self._expand_parent_with_diverse_sampling(
                    sample=sample,
                    parent=node,
                    path=path,
                    round_index=node.depth,
                    budget_snapshot=budget_snapshot,
                    first_node_id=first_node_id,
                    child_model_pools=child_model_pools,
                    force_submit=force_submit,
                    raw_call_counter=raw_call_counter,
                )
                sample_raw_calls.extend(discarded_raw_calls)
                total_tokens_spent += self._raw_call_tokens(discarded_raw_calls)
            else:
                child_results = await asyncio.gather(*[
                    self._expand_child(
                        sample=sample,
                        parent=node,
                        path=path,
                        round_index=node.depth,
                        node_counter=first_node_id + i,
                        budget_spent=budget_snapshot,
                        model_pool=mp,
                        force_submit=force_submit,
                        rng=node_rng,
                        raw_call_counter=raw_call_counter,
                    )
                    for i, mp in enumerate(child_model_pools)
                ])
                children = list(child_results)

            expandable_children: list[SearchNode] = []
            for child in children:
                sample_raw_calls.extend(child.raw_calls)
                total_tokens_spent += child.input_tokens + child.output_tokens
                budget_spent += child.cost
                model_calls += len(child.raw_call_ids)
                node.children.append(child)
                all_nodes[child.node_id] = child

                if child.is_terminal:
                    if self._is_submit_leaf(child):
                        final_leaves.append(child)
                    else:
                        failed_terminal_nodes.append(child)
                elif self._is_expandable_node(child):
                    expandable_children.append(child)

                self._progress_step(
                    increment=1,
                    task_id=sample.task_id,
                    round_index=child.depth,
                    leaf_count=len(final_leaves),
                    budget_spent=budget_spent,
                )

            open_frontier = [
                n for n in all_nodes.values()
                if self._is_expandable_node(n) and not n.children
            ]
            self._write_tree_snapshot(
                sample_dir=sample_dir,
                task_id=sample.task_id,
                all_nodes=all_nodes,
                budget_spent=budget_spent,
                final_leaves=final_leaves,
                failed_terminal_nodes=failed_terminal_nodes,
                open_frontier=open_frontier,
                sample_raw_calls=sample_raw_calls,
                total_tokens=total_tokens_spent,
                raw_call_counter_value=raw_call_counter["value"],
                node_counter=node_counter,
            )

            if self.config.target_leaf_trajectories is not None and len(final_leaves) >= self.config.target_leaf_trajectories:
                stop_event.set()
                return

            if budget_spent >= self.config.tree_budget_usd:
                return

            if expandable_children and not stop_event.is_set():
                await asyncio.gather(*[expand_node(child) for child in expandable_children])

        if start_nodes:
            await asyncio.gather(*[expand_node(node) for node in start_nodes])

        # Dynamic frontier replenishment: when all initial branches terminate
        # but we haven't reached the target leaf count, rescan the tree for
        # expandable nodes that were created during expansion and continue.
        while not stop_event.is_set() and budget_spent < self.config.tree_budget_usd:
            if self.config.target_leaf_trajectories is not None and len(final_leaves) >= self.config.target_leaf_trajectories:
                break
            new_frontier = [
                n for n in all_nodes.values()
                if self._is_expandable_node(n) and not n.children
            ]
            if not new_frontier:
                break
            selected, _ = self._select_nodes_for_expansion(
                frontier=new_frontier,
                all_nodes=all_nodes,
                rng=rng,
            )
            if not selected:
                break
            await asyncio.gather(*[expand_node(node) for node in selected])

        if stop_event.is_set() and self.config.target_leaf_trajectories is not None and len(final_leaves) >= self.config.target_leaf_trajectories:
            stop_reason: str = "target_leaf_trajectories_reached"
        elif budget_spent >= self.config.tree_budget_usd:
            stop_reason = "budget_exhausted"
        elif not start_nodes:
            stop_reason = "no_frontier"
        else:
            stop_reason = "all_leaves_finished"

        open_frontier_final = [
            n for n in all_nodes.values()
            if self._is_expandable_node(n) and not n.children
        ]
        rounds = self._reconstruct_rounds(all_nodes)
        trajectories = self._build_leaf_trajectories(final_leaves=final_leaves, all_nodes=all_nodes)
        best_leaf = self._best_leaf(final_leaves, all_nodes=all_nodes)
        majority_boxed_letter = self._majority_letter(trajectories)
        gold_answer_letter = chr(ord("A") + sample.answer_index)
        any_correct_leaf = any(node.is_correct for node in final_leaves)
        majority_correct = majority_boxed_letter == gold_answer_letter if majority_boxed_letter else False

        tree = {
            "task_id": sample.task_id,
            "budget_limit": self.config.tree_budget_usd,
            "budget_spent": budget_spent,
            "budget_exhausted": budget_spent >= self.config.tree_budget_usd,
            "stop_reason": stop_reason,
            "target_leaf_trajectories": self.config.target_leaf_trajectories,
            "branching_factor": self.config.branching_factor,
            "leaf_expand_ratio": self.config.leaf_expand_ratio,
            "frontier_limit": self.config.frontier_limit,
            "sibling_pool_strategy": self.config.sibling_pool_strategy,
            "path_max_steps": self.config.node_max_steps,
            "final_leaf_count": len(final_leaves),
            "open_leaf_count": len(open_frontier_final),
            "failed_terminal_count": len(failed_terminal_nodes),
            "correct_leaf_count": sum(1 for node in final_leaves if node.is_correct),
            "success": any_correct_leaf,
            "any_correct_leaf": any_correct_leaf,
            "best_leaf_correct": bool(best_leaf and best_leaf.is_correct),
            "majority_correct": majority_correct,
            "best_leaf_node_id": best_leaf.node_id if best_leaf is not None else None,
            "best_leaf_boxed_letter": best_leaf.boxed_letter if best_leaf is not None else None,
            "best_leaf_latest_delegate_confidence": (
                self._latest_delegate_confidence(self._reconstruct_path(best_leaf, all_nodes))
                if best_leaf is not None
                else None
            ),
            "majority_boxed_letter": majority_boxed_letter,
            "final_leaf_node_ids": [node.node_id for node in final_leaves],
            "failed_terminal_node_ids": [node.node_id for node in failed_terminal_nodes],
            "model_calls": max(len(sample_raw_calls), raw_call_counter["value"]),
            "total_tokens": total_tokens_spent,
            "cost": budget_spent,
            "rounds": rounds,
            "trajectories": trajectories,
            "open_frontier_node_ids": [node.node_id for node in open_frontier_final],
            "nodes": [node.to_json() for node in all_nodes.values()],
            "raw_calls": sample_raw_calls,
        }
        return tree

    async def _expand_round_unused(
        self,
        *,
        sample: ReasoningSample,
        sample_dir: Path,
        global_events: Path,
        all_nodes: dict[str, SearchNode],
        completed_rounds: list[dict[str, Any]],
        selected_parents: list[SearchNode],
        pending_frontier: list[SearchNode],
        active_frontier_count: int,
        round_index: int,
        selection_strategy: str,
        budget_spent: float,
        model_calls: int,
        node_counter: int,
        rng: random.Random,
        selected_count_requested: int | None = None,
        final_leaves: list[SearchNode],
        failed_terminal_nodes: list[SearchNode],
        sample_raw_calls: list[dict[str, Any]],
        raw_call_counter: dict[str, int],
        total_tokens_spent: int,
    ) -> tuple[dict[str, Any], float, int, int, int]:
        selected_parent_ids = [node.node_id for node in selected_parents]
        created_node_ids: list[str] = []
        created_final_leaf_node_ids: list[str] = []
        created_failed_terminal_node_ids: list[str] = []
        created_expandable_node_ids: list[str] = []

        self._append_events(
            sample_dir=sample_dir,
            global_events=global_events,
            event={
                "event": "expansion_round_started",
                "timestamp": self._now(),
                "task_id": sample.task_id,
                "round_index": round_index,
                "selection_strategy": selection_strategy,
                "selected_parent_ids": selected_parent_ids,
                "selected_count_requested": selected_count_requested,
                "budget_spent": budget_spent,
            },
        )

        expanded_parent_count = 0
        children_created = 0
        budget_exhausted = False
        force_submit_due_to_target = self._should_force_submit_round(
            final_leaf_count=len(final_leaves),
            active_parent_count=active_frontier_count,
        )

        for parent in selected_parents:
            if parent.action not in {"root", "delegate"}:
                continue
            if parent.depth >= self.config.node_max_steps:
                continue

            expanded_parent_count += 1
            path = self._reconstruct_path(parent, all_nodes)
            force_submit_due_to_budget = budget_spent >= self.config.tree_budget_usd
            child_model_pools = self._build_child_model_pools(
                rng,
                child_count=self.config.branching_factor,
                requires_image_inputs=self.config.main_use_images and bool(sample.images),
                sample=sample,
            )

            if self.config.orchestra_samples_per_prompt > 1:
                first_node_id = node_counter + 1
                (
                    sampled_children,
                    discarded_raw_calls,
                ) = await self._expand_parent_with_diverse_sampling(
                    sample=sample,
                    parent=parent,
                    path=path,
                    round_index=round_index,
                    budget_snapshot=budget_spent,
                    first_node_id=first_node_id,
                    child_model_pools=child_model_pools,
                    force_submit=(force_submit_due_to_target or force_submit_due_to_budget),
                    raw_call_counter=raw_call_counter,
                )
                node_counter = first_node_id + len(sampled_children) - 1
                for c in sampled_children:
                    budget_spent += c.cost
                    model_calls += len(c.raw_call_ids)
                budget_exhausted = budget_exhausted or (budget_spent >= self.config.tree_budget_usd)
                sample_raw_calls.extend(discarded_raw_calls)
                total_tokens_spent += self._raw_call_tokens(discarded_raw_calls)
                child_nodes = sampled_children
            else:
                child_nodes = []
                for model_pool in child_model_pools:
                    node_counter += 1
                    child = await self._expand_child(
                        sample=sample,
                        parent=parent,
                        path=path,
                        round_index=round_index,
                        node_counter=node_counter,
                        budget_spent=budget_spent,
                        model_pool=model_pool,
                        force_submit=(
                            force_submit_due_to_target
                            or budget_spent >= self.config.tree_budget_usd
                        ),
                        rng=rng,
                        raw_call_counter=raw_call_counter,
                    )
                    budget_spent += child.cost
                    model_calls += len(child.raw_call_ids)
                    child_nodes.append(child)
                    if budget_spent >= self.config.tree_budget_usd:
                        budget_exhausted = True

            for child in child_nodes:
                sample_raw_calls.extend(child.raw_calls)
                total_tokens_spent += child.input_tokens + child.output_tokens
                parent.children.append(child)
                all_nodes[child.node_id] = child
                created_node_ids.append(child.node_id)
                children_created += 1
                if child.is_terminal:
                    if self._is_submit_leaf(child):
                        created_final_leaf_node_ids.append(child.node_id)
                        final_leaves.append(child)
                    else:
                        created_failed_terminal_node_ids.append(child.node_id)
                        failed_terminal_nodes.append(child)
                elif child.action == "delegate" and child.depth < self.config.node_max_steps:
                    created_expandable_node_ids.append(child.node_id)

                self._append_events(
                    sample_dir=sample_dir,
                    global_events=global_events,
                    event={
                        "event": "node_expanded",
                        "timestamp": self._now(),
                        "task_id": sample.task_id,
                        "round_index": round_index,
                        "parent_id": parent.node_id,
                        "node": child.to_json(),
                        "budget_spent": budget_spent,
                    },
                )

                current_terminal_count = len(final_leaves)
                self._progress_step(
                    increment=1,
                    task_id=sample.task_id,
                    round_index=round_index,
                    leaf_count=current_terminal_count,
                    budget_spent=budget_spent,
                )

        submit_found = bool(created_final_leaf_node_ids)
        next_frontier = self._build_depth_first_frontier(
            pending_frontier=pending_frontier,
            created_expandable=[all_nodes[node_id] for node_id in created_expandable_node_ids],
            all_nodes=all_nodes,
        )
        next_count_requested = len(next_frontier)
        round_summary = {
            "round_index": round_index,
            "selection_strategy": selection_strategy,
            "selected_count_requested": selected_count_requested,
            "selected_parent_count": len(selected_parents),
            "selected_parent_ids": selected_parent_ids,
            "expanded_parent_count": expanded_parent_count,
            "children_created": children_created,
            "created_node_ids": created_node_ids,
            "created_final_leaf_node_ids": created_final_leaf_node_ids,
            "created_failed_terminal_node_ids": created_failed_terminal_node_ids,
            "created_expandable_node_ids": created_expandable_node_ids,
            "next_frontier_count_requested": next_count_requested,
            "next_frontier_node_ids": [node.node_id for node in next_frontier],
            "final_leaf_count_after": len(final_leaves),
            "failed_terminal_count_after": len(failed_terminal_nodes),
            "open_leaf_count_after": len(next_frontier),
            "budget_spent": budget_spent,
            "budget_exhausted": budget_exhausted,
            "submit_found": submit_found,
            "force_submit_due_to_target": force_submit_due_to_target,
            "force_submit_due_to_budget": budget_spent >= self.config.tree_budget_usd,
            "target_leaf_trajectories_reached": (
                self.config.target_leaf_trajectories is not None
                and len(final_leaves) >= self.config.target_leaf_trajectories
            ),
        }

        self._write_live_tree_snapshot(
            sample_dir=sample_dir,
            task_id=sample.task_id,
            all_nodes=all_nodes,
            completed_rounds=completed_rounds,
            current_round=round_summary,
            budget_spent=budget_spent,
            final_leaves=final_leaves,
            failed_terminal_nodes=failed_terminal_nodes,
            open_frontier=next_frontier,
            sample_raw_calls=sample_raw_calls,
            total_tokens=total_tokens_spent,
            raw_call_counter_value=raw_call_counter["value"],
            node_counter=node_counter,
        )

        self._append_events(
            sample_dir=sample_dir,
            global_events=global_events,
            event={
                "event": "expansion_round_completed",
                "timestamp": self._now(),
                "task_id": sample.task_id,
                "round_index": round_index,
                "summary": round_summary,
            },
        )
        return round_summary, budget_spent, total_tokens_spent, model_calls, node_counter

    def _instruction_similarity_encoder(self) -> MiniLMInstructionSimilarity:
        if self._instruction_similarity is None:
            self._instruction_similarity = MiniLMInstructionSimilarity(
                model_name=self.config.instruction_similarity_model_name,
                batch_size=self.config.instruction_similarity_batch_size,
                local_files_only=self.config.instruction_similarity_local_files_only,
            )
        return self._instruction_similarity

    def _compute_instruction_similarity_matrix(self, texts: list[str]) -> list[list[float]]:
        return self._instruction_similarity_encoder().similarity_matrix(texts)

    def _select_diverse_candidates(
        self,
        candidate_groups: list[list[SampledOrchestraCandidate]],
    ) -> list[SampledOrchestraCandidate]:
        if not candidate_groups or any(not group for group in candidate_groups):
            return []
        if len(candidate_groups) == 1:
            return [candidate_groups[0][0]]

        eligible_groups = [
            [candidate for candidate in group if candidate.delegate_instruction]
            for group in candidate_groups
        ]
        if any(not group for group in eligible_groups):
            return [group[0] for group in candidate_groups]

        flat_candidates = [candidate for group in eligible_groups for candidate in group]
        matrix = self._compute_instruction_similarity_matrix(
            [candidate.delegate_instruction for candidate in flat_candidates]
        )
        matrix_index = {
            candidate.candidate_id: index
            for index, candidate in enumerate(flat_candidates)
        }

        best_combo: tuple[SampledOrchestraCandidate, ...] | None = None
        best_key: tuple[float, float, tuple[str, ...]] | None = None
        # One candidate is chosen per prompt template; for two prompts this reduces
        # to the minimum cross-prompt cosine similarity pair requested by the user.
        for combo in itertools.product(*eligible_groups):
            similarities: list[float] = []
            for left_index in range(len(combo)):
                for right_index in range(left_index + 1, len(combo)):
                    left = combo[left_index]
                    right = combo[right_index]
                    similarities.append(
                        matrix[matrix_index[left.candidate_id]][matrix_index[right.candidate_id]]
                    )
            max_similarity = max(similarities) if similarities else -1.0
            mean_similarity = (
                sum(similarities) / len(similarities)
                if similarities
                else -1.0
            )
            key = (
                max_similarity,
                mean_similarity,
                tuple(candidate.candidate_id for candidate in combo),
            )
            if best_key is None or key < best_key:
                best_key = key
                best_combo = combo
        return list(best_combo or ())

    @staticmethod
    def _should_retry_llm_exception(exc: Exception) -> bool:
        message = f"{type(exc).__name__}: {exc}"
        retry_markers = (
            "APITimeoutError",
            "TimeoutError",
            "Request timed out",
            "RateLimitError",
            "rate limit",
            "APIConnectionError",
            "connection error",
            "Connection reset",
            "InternalServerError",
            "ServerTimeoutError",
            "ServiceUnavailableError",
            "502",
            "503",
            "504",
        )
        return any(marker in message for marker in retry_markers)

    async def _call_with_single_retry(self, operation, *, context: str):
        last_exc: Exception | None = None
        for attempt in range(1, 3):
            try:
                return await operation(), None, attempt
            except Exception as exc:
                last_exc = exc
                if attempt >= 2 or not self._should_retry_llm_exception(exc):
                    return None, exc, attempt
                print(f"[retry] {context} failed on attempt {attempt}, retrying once: {type(exc).__name__}: {exc}")
                await asyncio.sleep(0)
        return None, last_exc, 2

    @staticmethod
    def _format_retry_failure(context: str, exc: Exception, attempts: int) -> str:
        return f"{context} failed after {attempts} attempt(s): {type(exc).__name__}: {exc}"

    @staticmethod
    def _failed_delegate_result(error_message: str) -> DelegateResult:
        return DelegateResult(
            raw_answer_text="",
            answer=None,
            confidence=None,
            reasoning_summary="Delegate call failed.",
            thinking="",
            parse_ok=False,
            error=error_message,
            cost=0.0,
            input_tokens=0,
            output_tokens=0,
        )

    async def _sample_orchestra_candidate(
        self,
        *,
        sample: ReasoningSample,
        path: list[SearchNode],
        memory: MainMemory,
        round_index: int,
        candidate_id: str,
        instruction: str,
        model_pool: list[str],
        force_submit: bool,
        raw_call_counter: dict[str, int],
    ) -> SampledOrchestraCandidate:
        main_agent = MainAgent(
            llm=self.orchestra_client,
            main_model=self.config.orchestra_model,
            sub_models=model_pool,
            use_images=self.config.main_use_images,
            max_steps=self.config.node_max_steps,
        )
        node_sample = self._build_node_sample(
            sample=sample,
            path=path,
            instruction=instruction,
            model_pool=model_pool,
            round_index=round_index,
            node_id=candidate_id,
        )
        step_index = len(memory.attempts) + 1
        effective_force_submit = force_submit or (
            step_index == self.config.node_max_steps and bool(memory.attempts)
        )

        async def run_main_step():
            return await main_agent.step(
                sample=node_sample,
                memory=memory,
                step_index=step_index,
                force_submit=effective_force_submit,
            )

        action, main_exc, main_attempts = await self._call_with_single_retry(
            run_main_step,
            context=f"sample_orchestra_candidate:{candidate_id}",
        )
        if main_exc is not None or action is None:
            return SampledOrchestraCandidate(
                candidate_id=candidate_id,
                prompt_index=0,
                sample_index=0,
                node_instruction=instruction,
                model_pool=model_pool,
                error=self._format_retry_failure(
                    f"main_agent.step[{candidate_id}]",
                    main_exc or RuntimeError("unknown main_agent.step failure"),
                    main_attempts,
                ),
            )

        main_call_id = self._next_call_id(sample.task_id, raw_call_counter)
        raw_call = self._build_raw_call(
            call_id=main_call_id,
            task_id=sample.task_id,
            node_id=candidate_id,
            round_index=round_index,
            depth=step_index,
            actor="main",
            model=self.config.orchestra_model,
            system_prompt=action.system_prompt,
            user_prompt=action.user_prompt,
            raw_text=action.raw_response,
            thinking=action.thinking,
            parsed=action.parsed_payload,
            input_tokens=int(action.input_tokens),
            output_tokens=int(action.output_tokens),
            cost=action.cost,
        )
        return SampledOrchestraCandidate(
            candidate_id=candidate_id,
            prompt_index=0,
            sample_index=0,
            node_instruction=instruction,
            model_pool=model_pool,
            action=action,
            raw_call_id=main_call_id,
            raw_call=raw_call,
            cost=action.cost,
            input_tokens=int(action.input_tokens),
            output_tokens=int(action.output_tokens),
        )

    async def _run_submit_only_main_call(
        self,
        *,
        sample: ReasoningSample,
        memory: MainMemory,
        round_index: int,
        node_id: str,
        model_pool: list[str],
        raw_call_counter: dict[str, int],
    ) -> tuple[MainAction, dict[str, Any] | None, str | None]:
        main_agent = MainAgent(
            llm=self.orchestra_client,
            main_model=self.config.orchestra_model,
            sub_models=model_pool,
            use_images=self.config.main_use_images,
            max_steps=self.config.node_max_steps,
        )
        node_sample = self._build_node_sample(
            sample=sample,
            path=[],
            instruction="submit_only",
            model_pool=model_pool,
            round_index=round_index,
            node_id=node_id,
        )
        step_index = len(memory.attempts) + 1
        async def run_submit_only_step():
            return await main_agent.step(
                sample=node_sample,
                memory=memory,
                step_index=step_index,
                force_submit=True,
            )

        action, submit_exc, submit_attempts = await self._call_with_single_retry(
            run_submit_only_step,
            context=f"submit_only_main_call:{node_id}",
        )
        if submit_exc is not None or action is None:
            return (
                MainAction(
                    action="submit",
                    reasoning="Submit-only recovery call failed.",
                    submit_reason=self._format_retry_failure(
                        f"submit_only_main_call[{node_id}]",
                        submit_exc or RuntimeError("unknown submit-only failure"),
                        submit_attempts,
                    ),
                    final_answer=None,
                    final_boxed_letter=None,
                    cost=0.0,
                    input_tokens=0,
                    output_tokens=0,
                ),
                None,
                None,
            )

        call_id = self._next_call_id(sample.task_id, raw_call_counter)
        raw_call = self._build_raw_call(
            call_id=call_id,
            task_id=sample.task_id,
            node_id=node_id,
            round_index=round_index,
            depth=step_index,
            actor="main",
            model=self.config.orchestra_model,
            system_prompt=action.system_prompt,
            user_prompt=action.user_prompt,
            raw_text=action.raw_response,
            thinking=action.thinking,
            parsed=action.parsed_payload,
            input_tokens=int(action.input_tokens),
            output_tokens=int(action.output_tokens),
            cost=action.cost,
        )
        return action, raw_call, call_id

    async def _finalize_sampled_candidate(
        self,
        *,
        sample: ReasoningSample,
        memory: MainMemory,
        round_index: int,
        node_id: str,
        candidate: SampledOrchestraCandidate,
        budget_spent: float,
        raw_call_counter: dict[str, int],
    ) -> dict[str, Any]:
        raw_calls: list[dict[str, Any]] = []
        raw_call_ids: list[str] = []
        if candidate.raw_call is not None:
            main_raw_call = dict(candidate.raw_call)
            main_raw_call["node_id"] = node_id
            raw_calls.append(main_raw_call)
        if candidate.raw_call_id is not None:
            raw_call_ids.append(candidate.raw_call_id)

        if candidate.action is None:
            return {
                "action": "error",
                "status": "failed",
                "is_terminal": True,
                "orchestra_reasoning": "",
                "focus_question": None,
                "delegate_answer": None,
                "delegate_evidence": "",
                "delegate_confidence": None,
                "delegate_parse_ok": None,
                "submit_reason": None,
                "final_answer_text": "",
                "boxed_letter": None,
                "cost": candidate.cost,
                "input_tokens": candidate.input_tokens,
                "output_tokens": candidate.output_tokens,
                "error": candidate.error or "candidate_sampling_failed",
                "chosen_model": None,
                "raw_call_ids": raw_call_ids,
                "raw_calls": raw_calls,
            }

        action = candidate.action
        total_cost = candidate.cost
        total_input_tokens = candidate.input_tokens
        total_output_tokens = candidate.output_tokens

        if action.action == "submit":
            submit_reason = action.submit_reason or action.reasoning
            submit_result = self.submit_tool(action, memory.attempts, reason=submit_reason)
            return {
                "action": "submit",
                "status": "completed",
                "is_terminal": True,
                "orchestra_reasoning": action.reasoning,
                "focus_question": None,
                "delegate_answer": None,
                "delegate_evidence": "",
                "delegate_confidence": None,
                "delegate_parse_ok": None,
                "submit_reason": submit_reason,
                "final_answer_text": submit_result.final_answer_text,
                "boxed_letter": submit_result.final_boxed_letter,
                "cost": total_cost,
                "input_tokens": total_input_tokens,
                "output_tokens": total_output_tokens,
                "error": None,
                "chosen_model": None,
                "raw_call_ids": raw_call_ids,
                "raw_calls": raw_calls,
            }

        if budget_spent >= self.config.tree_budget_usd:
            submit_action, submit_raw_call, submit_call_id = await self._run_submit_only_main_call(
                sample=sample,
                memory=memory,
                round_index=round_index,
                node_id=node_id,
                model_pool=candidate.model_pool,
                raw_call_counter=raw_call_counter,
            )
            total_cost += submit_action.cost
            total_input_tokens += int(submit_action.input_tokens)
            total_output_tokens += int(submit_action.output_tokens)
            if submit_raw_call is not None:
                raw_calls.append(submit_raw_call)
            if submit_call_id is not None:
                raw_call_ids.append(submit_call_id)
            submit_reason = submit_action.submit_reason or submit_action.reasoning or "Budget-triggered submit."
            submit_result = self.submit_tool(submit_action, memory.attempts, reason=submit_reason)
            return {
                "action": "submit",
                "status": "completed",
                "is_terminal": True,
                "orchestra_reasoning": submit_action.reasoning,
                "focus_question": None,
                "delegate_answer": None,
                "delegate_evidence": "",
                "delegate_confidence": None,
                "delegate_parse_ok": None,
                "submit_reason": submit_reason,
                "final_answer_text": submit_result.final_answer_text,
                "boxed_letter": submit_result.final_boxed_letter,
                "cost": total_cost,
                "input_tokens": total_input_tokens,
                "output_tokens": total_output_tokens,
                "error": None,
                "chosen_model": None,
                "raw_call_ids": raw_call_ids,
                "raw_calls": raw_calls,
            }

        chosen_model = action.model or candidate.model_pool[0]
        api_model = chosen_model
        delegate_request = DelegateRequest(
            task_id=sample.task_id,
            question=sample.question,
            options=sample.options,
            images=sample.images if self.config.main_use_images else [],
            model=api_model,
            instruction=action.instruction or candidate.node_instruction,
            prior_attempts=list(memory.attempts),
        )
        sub_agent = SubAgentReasoning(llm=self._delegate_client_for_model(api_model))

        async def run_delegate_call():
            return await sub_agent.run(delegate_request)

        delegate_result, delegate_exc, delegate_attempts = await self._call_with_single_retry(
            run_delegate_call,
            context=f"delegate_call:{node_id}",
        )
        delegate_failed = delegate_exc is not None or delegate_result is None
        if delegate_failed:
            failure_exc = delegate_exc or RuntimeError("unknown delegate failure")
            delegate_result = self._failed_delegate_result(
                self._format_retry_failure(
                    f"delegate_call[{node_id}]",
                    failure_exc,
                    delegate_attempts,
                )
            )
            if self._is_model_not_available(str(failure_exc)) or self._is_model_image_incompatible(str(failure_exc)):
                self._disabled_delegate_models.add(chosen_model)

        total_cost += delegate_result.cost
        total_input_tokens += int(delegate_result.input_tokens)
        total_output_tokens += int(delegate_result.output_tokens)
        delegate_call_id = self._next_call_id(sample.task_id, raw_call_counter)
        raw_call_ids.append(delegate_call_id)
        raw_calls.append(
            self._build_raw_call(
                call_id=delegate_call_id,
                task_id=sample.task_id,
                node_id=node_id,
                round_index=round_index,
                depth=len(memory.attempts) + 1,
                actor="delegate",
                model=chosen_model,
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

        if delegate_failed:
            return {
                "action": "error",
                "status": "failed",
                "is_terminal": True,
                "orchestra_reasoning": action.reasoning,
                "focus_question": delegate_request.instruction,
                "delegate_answer": delegate_result.answer,
                "delegate_evidence": delegate_result.reasoning_summary,
                "delegate_confidence": delegate_result.confidence,
                "delegate_parse_ok": delegate_result.parse_ok,
                "submit_reason": None,
                "final_answer_text": "",
                "boxed_letter": None,
                "cost": total_cost,
                "input_tokens": total_input_tokens,
                "output_tokens": total_output_tokens,
                "error": delegate_result.error,
                "chosen_model": chosen_model,
                "raw_call_ids": raw_call_ids,
                "raw_calls": raw_calls,
            }

        return {
            "action": "delegate",
            "status": "completed" if delegate_result.parse_ok else "error",
            "is_terminal": False,
            "orchestra_reasoning": action.reasoning,
            "focus_question": delegate_request.instruction,
            "delegate_answer": delegate_result.answer,
            "delegate_evidence": delegate_result.reasoning_summary,
            "delegate_confidence": delegate_result.confidence,
            "delegate_parse_ok": delegate_result.parse_ok,
            "submit_reason": None,
            "final_answer_text": "",
            "boxed_letter": None,
            "cost": total_cost,
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
            "error": delegate_result.error,
            "chosen_model": chosen_model,
            "raw_call_ids": raw_call_ids,
            "raw_calls": raw_calls,
        }

    async def _expand_parent_with_diverse_sampling(
        self,
        *,
        sample: ReasoningSample,
        parent: SearchNode,
        path: list[SearchNode],
        round_index: int,
        budget_snapshot: float,
        first_node_id: int,
        child_model_pools: list[list[str]],
        force_submit: bool,
        raw_call_counter: dict[str, int],
    ) -> tuple[list[SearchNode], list[dict[str, Any]]]:
        memory = self._memory_from_path(path)

        # Build all oracle sampling tasks upfront
        sampling_tasks = []
        task_coords: list[tuple[int, int, list[str]]] = []
        for prompt_index, model_pool in enumerate(child_model_pools):
            prompt_instruction = build_node_instruction(
                is_last_step=(parent.depth + 1 >= self.config.node_max_steps),
                is_budget_exhausted=(budget_snapshot >= self.config.tree_budget_usd),
            )
            for sample_index in range(self.config.orchestra_samples_per_prompt):
                sampling_tasks.append(self._sample_orchestra_candidate(
                    sample=sample,
                    path=path,
                    memory=memory,
                    round_index=round_index,
                    candidate_id=f"{parent.node_id}_r{round_index}_p{prompt_index}_s{sample_index}",
                    instruction=prompt_instruction,
                    model_pool=model_pool,
                    force_submit=force_submit,
                    raw_call_counter=raw_call_counter,
                ))
                task_coords.append((prompt_index, sample_index, model_pool))

        # All oracle calls in parallel
        all_sampled: list[SampledOrchestraCandidate] = list(await asyncio.gather(*sampling_tasks))

        # Organise into per-prompt groups
        groups: dict[int, list[SampledOrchestraCandidate]] = {}
        for (prompt_index, sample_index, _), candidate in zip(task_coords, all_sampled):
            candidate.prompt_index = prompt_index
            candidate.sample_index = sample_index
            groups.setdefault(prompt_index, []).append(candidate)
        candidate_groups = [groups[i] for i in sorted(groups)]

        selected_candidates = self._select_diverse_candidates(candidate_groups)
        selected_ids = {c.candidate_id for c in selected_candidates}
        discarded_raw_calls = [
            c.raw_call
            for group in candidate_groups
            for c in group
            if c.raw_call is not None and c.candidate_id not in selected_ids
        ]

        sorted_selected = sorted(selected_candidates, key=lambda c: (c.prompt_index, c.sample_index))

        # Finalize selected candidates (delegate calls) in parallel
        async def finalize_one(idx: int, candidate: SampledOrchestraCandidate) -> tuple[str, dict[str, Any], SampledOrchestraCandidate]:
            node_id = f"node_{first_node_id + idx:04d}"
            result = await self._finalize_sampled_candidate(
                sample=sample,
                memory=memory,
                round_index=round_index,
                node_id=node_id,
                candidate=candidate,
                budget_spent=budget_snapshot,
                raw_call_counter=raw_call_counter,
            )
            return node_id, result, candidate

        finalized = list(await asyncio.gather(*[
            finalize_one(i, c) for i, c in enumerate(sorted_selected)
        ]))

        children: list[SearchNode] = []
        for node_id, node_result, candidate in finalized:
            child = SearchNode(
                node_id=node_id,
                depth=parent.depth + 1,
                round_index=round_index,
                parent_id=parent.node_id,
                instruction=candidate.node_instruction,
                model_pool=candidate.model_pool,
                action=node_result["action"],
                main_model=self.config.orchestra_model,
                chosen_model=node_result["chosen_model"],
                orchestra_reasoning=node_result["orchestra_reasoning"],
                focus_question=node_result["focus_question"],
                delegate_answer=node_result["delegate_answer"],
                delegate_evidence=node_result["delegate_evidence"],
                delegate_confidence=node_result["delegate_confidence"],
                delegate_parse_ok=node_result["delegate_parse_ok"],
                submit_reason=node_result["submit_reason"],
                final_answer_text=node_result["final_answer_text"],
                boxed_letter=node_result["boxed_letter"],
                cost=node_result["cost"],
                input_tokens=node_result["input_tokens"],
                output_tokens=node_result["output_tokens"],
                raw_call_ids=node_result["raw_call_ids"],
                is_terminal=node_result["is_terminal"],
                status=node_result["status"],
                is_correct=bool(node_result["final_answer_text"])
                and compute_mca(sample, node_result["final_answer_text"]) > 0.5,
                error=node_result["error"],
                raw_calls=node_result["raw_calls"],
            )
            children.append(child)

        return children, discarded_raw_calls

    async def _expand_child(
        self,
        *,
        sample: ReasoningSample,
        parent: SearchNode,
        path: list[SearchNode],
        round_index: int,
        node_counter: int,
        budget_spent: float,
        model_pool: list[str],
        force_submit: bool,
        rng: random.Random,
        raw_call_counter: dict[str, int],
    ) -> SearchNode:
        node_id = f"node_{node_counter:04d}"
        chosen_model = model_pool[0]
        instruction = build_node_instruction(
            is_last_step=(parent.depth + 1 >= self.config.node_max_steps),
            is_budget_exhausted=(budget_spent >= self.config.tree_budget_usd),
        )

        node_result = await self._run_orchestra_node(
            sample=sample,
            path=path,
            round_index=round_index,
            node_id=node_id,
            instruction=instruction,
            model_pool=model_pool,
            budget_spent=budget_spent,
            force_submit=force_submit,
            raw_call_counter=raw_call_counter,
        )

        final_answer_text = node_result["final_answer_text"]
        is_correct = bool(final_answer_text) and compute_mca(sample, final_answer_text) > 0.5
        return SearchNode(
            node_id=node_id,
            depth=parent.depth + 1,
            round_index=round_index,
            parent_id=parent.node_id,
            instruction=instruction,
            model_pool=model_pool,
            action=node_result["action"],
            main_model=self.config.orchestra_model,
            chosen_model=node_result["chosen_model"],
            orchestra_reasoning=node_result["orchestra_reasoning"],
            focus_question=node_result["focus_question"],
            delegate_answer=node_result["delegate_answer"],
            delegate_evidence=node_result["delegate_evidence"],
            delegate_confidence=node_result["delegate_confidence"],
            delegate_parse_ok=node_result["delegate_parse_ok"],
            submit_reason=node_result["submit_reason"],
            final_answer_text=final_answer_text,
            boxed_letter=node_result["boxed_letter"],
            cost=node_result["cost"],
            input_tokens=node_result["input_tokens"],
            output_tokens=node_result["output_tokens"],
            raw_call_ids=node_result["raw_call_ids"],
            is_terminal=node_result["is_terminal"],
            status=node_result["status"],
            is_correct=is_correct,
            error=node_result["error"],
            raw_calls=node_result["raw_calls"],
        )

    async def _run_orchestra_node(
        self,
        *,
        sample: ReasoningSample,
        path: list[SearchNode],
        round_index: int,
        node_id: str,
        instruction: str,
        model_pool: list[str],
        budget_spent: float,
        force_submit: bool = False,
        raw_call_counter: dict[str, int],
    ) -> dict[str, Any]:
        main_agent = MainAgent(
            llm=self.orchestra_client,
            main_model=self.config.orchestra_model,
            sub_models=model_pool,
            use_images=self.config.main_use_images,
            max_steps=self.config.node_max_steps,
        )
        memory = self._memory_from_path(path)
        raw_calls: list[dict[str, Any]] = []
        raw_call_ids: list[str] = []

        node_sample = self._build_node_sample(
            sample=sample,
            path=path,
            instruction=instruction,
            model_pool=model_pool,
            round_index=round_index,
            node_id=node_id,
        )

        step_index = len(memory.attempts) + 1
        force_submit = force_submit or (step_index == self.config.node_max_steps and bool(memory.attempts))

        async def run_main_step():
            return await main_agent.step(
                sample=node_sample,
                memory=memory,
                step_index=step_index,
                force_submit=force_submit,
            )

        action, main_exc, main_attempts = await self._call_with_single_retry(
            run_main_step,
            context=f"run_orchestra_node:{node_id}",
        )
        if main_exc is not None or action is None:
            failure_exc = main_exc or RuntimeError("unknown main_agent.step failure")
            return {
                "action": "error",
                "status": "failed",
                "is_terminal": True,
                "orchestra_reasoning": "",
                "focus_question": None,
                "delegate_answer": None,
                "delegate_evidence": "",
                "delegate_confidence": None,
                "delegate_parse_ok": None,
                "submit_reason": None,
                "final_answer_text": "",
                "boxed_letter": None,
                "cost": 0.0,
                "input_tokens": 0,
                "output_tokens": 0,
                "error": self._format_retry_failure(
                    f"main_agent.step[{node_id}]",
                    failure_exc,
                    main_attempts,
                ),
                "chosen_model": None,
                "raw_call_ids": [],
                "raw_calls": [],
            }

        total_cost = action.cost
        total_input_tokens = int(action.input_tokens)
        total_output_tokens = int(action.output_tokens)

        main_call_id = self._next_call_id(sample.task_id, raw_call_counter)
        raw_call_ids.append(main_call_id)
        raw_calls.append(
            self._build_raw_call(
                call_id=main_call_id,
                task_id=sample.task_id,
                node_id=node_id,
                round_index=round_index,
                depth=step_index,
                actor="main",
                model=self.config.orchestra_model,
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

        if action.action == "submit":
            submit_reason = action.submit_reason or action.reasoning
            submit_result = self.submit_tool(action, memory.attempts, reason=submit_reason)
            return {
                "action": "submit",
                "status": "completed",
                "is_terminal": True,
                "orchestra_reasoning": action.reasoning,
                "focus_question": None,
                "delegate_answer": None,
                "delegate_evidence": "",
                "delegate_confidence": None,
                "delegate_parse_ok": None,
                "submit_reason": submit_reason,
                "final_answer_text": submit_result.final_answer_text,
                "boxed_letter": submit_result.final_boxed_letter,
                "cost": total_cost,
                "input_tokens": total_input_tokens,
                "output_tokens": total_output_tokens,
                "error": None,
                "chosen_model": None,
                "raw_call_ids": raw_call_ids,
                "raw_calls": raw_calls,
            }

        if budget_spent + total_cost >= self.config.tree_budget_usd:
            submit_action, submit_raw_call, submit_call_id = await self._run_submit_only_main_call(
                sample=sample,
                memory=memory,
                round_index=round_index,
                node_id=node_id,
                model_pool=model_pool,
                raw_call_counter=raw_call_counter,
            )
            total_cost += submit_action.cost
            total_input_tokens += int(submit_action.input_tokens)
            total_output_tokens += int(submit_action.output_tokens)
            if submit_call_id is not None:
                raw_call_ids.append(submit_call_id)
            if submit_raw_call is not None:
                raw_calls.append(submit_raw_call)
            submit_reason = submit_action.submit_reason or submit_action.reasoning or "Budget-triggered submit."
            submit_result = self.submit_tool(submit_action, memory.attempts, reason=submit_reason)
            return {
                "action": "submit",
                "status": "completed",
                "is_terminal": True,
                "orchestra_reasoning": submit_action.reasoning,
                "focus_question": None,
                "delegate_answer": None,
                "delegate_evidence": "",
                "delegate_confidence": None,
                "delegate_parse_ok": None,
                "submit_reason": submit_reason,
                "final_answer_text": submit_result.final_answer_text,
                "boxed_letter": submit_result.final_boxed_letter,
                "cost": total_cost,
                "input_tokens": total_input_tokens,
                "output_tokens": total_output_tokens,
                "error": None,
                "chosen_model": None,
                "raw_call_ids": raw_call_ids,
                "raw_calls": raw_calls,
            }

        chosen_model = action.model or model_pool[0]
        api_model = chosen_model
        delegate_request = DelegateRequest(
            task_id=sample.task_id,
            question=sample.question,
            options=sample.options,
            images=sample.images if self.config.main_use_images else [],
            model=api_model,
            instruction=action.instruction or instruction,
            prior_attempts=list(memory.attempts),
        )
        sub_agent = SubAgentReasoning(llm=self._delegate_client_for_model(api_model))

        async def run_delegate_call():
            return await sub_agent.run(delegate_request)

        delegate_result, delegate_exc, delegate_attempts = await self._call_with_single_retry(
            run_delegate_call,
            context=f"delegate_call:{node_id}",
        )
        delegate_failed = delegate_exc is not None or delegate_result is None
        if delegate_failed:
            failure_exc = delegate_exc or RuntimeError("unknown delegate failure")
            delegate_result = self._failed_delegate_result(
                self._format_retry_failure(
                    f"delegate_call[{node_id}]",
                    failure_exc,
                    delegate_attempts,
                )
            )
            if self._is_model_not_available(str(failure_exc)) or self._is_model_image_incompatible(str(failure_exc)):
                self._disabled_delegate_models.add(chosen_model)

        total_cost += delegate_result.cost
        total_input_tokens += int(delegate_result.input_tokens)
        total_output_tokens += int(delegate_result.output_tokens)
        delegate_call_id = self._next_call_id(sample.task_id, raw_call_counter)
        raw_call_ids.append(delegate_call_id)
        raw_calls.append(
            self._build_raw_call(
                call_id=delegate_call_id,
                task_id=sample.task_id,
                node_id=node_id,
                round_index=round_index,
                depth=step_index,
                actor="delegate",
                model=chosen_model,
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

        if delegate_failed:
            return {
                "action": "error",
                "status": "failed",
                "is_terminal": True,
                "orchestra_reasoning": action.reasoning,
                "focus_question": delegate_request.instruction,
                "delegate_answer": delegate_result.answer,
                "delegate_evidence": delegate_result.reasoning_summary,
                "delegate_confidence": delegate_result.confidence,
                "delegate_parse_ok": delegate_result.parse_ok,
                "submit_reason": None,
                "final_answer_text": "",
                "boxed_letter": None,
                "cost": total_cost,
                "input_tokens": total_input_tokens,
                "output_tokens": total_output_tokens,
                "error": delegate_result.error,
                "chosen_model": chosen_model,
                "raw_call_ids": raw_call_ids,
                "raw_calls": raw_calls,
            }

        return {
            "action": "delegate",
            "status": "completed" if delegate_result.parse_ok else "error",
            "is_terminal": False,
            "orchestra_reasoning": action.reasoning,
            "focus_question": delegate_request.instruction,
            "delegate_answer": delegate_result.answer,
            "delegate_evidence": delegate_result.reasoning_summary,
            "delegate_confidence": delegate_result.confidence,
            "delegate_parse_ok": delegate_result.parse_ok,
            "submit_reason": None,
            "final_answer_text": "",
            "boxed_letter": None,
            "cost": total_cost,
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
            "error": delegate_result.error,
            "chosen_model": chosen_model,
            "raw_call_ids": raw_call_ids,
            "raw_calls": raw_calls,
        }

    def _build_node_sample(
        self,
        *,
        sample: ReasoningSample,
        path: list[SearchNode],
        instruction: str,
        model_pool: list[str],
        round_index: int,
        node_id: str,
    ) -> ReasoningSample:
        del path, instruction, model_pool, round_index, node_id
        question = sample.question
        return ReasoningSample(
            task_id=sample.task_id,
            question=question,
            options=sample.options,
            answer_index=sample.answer_index,
            steps=sample.steps,
            discipline=sample.discipline,
            images=sample.images,
        )

    def _available_delegate_models(self, *, requires_image_inputs: bool = False) -> list[str]:
        available = [model for model in self.config.candidate_models if model not in self._disabled_delegate_models]
        if requires_image_inputs:
            available = [model for model in available if supports_image_inputs(model)]
            if not available:
                raise RuntimeError("No enabled candidate models support image inputs for this node.")
        if not available:
            available = list(self.config.candidate_models)
        return available





    def _load_correct_models(self, scored_dir: Path) -> dict[str, list[str]]:
        candidate_set = set(self.config.candidate_models)
        result: dict[str, list[str]] = {}
        for scored_file in sorted(scored_dir.glob("*.json")):
            model_name = scored_file.stem
            if model_name not in candidate_set:
                continue
            try:
                rows = json.loads(scored_file.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(rows, list):
                continue
            for row in rows:
                if not isinstance(row, dict) or not row.get("scoring_answer"):
                    continue
                key = _normalize_question_for_pool(row.get("problem", ""))
                if not key:
                    continue
                if model_name not in result.get(key, []):
                    result.setdefault(key, []).append(model_name)
        return result

    def _delegate_client_for_model(self, model_name: str) -> Any:
        return self.delegate_client

    def _sample_model_pool(
        self,
        rng: random.Random,
        *,
        requires_image_inputs: bool = False,
        available_models: list[str] | None = None,
    ) -> list[str]:
        available = list(available_models) if available_models is not None else self._available_delegate_models(
            requires_image_inputs=requires_image_inputs
        )
        pool_size = min(len(available), max(1, self.config.node_model_pool_size))
        pool = rng.sample(available, pool_size)
        if pool:
            rng.shuffle(pool)
        return pool

    def _sample_disjoint_model_pools(
        self,
        rng: random.Random,
        *,
        child_count: int,
        available_models: list[str],
    ) -> list[list[str]]:
        """Sample sibling pools without overlap within one parent expansion."""
        if child_count <= 0:
            return []
        pool_size = min(len(available_models), max(1, self.config.node_model_pool_size))
        required = child_count * pool_size
        if required > len(available_models):
            raise RuntimeError(
                "Cannot build disjoint sibling model pools: required "
                f"{required} unique models ({child_count} siblings x pool_size {pool_size}), "
                f"but only {len(available_models)} available. "
                "Reduce branching_factor/node_model_pool_size or increase candidate_models."
            )
        shuffled = list(available_models)
        rng.shuffle(shuffled)
        pools: list[list[str]] = []
        cursor = 0
        for _ in range(child_count):
            pool = shuffled[cursor: cursor + pool_size]
            cursor += pool_size
            rng.shuffle(pool)
            pools.append(pool)
        return pools

    def _build_child_model_pools(
        self,
        rng: random.Random,
        *,
        child_count: int,
        requires_image_inputs: bool = False,
        sample: Any = None,
    ) -> list[list[str]]:
        if child_count <= 0:
            return []

        available = self._available_delegate_models(requires_image_inputs=requires_image_inputs)

        if self.config.sibling_pool_strategy == "correct_50_50":
            pools = self._build_correct_50_50_pools(rng, child_count=child_count, available=available, sample=sample)
            return pools

        if self.config.sibling_pool_strategy == "sample":
            pools = self._sample_disjoint_model_pools(
                rng,
                child_count=child_count,
                available_models=available,
            )
            return pools

        if self.config.sibling_pool_strategy != "random_partition":
            pools = [
                self._sample_model_pool(rng, available_models=available)
                for _ in range(child_count)
            ]
            return pools

        shuffled = list(available)
        rng.shuffle(shuffled)
        pools: list[list[str]] = [[] for _ in range(child_count)]
        for index, model in enumerate(shuffled):
            pools[index % child_count].append(model)

        for index, pool in enumerate(pools):
            if pool:
                rng.shuffle(pool)
                continue
            pools[index] = self._sample_model_pool(rng, available_models=available)
        return pools

    def _build_correct_50_50_pools(
        self,
        rng: random.Random,
        *,
        child_count: int,
        available: list[str],
        sample: Any,
    ) -> list[list[str]]:
        correct_models: list[str] = []
        if sample is not None and self._correct_models_by_question:
            key = _normalize_question_for_pool(sample.question)
            available_set = set(available)
            correct_models = [m for m in self._correct_models_by_question.get(key, []) if m in available_set]

        if not correct_models:
            return [self._sample_model_pool(rng, available_models=available) for _ in range(child_count)]

        if len(correct_models) == 1:
            single_pool = self._sample_model_pool(rng, available_models=correct_models)
            return [single_pool for _ in range(child_count)]

        half = max(1, child_count // 2)
        pools: list[list[str]] = []
        for i in range(child_count):
            if i < half:
                pools.append(self._sample_model_pool(rng, available_models=correct_models))
            else:
                pools.append(self._sample_model_pool(rng, available_models=available))
        return pools

    def _should_force_submit(self, final_leaf_count: int) -> bool:
        if self.config.target_leaf_trajectories is None:
            return False
        return final_leaf_count + self.config.branching_factor >= self.config.target_leaf_trajectories

    def _reconstruct_rounds(self, all_nodes: dict[str, SearchNode]) -> list[dict[str, Any]]:
        """Group nodes by depth and produce round summaries for output compatibility."""
        by_depth: dict[int, list[SearchNode]] = {}
        for node in all_nodes.values():
            if node.node_id == "root":
                continue
            by_depth.setdefault(node.depth, []).append(node)
        rounds = []
        for depth in sorted(by_depth.keys()):
            nodes = by_depth[depth]
            final_leaf_nodes = [n for n in nodes if self._is_submit_leaf(n)]
            failed_nodes = [n for n in nodes if n.is_terminal and not self._is_submit_leaf(n)]
            expandable_nodes = [n for n in nodes if not n.is_terminal]
            rounds.append({
                "round_index": depth,
                "children_created": len(nodes),
                "created_node_ids": [n.node_id for n in nodes],
                "created_final_leaf_node_ids": [n.node_id for n in final_leaf_nodes],
                "created_failed_terminal_node_ids": [n.node_id for n in failed_nodes],
                "created_expandable_node_ids": [n.node_id for n in expandable_nodes],
                "final_leaf_count_after": len(final_leaf_nodes),
                "budget_spent": sum(n.cost for n in nodes),
            })
        return rounds

    def _should_force_submit_round(self, *, final_leaf_count: int, active_parent_count: int) -> bool:
        if self.config.target_leaf_trajectories is None:
            return False
        if active_parent_count <= 0:
            return False
        return final_leaf_count + (active_parent_count * self.config.branching_factor) >= self.config.target_leaf_trajectories

    def _memory_from_path(self, path: list[SearchNode]) -> MainMemory:
        memory = MainMemory()
        for node in path[1:]:
            if node.action != "delegate":
                continue
            delegate_result = DelegateResult(
                raw_answer_text="",
                answer=node.delegate_answer,
                confidence=node.delegate_confidence,
                reasoning_summary=node.delegate_evidence,
                parse_ok=bool(node.delegate_parse_ok),
                error=node.error,
                cost=0.0,
                input_tokens=0,
                output_tokens=0,
            )
            memory.add_attempt(
                AttemptRecord(
                    attempt_index=node.depth,
                    model=node.chosen_model or "-",
                    instruction=node.focus_question or node.instruction,
                    delegate_result=delegate_result,
                    main_reasoning=node.orchestra_reasoning,
                )
            )
        return memory

    def _select_nodes_for_expansion(
        self,
        *,
        frontier: list[SearchNode],
        all_nodes: dict[str, SearchNode],
        rng: random.Random | None = None,
    ) -> tuple[list[SearchNode], int]:
        expandable = [node for node in frontier if self._is_expandable_node(node)]
        if not expandable:
            return [], 0
        if self.config.frontier_limit is not None:
            requested = min(len(expandable), self.config.frontier_limit)
        else:
            requested = len(expandable)
        ranked = self._rank_nodes(nodes=expandable, all_nodes=all_nodes)
        if rng is None or len(ranked) <= requested:
            return ranked[:requested], requested
        # Rank-based weighted sampling: top-ranked node gets weight N, second gets N-1, etc.
        n = len(ranked)
        weights = [float(n - i) for i in range(n)]
        seen: set[str] = set()
        selected: list[SearchNode] = []
        remaining = list(ranked)
        remaining_weights = list(weights)
        while len(selected) < requested and remaining:
            chosen = rng.choices(remaining, weights=remaining_weights, k=1)[0]
            idx = next(i for i, node in enumerate(remaining) if node.node_id == chosen.node_id)
            selected.append(chosen)
            seen.add(chosen.node_id)
            remaining.pop(idx)
            remaining_weights.pop(idx)
        return selected, requested

    def _build_depth_first_frontier(
        self,
        *,
        pending_frontier: list[SearchNode],
        created_expandable: list[SearchNode],
        all_nodes: dict[str, SearchNode],
    ) -> list[SearchNode]:
        carryover = [node for node in pending_frontier if self._is_expandable_node(node)]
        if not created_expandable:
            return carryover

        ranked_created = self._rank_nodes(nodes=created_expandable, all_nodes=all_nodes)
        # Treat frontier as a stack: append new children at the end so the best
        # freshly-created branch is explored immediately on the next iteration.
        return carryover + list(reversed(ranked_created))

    def _reconstruct_path(self, node: SearchNode, all_nodes: dict[str, SearchNode]) -> list[SearchNode]:
        path: list[SearchNode] = []
        current: SearchNode | None = node
        while current is not None:
            path.append(current)
            if current.parent_id is None:
                break
            current = all_nodes.get(current.parent_id)
        path.reverse()
        return path

    def _build_leaf_trajectories(
        self,
        *,
        final_leaves: list[SearchNode],
        all_nodes: dict[str, SearchNode],
    ) -> list[dict[str, Any]]:
        ranked_leaves = self._rank_nodes(nodes=final_leaves, all_nodes=all_nodes)
        trajectories: list[dict[str, Any]] = []
        for index, leaf in enumerate(ranked_leaves, start=1):
            path = self._reconstruct_path(leaf, all_nodes)
            score_details = self._node_score_details(leaf, all_nodes=all_nodes, nodes=final_leaves)
            trajectories.append(
                {
                    "trajectory_index": index,
                    "leaf_node_id": leaf.node_id,
                    "node_ids": [item.node_id for item in path],
                    "depth": leaf.depth,
                    "boxed_letter": leaf.boxed_letter,
                    "latest_delegate_confidence": score_details["latest_delegate_confidence"],
                    "selection_score": score_details["score"],
                    "trajectory_cost": score_details["trajectory_cost"],
                    "trajectory_tokens": score_details["trajectory_tokens"],
                    "final_answer_text": leaf.final_answer_text,
                    "correct": leaf.is_correct,
                    "actions": [item.action for item in path[1:]],
                }
            )
        return trajectories

    def _best_leaf(self, leaves: list[SearchNode], *, all_nodes: dict[str, SearchNode]) -> SearchNode | None:
        if not leaves:
            return None
        return self._rank_nodes(nodes=leaves, all_nodes=all_nodes)[0]

    def _rank_nodes(
        self,
        *,
        nodes: list[SearchNode],
        all_nodes: dict[str, SearchNode],
    ) -> list[SearchNode]:
        score_details = {
            node.node_id: self._node_score_details(node, all_nodes=all_nodes, nodes=nodes)
            for node in nodes
        }
        return sorted(
            nodes,
            key=lambda item: (-score_details[item.node_id]["score"], item.node_id),
        )

    def _node_score_details(
        self,
        node: SearchNode,
        *,
        all_nodes: dict[str, SearchNode],
        nodes: list[SearchNode],
    ) -> dict[str, float]:
        metrics = {
            item.node_id: self._node_metrics(item, all_nodes=all_nodes)
            for item in nodes
        }
        latest_delegate_confidence_norm = self._normalize_metric(
            {node_id: values["latest_delegate_confidence"] for node_id, values in metrics.items()}
        )
        cost_norm = self._normalize_metric(
            {node_id: values["trajectory_cost"] for node_id, values in metrics.items()}
        )
        token_norm = self._normalize_metric(
            {node_id: values["trajectory_tokens"] for node_id, values in metrics.items()}
        )
        node_id = node.node_id
        score = latest_delegate_confidence_norm[node_id] - cost_norm[node_id] - token_norm[node_id]
        return {
            "score": score,
            "latest_delegate_confidence": metrics[node_id]["latest_delegate_confidence"],
            "latest_delegate_confidence_norm": latest_delegate_confidence_norm[node_id],
            "cost_norm": cost_norm[node_id],
            "token_norm": token_norm[node_id],
            "trajectory_cost": metrics[node_id]["trajectory_cost"],
            "trajectory_tokens": metrics[node_id]["trajectory_tokens"],
        }

    def _node_metrics(
        self,
        node: SearchNode,
        *,
        all_nodes: dict[str, SearchNode],
    ) -> dict[str, float]:
        path = self._reconstruct_path(node, all_nodes)
        return {
            "latest_delegate_confidence": self._latest_delegate_confidence(path),
            "trajectory_cost": sum(item.cost for item in path),
            "trajectory_tokens": float(sum(item.input_tokens + item.output_tokens for item in path)),
        }

    @staticmethod
    def _latest_delegate_confidence(path: list[SearchNode]) -> float:
        for node in reversed(path):
            if node.action == "delegate" and node.delegate_parse_ok and node.delegate_confidence is not None:
                return float(node.delegate_confidence)
        return 0.0

    def _is_expandable_node(self, node: SearchNode) -> bool:
        return node.action in {"root", "delegate"} and not node.is_terminal and node.depth < self.config.node_max_steps

    @staticmethod
    def _is_submit_leaf(node: SearchNode) -> bool:
        return node.is_terminal and node.action == "submit" and bool(node.boxed_letter)

    @staticmethod
    def _normalize_metric(values: dict[str, float]) -> dict[str, float]:
        if not values:
            return {}
        minimum = min(values.values())
        maximum = max(values.values())
        if math.isclose(minimum, maximum):
            return {key: 0.0 for key in values}
        scale = maximum - minimum
        return {key: (value - minimum) / scale for key, value in values.items()}

    def _majority_letter(self, trajectories: list[dict[str, Any]]) -> str | None:
        votes: dict[str, int] = {}
        for item in trajectories:
            letter = item.get("boxed_letter")
            if not letter:
                continue
            votes[letter] = votes.get(letter, 0) + 1
        if not votes:
            return None
        return sorted(votes.items(), key=lambda item: (-item[1], item[0]))[0][0]

    @staticmethod
    def _raw_call_tokens(raw_calls: list[dict[str, Any]]) -> int:
        return sum(
            int(item.get("input_tokens", 0)) + int(item.get("output_tokens", 0))
            for item in raw_calls
        )

    @staticmethod
    def _extract_node_counter(node_id: str) -> int:
        prefix = "node_"
        if not isinstance(node_id, str) or not node_id.startswith(prefix):
            return 0
        suffix = node_id[len(prefix):]
        return int(suffix) if suffix.isdigit() else 0

    @staticmethod
    def _extract_call_counter(call_id: str) -> int:
        marker = "_call_"
        if not isinstance(call_id, str) or marker not in call_id:
            return 0
        suffix = call_id.rsplit(marker, maxsplit=1)[-1]
        return int(suffix) if suffix.isdigit() else 0

    @staticmethod
    def _normalize_resumed_rounds(raw_rounds: Any) -> list[dict[str, Any]]:
        if not isinstance(raw_rounds, list):
            return []
        rounds: list[dict[str, Any]] = []
        for item in raw_rounds:
            if not isinstance(item, dict):
                continue
            normalized = dict(item)
            normalized.pop("status", None)
            rounds.append(normalized)
        return rounds

    @staticmethod
    def _stop_reason_for_finished_frontier(rounds: list[dict[str, Any]]) -> str:
        if rounds:
            last_round = rounds[-1]
            if last_round.get("target_leaf_trajectories_reached"):
                return "target_leaf_trajectories_reached"
            if int(last_round.get("children_created", 0)) == 0:
                return "no_children_created"
        return "all_leaves_finished"

    def _load_resume_state(self, sample_dir: Path) -> ResumedTreeState | None:
        latest_path = sample_dir / "latest.json"
        if not latest_path.exists():
            return None
        payload = self._read_json(latest_path)
        if not isinstance(payload, dict):
            return None
        node_payloads = payload.get("nodes")
        open_frontier_node_ids = payload.get("open_frontier_node_ids")
        if not isinstance(node_payloads, list) or not isinstance(open_frontier_node_ids, list):
            return None

        all_nodes: dict[str, SearchNode] = {}
        children_by_node: dict[str, list[str]] = {}
        for item in node_payloads:
            if not isinstance(item, dict) or "node_id" not in item:
                continue
            node = SearchNode.from_json(item)
            all_nodes[node.node_id] = node
            children_by_node[node.node_id] = [str(child_id) for child_id in item.get("children_ids", [])]
        if "root" not in all_nodes:
            return None

        for node_id, child_ids in children_by_node.items():
            all_nodes[node_id].children = [
                all_nodes[child_id]
                for child_id in child_ids
                if child_id in all_nodes
            ]

        frontier = [
            all_nodes[node_id]
            for node_id in open_frontier_node_ids
            if node_id in all_nodes and self._is_expandable_node(all_nodes[node_id])
        ]
        final_leaves = [
            node
            for node in all_nodes.values()
            if node.node_id != "root" and self._is_submit_leaf(node)
        ]
        failed_terminal_nodes = [
            node
            for node in all_nodes.values()
            if node.node_id != "root" and node.is_terminal and not self._is_submit_leaf(node)
        ]

        calls_path = sample_dir / "calls.partial.jsonl"
        if not calls_path.exists():
            calls_path = sample_dir / "calls.jsonl"
        sample_raw_calls = self._read_jsonl(calls_path) if calls_path.exists() else []

        raw_call_counter_value = len(sample_raw_calls)
        if raw_call_counter_value <= 0:
            raw_call_counter_value = int(payload.get("raw_call_count", 0) or 0)
        if raw_call_counter_value <= 0:
            raw_call_counter_value = max(
                (
                    self._extract_call_counter(call_id)
                    for node in all_nodes.values()
                    for call_id in node.raw_call_ids
                ),
                default=0,
            )

        node_counter = int(payload.get("node_counter", 0) or 0)
        if node_counter <= 0:
            node_counter = max(
                (self._extract_node_counter(node.node_id) for node in all_nodes.values()),
                default=0,
            )

        total_tokens = int(payload.get("total_tokens", 0) or 0)
        if total_tokens <= 0 and sample_raw_calls:
            total_tokens = self._raw_call_tokens(sample_raw_calls)

        return ResumedTreeState(
            all_nodes=all_nodes,
            frontier=frontier,
            final_leaves=final_leaves,
            failed_terminal_nodes=failed_terminal_nodes,
            sample_raw_calls=sample_raw_calls,
            budget_spent=float(payload.get("budget_spent", 0.0)),
            total_tokens=total_tokens,
            raw_call_counter_value=raw_call_counter_value,
            node_counter=node_counter,
            rounds=self._normalize_resumed_rounds(payload.get("rounds", [])),
        )

    def _write_tree_snapshot(
        self,
        *,
        sample_dir: Path,
        task_id: str,
        all_nodes: dict[str, SearchNode],
        budget_spent: float,
        final_leaves: list[SearchNode],
        failed_terminal_nodes: list[SearchNode],
        open_frontier: list[SearchNode],
        sample_raw_calls: list[dict[str, Any]],
        total_tokens: int,
        raw_call_counter_value: int,
        node_counter: int,
    ) -> None:
        snapshot = {
            "task_id": task_id,
            "budget_spent": budget_spent,
            "total_tokens": total_tokens,
            "raw_call_count": raw_call_counter_value,
            "node_counter": node_counter,
            "final_leaf_count": len(final_leaves),
            "failed_terminal_count": len(failed_terminal_nodes),
            "open_leaf_count": len(open_frontier),
            "rounds": [],
            "open_frontier_node_ids": [node.node_id for node in open_frontier],
            "final_trajectories": self._build_leaf_trajectories(final_leaves=final_leaves, all_nodes=all_nodes),
            "nodes": [node.to_json() for node in all_nodes.values()],
        }
        self._write_json(sample_dir / "latest.json", snapshot)
        self._write_jsonl(sample_dir / "calls.partial.jsonl", sample_raw_calls)

    def _write_live_tree_snapshot(
        self,
        *,
        sample_dir: Path,
        task_id: str,
        all_nodes: dict[str, SearchNode],
        completed_rounds: list[dict[str, Any]],
        current_round: dict[str, Any],
        budget_spent: float,
        final_leaves: list[SearchNode],
        failed_terminal_nodes: list[SearchNode],
        open_frontier: list[SearchNode],
        sample_raw_calls: list[dict[str, Any]],
        total_tokens: int,
        raw_call_counter_value: int,
        node_counter: int,
    ) -> None:
        live_round = dict(current_round)
        live_round["status"] = "in_progress"
        snapshot = {
            "task_id": task_id,
            "budget_spent": budget_spent,
            "total_tokens": total_tokens,
            "raw_call_count": raw_call_counter_value,
            "node_counter": node_counter,
            "final_leaf_count": len(final_leaves),
            "failed_terminal_count": len(failed_terminal_nodes),
            "open_leaf_count": len(open_frontier),
            "rounds": [*completed_rounds, live_round],
            "open_frontier_node_ids": [node.node_id for node in open_frontier],
            "final_trajectories": self._build_leaf_trajectories(final_leaves=final_leaves, all_nodes=all_nodes),
            "nodes": [node.to_json() for node in all_nodes.values()],
            "last_updated_event": "node_expanded",
        }
        self._write_json(sample_dir / "latest.json", snapshot)
        self._write_jsonl(sample_dir / "calls.partial.jsonl", sample_raw_calls)

    def _build_summary(self, *, records: list[dict[str, Any]], run_dir: Path) -> dict[str, Any]:
        total_cost = sum(float(item["total_cost"]) for item in records)
        return {
            "total_samples": len(records),
            "success_count": sum(1 for item in records if item["success"]),
            "failure_count": sum(1 for item in records if not item["success"]),
            "success_rate": (sum(1 for item in records if item["success"]) / len(records)) if records else 0.0,
            "avg_expansion_rounds": (
                sum(int(item["expansion_rounds_ran"]) for item in records) / len(records)
            )
            if records
            else 0.0,
            "avg_final_leaf_count": (
                sum(int(item["final_leaf_count"]) for item in records) / len(records)
            )
            if records
            else 0.0,
            "avg_total_cost": (total_cost / len(records)) if records else 0.0,
            "total_cost": total_cost,
            "total_tokens": sum(int(item["total_tokens"]) for item in records),
            "total_model_calls": sum(int(item["total_model_calls"]) for item in records),
            "selected_task_ids": [item["task_id"] for item in records],
            "dataset_name": self.config.dataset_name,
            "dataset_split": self.config.dataset_split,
            "discipline": self.config.discipline,
            "sample_count": len(records),
            "sample_seed": self.config.sample_seed,
            "orchestra_model": self.config.orchestra_model,
            "output_dir": str(run_dir),
        }

    def _append_events(self, *, sample_dir: Path, global_events: Path, event: dict[str, Any]) -> None:
        return None

    def _progress_step(
        self,
        *,
        increment: int,
        task_id: str,
        round_index: int,
        leaf_count: int,
        budget_spent: float,
    ) -> None:
        if self._progress is None:
            return
        delta = budget_spent - self._last_task_cost.get(task_id, 0.0)
        self._last_task_cost[task_id] = budget_spent
        self._total_cost += delta
        self._progress.update(increment)
        self._progress.set_postfix_str(
            f"total=${self._total_cost:.3f} task={task_id} round={round_index} leaves={leaf_count}"
        )

    @staticmethod
    def _write_json(path: Path, data: Any) -> None:
        dump_json(path, data)

    @staticmethod
    def _read_json(path: Path) -> Any:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _read_jsonl(path: Path) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return rows

    @staticmethod
    def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False))
                f.write("\n")

    @staticmethod
    def _append_jsonl(path: Path, data: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False))
            f.write("\n")


    @staticmethod
    def _is_model_not_available(error: str) -> bool:
        lowered = error.lower()
        return "model_not_found" in lowered or "no available channel for model" in lowered

    @staticmethod
    def _is_model_image_incompatible(error: str) -> bool:
        lowered = error.lower()
        return "does not support image inputs" in lowered or "vision model" in lowered

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _sanitize_model_dir_name(model_name: str) -> str:
        return model_name.strip().replace("/", "__").replace("\\", "__")

    @staticmethod
    def _stable_seed(task_id: str, base_seed: int) -> int:
        text = f"{base_seed}:{task_id}"
        seed = 0
        for char in text:
            seed = (seed * 131 + ord(char)) % (2**32)
        return seed
