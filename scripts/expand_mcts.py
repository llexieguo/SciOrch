#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
from contextlib import suppress
import contextvars
import json
import os
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None


DEFAULT_SCIORCH_ROOT = Path(__file__).resolve().parents[1]
if str(DEFAULT_SCIORCH_ROOT) not in sys.path:
    sys.path.insert(0, str(DEFAULT_SCIORCH_ROOT))

from mcts.config import MCTSConfig
from mcts.runner import MCTSReasoningRunner, ResumedTreeState, SearchNode


@dataclass
class StagePlan:
    stage_name: str
    task_ids: list[str]
    fallback_random_pool_task_ids: list[str]
    reopen_nodes: dict[str, str]
    reopen_node_selection: dict[str, dict[str, Any]]
    leaf_acc_before: dict[str, float]


@dataclass
class StageSpec:
    stage_name: str
    selector: str
    reopen_node_strategy: str
    child_pool_strategy: str
    leaf_acc_threshold: float
    tree_budget_usd: float
    max_final_leaf_count: int | None
    frontier_limit: int
    random_seed_offset: int = 0
    child_pool_size: int = 1
    leaf_acc_min_exclusive: float | None = None
    leaf_acc_max_exclusive: float | None = None


class FixedPoolExpansionRunner(MCTSReasoningRunner):
    def __init__(
        self,
        config: MCTSConfig,
        *,
        pools_by_task: dict[str, list[str]],
        leaf_acc_threshold: float,
        max_final_leaf_count: int | None = None,
        progress_desc: str | None = None,
        stage_name: str = "below_threshold",
        reopen_node_strategy: str = "child_expected_acc_gap_weighted",
        child_pool_strategy: str = "single_random_from_task_pool",
        child_pool_size: int = 1,
    ) -> None:
        super().__init__(config)
        self.pools_by_task = pools_by_task
        self.leaf_acc_threshold = float(leaf_acc_threshold)
        self.max_final_leaf_count = (
            int(max_final_leaf_count)
            if max_final_leaf_count is not None
            else None
        )
        self.progress_desc = progress_desc or "Expand"
        self.stage_name = stage_name
        self.reopen_node_strategy = reopen_node_strategy
        self.child_pool_strategy = child_pool_strategy
        self.child_pool_size = max(1, int(child_pool_size))
        self._active_task_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
            "active_task_id",
            default=None,
        )
        self._running_task_ids: set[str] = set()
        self._folder_total_cost = 0.0

    def _effective_submit_leaf_cap(self) -> int | None:
        """Upper bound on submit-terminal leaves during expand.

        Stage YAML sets ``max_final_leaf_count``; the copied reasoning snapshot still carries
        ``target_leaf_trajectories``. Expand previously enforced only ``max_final_leaf_count``,
        so a larger reasoning target could make it look like expand ignored its own cap.

        Use the **minimum** of all configured caps so the tighter budget always applies.
        """
        caps: list[int] = []
        if self.max_final_leaf_count is not None:
            caps.append(int(self.max_final_leaf_count))
        traj = getattr(self.config, "target_leaf_trajectories", None)
        if traj is not None:
            caps.append(int(traj))
        if not caps:
            return None
        return min(caps)

    @staticmethod
    def _format_reopen_context(sample_dir: Path) -> str:
        latest_path = sample_dir / "latest.json"
        if not latest_path.exists():
            return "reopen_node=? stage=?"
        try:
            payload = json.loads(latest_path.read_text(encoding="utf-8"))
        except Exception:
            return "reopen_node=? stage=?"
        reopen_node_id = str(payload.get("reopen_node_id") or payload.get("reopen_from_node_id") or "?")
        reopen_stage = str(payload.get("reopen_stage") or "?")
        open_leaf_count = payload.get("open_leaf_count")
        return f"reopen_node={reopen_node_id} stage={reopen_stage} open_leaf_count={open_leaf_count}"

    @staticmethod
    def _format_running_task_ids(task_ids: set[str]) -> str:
        return "[" + ", ".join(sorted(task_ids)) + "]"

    @staticmethod
    def _log_sample_event(message: str) -> None:
        del message

    @staticmethod
    def _read_cost_from_sample_dir(sample_dir: Path) -> float:
        result_path = sample_dir / "result.json"
        if result_path.exists():
            try:
                payload = json.loads(result_path.read_text(encoding="utf-8"))
                if isinstance(payload, dict):
                    return float(payload.get("total_cost", payload.get("budget_spent", 0.0)) or 0.0)
            except Exception:
                return 0.0

        latest_path = sample_dir / "latest.json"
        if latest_path.exists():
            try:
                payload = json.loads(latest_path.read_text(encoding="utf-8"))
                if isinstance(payload, dict):
                    return float(payload.get("budget_spent", payload.get("total_cost", 0.0)) or 0.0)
            except Exception:
                return 0.0

        return 0.0

    def _collect_cost_totals(self, *, run_dir: Path, selected_task_ids: list[str]) -> tuple[float, float]:
        samples_dir = run_dir / "samples"
        if not samples_dir.is_dir():
            return 0.0, 0.0

        selected = set(selected_task_ids)
        selected_total = 0.0
        folder_total = 0.0
        for sample_dir in sorted(path for path in samples_dir.iterdir() if path.is_dir()):
            sample_cost = self._read_cost_from_sample_dir(sample_dir)
            folder_total += sample_cost
            if sample_dir.name in selected:
                selected_total += sample_cost
        return selected_total, folder_total

    async def _run_sample(self, *, run_dir: Path, sample: Any, global_events: Path) -> dict[str, Any]:
        token = self._active_task_id.set(sample.task_id)
        try:
            return await super()._run_sample(run_dir=run_dir, sample=sample, global_events=global_events)
        finally:
            self._active_task_id.reset(token)

    async def _expand_round(self, **kwargs: Any) -> tuple[dict[str, Any], float, int, int, int]:
        # Keep the expansion script compatible with the current runner, where
        # the old _expand_round entrypoint was renamed to _expand_round_unused.
        return await super()._expand_round_unused(**kwargs)

    def _build_child_model_pools(
        self,
        rng: random.Random,
        *,
        child_count: int,
        requires_image_inputs: bool = False,
        sample: Any = None,
    ) -> list[list[str]]:
        child_total = max(0, child_count)
        if child_total <= 0:
            return []

        available = self._available_delegate_models(requires_image_inputs=requires_image_inputs)

        if self.child_pool_strategy == "random_k_from_available":
            if not available:
                raise RuntimeError("No delegate models available for child expansion.")
            return [
                self._sample_random_pool(
                    rng,
                    available_models=available,
                    pool_size=self.child_pool_size,
                )
                for _ in range(child_total)
            ]

        if self.child_pool_strategy == "single_random_from_task_pool":
            task_id = self._active_task_id.get()
            task_pool = list(self.pools_by_task.get(task_id, [])) if task_id else []
            if task_pool:
                available_set = set(available)
                task_pool = [model for model in task_pool if model in available_set]
            if not task_pool:
                task_pool = available
            if not task_pool:
                raise RuntimeError("No delegate models available for child expansion.")
            return [[rng.choice(task_pool)] for _ in range(child_total)]

        raise ValueError(f"Unsupported child_pool_strategy: {self.child_pool_strategy}")

    @staticmethod
    def _sample_random_pool(
        rng: random.Random,
        *,
        available_models: list[str],
        pool_size: int,
    ) -> list[str]:
        choices = list(available_models)
        if not choices:
            return []
        effective_pool_size = min(len(choices), max(1, pool_size))
        sampled = rng.sample(choices, effective_pool_size)
        if sampled:
            rng.shuffle(sampled)
        return sampled

    @staticmethod
    def _model_supports_images(model_name: str) -> bool:
        from sciorch.llm.model_capabilities import supports_image_inputs
        return supports_image_inputs(model_name)

    def _leaf_acc(self, final_leaves: list[SearchNode]) -> float:
        if not final_leaves:
            return 0.0
        correct = sum(1 for node in final_leaves if node.is_correct)
        return correct / len(final_leaves)

    @staticmethod
    def _descendant_reward_stats_from_tree(all_nodes: dict[str, SearchNode]) -> dict[str, dict[str, Any]]:
        memo: dict[str, dict[str, Any]] = {}

        def resolve(node: SearchNode) -> dict[str, Any]:
            cached = memo.get(node.node_id)
            if cached is not None:
                return cached

            if not node.children:
                if node.is_terminal:
                    is_submit = node.action == "submit" and bool(node.boxed_letter)
                    terminal_count = 1
                    correct_count = 1 if is_submit and bool(node.is_correct) else 0
                    stats = {
                        "expected_acc": (correct_count / terminal_count),
                        "terminal_count": terminal_count,
                        "correct_count": correct_count,
                    }
                else:
                    stats = {
                        "expected_acc": 0.0,
                        "terminal_count": 0,
                        "correct_count": 0,
                    }
                memo[node.node_id] = stats
                return stats

            terminal_count = 0
            correct_count = 0
            for child in node.children:
                child_stats = resolve(child)
                terminal_count += int(child_stats["terminal_count"])
                correct_count += int(child_stats["correct_count"])
            expected_acc = (correct_count / terminal_count) if terminal_count > 0 else 0.0
            stats = {
                "expected_acc": expected_acc,
                "terminal_count": terminal_count,
                "correct_count": correct_count,
            }
            memo[node.node_id] = stats
            return stats

        for node in all_nodes.values():
            resolve(node)
        return memo

    def _choose_reopen_node_from_tree(
        self,
        *,
        all_nodes: dict[str, SearchNode],
        rng: random.Random,
    ) -> SearchNode | None:
        candidates = sorted(
            [node for node in all_nodes.values() if self._is_expandable_node(node)],
            key=lambda item: item.node_id,
        )
        if not candidates:
            return None

        if self.reopen_node_strategy == "uniform_random_expandable_node":
            weights = [1.0 / float(node.depth + 1) for node in candidates]
            if any(weight > 0.0 for weight in weights):
                return rng.choices(candidates, weights=weights, k=1)[0]
            return rng.choice(candidates)

        if self.reopen_node_strategy == "child_expected_acc_gap_weighted":
            reward_stats = self._descendant_reward_stats_from_tree(all_nodes)
            weights: list[float] = []
            for node in candidates:
                child_expected_accs = [
                    float(reward_stats.get(child.node_id, {}).get("expected_acc", 0.0))
                    for child in node.children
                ]
                variance = 0.0
                if len(child_expected_accs) >= 2:
                    mean_value = sum(child_expected_accs) / float(len(child_expected_accs))
                    variance = sum((value - mean_value) ** 2 for value in child_expected_accs) / float(len(child_expected_accs))
                depth_bias = 1.0 / float(node.depth + 1)
                weights.append(max(variance * depth_bias, 0.0))

            if any(weight > 0.0 for weight in weights):
                return rng.choices(candidates, weights=weights, k=1)[0]
            return rng.choice(candidates)

        raise ValueError(f"Unsupported reopen_node_strategy: {self.reopen_node_strategy}")

    async def run_with_leaf_acc_stop(self) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        run_dir = self._build_run_dir()
        run_dir.mkdir(parents=True, exist_ok=True)
        self._write_json(run_dir / "config.snapshot.json", self.config.to_json())

        samples = self._select_samples(self._load_dataset_samples())
        selected_task_ids = [sample.task_id for sample in samples]

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
                    {"task_id": sample.task_id, "discipline": sample.discipline}
                    for sample in samples
                ],
            },
        )

        semaphore = asyncio.Semaphore(max(1, self.config.max_concurrency))
        existing_total_cost, existing_last_task_cost, _ = self._load_existing_progress_state(
            run_dir=run_dir,
            selected_task_ids=selected_task_ids,
        )
        self._total_cost = existing_total_cost
        self._last_task_cost = existing_last_task_cost
        live_selected_cost, live_folder_cost = self._collect_cost_totals(
            run_dir=run_dir,
            selected_task_ids=selected_task_ids,
        )
        self._total_cost = live_selected_cost
        self._folder_total_cost = live_folder_cost

        progress = None
        if self.config.show_progress and tqdm is not None:
            progress = tqdm(
                total=len(samples),
                desc=self.progress_desc,
                unit="tree",
                dynamic_ncols=True,
            )
            progress.set_postfix_str(
                f"expand=${self._total_cost:.3f} folder=${self._folder_total_cost:.3f}"
            )

        stop_postfix_refresh = asyncio.Event()
        postfix_refresh_task: asyncio.Task[None] | None = None

        async def refresh_postfix() -> None:
            if progress is None:
                return
            while not stop_postfix_refresh.is_set():
                await asyncio.sleep(10)
                live_expand_cost, live_folder_cost_inner = self._collect_cost_totals(
                    run_dir=run_dir,
                    selected_task_ids=selected_task_ids,
                )
                self._total_cost = live_expand_cost
                self._folder_total_cost = live_folder_cost_inner
                progress.set_postfix_str(
                    f"expand=${self._total_cost:.3f} "
                    f"folder=${self._folder_total_cost:.3f} "
                    f"running={len(self._running_task_ids)}"
                )
                progress.refresh()

        if progress is not None:
            postfix_refresh_task = asyncio.create_task(refresh_postfix())

        async def run_limited(sample: Any) -> dict[str, Any]:
            async with semaphore:
                return await self._run_sample_with_leaf_acc_stop(
                    run_dir=run_dir,
                    sample=sample,
                    global_events=run_dir,
                )

        tasks = [asyncio.create_task(run_limited(sample)) for sample in samples]
        try:
            for future in asyncio.as_completed(tasks):
                task_result = await future
                summary = task_result.get("summary", {}) if isinstance(task_result, dict) else {}
                task_id = str(summary.get("task_id") or "")
                task_cost = float(summary.get("total_cost", 0.0) or 0.0)
                delta = task_cost - self._last_task_cost.get(task_id, 0.0)
                self._last_task_cost[task_id] = task_cost
                self._total_cost += delta
                self._folder_total_cost += delta
                if progress is not None:
                    progress.update(1)
                    progress.set_postfix_str(
                        f"expand=${self._total_cost:.3f} "
                        f"folder=${self._folder_total_cost:.3f} "
                        f"running={len(self._running_task_ids)}"
                    )
        finally:
            stop_postfix_refresh.set()
            if postfix_refresh_task is not None:
                postfix_refresh_task.cancel()
                with suppress(asyncio.CancelledError):
                    await postfix_refresh_task
            if progress is not None:
                progress.close()

        records = self._collect_sample_results(run_dir)
        all_raw_calls = self._collect_sample_raw_calls(run_dir)
        summary = self._build_summary(records=records, run_dir=run_dir)
        self._write_json(run_dir / "scored.json", records)
        self._write_json(run_dir / "summary.json", summary)
        with (run_dir / "raw_calls.jsonl").open("w", encoding="utf-8") as handle:
            for row in all_raw_calls:
                handle.write(json.dumps(row, ensure_ascii=False))
                handle.write("\n")
        return records, summary

    async def _run_sample_with_leaf_acc_stop(
        self,
        *,
        run_dir: Path,
        sample: Any,
        global_events: Path,
    ) -> dict[str, Any]:
        sample_dir = run_dir / "samples" / sample.task_id
        resume_state: ResumedTreeState | None = None
        if sample_dir.exists():
            result_path = sample_dir / "result.json"
            latest_path = sample_dir / "latest.json"
            if self.config.resume and result_path.exists():
                payload = self._read_json(result_path)
                return {"summary": payload if isinstance(payload, dict) else {"task_id": sample.task_id}, "raw_calls": []}
            if self.config.resume and latest_path.exists():
                resume_state = self._load_resume_state(sample_dir)
                if resume_state is None:
                    raise RuntimeError(f"Non-resumable latest snapshot for {sample.task_id}: {latest_path}")
        sample_dir.mkdir(parents=True, exist_ok=True)
        reopen_context = self._format_reopen_context(sample_dir)
        self._running_task_ids.add(sample.task_id)
        self._log_sample_event(
            f"[tree:start] task_id={sample.task_id} {reopen_context} "
            f"running={self._format_running_task_ids(self._running_task_ids)}"
        )

        task_token = self._active_task_id.set(sample.task_id)
        try:
            sample_start_tree = await self._run_tree_until_leaf_acc(
                sample=sample,
                sample_dir=sample_dir,
                global_events=global_events,
                rng=random.Random(self._stable_seed(sample.task_id, self.config.tree_seed)),
                resume_state=resume_state,
            )
            result = {
                "task_id": sample.task_id,
                "discipline": sample.discipline,
                "status": "completed",
                "question": sample.question,
                "options": sample.options,
                "gold_answer_letter": chr(ord("A") + sample.answer_index),
                "orchestra_model": self.config.orchestra_model,
                "success": sample_start_tree["success"],
                "any_correct_leaf": sample_start_tree["any_correct_leaf"],
                "best_leaf_correct": sample_start_tree["best_leaf_correct"],
                "majority_correct": sample_start_tree["majority_correct"],
                "correct_leaf_count": sample_start_tree["correct_leaf_count"],
                "final_leaf_count": sample_start_tree["final_leaf_count"],
                "open_leaf_count": sample_start_tree["open_leaf_count"],
                "target_leaf_trajectories": self.config.target_leaf_trajectories,
                "branching_factor": self.config.branching_factor,
                "leaf_expand_ratio": self.config.leaf_expand_ratio,
                "frontier_limit": self.config.frontier_limit,
                "sibling_pool_strategy": self.config.sibling_pool_strategy,
                "path_max_steps": self.config.node_max_steps,
                "budget_limit": self.config.tree_budget_usd,
                "budget_spent": sample_start_tree["budget_spent"],
                "budget_exhausted": sample_start_tree["budget_exhausted"],
                "stop_reason": sample_start_tree["stop_reason"],
                "expansion_rounds_ran": len(sample_start_tree["rounds"]),
                "best_leaf_node_id": sample_start_tree["best_leaf_node_id"],
                "best_leaf_boxed_letter": sample_start_tree["best_leaf_boxed_letter"],
                "best_leaf_latest_delegate_confidence": sample_start_tree["best_leaf_latest_delegate_confidence"],
                "majority_boxed_letter": sample_start_tree["majority_boxed_letter"],
                "latency_seconds": 0.0,
                "total_cost": sample_start_tree["cost"],
                "total_tokens": sample_start_tree["total_tokens"],
                "total_model_calls": sample_start_tree["model_calls"],
                "failed_terminal_count": sample_start_tree["failed_terminal_count"],
                "leaf_acc": self._leaf_acc_from_counts(
                    sample_start_tree["correct_leaf_count"],
                    sample_start_tree["final_leaf_count"],
                ),
            }
            self._write_json(sample_dir / "result.json", result)
            self._write_jsonl(sample_dir / "calls.jsonl", sample_start_tree["raw_calls"])
            self._write_jsonl(sample_dir / "nodes.jsonl", sample_start_tree["nodes"])
            self._write_json(
                sample_dir / "view.json",
                {
                    "task_id": sample.task_id,
                    "status": "completed",
                    "rounds": sample_start_tree["rounds"],
                    "trajectories": sample_start_tree["trajectories"],
                    "final_leaf_node_ids": sample_start_tree["final_leaf_node_ids"],
                    "open_frontier_node_ids": sample_start_tree["open_frontier_node_ids"],
                    "failed_terminal_node_ids": sample_start_tree["failed_terminal_node_ids"],
                    "final_summary": {
                        "stop_reason": sample_start_tree["stop_reason"],
                        "leaf_acc": result["leaf_acc"],
                        "success": result["success"],
                    },
                    "metrics": {
                        "budget_spent": result["budget_spent"],
                        "final_leaf_count": result["final_leaf_count"],
                        "correct_leaf_count": result["correct_leaf_count"],
                        "leaf_acc": result["leaf_acc"],
                    },
                },
            )
            self._running_task_ids.discard(sample.task_id)
            self._log_sample_event(
                "[tree:done] task_id={task_id} stop_reason={stop_reason} leaf_acc={leaf_acc:.4f} "
                "correct_leafs={correct_leaf_count}/{final_leaf_count} open_leafs={open_leaf_count} "
                "running={running}".format(
                    task_id=sample.task_id,
                    stop_reason=sample_start_tree.get("stop_reason", "?"),
                    leaf_acc=float(result["leaf_acc"]),
                    correct_leaf_count=int(result["correct_leaf_count"]),
                    final_leaf_count=int(result["final_leaf_count"]),
                    open_leaf_count=int(result["open_leaf_count"]),
                    running=self._format_running_task_ids(self._running_task_ids),
                )
            )
            return {"summary": result, "raw_calls": sample_start_tree["raw_calls"]}
        except Exception as exc:
            self._running_task_ids.discard(sample.task_id)
            self._log_sample_event(
                f"[tree:error] task_id={sample.task_id} error={exc} "
                f"running={self._format_running_task_ids(self._running_task_ids)}"
            )
            raise
        finally:
            self._active_task_id.reset(task_token)

    @staticmethod
    def _leaf_acc_from_counts(correct_leaf_count: int, final_leaf_count: int) -> float:
        if final_leaf_count <= 0:
            return 0.0
        return correct_leaf_count / final_leaf_count

    async def _run_tree_until_leaf_acc(
        self,
        *,
        sample: Any,
        sample_dir: Path,
        global_events: Path,
        rng: random.Random,
        resume_state: ResumedTreeState | None = None,
    ) -> dict[str, Any]:
        if resume_state is None:
            raise RuntimeError("This script only supports resume-from-copied-tree mode.")

        all_nodes = resume_state.all_nodes
        frontier = resume_state.frontier
        final_leaves = resume_state.final_leaves
        failed_terminal_nodes = resume_state.failed_terminal_nodes
        sample_raw_calls = resume_state.sample_raw_calls
        raw_call_counter = {"value": resume_state.raw_call_counter_value}
        budget_spent = resume_state.budget_spent
        total_tokens_spent = resume_state.total_tokens
        model_calls = max(len(sample_raw_calls), raw_call_counter["value"])
        node_counter = resume_state.node_counter
        rounds = resume_state.rounds

        stop_reason: str | None = None
        round_index = len(rounds)
        reopen_count = 0
        while stop_reason is None:
            if budget_spent >= self.config.tree_budget_usd:
                stop_reason = "budget_exhausted"
                break
            if self._leaf_acc(final_leaves) >= self.leaf_acc_threshold:
                stop_reason = "leaf_acc_threshold_reached"
                break
            _cap = self._effective_submit_leaf_cap()
            if _cap is not None and (len(final_leaves) + len(failed_terminal_nodes)) >= _cap:
                stop_reason = "max_final_leaf_count_reached"
                break
            if not frontier:
                reopen_node = self._choose_reopen_node_from_tree(all_nodes=all_nodes, rng=rng)
                if reopen_node is None:
                    stop_reason = (
                        "budget_exhausted_no_reopen_candidates"
                        if budget_spent >= self.config.tree_budget_usd
                        else "no_reopen_candidates"
                    )
                    break
                frontier = [reopen_node]
                reopen_count += 1

            round_index += 1
            selected_nodes, selected_count_requested = self._select_nodes_for_expansion(
                frontier=frontier,
                all_nodes=all_nodes,
                rng=rng,
            )
            if not selected_nodes:
                stop_reason = "no_frontier_selected"
                break
            selected_node_ids = {node.node_id for node in selected_nodes}
            pending_frontier = [node for node in frontier if node.node_id not in selected_node_ids]

            (
                round_summary,
                budget_spent,
                total_tokens_spent,
                model_calls,
                node_counter,
            ) = await self._expand_round(
                sample=sample,
                sample_dir=sample_dir,
                global_events=global_events,
                all_nodes=all_nodes,
                completed_rounds=rounds,
                selected_parents=selected_nodes,
                pending_frontier=pending_frontier,
                active_frontier_count=len(selected_nodes),
                round_index=round_index,
                selection_strategy="frontier_rank_expand",
                budget_spent=budget_spent,
                model_calls=model_calls,
                node_counter=node_counter,
                rng=rng,
                selected_count_requested=selected_count_requested,
                final_leaves=final_leaves,
                failed_terminal_nodes=failed_terminal_nodes,
                sample_raw_calls=sample_raw_calls,
                raw_call_counter=raw_call_counter,
                total_tokens_spent=total_tokens_spent,
            )
            rounds.append(round_summary)
            frontier = [all_nodes[node_id] for node_id in round_summary["next_frontier_node_ids"]]
            self._write_tree_snapshot(
                sample_dir=sample_dir,
                task_id=sample.task_id,
                all_nodes=all_nodes,
                budget_spent=budget_spent,
                final_leaves=final_leaves,
                failed_terminal_nodes=failed_terminal_nodes,
                open_frontier=frontier,
                sample_raw_calls=sample_raw_calls,
                total_tokens=total_tokens_spent,
                raw_call_counter_value=raw_call_counter["value"],
                node_counter=node_counter,
            )
            if self._leaf_acc(final_leaves) >= self.leaf_acc_threshold:
                stop_reason = "leaf_acc_threshold_reached"
                break
            _cap = self._effective_submit_leaf_cap()
            if _cap is not None and (len(final_leaves) + len(failed_terminal_nodes)) >= _cap:
                stop_reason = "max_final_leaf_count_reached"
                break
            if budget_spent >= self.config.tree_budget_usd:
                stop_reason = "budget_exhausted"
                break

        trajectories = self._build_leaf_trajectories(final_leaves=final_leaves, all_nodes=all_nodes)
        best_leaf = self._best_leaf(final_leaves, all_nodes=all_nodes)
        majority_boxed_letter = self._majority_letter(trajectories)
        gold_answer_letter = chr(ord("A") + sample.answer_index)
        any_correct_leaf = any(node.is_correct for node in final_leaves)
        majority_correct = majority_boxed_letter == gold_answer_letter if majority_boxed_letter else False
        return {
            "task_id": sample.task_id,
            "budget_limit": self.config.tree_budget_usd,
            "budget_spent": budget_spent,
            "budget_exhausted": budget_spent >= self.config.tree_budget_usd,
            "stop_reason": stop_reason or "completed",
            "target_leaf_trajectories": self.config.target_leaf_trajectories,
            "branching_factor": self.config.branching_factor,
            "leaf_expand_ratio": self.config.leaf_expand_ratio,
            "frontier_limit": self.config.frontier_limit,
            "sibling_pool_strategy": self.config.sibling_pool_strategy,
            "path_max_steps": self.config.node_max_steps,
            "final_leaf_count": len(final_leaves),
            "open_leaf_count": len(frontier),
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
            "reopen_count": reopen_count,
            "rounds": rounds,
            "trajectories": trajectories,
            "open_frontier_node_ids": [node.node_id for node in frontier],
            "nodes": [node.to_json() for node in all_nodes.values()],
            "raw_calls": sample_raw_calls,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Continue an existing MCTS run in place with fixed per-task model pools."
    )
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument(
        "--start-stage",
        default=None,
        help="Optional stage name override. Defaults to start_stage in config or the first configured stage.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Only rewrite selected sample snapshots to prepare reopen frontier, do not make model calls.",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return payload


def _load_env_file(env_path: Path) -> None:
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if len(value) >= 2 and (
            (value[0] == '"' and value[-1] == '"')
            or (value[0] == "'" and value[-1] == "'")
        ):
            value = value[1:-1]
        os.environ.setdefault(key, value)


def _load_env_candidates(config_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    explicit_repo_env = project_root / ".env"
    candidates = [
        explicit_repo_env,
        project_root / ".env",
        Path.cwd() / ".env",
        config_path.parent / ".env",
        config_path.parent.parent / ".env",
    ]
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        _load_env_file(resolved)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def normalize_model_pools(summary_path: Path) -> dict[str, list[str]]:
    payload = load_json(summary_path)
    tasks = payload.get("tasks", [])
    pools: dict[str, list[str]] = {}
    for item in tasks:
        if not isinstance(item, dict):
            continue
        task_id = str(item.get("task_id") or "").strip()
        models = [str(model) for model in item.get("correct_models", []) if str(model).strip()]
        if task_id:
            pools[task_id] = models
    return pools


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _normalize_stage_selector(selector: str) -> str:
    selector_name = str(selector or "").strip()
    if selector_name == "below_threshold":
        return "leaf_acc_band"
    return selector_name


def _default_reopen_node_strategy(selector: str) -> str:
    normalized = _normalize_stage_selector(selector)
    if normalized == "all_wrong":
        return "uniform_random_expandable_node"
    if normalized == "leaf_acc_band":
        return "child_expected_acc_gap_weighted"
    raise ValueError(f"Unsupported stage selector: {selector}")


def _default_child_pool_strategy(selector: str) -> str:
    normalized = _normalize_stage_selector(selector)
    if normalized == "all_wrong":
        return "random_k_from_available"
    if normalized == "leaf_acc_band":
        return "single_random_from_task_pool"
    raise ValueError(f"Unsupported stage selector: {selector}")


def stage_requires_task_pool(stage_spec: StageSpec) -> bool:
    return "task_pool" in str(stage_spec.child_pool_strategy)


def ensure_pool_summary_exists(
    *,
    config_path: Path,
    pool_summary_path: Path,
    active_stage_specs: list[StageSpec],
) -> None:
    needs_task_pool = any(stage_requires_task_pool(stage_spec) for stage_spec in active_stage_specs)
    if not needs_task_pool or pool_summary_path.exists():
        return

    from build_model_pools import build_correct_model_pools_from_config

    print(
        f"[auto-build-pools] missing pool summary for task-pool stages, building in-process from: {config_path}",
        flush=True,
    )
    build_correct_model_pools_from_config(config_path, verbose=True)
    if not pool_summary_path.exists():
        raise FileNotFoundError(
            f"Pool summary was not created by auto-build step: {pool_summary_path}"
        )


def load_stage_specs(config: dict[str, Any]) -> list[StageSpec]:
    default_leaf_acc_threshold = float(config.get("leaf_acc_threshold", 0.3))
    default_tree_budget_usd = float(config.get("tree_budget_usd", 5.0))
    default_max_final_leaf_count = _optional_int(config.get("max_final_leaf_count", 128))
    default_frontier_limit = int(config.get("frontier_limit", 1))

    stages_payload = config.get("stages")
    if not stages_payload:
        return [
            StageSpec(
                stage_name="all_wrong",
                selector="all_wrong",
                reopen_node_strategy="uniform_random_expandable_node",
                child_pool_strategy="random_k_from_available",
                child_pool_size=3,
                leaf_acc_threshold=default_leaf_acc_threshold,
                tree_budget_usd=default_tree_budget_usd,
                max_final_leaf_count=default_max_final_leaf_count,
                frontier_limit=default_frontier_limit,
                random_seed_offset=0,
            ),
            StageSpec(
                stage_name="below_threshold",
                selector="leaf_acc_band",
                reopen_node_strategy="child_expected_acc_gap_weighted",
                child_pool_strategy="single_random_from_task_pool",
                child_pool_size=1,
                leaf_acc_threshold=default_leaf_acc_threshold,
                tree_budget_usd=default_tree_budget_usd,
                max_final_leaf_count=default_max_final_leaf_count,
                frontier_limit=default_frontier_limit,
                random_seed_offset=1,
                leaf_acc_min_exclusive=0.0,
                leaf_acc_max_exclusive=default_leaf_acc_threshold,
            ),
        ]

    if not isinstance(stages_payload, list) or not stages_payload:
        raise ValueError("stages must be a non-empty list when provided.")

    stage_specs: list[StageSpec] = []
    stage_names: set[str] = set()
    for index, item in enumerate(stages_payload):
        if not isinstance(item, dict):
            raise ValueError(f"stages[{index}] must be a mapping.")
        stage_name = str(item.get("stage_name") or item.get("name") or "").strip()
        if not stage_name:
            raise ValueError(f"stages[{index}] is missing stage_name.")
        if stage_name in stage_names:
            raise ValueError(f"Duplicate stage_name in config: {stage_name}")
        stage_names.add(stage_name)

        selector = str(item.get("selector") or stage_name).strip()
        normalized_selector = _normalize_stage_selector(selector)
        reopen_node_strategy = str(
            item.get("reopen_node_strategy") or _default_reopen_node_strategy(selector)
        ).strip()
        child_pool_strategy = str(
            item.get("child_pool_strategy") or _default_child_pool_strategy(selector)
        ).strip()
        leaf_acc_threshold = float(item.get("leaf_acc_threshold", default_leaf_acc_threshold))
        tree_budget_usd = float(item.get("tree_budget_usd", default_tree_budget_usd))
        max_final_leaf_count = _optional_int(
            item.get("max_final_leaf_count", default_max_final_leaf_count)
        )
        frontier_limit = int(item.get("frontier_limit", default_frontier_limit))
        child_pool_size = int(
            item.get(
                "child_pool_size",
                3 if child_pool_strategy == "random_k_from_available" else 1,
            )
        )
        leaf_acc_min_exclusive = _optional_float(item.get("leaf_acc_min_exclusive"))
        leaf_acc_max_exclusive = _optional_float(item.get("leaf_acc_max_exclusive"))
        if normalized_selector == "leaf_acc_band":
            if leaf_acc_min_exclusive is None:
                leaf_acc_min_exclusive = 0.0
            if leaf_acc_max_exclusive is None:
                leaf_acc_max_exclusive = leaf_acc_threshold

        stage_specs.append(
            StageSpec(
                stage_name=stage_name,
                selector=selector,
                reopen_node_strategy=reopen_node_strategy,
                child_pool_strategy=child_pool_strategy,
                child_pool_size=child_pool_size,
                leaf_acc_threshold=leaf_acc_threshold,
                tree_budget_usd=tree_budget_usd,
                max_final_leaf_count=max_final_leaf_count,
                frontier_limit=frontier_limit,
                random_seed_offset=int(item.get("random_seed_offset", index)),
                leaf_acc_min_exclusive=leaf_acc_min_exclusive,
                leaf_acc_max_exclusive=leaf_acc_max_exclusive,
            )
        )

    return stage_specs


def row_matches_stage_spec(row: dict[str, Any], stage_spec: StageSpec) -> bool:
    if not can_continue_expansion(row, stage_spec.max_final_leaf_count):
        return False

    selector = _normalize_stage_selector(stage_spec.selector)
    if selector == "all_wrong":
        return int(row.get("correct_leaf_count", 0) or 0) == 0

    if selector == "leaf_acc_band":
        leaf_acc = leaf_acc_from_result(row)
        if stage_spec.leaf_acc_min_exclusive is not None and leaf_acc <= stage_spec.leaf_acc_min_exclusive:
            return False
        if stage_spec.leaf_acc_max_exclusive is not None and leaf_acc >= stage_spec.leaf_acc_max_exclusive:
            return False
        return True

    raise ValueError(f"Unsupported stage selector: {stage_spec.selector}")


def load_results(run_dir: Path) -> list[dict[str, Any]]:
    rows = load_json(run_dir / "scored.json")
    if not isinstance(rows, list):
        raise ValueError(f"Expected a list in {run_dir / 'scored.json'}")
    return [row for row in rows if isinstance(row, dict)]


def load_latest(sample_dir: Path) -> dict[str, Any]:
    payload = load_json(sample_dir / "latest.json")
    if not isinstance(payload, dict):
        raise ValueError(f"Expected an object in {sample_dir / 'latest.json'}")
    return payload


def leaf_acc_from_result(row: dict[str, Any]) -> float:
    final_leaf_count = int(row.get("final_leaf_count", 0) or 0)
    correct_leaf_count = int(row.get("correct_leaf_count", 0) or 0)
    if final_leaf_count <= 0:
        return 0.0
    return correct_leaf_count / final_leaf_count


def can_continue_expansion(row: dict[str, Any], max_final_leaf_count: int | None) -> bool:
    if max_final_leaf_count is None:
        return True
    final_leaf_count = int(row.get("final_leaf_count", 0) or 0)
    return final_leaf_count < max_final_leaf_count


def _descendant_reward_stats_by_node(latest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    nodes_by_id: dict[str, dict[str, Any]] = {}
    for item in latest.get("nodes", []):
        if not isinstance(item, dict):
            continue
        node_id = str(item.get("node_id") or "").strip()
        if node_id:
            nodes_by_id[node_id] = item

    memo: dict[str, dict[str, Any]] = {}

    def resolve(node_id: str) -> dict[str, Any]:
        cached = memo.get(node_id)
        if cached is not None:
            return cached

        node = nodes_by_id[node_id]
        child_ids = [str(child_id) for child_id in node.get("children_ids", []) if str(child_id) in nodes_by_id]
        if not child_ids:
            if bool(node.get("is_terminal")):
                is_submit = str(node.get("action") or "") == "submit" and bool(node.get("boxed_letter"))
                terminal_count = 1
                correct_count = 1 if is_submit and bool(node.get("is_correct")) else 0
                stats = {
                    "expected_acc": correct_count / terminal_count,
                    "terminal_count": terminal_count,
                    "correct_count": correct_count,
                }
            else:
                stats = {
                    "expected_acc": 0.0,
                    "terminal_count": 0,
                    "correct_count": 0,
                }
            memo[node_id] = stats
            return stats

        terminal_count = 0
        correct_count = 0
        for child_id in child_ids:
            child_stats = resolve(child_id)
            terminal_count += int(child_stats["terminal_count"])
            correct_count += int(child_stats["correct_count"])
        expected_acc = (correct_count / terminal_count) if terminal_count > 0 else 0.0
        stats = {
            "expected_acc": expected_acc,
            "terminal_count": terminal_count,
            "correct_count": correct_count,
        }
        memo[node_id] = stats
        return stats

    for node_id in nodes_by_id:
        resolve(node_id)
    return memo


def expandable_nodes_with_stats(latest: dict[str, Any], *, node_max_steps: int) -> list[dict[str, Any]]:
    reward_stats = _descendant_reward_stats_by_node(latest)
    candidates: list[dict[str, Any]] = []
    for item in latest.get("nodes", []):
        if not isinstance(item, dict):
            continue
        action = str(item.get("action") or "")
        is_terminal = bool(item.get("is_terminal", False))
        depth = int(item.get("depth", 0) or 0)
        node_id = str(item.get("node_id") or "").strip()
        if action not in {"root", "delegate"} or is_terminal or depth >= node_max_steps or not node_id:
            continue
        stats = reward_stats.get(
            node_id,
            {"expected_acc": 0.0, "terminal_count": 0, "correct_count": 0},
        )
        candidates.append(
            {
                "node_id": node_id,
                "depth": depth,
                "expected_acc": float(stats["expected_acc"]),
                "terminal_count": int(stats["terminal_count"]),
                "correct_count": int(stats["correct_count"]),
            }
        )
    return sorted(candidates, key=lambda item: (int(item["depth"]), str(item["node_id"])))


def choose_reopen_node(
    latest: dict[str, Any],
    *,
    task_id: str,
    node_max_steps: int,
    random_seed: int,
    stage_name: str,
    reopen_node_strategy: str,
) -> dict[str, Any] | None:
    candidates = expandable_nodes_with_stats(latest, node_max_steps=node_max_steps)
    if not candidates:
        return None

    rng = random.Random(f"{random_seed}:{stage_name}:{task_id}")
    if reopen_node_strategy == "uniform_random_expandable_node":
        weighted_candidates = [
            {
                **item,
                "depth_bias": 1.0 / float(int(item["depth"]) + 1),
                "selection_weight": 1.0 / float(int(item["depth"]) + 1),
            }
            for item in candidates
        ]
        weights = [float(item["selection_weight"]) for item in weighted_candidates]
        selected = rng.choices(weighted_candidates, weights=weights, k=1)[0] if any(weight > 0.0 for weight in weights) else rng.choice(weighted_candidates)
        top_candidates = sorted(
            weighted_candidates,
            key=lambda item: (-float(item["selection_weight"]), int(item["depth"]), str(item["node_id"])),
        )[:8]
        return {
            "node_id": selected["node_id"],
            "selection_mode": "uniform_random_depth_biased",
            "candidate_count": len(candidates),
            "selected_depth": int(selected["depth"]),
            "selected_expected_acc": selected["expected_acc"],
            "selected_depth_bias": selected["depth_bias"],
            "selected_terminal_count": selected["terminal_count"],
            "selected_correct_count": selected["correct_count"],
            "selected_weight": selected["selection_weight"],
            "top_weighted_candidates": top_candidates,
        }

    if reopen_node_strategy == "child_expected_acc_gap_weighted":
        nodes_by_id: dict[str, dict[str, Any]] = {}
        for item in latest.get("nodes", []):
            if not isinstance(item, dict):
                continue
            node_id = str(item.get("node_id") or "").strip()
            if node_id:
                nodes_by_id[node_id] = item
        reward_stats = _descendant_reward_stats_by_node(latest)

        weighted_candidates: list[dict[str, Any]] = []
        for item in candidates:
            node_id = str(item["node_id"])
            node_payload = nodes_by_id.get(node_id, {})
            child_ids = [
                str(child_id)
                for child_id in node_payload.get("children_ids", [])
                if str(child_id) in nodes_by_id
            ]
            child_expected_accs = [
                float(reward_stats.get(child_id, {}).get("expected_acc", 0.0))
                for child_id in child_ids
            ]
            child_variance = 0.0
            if len(child_expected_accs) >= 2:
                mean_value = sum(child_expected_accs) / float(len(child_expected_accs))
                child_variance = sum((value - mean_value) ** 2 for value in child_expected_accs) / float(len(child_expected_accs))
            depth_bias = 1.0 / float(int(item["depth"]) + 1)
            weighted_candidates.append(
                {
                    **item,
                    "child_expected_acc_variance": child_variance,
                    "child_expected_accs": child_expected_accs,
                    "depth_bias": depth_bias,
                    "selection_weight": max(child_variance * depth_bias, 0.0),
                }
            )

        weights = [float(item["selection_weight"]) for item in weighted_candidates]
        has_child_variance_signal = any(weight > 0.0 for weight in weights)
        if has_child_variance_signal:
            selected = rng.choices(weighted_candidates, weights=weights, k=1)[0]
        else:
            selected = rng.choice(weighted_candidates)

        top_candidates = sorted(
            weighted_candidates,
            key=lambda item: (-float(item["selection_weight"]), int(item["depth"]), str(item["node_id"])),
        )[:8]

        return {
            "node_id": selected["node_id"],
            "selection_mode": "child_expected_acc_variance_depth_biased",
            "has_child_variance_signal": has_child_variance_signal,
            "selected_depth": int(selected["depth"]),
            "candidate_count": len(candidates),
            "selected_expected_acc": selected["expected_acc"],
            "selected_child_expected_acc_variance": selected["child_expected_acc_variance"],
            "selected_child_expected_accs": selected["child_expected_accs"],
            "selected_depth_bias": selected["depth_bias"],
            "selected_terminal_count": selected["terminal_count"],
            "selected_correct_count": selected["correct_count"],
            "selected_weight": selected["selection_weight"],
            "top_weighted_candidates": top_candidates,
        }

    raise ValueError(f"Unsupported reopen_node_strategy: {reopen_node_strategy}")


def prepare_sample_dir_for_stage(
    sample_dir: Path,
    *,
    task_id: str,
    reopen_node_id: str,
    stage_name: str,
) -> None:
    latest = load_latest(sample_dir)
    latest["open_frontier_node_ids"] = [reopen_node_id]
    latest["open_leaf_count"] = 1
    latest["reopen_stage"] = stage_name
    latest["reopen_node_id"] = reopen_node_id
    write_json(sample_dir / "latest.json", latest)

    calls_partial = sample_dir / "calls.partial.jsonl"
    calls_jsonl = sample_dir / "calls.jsonl"
    if not calls_partial.exists() and calls_jsonl.exists():
        calls_partial.write_text(calls_jsonl.read_text(encoding="utf-8"), encoding="utf-8")

    for name in ("result.json", "view.json", "nodes.jsonl", "calls.jsonl"):
        target = sample_dir / name
        if target.exists():
            target.unlink()

    stage_marker = {
        "task_id": task_id,
        "stage_name": stage_name,
        "reopen_node_id": reopen_node_id,
    }
    write_json(sample_dir / f"reopen_{stage_name}.json", stage_marker)


def build_stage_plan(
    *,
    stage_spec: StageSpec,
    rows: list[dict[str, Any]],
    run_dir: Path,
    pools_by_task: dict[str, list[str]],
    node_max_steps: int,
    random_seed: int,
) -> StagePlan:
    selected_rows = [row for row in rows if row_matches_stage_spec(row, stage_spec)]

    task_ids: list[str] = []
    fallback_random_pool_task_ids: list[str] = []
    reopen_nodes: dict[str, str] = {}
    reopen_node_selection: dict[str, dict[str, Any]] = {}
    leaf_acc_before: dict[str, float] = {}

    for row in selected_rows:
        task_id = str(row.get("task_id") or "").strip()
        if not task_id:
            continue
        leaf_acc_before[task_id] = leaf_acc_from_result(row)
        pool = pools_by_task.get(task_id, [])
        if not pool:
            fallback_random_pool_task_ids.append(task_id)
        sample_dir = run_dir / "samples" / task_id
        latest = load_latest(sample_dir)
        reopen_selection = choose_reopen_node(
            latest,
            task_id=task_id,
            node_max_steps=node_max_steps,
            random_seed=random_seed,
            stage_name=stage_spec.stage_name,
            reopen_node_strategy=stage_spec.reopen_node_strategy,
        )
        if reopen_selection is None:
            continue
        task_ids.append(task_id)
        reopen_nodes[task_id] = str(reopen_selection["node_id"])
        reopen_node_selection[task_id] = reopen_selection

    return StagePlan(
        stage_name=stage_spec.stage_name,
        task_ids=task_ids,
        fallback_random_pool_task_ids=fallback_random_pool_task_ids,
        reopen_nodes=reopen_nodes,
        reopen_node_selection=reopen_node_selection,
        leaf_acc_before=leaf_acc_before,
    )


def build_stage_config_payload(
    *,
    base_snapshot: dict[str, Any],
    output_parent_dir: Path,
    task_ids: list[str],
    all_candidate_models: list[str],
    tree_budget_usd: float,
    frontier_limit: int,
    max_concurrency: int,
) -> dict[str, Any]:
    payload = dict(base_snapshot)
    payload["output_dir"] = str(output_parent_dir)
    payload["task_ids"] = task_ids
    payload["sample_count"] = len(task_ids)
    payload["resume"] = True
    # Disable self delegation during rescue expansion so only delegate pools are explored.
    payload["allow_orchestra_model_delegation"] = False
    # Expansion currently relies on the runner's stable single-sample child expansion path.
    # Force this to 1 so rescue runs do not depend on the older diverse-sampling round API.
    payload["orchestra_samples_per_prompt"] = 4
    payload["tree_budget_usd"] = tree_budget_usd
    payload["frontier_limit"] = frontier_limit
    payload["max_concurrency"] = max_concurrency
    payload["candidate_models"] = list(all_candidate_models)
    payload.pop("exclude_task_ids", None)
    payload.pop("exclude_task_ids_path", None)
    # During expand, max_final_leaf_count (from stage spec) is the intended cap.
    # Clear target_leaf_trajectories so _effective_submit_leaf_cap does not
    # pick up the (usually lower) reasoning-phase default and silently shrink the cap.
    payload["target_leaf_trajectories"] = None
    return payload


async def run_stage(
    *,
    stage_spec: StageSpec,
    stage_plan: StagePlan,
    run_dir: Path,
    output_parent_dir: Path,
    base_snapshot: dict[str, Any],
    all_candidate_models: list[str],
    max_concurrency: int,
    pools_by_task: dict[str, list[str]],
    meta_dir: Path,
) -> dict[str, Any]:
    if not stage_plan.task_ids:
        summary = {
            "stage_name": stage_plan.stage_name,
            "selected_task_count": 0,
            "selected_task_ids": [],
            "fallback_random_pool_task_ids": stage_plan.fallback_random_pool_task_ids,
            "stage_config": asdict(stage_spec),
        }
        write_json(meta_dir / f"{stage_plan.stage_name}_summary.json", summary)
        return summary

    for task_id in stage_plan.task_ids:
        prepare_sample_dir_for_stage(
            run_dir / "samples" / task_id,
            task_id=task_id,
            reopen_node_id=stage_plan.reopen_nodes[task_id],
            stage_name=stage_plan.stage_name,
        )

    config_payload = build_stage_config_payload(
        base_snapshot=base_snapshot,
        output_parent_dir=output_parent_dir,
        task_ids=stage_plan.task_ids,
        all_candidate_models=all_candidate_models,
        tree_budget_usd=stage_spec.tree_budget_usd,
        frontier_limit=stage_spec.frontier_limit,
        max_concurrency=max_concurrency,
    )
    stage_config_path = meta_dir / f"{stage_plan.stage_name}.runner.yaml"
    stage_config_path.parent.mkdir(parents=True, exist_ok=True)
    stage_config_path.write_text(
        yaml.safe_dump(config_payload, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    config = MCTSConfig.load(stage_config_path)
    runner = FixedPoolExpansionRunner(
        config,
        pools_by_task=pools_by_task,
        leaf_acc_threshold=stage_spec.leaf_acc_threshold,
        max_final_leaf_count=stage_spec.max_final_leaf_count,
        progress_desc=f"Expand-{stage_plan.stage_name}",
        stage_name=stage_plan.stage_name,
        reopen_node_strategy=stage_spec.reopen_node_strategy,
        child_pool_strategy=stage_spec.child_pool_strategy,
        child_pool_size=stage_spec.child_pool_size,
    )
    records, summary = await runner.run_with_leaf_acc_stop()
    stage_summary = {
        "stage_name": stage_plan.stage_name,
        "stage_config": asdict(stage_spec),
        "selected_task_count": len(stage_plan.task_ids),
        "selected_task_ids": stage_plan.task_ids,
        "reopen_nodes": stage_plan.reopen_nodes,
        "fallback_random_pool_task_ids": stage_plan.fallback_random_pool_task_ids,
        "runner_summary": summary,
        "post_stage_leaf_acc": {
            str(row.get("task_id")): leaf_acc_from_result(row)
            for row in records
            if str(row.get("task_id")) in set(stage_plan.task_ids)
        },
    }
    write_json(meta_dir / f"{stage_plan.stage_name}_summary.json", stage_summary)
    return stage_summary


def main() -> int:
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()
    _load_env_candidates(config_path)
    config = load_yaml(config_path)
    stage_specs = load_stage_specs(config)
    stage_names = [spec.stage_name for spec in stage_specs]

    start_stage = args.start_stage or str(config.get("start_stage", stage_names[0]))
    if start_stage not in stage_names:
        raise ValueError(f"start_stage must be one of: {', '.join(stage_names)}")

    run_dir = Path(config.get("run_dir", config["base_run_dir"])).expanduser().resolve()
    pool_summary_path = Path(config["pool_summary_path"]).expanduser().resolve()
    max_concurrency = int(config.get("max_concurrency", 1))
    random_seed = int(config.get("random_seed", 42))

    start_index = stage_names.index(start_stage)
    active_stage_specs = stage_specs[start_index:]
    ensure_pool_summary_exists(
        config_path=config_path,
        pool_summary_path=pool_summary_path,
        active_stage_specs=active_stage_specs,
    )
    pools_by_task = normalize_model_pools(pool_summary_path) if pool_summary_path.exists() else {}
    base_snapshot = load_json(run_dir / "config.snapshot.json")
    output_parent_dir = run_dir.parent
    config_candidate_models = config.get("candidate_models")
    if config_candidate_models and isinstance(config_candidate_models, list):
        all_candidate_models = [str(m).strip() for m in config_candidate_models if str(m).strip()]
    else:
        all_candidate_models = sorted(
            {
                *[str(item) for item in base_snapshot.get("candidate_models", [])],
                *[model for pool in pools_by_task.values() for model in pool],
            }
        )
    if not all_candidate_models:
        raise ValueError("No candidate models available after merging base snapshot and task pools.")

    if not run_dir.exists():
        raise FileNotFoundError(f"run_dir does not exist: {run_dir}")

    meta_dir = run_dir / "expansion_meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    initial_rows = load_results(run_dir)
    initial_summary = {
        "run_dir": str(run_dir),
        "task_count": len(initial_rows),
        "start_stage": start_stage,
        "max_concurrency": max_concurrency,
        "stage_configs": [asdict(spec) for spec in stage_specs],
    }
    write_json(meta_dir / "initial_summary.json", initial_summary)

    stage_summaries: list[dict[str, Any]] = []
    planned_stages: list[tuple[StageSpec, StagePlan]] = []
    rows_for_stage = initial_rows
    node_max_steps = int(base_snapshot.get("node_max_steps", 8))

    for stage_spec in active_stage_specs:
        stage_plan = build_stage_plan(
            stage_spec=stage_spec,
            rows=rows_for_stage,
            run_dir=run_dir,
            pools_by_task=pools_by_task,
            node_max_steps=node_max_steps,
            random_seed=random_seed + stage_spec.random_seed_offset,
        )
        write_json(
            meta_dir / f"{stage_spec.stage_name}_plan.json",
            {
                "stage_config": asdict(stage_spec),
                "plan": asdict(stage_plan),
            },
        )
        planned_stages.append((stage_spec, stage_plan))

        if args.prepare_only:
            break

    if not args.prepare_only:
        async def _run_all_stages():
            return list(await asyncio.gather(*[
                run_stage(
                    stage_spec=spec,
                    stage_plan=plan,
                    run_dir=run_dir,
                    output_parent_dir=output_parent_dir,
                    base_snapshot=base_snapshot,
                    all_candidate_models=all_candidate_models,
                    max_concurrency=max_concurrency,
                    pools_by_task=pools_by_task,
                    meta_dir=meta_dir,
                )
                for spec, plan in planned_stages
            ]))
        stage_summaries = asyncio.run(_run_all_stages())

    if args.prepare_only:
        prepared_spec, prepared_stage = planned_stages[0]
        for task_id in prepared_stage.task_ids:
            prepare_sample_dir_for_stage(
                run_dir / "samples" / task_id,
                task_id=task_id,
                reopen_node_id=prepared_stage.reopen_nodes[task_id],
                stage_name=prepared_stage.stage_name,
            )
        write_json(
            meta_dir / "prepare_only_summary.json",
            {
                "run_dir": str(run_dir),
                "start_stage": start_stage,
                "prepared_stage": prepared_stage.stage_name,
                "stage_config": asdict(prepared_spec),
                "prepared_task_ids": prepared_stage.task_ids,
                "fallback_random_pool_task_ids": prepared_stage.fallback_random_pool_task_ids,
            },
        )
        print("prepare_only: true")
        print(f"start_stage: {start_stage}")
        print(f"run_dir: {run_dir}")
        print(f"prepared_task_count: {len(prepared_stage.task_ids)}")
        print(f"meta_dir: {meta_dir}")
        return 0

    final_rows = load_results(run_dir)
    final_remaining_task_ids_by_stage = {
        spec.stage_name: [
            str(row.get("task_id"))
            for row in final_rows
            if row_matches_stage_spec(row, spec)
        ]
        for spec in stage_specs
    }
    final_summary = {
        "run_dir": str(run_dir),
        "start_stage": start_stage,
        "stage_configs": [asdict(spec) for spec in stage_specs],
        "stage_summaries": stage_summaries,
        "final_remaining_task_ids_by_stage": final_remaining_task_ids_by_stage,
        "final_below_threshold_task_ids": final_remaining_task_ids_by_stage.get("below_threshold", []),
        "final_leaf_acc": {
            str(row.get("task_id")): leaf_acc_from_result(row)
            for row in final_rows
        },
    }
    write_json(meta_dir / "expansion_summary.json", final_summary)

    print(f"run_dir: {run_dir}")
    print(f"start_stage: {start_stage}")
    for stage_summary in stage_summaries:
        print(f"{stage_summary['stage_name']}_task_count: {stage_summary['selected_task_count']}")
    print(f"expansion_summary_path: {meta_dir / 'expansion_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
