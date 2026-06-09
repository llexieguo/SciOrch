#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import copy
import json
import re
import shutil
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None


STEP_ORDER = ("reasoning", "expand", "prune", "export")


def _tree_phase_costs(
    reasoning_result: dict[str, Any] | None,
    expand_result: dict[str, Any] | None,
    *,
    expand_enabled: bool,
) -> tuple[float, float, float]:
    """Return (reasoning_cost, expand_cost, combined_total) for one tree.

    Progress reporting should use combined_total = reasoning + expand when both phases run.
    Expand-only totals were misleading in the tree-mode progress bar / pipeline summary.
    """
    reasoning_cost = float((reasoning_result or {}).get("total_cost", 0.0) or 0.0)
    if not expand_enabled:
        return reasoning_cost, 0.0, reasoning_cost
    if expand_result is None:
        return reasoning_cost, 0.0, reasoning_cost
    is_legacy_expand = expand_result.get("stop_reason") == "legacy_latest_resume_state"
    expand_cost = (
        0.0
        if is_legacy_expand
        else float(expand_result.get("total_cost", 0.0) or 0.0)
    )
    return reasoning_cost, expand_cost, reasoning_cost + expand_cost


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one MCTS data-generation iteration: reasoning -> expand -> prune -> export."
    )
    parser.add_argument("--config", required=True, help="Path to unified YAML config.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only render generated configs and commands, do not execute subprocesses.",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    return payload


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _as_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return bool(value)


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sanitize_model_dir_name(model_name: str) -> str:
    return str(model_name).strip().replace("/", "__").replace("\\", "__")


def _resolve_path(path_like: Any, *, base_dir: Path) -> Path:
    path = Path(str(path_like)).expanduser()
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _resolve_optional_path(path_like: Any, *, base_dir: Path) -> Path | None:
    if path_like is None:
        return None
    text = str(path_like).strip()
    if not text:
        return None
    return _resolve_path(text, base_dir=base_dir)


def _resolve_resume_iteration_dir(
    resume_value: Any,
    *,
    base_output_dir: Path,
) -> Path | None:
    if resume_value is None:
        return None
    text = str(resume_value).strip()
    if not text:
        return None
    path = Path(text).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (base_output_dir / path).resolve()


def _ensure_mapping(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key) or {}
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a mapping")
    return value


def _ensure_step_order(start_from: str) -> list[str]:
    if start_from not in STEP_ORDER:
        allowed = ", ".join(STEP_ORDER)
        raise ValueError(f"steps.start_from must be one of: {allowed}")
    start_index = STEP_ORDER.index(start_from)
    return list(STEP_ORDER[start_index:])


def _detect_resume_start_from(
    *,
    reasoning_run_dir: Path,
    prune_output_dir: Path,
) -> tuple[str | None, str]:
    prune_has_output = (prune_output_dir / "samples").is_dir() or (prune_output_dir / "scored.json").exists()
    if prune_has_output:
        return None, "detected_prune_output"

    expand_meta_dir = reasoning_run_dir / "expansion_meta"
    if expand_meta_dir.is_dir():
        return "expand", "detected_expand_meta"

    reasoning_has_output = (reasoning_run_dir / "samples").is_dir() or (reasoning_run_dir / "scored.json").exists()
    if reasoning_has_output:
        return "reasoning", "detected_reasoning_output"

    return "reasoning", "detected_empty_iteration"


def _collect_candidate_models(
    section_models: Any,
    shared_models: list[str],
) -> list[str]:
    models: list[str] = []
    if isinstance(section_models, list):
        models = [str(model).strip() for model in section_models if str(model).strip()]
    if not models:
        models = [str(model).strip() for model in shared_models if str(model).strip()]
    if not models:
        raise ValueError("candidate_models cannot be empty after merge")
    return models


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _run_command(command: list[str], *, dry_run: bool) -> None:
    print("$ " + " ".join(command), flush=True)
    if dry_run:
        return
    subprocess.run(command, check=True)


def _run_command_logged(command: list[str], *, log_path: Path, dry_run: bool) -> None:
    if dry_run:
        print("$ " + " ".join(command), flush=True)
        print(f"  -> log: {log_path}", flush=True)
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"\n[{_now_utc()}] $ {' '.join(command)}\n")
        handle.flush()
        subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=True)


def _next_version_dir(base_output_dir: Path, version_prefix: str) -> Path:
    version_pattern = re.compile(rf"^{re.escape(version_prefix)}(\d+)$")
    max_version = -1
    if base_output_dir.exists():
        for child in base_output_dir.iterdir():
            if not child.is_dir():
                continue
            match = version_pattern.match(child.name)
            if match is None:
                continue
            max_version = max(max_version, int(match.group(1)))
    return base_output_dir / f"{version_prefix}{max_version + 1}"


def _summarize_and_visualize(
    *,
    script_path: Path,
    run_dir: Path,
    viz_dir: Path,
    json_path: Path,
    dry_run: bool,
) -> None:
    command = [
        sys.executable,
        str(script_path),
        "--run-dir",
        str(run_dir),
        "--visualize",
        "--viz-dir",
        str(viz_dir),
        "--json",
        "--json-output-path",
        str(json_path),
    ]
    _run_command(command, dry_run=dry_run)


def _iter_sample_dirs(run_dir: Path) -> list[Path]:
    samples_dir = run_dir / "samples"
    if not samples_dir.is_dir():
        return []
    return sorted(path for path in samples_dir.iterdir() if path.is_dir())


def _sync_sample_dir(*, source_run_dir: Path, dest_run_dir: Path, task_id: str) -> None:
    source_sample_dir = source_run_dir / "samples" / task_id
    if not source_sample_dir.is_dir():
        raise FileNotFoundError(f"Missing sample dir for task {task_id}: {source_sample_dir}")
    dest_sample_dir = dest_run_dir / "samples" / task_id
    if dest_sample_dir.exists():
        shutil.rmtree(dest_sample_dir)
    dest_sample_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_sample_dir, dest_sample_dir)


def _materialize_legacy_tree_reasoning_runs(
    *,
    aggregate_reasoning_run_dir: Path,
    iteration_dir: Path,
    model_dir_name: str,
    reasoning_payload: dict[str, Any],
    task_ids: list[str],
    dry_run: bool,
) -> dict[str, Any]:
    if not aggregate_reasoning_run_dir.is_dir():
        return {"materialized_count": 0, "skipped_count": 0, "missing_task_ids": []}

    available_task_ids = {sample_dir.name for sample_dir in _iter_sample_dirs(aggregate_reasoning_run_dir)}
    if not available_task_ids:
        return {"materialized_count": 0, "skipped_count": 0, "missing_task_ids": task_ids}

    materialized_count = 0
    skipped_count = 0
    missing_task_ids: list[str] = []
    for task_id in task_ids:
        if task_id not in available_task_ids:
            missing_task_ids.append(task_id)
            continue
        paths = _build_tree_task_dirs(iteration_dir=iteration_dir, model_dir_name=model_dir_name, task_id=task_id)
        if paths["reasoning_sample_dir"].exists():
            skipped_count += 1
            continue
        if dry_run:
            materialized_count += 1
            continue
        _sync_sample_dir(
            source_run_dir=aggregate_reasoning_run_dir,
            dest_run_dir=paths["reasoning_run_dir"],
            task_id=task_id,
        )
        single_task_payload = _build_single_task_reasoning_payload(
            base_payload=reasoning_payload,
            task_id=task_id,
            output_dir=paths["reasoning_base_dir"],
        )
        _rebuild_aggregate_run_outputs(
            run_dir=paths["reasoning_run_dir"],
            orchestra_model=str(reasoning_payload["orchestra_model"]),
            config_snapshot=single_task_payload,
            selected_task_ids=[task_id],
        )
        _write_json(
            paths["tree_root"] / "migrated_from_aggregate_reasoning.json",
            {
                "generated_at": _now_utc(),
                "source_run_dir": str(aggregate_reasoning_run_dir),
                "task_id": task_id,
            },
        )
        materialized_count += 1

    return {
        "materialized_count": materialized_count,
        "skipped_count": skipped_count,
        "missing_task_ids": missing_task_ids,
    }


def _tail_text(path: Path, *, max_lines: int = 40) -> str:
    if not path.exists():
        return ""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-max_lines:])

def _resolve_tree_task_ids(
    *,
    manifest_path: Path,
    reasoning_payload: dict[str, Any],
    reasoning_config_path: Path,
) -> list[str]:
    if manifest_path.exists():
        payload = load_json(manifest_path)
        task_ids = [
            str(task_id).strip()
            for task_id in payload.get("task_ids", [])
            if str(task_id).strip()
        ]
        if task_ids:
            return task_ids

    explicit_task_ids = reasoning_payload.get("task_ids")
    if isinstance(explicit_task_ids, list):
        task_ids = [str(task_id).strip() for task_id in explicit_task_ids if str(task_id).strip()]
        if task_ids:
            _write_json(
                manifest_path,
                {
                    "generated_at": _now_utc(),
                    "source": "explicit_task_ids",
                    "task_ids": task_ids,
                },
            )
            return task_ids

    from mcts.config import MCTSConfig
    from mcts.runner import MCTSReasoningRunner

    config = MCTSConfig.load(reasoning_config_path)
    runner = MCTSReasoningRunner(config)
    samples = runner._select_samples(runner._load_dataset_samples())
    task_ids = [str(sample.task_id).strip() for sample in samples if str(sample.task_id).strip()]
    _write_json(
        manifest_path,
        {
            "generated_at": _now_utc(),
            "source": "runner_selection",
            "task_ids": task_ids,
        },
    )
    return task_ids


def _build_tree_task_dirs(
    *,
    iteration_dir: Path,
    model_dir_name: str,
    task_id: str,
) -> dict[str, Path]:
    tree_root = iteration_dir / "_tree_runs" / task_id
    reasoning_base_dir = tree_root / "reasoning"
    reasoning_run_dir = reasoning_base_dir / model_dir_name
    expand_base_dir = tree_root / "expand"
    expand_run_dir = expand_base_dir / model_dir_name
    return {
        "tree_root": tree_root,
        "config_dir": tree_root / "configs",
        "log_dir": tree_root / "logs",
        "reasoning_base_dir": reasoning_base_dir,
        "reasoning_run_dir": reasoning_run_dir,
        "reasoning_sample_dir": reasoning_run_dir / "samples" / task_id,
        "expand_base_dir": expand_base_dir,
        "expand_run_dir": expand_run_dir,
        "expand_sample_dir": expand_run_dir / "samples" / task_id,
        "expand_pool_dir": expand_base_dir / "model_pools",
    }


def _build_single_task_reasoning_payload(
    *,
    base_payload: dict[str, Any],
    task_id: str,
    output_dir: Path,
) -> dict[str, Any]:
    payload = copy.deepcopy(base_payload)
    payload["output_dir"] = str(output_dir)
    payload["task_ids"] = [task_id]
    payload["sample_count"] = 1
    payload["resume"] = True
    payload["max_concurrency"] = 1
    payload["show_progress"] = False
    return payload


def _build_single_task_expand_payload(
    *,
    base_payload: dict[str, Any],
    run_dir: Path,
    pool_output_dir: Path,
) -> dict[str, Any]:
    payload = copy.deepcopy(base_payload)
    payload["base_run_dir"] = str(run_dir)
    payload["run_dir"] = str(run_dir)
    payload["max_concurrency"] = 1
    return payload


def _load_legacy_task_result(sample_dir: Path) -> dict[str, Any] | None:
    latest_path = sample_dir / "latest.json"
    if not latest_path.exists():
        return None
    payload = load_json(latest_path)
    if not isinstance(payload, dict):
        return None

    task_id = str(payload.get("task_id") or sample_dir.name)
    final_trajectories = payload.get("final_trajectories")
    if not isinstance(final_trajectories, list):
        final_trajectories = []
    correct_leaf_count = sum(
        1 for item in final_trajectories if isinstance(item, dict) and bool(item.get("correct"))
    )
    final_leaf_count = int(payload.get("final_leaf_count", len(final_trajectories)) or 0)
    if final_leaf_count <= 0:
        final_leaf_count = len(final_trajectories)
    any_correct_leaf = correct_leaf_count > 0
    leaf_acc = (correct_leaf_count / final_leaf_count) if final_leaf_count > 0 else 0.0

    return {
        "task_id": task_id,
        "status": "completed",
        "success": any_correct_leaf,
        "any_correct_leaf": any_correct_leaf,
        "best_leaf_correct": any_correct_leaf,
        "majority_correct": False,
        "correct_leaf_count": correct_leaf_count,
        "final_leaf_count": final_leaf_count,
        "open_leaf_count": int(payload.get("open_leaf_count", 0) or 0),
        "failed_terminal_count": int(payload.get("failed_terminal_count", 0) or 0),
        "expansion_rounds_ran": len(payload.get("rounds", [])) if isinstance(payload.get("rounds"), list) else 0,
        "stop_reason": "legacy_latest_resume_state",
        "budget_spent": float(payload.get("budget_spent", payload.get("total_cost", 0.0)) or 0.0),
        "total_cost": float(payload.get("budget_spent", payload.get("total_cost", 0.0)) or 0.0),
        "total_tokens": int(payload.get("total_tokens", 0) or 0),
        "total_model_calls": int(payload.get("raw_call_count", payload.get("total_model_calls", 0)) or 0),
        "leaf_acc": leaf_acc,
        "legacy_resume_source": str(latest_path),
    }


def _load_task_result(sample_dir: Path) -> dict[str, Any] | None:
    result_path = sample_dir / "result.json"
    if result_path.exists():
        payload = load_json(result_path)
        return payload if isinstance(payload, dict) else None
    return _load_legacy_task_result(sample_dir)


def _load_result_rows_from_run(run_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample_dir in _iter_sample_dirs(run_dir):
        payload = _load_task_result(sample_dir)
        if isinstance(payload, dict):
            rows.append(payload)
    return sorted(rows, key=lambda row: str(row.get("task_id") or ""))


def _rebuild_aggregate_run_outputs(
    *,
    run_dir: Path,
    orchestra_model: str,
    config_snapshot: dict[str, Any],
    selected_task_ids: list[str],
) -> None:
    rows = _load_result_rows_from_run(run_dir)
    sample_count = len(rows)
    success_count = sum(1 for row in rows if bool(row.get("success")))
    total_cost = sum(float(row.get("total_cost", 0.0) or 0.0) for row in rows)
    total_tokens = sum(int(row.get("total_tokens", 0) or 0) for row in rows)
    total_model_calls = sum(int(row.get("total_model_calls", 0) or 0) for row in rows)
    avg_final_leaf_count = (
        sum(float(row.get("final_leaf_count", 0) or 0.0) for row in rows) / sample_count if sample_count else 0.0
    )
    avg_expansion_rounds = (
        sum(float(row.get("expansion_rounds_ran", 0) or 0.0) for row in rows) / sample_count if sample_count else 0.0
    )
    avg_total_cost = (total_cost / sample_count) if sample_count else 0.0
    summary = {
        "output_dir": str(run_dir),
        "orchestra_model": orchestra_model,
        "sample_count": sample_count,
        "success_count": success_count,
        "failure_count": sample_count - success_count,
        "success_rate": (success_count / sample_count) if sample_count else 0.0,
        "total_cost": total_cost,
        "total_tokens": total_tokens,
        "total_model_calls": total_model_calls,
        "avg_final_leaf_count": avg_final_leaf_count,
        "avg_expansion_rounds": avg_expansion_rounds,
        "avg_total_cost": avg_total_cost,
        "selected_task_ids": selected_task_ids,
        "generated_at": _now_utc(),
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(run_dir / "scored.json", rows)
    _write_json(run_dir / "summary.json", summary)
    _write_json(run_dir / "config.snapshot.json", config_snapshot)
    _write_json(
        run_dir / "selected_tasks.json",
        {
            "sample_count": len(selected_task_ids),
            "selected_task_ids": selected_task_ids,
        },
    )


def _copy_reasoning_tree_to_expand(*, reasoning_run_dir: Path, expand_base_dir: Path, expand_run_dir: Path) -> None:
    if expand_base_dir.exists():
        shutil.rmtree(expand_base_dir)
    expand_run_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(reasoning_run_dir, expand_run_dir)
    # Remove terminal files so expand doesn't mistake copied reasoning results as expand-done
    samples_dir = expand_run_dir / "samples"
    if samples_dir.exists():
        for _rm in ("result.json", "calls.jsonl", "nodes.jsonl", "view.json"):
            for _f in samples_dir.glob(f"*/{_rm}"):
                _f.unlink(missing_ok=True)


def _run_single_tree_full_flow(
    *,
    task_id: str,
    iteration_dir: Path,
    model_dir_name: str,
    reasoning_payload: dict[str, Any],
    expand_payload: dict[str, Any],
    tree_start_phase: str,
    expand_enabled: bool,
    shared_root_dir: Path,
    generated_cfg_dir: Path,
    dry_run: bool,
) -> dict[str, Any]:
    paths = _build_tree_task_dirs(iteration_dir=iteration_dir, model_dir_name=model_dir_name, task_id=task_id)
    config_dir = paths["config_dir"]
    log_dir = paths["log_dir"]
    reasoning_config_path = config_dir / "reasoning.generated.yaml"
    expand_config_path = config_dir / "expand.generated.yaml"
    reasoning_log_path = log_dir / "reasoning.log"
    expand_log_path = log_dir / "expand.log"

    reasoning_result = _load_task_result(paths["reasoning_sample_dir"])
    _exp_path = paths["expand_sample_dir"] / "result.json"
    expand_result = (load_json(_exp_path) if _exp_path.exists() else None)

    if dry_run:
        reasoning_task_payload = _build_single_task_reasoning_payload(
            base_payload=reasoning_payload,
            task_id=task_id,
            output_dir=paths["reasoning_base_dir"],
        )
        _write_yaml(reasoning_config_path, reasoning_task_payload)
        if expand_enabled:
            expand_task_payload = _build_single_task_expand_payload(
                base_payload=expand_payload,
                run_dir=paths["expand_run_dir"],
                pool_output_dir=paths["expand_pool_dir"],
            )
            _write_yaml(expand_config_path, expand_task_payload)
        _, _, _tc = _tree_phase_costs(
            reasoning_result, expand_result, expand_enabled=expand_enabled
        )
        return {
            "task_id": task_id,
            "status": "dry_run",
            "reasoning_result": reasoning_result,
            "expand_result": expand_result,
            "reasoning_run_dir": str(paths["reasoning_run_dir"]),
            "expand_run_dir": str(paths["expand_run_dir"]),
            "reasoning_log_path": str(reasoning_log_path),
            "expand_log_path": str(expand_log_path),
            "total_cost": _tc,
        }

    if expand_enabled and expand_result is not None:
        _, _, _tc = _tree_phase_costs(
            reasoning_result, expand_result, expand_enabled=True
        )
        return {
            "task_id": task_id,
            "status": "completed",
            "reasoning_result": reasoning_result,
            "expand_result": expand_result,
            "reasoning_run_dir": str(paths["reasoning_run_dir"]),
            "expand_run_dir": str(paths["expand_run_dir"]),
            "reasoning_log_path": str(reasoning_log_path),
            "expand_log_path": str(expand_log_path),
            "total_cost": _tc,
        }

    if not expand_enabled and reasoning_result is not None:
        _, _, _tc = _tree_phase_costs(
            reasoning_result, None, expand_enabled=False
        )
        return {
            "task_id": task_id,
            "status": "completed",
            "reasoning_result": reasoning_result,
            "expand_result": None,
            "reasoning_run_dir": str(paths["reasoning_run_dir"]),
            "expand_run_dir": None,
            "reasoning_log_path": str(reasoning_log_path),
            "expand_log_path": None,
            "total_cost": _tc,
        }

    if reasoning_result is None:
        if tree_start_phase == "expand":
            raise RuntimeError(f"Task {task_id} cannot start from expand without a completed reasoning tree.")
        reasoning_task_payload = _build_single_task_reasoning_payload(
            base_payload=reasoning_payload,
            task_id=task_id,
            output_dir=paths["reasoning_base_dir"],
        )
        _write_yaml(reasoning_config_path, reasoning_task_payload)
        reasoning_command = [
            sys.executable,
            "-m",
            "mcts.cli.run_mcts_reasoning",
            "--config",
            str(reasoning_config_path),
        ]
        try:
            _run_command_logged(reasoning_command, log_path=reasoning_log_path, dry_run=False)
        except subprocess.CalledProcessError as exc:
            tail = _tail_text(reasoning_log_path)
            raise RuntimeError(
                f"Reasoning failed for {task_id} (exit={exc.returncode}).\n"
                f"log_path: {reasoning_log_path}\n{tail}"
            ) from exc
        reasoning_result = _load_task_result(paths["reasoning_sample_dir"])
        if reasoning_result is None:
            raise RuntimeError(f"Reasoning finished for {task_id} but result.json is missing.")

    if not expand_enabled:
        _, _, _tc = _tree_phase_costs(
            reasoning_result, None, expand_enabled=False
        )
        return {
            "task_id": task_id,
            "status": "completed",
            "reasoning_result": reasoning_result,
            "expand_result": None,
            "reasoning_run_dir": str(paths["reasoning_run_dir"]),
            "expand_run_dir": None,
            "reasoning_log_path": str(reasoning_log_path),
            "expand_log_path": None,
            "total_cost": _tc,
        }

    reasoning_snap = paths["reasoning_run_dir"] / "config.snapshot.json"
    expand_snap = paths["expand_run_dir"] / "config.snapshot.json"
    # Expand requires config.snapshot.json at run_dir root (see expand_mcts.main).
    # Resume can leave expand_run_dir + samples/ populated (e.g. scored.json rebuilt from legacy)
    # without ever copying reasoning — then expand_snap is missing and expand crashes.
    need_expand_seed = (
        not paths["expand_run_dir"].exists()
        or not paths["expand_sample_dir"].exists()
        or not expand_snap.exists()
    )
    if need_expand_seed:
        if not reasoning_snap.exists():
            raise RuntimeError(
                f"Cannot prepare expand for {task_id}: missing {reasoning_snap}. "
                "Finish or repair reasoning for this tree before expand."
            )
        _copy_reasoning_tree_to_expand(
            reasoning_run_dir=paths["reasoning_run_dir"],
            expand_base_dir=paths["expand_base_dir"],
            expand_run_dir=paths["expand_run_dir"],
        )

    _exp2 = paths["expand_sample_dir"] / "result.json"
    expand_result = (_exp2.exists() and __import__('json').loads(_exp2.read_text())) or None
    if expand_result is None:
        # Ensure exp_run/scored.json exists — expand subprocess needs it to determine eligibility.
        # If missing (e.g. reasoning was interrupted before writing scored.json), rebuild from latest.json.
        _scored_path = paths["expand_run_dir"] / "scored.json"
        if not _scored_path.exists():
            _legacy = _load_legacy_task_result(paths["expand_sample_dir"]) or _load_legacy_task_result(paths["reasoning_sample_dir"])
            if _legacy is not None:
                _write_json(_scored_path, [_legacy])
        expand_task_payload = _build_single_task_expand_payload(
            base_payload=expand_payload,
            run_dir=paths["expand_run_dir"],
            pool_output_dir=paths["expand_pool_dir"],
        )
        _write_yaml(expand_config_path, expand_task_payload)
        # Force-update config.snapshot.json so resumed expand picks up new settings
        # (e.g. orchestra_samples_per_prompt, branching_factor changes).
        _write_json(paths["expand_run_dir"] / "config.snapshot.json", expand_task_payload)
        expand_command = [
            sys.executable,
            str(shared_root_dir / "scripts" / "expand_mcts.py"),
            "--config",
            str(expand_config_path),
        ]
        try:
            _run_command_logged(expand_command, log_path=expand_log_path, dry_run=False)
        except subprocess.CalledProcessError as exc:
            tail = _tail_text(expand_log_path)
            raise RuntimeError(
                f"Expand failed for {task_id} (exit={exc.returncode}).\n"
                f"log_path: {expand_log_path}\n{tail}"
            ) from exc
        expand_result = _load_task_result(paths["expand_sample_dir"])
        if expand_result is None:
            raise RuntimeError(f"Expand finished for {task_id} but result.json is missing.")

    _, _, _tc = _tree_phase_costs(
        reasoning_result, expand_result, expand_enabled=True
    )

    return {
        "task_id": task_id,
        "status": "completed",
        "reasoning_result": reasoning_result,
        "expand_result": expand_result,
        "reasoning_run_dir": str(paths["reasoning_run_dir"]),
        "expand_run_dir": str(paths["expand_run_dir"]),
        "reasoning_log_path": str(reasoning_log_path),
        "expand_log_path": str(expand_log_path),
        "total_cost": _tc,
    }


def _completed_tree_summary_row_from_disk(
    *,
    iteration_dir: Path,
    model_dir_name: str,
    task_id: str,
    expand_enabled: bool,
) -> dict[str, Any]:
    """Build the same per-tree summary row used in the progress bar, for trees already finished on disk."""
    paths = _build_tree_task_dirs(iteration_dir=iteration_dir, model_dir_name=model_dir_name, task_id=task_id)
    log_dir = paths["log_dir"]
    reasoning_log_path = log_dir / "reasoning.log"
    expand_log_path = log_dir / "expand.log"
    reasoning_result = _load_task_result(paths["reasoning_sample_dir"]) or {}
    _exp_path = paths["expand_sample_dir"] / "result.json"
    expand_result: dict[str, Any] | None
    if _exp_path.exists():
        raw = load_json(_exp_path)
        expand_result = raw if isinstance(raw, dict) else None
    else:
        expand_result = None

    reasoning_cost, expand_cost, total_cost = _tree_phase_costs(
        reasoning_result if reasoning_result else None,
        expand_result,
        expand_enabled=expand_enabled,
    )

    return {
        "task_id": task_id,
        "status": "completed",
        "reasoning_log_path": str(reasoning_log_path),
        "expand_log_path": str(expand_log_path),
        "reasoning_run_dir": str(paths["reasoning_run_dir"]),
        "expand_run_dir": str(paths["expand_run_dir"]) if expand_enabled else None,
        "reasoning_cost": reasoning_cost,
        "expand_cost": expand_cost,
        "total_cost": total_cost,
        "updated_at": _now_utc(),
    }


def _run_tree_pipeline(
    *,
    task_ids: list[str],
    tree_max_concurrency: int,
    iteration_dir: Path,
    model_dir_name: str,
    reasoning_payload: dict[str, Any],
    expand_payload: dict[str, Any],
    reasoning_aggregate_run_dir: Path,
    expand_aggregate_run_dir: Path,
    shared_root_dir: Path,
    generated_cfg_dir: Path,
    tree_start_phase: str,
    expand_enabled: bool,
    dry_run: bool,
) -> dict[str, Any]:
    tree_summary_path = generated_cfg_dir / "tree_pipeline_summary.json"
    tree_results: dict[str, dict[str, Any]] = {}
    total_cost = 0.0

    # Pre-scan: classify tasks so we can skip already-done ones and
    # prioritise in-progress trees (reasoning done, expand pending).
    already_complete: list[str] = []
    in_progress: list[str] = []
    fresh: list[str] = []
    for _tid in task_ids:
        _p = _build_tree_task_dirs(iteration_dir=iteration_dir, model_dir_name=model_dir_name, task_id=_tid)
        _exp_done = (_p["expand_sample_dir"] / "result.json").exists()
        _rea_done = (_p["reasoning_sample_dir"] / "result.json").exists()
        if expand_enabled:
            if _exp_done:
                already_complete.append(_tid)
            elif _rea_done:
                in_progress.append(_tid)
            else:
                fresh.append(_tid)
        else:
            if _rea_done:
                already_complete.append(_tid)
            else:
                fresh.append(_tid)

    pending_task_ids = in_progress + fresh  # in-progress first, then fresh
    completed = len(already_complete)

    for _tid in already_complete:
        tree_results[_tid] = _completed_tree_summary_row_from_disk(
            iteration_dir=iteration_dir,
            model_dir_name=model_dir_name,
            task_id=_tid,
            expand_enabled=expand_enabled,
        )

    progress = None
    if tqdm is not None:
        progress = tqdm(total=len(task_ids), initial=completed, desc="Trees", unit="tree")
        if tree_results:
            progress.set_postfix_str(
                f"cost=${sum(float(item.get('total_cost', 0.0) or 0.0) for item in tree_results.values()):.3f}"
            )

    # Sync already-complete tasks to aggregate dirs so prune/export can see them
    if not dry_run:
        _ac_snap = copy.deepcopy(reasoning_payload)
        _ac_exp_snap = copy.deepcopy(expand_payload)
        _ac_exp_snap["base_run_dir"] = str(expand_aggregate_run_dir)
        _ac_exp_snap["run_dir"] = str(expand_aggregate_run_dir)
        for _tid in already_complete:
            _p = _build_tree_task_dirs(iteration_dir=iteration_dir, model_dir_name=model_dir_name, task_id=_tid)
            _sync_sample_dir(source_run_dir=_p["reasoning_run_dir"], dest_run_dir=reasoning_aggregate_run_dir, task_id=_tid)
            if expand_enabled and _p["expand_run_dir"].exists():
                _sync_sample_dir(source_run_dir=_p["expand_run_dir"], dest_run_dir=expand_aggregate_run_dir, task_id=_tid)
        if already_complete:
            _rebuild_aggregate_run_outputs(run_dir=reasoning_aggregate_run_dir, orchestra_model=str(reasoning_payload["orchestra_model"]), config_snapshot=_ac_snap, selected_task_ids=task_ids)
            if expand_enabled:
                _rebuild_aggregate_run_outputs(run_dir=expand_aggregate_run_dir, orchestra_model=str(reasoning_payload["orchestra_model"]), config_snapshot=_ac_exp_snap, selected_task_ids=task_ids)

    reasoning_snapshot = copy.deepcopy(reasoning_payload)
    expand_snapshot = copy.deepcopy(expand_payload)
    expand_snapshot["base_run_dir"] = str(expand_aggregate_run_dir)
    expand_snapshot["run_dir"] = str(expand_aggregate_run_dir)
    aggregate_lock = threading.Lock()

    def sync_outcome(outcome: dict[str, Any]) -> None:
        nonlocal total_cost
        if dry_run:
            return
        task_id = str(outcome["task_id"])
        with aggregate_lock:
            _sync_sample_dir(
                source_run_dir=Path(outcome["reasoning_run_dir"]),
                dest_run_dir=reasoning_aggregate_run_dir,
                task_id=task_id,
            )
            _rebuild_aggregate_run_outputs(
                run_dir=reasoning_aggregate_run_dir,
                orchestra_model=str(reasoning_payload["orchestra_model"]),
                config_snapshot=reasoning_snapshot,
                selected_task_ids=task_ids,
            )
            if expand_enabled and outcome.get("expand_run_dir"):
                _sync_sample_dir(
                    source_run_dir=Path(outcome["expand_run_dir"]),
                    dest_run_dir=expand_aggregate_run_dir,
                    task_id=task_id,
                )
                _rebuild_aggregate_run_outputs(
                    run_dir=expand_aggregate_run_dir,
                    orchestra_model=str(reasoning_payload["orchestra_model"]),
                    config_snapshot=expand_snapshot,
                    selected_task_ids=task_ids,
                )
        total_cost = sum(float(item.get("total_cost", 0.0) or 0.0) for item in tree_results.values())

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, tree_max_concurrency)) as executor:
        future_to_task = {
            executor.submit(
                _run_single_tree_full_flow,
                task_id=task_id,
                iteration_dir=iteration_dir,
                model_dir_name=model_dir_name,
                reasoning_payload=reasoning_payload,
                expand_payload=expand_payload,
                tree_start_phase=tree_start_phase,
                expand_enabled=expand_enabled,
                shared_root_dir=shared_root_dir,
                generated_cfg_dir=generated_cfg_dir,
                dry_run=dry_run,
            ): task_id
            for task_id in pending_task_ids
        }

        for future in concurrent.futures.as_completed(future_to_task):
            task_id = future_to_task[future]
            outcome = future.result()
            tree_results[task_id] = outcome
            sync_outcome(outcome)
            completed += 1
            rc, ec, tc = _tree_phase_costs(
                outcome.get("reasoning_result"),
                outcome.get("expand_result"),
                expand_enabled=expand_enabled,
            )
            summary_row = {
                "task_id": task_id,
                "status": outcome.get("status"),
                "reasoning_log_path": outcome.get("reasoning_log_path"),
                "expand_log_path": outcome.get("expand_log_path"),
                "reasoning_run_dir": outcome.get("reasoning_run_dir"),
                "expand_run_dir": outcome.get("expand_run_dir"),
                "reasoning_cost": rc,
                "expand_cost": ec,
                "total_cost": tc,
                "updated_at": _now_utc(),
            }
            tree_results[task_id] = summary_row
            _write_json(
                tree_summary_path,
                {
                    "generated_at": _now_utc(),
                    "tree_max_concurrency": tree_max_concurrency,
                    "completed_tree_count": completed,
                    "tree_count": len(task_ids),
                    "total_cost": sum(float(item.get("total_cost", 0.0) or 0.0) for item in tree_results.values()),
                    "tasks": [tree_results[k] for k in sorted(tree_results)],
                },
            )
            if progress is not None:
                progress.update(1)
                progress.set_postfix_str(f"cost=${sum(float(item.get('total_cost', 0.0) or 0.0) for item in tree_results.values()):.3f}")
            else:
                print(
                    f"[tree-progress] completed={completed}/{len(task_ids)} task_id={task_id} "
                    f"total_cost=${sum(float(item.get('total_cost', 0.0) or 0.0) for item in tree_results.values()):.3f}",
                    flush=True,
                )

    if progress is not None:
        progress.close()

    return {
        "tree_count": len(task_ids),
        "completed_tree_count": completed,
        "tree_max_concurrency": tree_max_concurrency,
        "total_cost": sum(float(item.get("total_cost", 0.0) or 0.0) for item in tree_results.values()),
        "tree_summary_path": str(tree_summary_path),
    }


def main() -> int:
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()
    config_dir = config_path.parent
    raw = load_yaml(config_path)

    run_cfg = _ensure_mapping(raw, "run")
    steps_cfg = _ensure_mapping(raw, "steps")
    shared_cfg = _ensure_mapping(raw, "shared")
    reasoning_cfg = _ensure_mapping(raw, "reasoning")
    expand_cfg = _ensure_mapping(raw, "expand")
    prune_cfg = _ensure_mapping(raw, "prune")
    export_cfg = _ensure_mapping(raw, "export")

    base_dir_value = run_cfg.get("output_dir", run_cfg.get("base_dir"))
    if not base_dir_value:
        raise ValueError("run.output_dir is required")
    base_output_dir = _resolve_path(base_dir_value, base_dir=config_dir)
    iteration_name = str(run_cfg.get("iteration_name") or run_cfg.get("name") or "iteration1").strip()
    if not iteration_name:
        raise ValueError("run.iteration_name must not be empty")
    version_prefix = str(run_cfg.get("version_prefix") or "v").strip()
    if not version_prefix:
        raise ValueError("run.version_prefix must not be empty")
    resume_iteration_dir = _resolve_resume_iteration_dir(
        run_cfg.get("resume_iteration"),
        base_output_dir=base_output_dir,
    )
    resume_enabled = resume_iteration_dir is not None
    if resume_enabled:
        iteration_dir = resume_iteration_dir
        version_dir = iteration_dir.parent
        if not iteration_dir.exists():
            raise FileNotFoundError(f"resume iteration dir does not exist: {iteration_dir}")
    else:
        version_dir = _next_version_dir(base_output_dir, version_prefix)
        iteration_dir = version_dir / iteration_name

    shared_root_dir = _resolve_optional_path(shared_cfg.get("root_dir"), base_dir=config_dir)
    if shared_root_dir is None:
        shared_root_dir = config_dir.parent
    summarize_script_path = shared_root_dir / "scripts" / "summarize_mcts.py"
    shared_eval_dir = _resolve_optional_path(shared_cfg.get("evaluation_dir"), base_dir=config_dir)
    if shared_eval_dir is None:
        raise ValueError("shared.evaluation_dir is required")
    from sciorch.model_list import resolve_model_list
    shared_candidate_models = resolve_model_list(shared_cfg.get("candidate_models"))

    enabled_steps = {
        name: _as_bool((_ensure_mapping(steps_cfg, name)).get("enabled"), True)
        for name in STEP_ORDER
    }
    execution_mode = str(steps_cfg.get("execution_mode", "stage") or "stage").strip().lower()
    if execution_mode not in {"stage", "tree"}:
        raise ValueError("steps.execution_mode must be one of: stage, tree")

    generated_cfg_dir = iteration_dir / "_generated_configs"
    if not args.dry_run:
        generated_cfg_dir.mkdir(parents=True, exist_ok=resume_enabled)

    reasoning_payload = copy.deepcopy(reasoning_cfg)
    reasoning_orchestra_model = str(
        reasoning_payload.get("orchestra_model")
        or shared_cfg.get("orchestra_model")
        or ""
    ).strip()
    if not reasoning_orchestra_model:
        raise ValueError("reasoning.orchestra_model (or shared.orchestra_model) is required")
    reasoning_payload["orchestra_model"] = reasoning_orchestra_model
    reasoning_payload["output_dir"] = str(iteration_dir / "reasoning")
    if resume_enabled:
        reasoning_payload["resume"] = True
    reasoning_payload["candidate_models"] = _collect_candidate_models(
        reasoning_payload.get("candidate_models"),
        shared_candidate_models,
    )
    model_dir_name = _sanitize_model_dir_name(reasoning_orchestra_model)
    reasoning_run_dir = Path(reasoning_payload["output_dir"]) / model_dir_name
    reasoning_config_path = generated_cfg_dir / "reasoning.generated.yaml"
    if not args.dry_run:
        _write_yaml(reasoning_config_path, reasoning_payload)

    expand_payload = copy.deepcopy(expand_cfg)
    expand_run_dir = iteration_dir / "expand" / model_dir_name
    expand_pool_dir = iteration_dir / "expand" / "model_pools"
    expand_payload["base_run_dir"] = str(reasoning_run_dir)
    expand_payload["run_dir"] = str(reasoning_run_dir)
    expand_payload["pool_output_dir"] = str(expand_pool_dir)
    expand_payload["pool_summary_path"] = str(expand_pool_dir / "summary.json")
    expand_payload["evaluation_dir"] = str(
        (_resolve_optional_path(expand_payload.get("evaluation_dir"), base_dir=config_dir) or shared_eval_dir)
    )
    expand_payload["candidate_models"] = _collect_candidate_models(
        expand_payload.get("candidate_models"),
        shared_candidate_models,
    )
    expand_config_path = generated_cfg_dir / "expand.generated.yaml"
    if not args.dry_run:
        _write_yaml(expand_config_path, expand_payload)

    prune_output_override = _resolve_optional_path(
        prune_cfg.get("output_run_dir"),
        base_dir=config_dir,
    )
    prune_output_dir = prune_output_override or (iteration_dir / "pruned")
    prune_args = copy.deepcopy(prune_cfg)
    if "output_run_dir" in prune_args:
        del prune_args["output_run_dir"]

    export_payload = copy.deepcopy(export_cfg)
    export_output_dir = _resolve_optional_path(
        export_payload.pop("output_dir", None),
        base_dir=config_dir,
    ) or (iteration_dir / "msswift_export")
    export_images_output_dir = _resolve_optional_path(
        export_payload.pop("images_output_dir", None),
        base_dir=config_dir,
    ) or (export_output_dir / "images")
    export_prompt_config_path = _resolve_optional_path(
        export_payload.pop("prompt_config", None),
        base_dir=config_dir,
    ) or (shared_root_dir / "configs" / "reasoning.yaml")
    export_args: dict[str, Any] = {
        "dataset_type": "ppo",
        "completion_mode": "full",
        "record_style": "minimal",
        "prompt_source": "template",
        "prompt_config": str(export_prompt_config_path),
        "filter_fallback": True,
        "actor": "main",
        "include_self_delegate_with_main": True,
        "resolve_images": "auto",
        "image_storage_mode": "file_paths",
        "image_path_style": "absolute",
        "images_output_dir": str(export_images_output_dir),
        "overwrite_images_dir": True,
        "output_dir": str(export_output_dir),
    }
    for key, value in export_payload.items():
        export_args[str(key)] = value

    # export_msswift reads dataset_name/split from run dir config or CLI.
    # Pruned dirs lack dataset fields in config.snapshot.json — inherit from reasoning (same data as MCTS).
    def _export_ds_nonempty(key: str) -> bool:
        return bool(str(export_args.get(key) or "").strip())

    if not _export_ds_nonempty("dataset_name"):
        name = reasoning_cfg.get("dataset_name") or shared_cfg.get("dataset_name")
        if name:
            export_args["dataset_name"] = str(name).strip()
    if not _export_ds_nonempty("dataset_split"):
        split = reasoning_cfg.get("dataset_split") or shared_cfg.get("dataset_split")
        if split:
            export_args["dataset_split"] = str(split).strip()

    start_from_raw = str(
        steps_cfg.get("start_from") or ("auto" if resume_enabled else "reasoning")
    ).strip().lower()
    auto_resume_reason: str | None = None
    if start_from_raw == "auto" and not resume_enabled:
        start_from = "reasoning"
        step_sequence = _ensure_step_order(start_from)
        auto_resume_reason = "disabled_without_resume_iteration"
    elif resume_enabled and start_from_raw == "auto":
        detected_start_from, auto_resume_reason = _detect_resume_start_from(
            reasoning_run_dir=reasoning_run_dir,
            prune_output_dir=prune_output_dir,
        )
        if detected_start_from is None:
            start_from = "prune"
            step_sequence: list[str] = []
        else:
            start_from = detected_start_from
            step_sequence = _ensure_step_order(start_from)
    else:
        start_from = start_from_raw
        step_sequence = _ensure_step_order(start_from)

    default_tree_concurrency = int(
        steps_cfg.get(
            "tree_max_concurrency",
            reasoning_payload.get("max_concurrency") or expand_payload.get("max_concurrency") or 1,
        )
    )
    tree_max_concurrency = max(1, default_tree_concurrency)

    plan_payload = {
        "generated_at": _now_utc(),
        "source_config": str(config_path),
        "iteration_dir": str(iteration_dir),
        "steps_enabled": enabled_steps,
        "execution_mode": execution_mode,
        "tree_max_concurrency": tree_max_concurrency,
        "start_from": start_from,
        "requested_start_from": start_from_raw,
        "auto_resume_reason": auto_resume_reason,
        "resume_iteration_dir": str(iteration_dir) if resume_enabled else None,
        "generated_configs": {
            "reasoning": str(reasoning_config_path),
            "expand": str(expand_config_path),
        },
        "paths": {
            "reasoning_run_dir": str(reasoning_run_dir),
            "expand_run_dir": str(expand_run_dir),
            "expand_pool_summary": str(Path(expand_payload["pool_summary_path"]).resolve()),
            "prune_output_dir": str(prune_output_dir),
            "export_output_dir": str(export_output_dir),
        },
    }
    if not args.dry_run:
        _write_json(generated_cfg_dir / "generate.plan.json", plan_payload)

    print("=== Generate iteration plan ===")
    print(f"base_output_dir: {base_output_dir}")
    print(f"version_dir: {version_dir}")
    print(f"iteration_dir: {iteration_dir}")
    print(f"resume_enabled: {resume_enabled}")
    print(f"requested_start_from: {start_from_raw}")
    if auto_resume_reason:
        print(f"auto_resume_reason: {auto_resume_reason}")
    print(f"effective_start_from: {start_from}")
    print(f"execution_mode: {execution_mode}")
    print(f"tree_max_concurrency: {tree_max_concurrency}")
    print(f"reasoning_run_dir: {reasoning_run_dir}")
    print(f"expand_run_dir: {expand_run_dir}")
    print(f"prune_output_dir: {prune_output_dir}")
    print(f"export_output_dir: {export_output_dir}")
    print(f"dry_run: {args.dry_run}")
    print("===============================")

    current_run_dir_for_downstream = (
        expand_run_dir if execution_mode == "tree" and enabled_steps.get("expand", True) else reasoning_run_dir
    )
    tree_pipeline_should_run = (
        execution_mode == "tree"
        and any(step_name in {"reasoning", "expand"} for step_name in step_sequence)
    )

    if tree_pipeline_should_run:
        tree_task_manifest_path = generated_cfg_dir / "tree_task_ids.json"
        task_ids = _resolve_tree_task_ids(
            manifest_path=tree_task_manifest_path,
            reasoning_payload=reasoning_payload,
            reasoning_config_path=reasoning_config_path,
        )
        if not task_ids:
            raise ValueError("No task_ids resolved for tree execution mode.")

        if not args.dry_run:
            plan_payload["tree_task_ids"] = task_ids
            _write_json(generated_cfg_dir / "generate.plan.json", plan_payload)

        legacy_materialization = _materialize_legacy_tree_reasoning_runs(
            aggregate_reasoning_run_dir=reasoning_run_dir,
            iteration_dir=iteration_dir,
            model_dir_name=model_dir_name,
            reasoning_payload=reasoning_payload,
            task_ids=task_ids,
            dry_run=args.dry_run,
        )

        tree_start_phase = "expand" if step_sequence and step_sequence[0] == "expand" else "reasoning"
        expand_enabled_in_pipeline = enabled_steps.get("expand", True)

        # Pre-build global pool summary once so per-task expand configs find it
        if expand_enabled_in_pipeline and not args.dry_run:
            _pool_path = Path(expand_payload["pool_summary_path"])
            if not _pool_path.exists():
                _pool_path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    import sys as _sys
                    _sys.path.insert(0, str(Path(__file__).parent))
                    from expand_mcts import load_stage_specs as _lss, ensure_pool_summary_exists as _eps
                    # If pool_base_run_dir is configured, write a separate pool-build config using that
                    # as base_run_dir so the pool can be built before reasoning output exists.
                    _pool_base_run_dir = _resolve_optional_path(
                        expand_cfg.get("pool_base_run_dir"), base_dir=config_dir
                    )
                    if _pool_base_run_dir is not None:
                        import copy as _copy
                        _pool_build_payload = _copy.deepcopy(expand_payload)
                        _pool_build_payload["base_run_dir"] = str(_pool_base_run_dir)
                        _pool_build_config_path = generated_cfg_dir / "pool_build.generated.yaml"
                        _write_yaml(_pool_build_config_path, _pool_build_payload)
                        _pool_config_path = _pool_build_config_path
                    else:
                        _pool_config_path = expand_config_path
                    _eps(config_path=_pool_config_path, pool_summary_path=_pool_path, active_stage_specs=_lss(expand_payload))
                except Exception as _e:
                    import json as _json
                    _pool_path.write_text(_json.dumps({"tasks": [], "source": "empty_fallback", "reason": str(_e)}, indent=2))
                    print(f"[pool] no evaluation data yet, using empty pool (fallback to random): {_e}", flush=True)

        pipeline_summary = _run_tree_pipeline(
            task_ids=task_ids,
            tree_max_concurrency=tree_max_concurrency,
            iteration_dir=iteration_dir,
            model_dir_name=model_dir_name,
            reasoning_payload=reasoning_payload,
            expand_payload=expand_payload,
            reasoning_aggregate_run_dir=reasoning_run_dir,
            expand_aggregate_run_dir=expand_run_dir,
            shared_root_dir=shared_root_dir,
            generated_cfg_dir=generated_cfg_dir,
            tree_start_phase=tree_start_phase,
            expand_enabled=expand_enabled_in_pipeline,
            dry_run=args.dry_run,
        )
        print("=== Tree pipeline summary ===")
        print(f"tree_count: {pipeline_summary['tree_count']}")
        print(f"completed_tree_count: {pipeline_summary['completed_tree_count']}")
        print(f"total_cost: ${pipeline_summary['total_cost']:.6f}")
        print(f"tree_summary_path: {pipeline_summary['tree_summary_path']}")
        print("============================")

        if enabled_steps.get("reasoning", True):
            _summarize_and_visualize(
                script_path=summarize_script_path,
                run_dir=reasoning_run_dir,
                viz_dir=iteration_dir / "visualizations" / "reasoning",
                json_path=iteration_dir / "visualizations" / "reasoning_summary.json",
                dry_run=args.dry_run,
            )
        if expand_enabled_in_pipeline:
            _summarize_and_visualize(
                script_path=summarize_script_path,
                run_dir=expand_run_dir,
                viz_dir=iteration_dir / "visualizations" / "expand",
                json_path=iteration_dir / "visualizations" / "expand_summary.json",
                dry_run=args.dry_run,
            )
            current_run_dir_for_downstream = expand_run_dir
        else:
            current_run_dir_for_downstream = reasoning_run_dir

        step_sequence = [step_name for step_name in step_sequence if step_name not in {"reasoning", "expand"}]

    for step_name in step_sequence:
        if not enabled_steps.get(step_name, True):
            continue

        if execution_mode == "stage" and step_name == "reasoning":
            command = [
                sys.executable,
                "-m",
                "mcts.cli.run_mcts_reasoning",
                "--config",
                str(reasoning_config_path),
            ]
            _run_command(command, dry_run=args.dry_run)
            _summarize_and_visualize(
                script_path=summarize_script_path,
                run_dir=reasoning_run_dir,
                viz_dir=iteration_dir / "visualizations" / "reasoning",
                json_path=iteration_dir / "visualizations" / "reasoning_summary.json",
                dry_run=args.dry_run,
            )
            current_run_dir_for_downstream = reasoning_run_dir
            continue

        if execution_mode == "stage" and step_name == "expand":
            command = [
                sys.executable,
                str(shared_root_dir / "scripts" / "expand_mcts.py"),
                "--config",
                str(expand_config_path),
            ]
            _run_command(command, dry_run=args.dry_run)
            _summarize_and_visualize(
                script_path=summarize_script_path,
                run_dir=reasoning_run_dir,
                viz_dir=iteration_dir / "visualizations" / "expand",
                json_path=iteration_dir / "visualizations" / "expand_summary.json",
                dry_run=args.dry_run,
            )
            current_run_dir_for_downstream = reasoning_run_dir
            continue

        if step_name == "prune":
            command = [
                sys.executable,
                str(shared_root_dir / "scripts" / "prune_mcts.py"),
                "--input-run-dir",
                str(current_run_dir_for_downstream),
                "--output-run-dir",
                str(prune_output_dir),
            ]
            for key, value in prune_args.items():
                if key in {"input_run_dir", "output_run_dir"}:
                    continue
                if (
                    str(prune_args.get("selection_mode", "")).strip().lower() != "sample"
                    and key in {"sample_uniform_mix", "sample_temperature", "sample_seed"}
                ):
                    continue
                flag = "--" + str(key).replace("_", "-")
                if isinstance(value, bool):
                    if value:
                        command.append(flag)
                    continue
                if value is None:
                    continue
                command.extend([flag, str(value)])
            _run_command(command, dry_run=args.dry_run)
            _summarize_and_visualize(
                script_path=summarize_script_path,
                run_dir=prune_output_dir,
                viz_dir=iteration_dir / "visualizations" / "prune",
                json_path=iteration_dir / "visualizations" / "prune_summary.json",
                dry_run=args.dry_run,
            )
            continue

        if step_name == "export":
            command = [
                sys.executable,
                str(shared_root_dir / "scripts" / "export_msswift.py"),
                str(prune_output_dir),
            ]
            for key, value in export_args.items():
                flag = "--" + str(key).replace("_", "-")
                if isinstance(value, bool):
                    if value:
                        command.append(flag)
                    continue
                if value is None:
                    continue
                command.extend([flag, str(value)])
            _run_command(command, dry_run=args.dry_run)
            continue

    print("Generate iteration finished.")
    print(f"plan_file: {generated_cfg_dir / 'generate.plan.json'}")
    print(f"pruned_output: {prune_output_dir}")
    print(f"export_output: {export_output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
