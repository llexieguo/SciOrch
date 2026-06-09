#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mcts.runner import SearchNode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy an MCTS run directory, then prune each node to at most N children by "
            "descending descendant expected accuracy."
        )
    )
    parser.add_argument("--input-run-dir", required=True, type=Path, help="Original run directory.")
    parser.add_argument("--output-run-dir", required=True, type=Path, help="New copied-and-pruned run directory.")
    parser.add_argument("--max-children", type=int, default=8, help="Maximum children to keep per node.")
    parser.add_argument(
        "--spread-max-children",
        type=int,
        default=8,
        help="For selection-mode=spread, cap dynamic keep count at this value.",
    )
    parser.add_argument(
        "--selection-mode",
        choices=("topk", "sample", "spread"),
        default="topk",
        help="How to choose kept children when a parent has more than max_children.",
    )
    parser.add_argument(
        "--sample-uniform-mix",
        type=float,
        default=0.35,
        help="For selection-mode=sample, mix this much uniform probability into reward-based sampling.",
    )
    parser.add_argument(
        "--sample-temperature",
        type=float,
        default=1.0,
        help="For selection-mode=sample, larger means flatter reward-based sampling.",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=42,
        help="Random seed used by selection-mode=sample.",
    )
    parser.add_argument(
        "--drop-error-terminal-nodes",
        action="store_true",
        help="Remove terminal error nodes (for example API/format failures) before child pruning.",
    )
    parser.add_argument(
        "--drop-invalid-submit-nodes",
        action="store_true",
        help="Remove terminal submit nodes with empty boxed answers before child pruning.",
    )
    parser.add_argument(
        "--drop-all-wrong-trees",
        action="store_true",
        help="After pruning, delete sample dirs whose pruned tree still has zero correct final leaves.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow deleting the output directory first if it already exists.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    content = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    path.write_text(content, encoding="utf-8")


def ensure_output_dir(input_run_dir: Path, output_run_dir: Path, overwrite: bool) -> None:
    input_resolved = input_run_dir.resolve()
    output_resolved = output_run_dir.resolve()
    if input_resolved == output_resolved:
        raise ValueError("--output-run-dir must be different from --input-run-dir")
    if output_resolved.is_relative_to(input_resolved):
        raise ValueError("--output-run-dir cannot be inside --input-run-dir")
    if output_run_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Output directory already exists: {output_run_dir}")
        shutil.rmtree(output_run_dir)


def iter_sample_dirs(run_dir: Path) -> list[Path]:
    samples_dir = run_dir / "samples"
    return sorted(path.parent for path in samples_dir.rglob("latest.json"))


def descendant_reward_stats_by_node(nodes_by_id: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
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
            else:
                terminal_count = 0
                correct_count = 0
            stats = {
                "expected_acc": (correct_count / terminal_count) if terminal_count > 0 else 0.0,
                "terminal_count": terminal_count,
                "correct_count": correct_count,
            }
            memo[node_id] = stats
            return stats

        terminal_count = 0
        correct_count = 0
        for child_id in child_ids:
            child_stats = resolve(child_id)
            terminal_count += int(child_stats["terminal_count"])
            correct_count += int(child_stats["correct_count"])
        stats = {
            "expected_acc": (correct_count / terminal_count) if terminal_count > 0 else 0.0,
            "terminal_count": terminal_count,
            "correct_count": correct_count,
        }
        memo[node_id] = stats
        return stats

    for node_id in nodes_by_id:
        resolve(node_id)
    return memo


def is_error_terminal_node(node: dict[str, Any]) -> bool:
    if not isinstance(node, dict):
        return False
    if not bool(node.get("is_terminal")):
        return False
    action = str(node.get("action") or "")
    if action == "error":
        return True
    return bool(node.get("error"))


def is_invalid_submit_terminal_node(node: dict[str, Any]) -> bool:
    if not isinstance(node, dict):
        return False
    if not bool(node.get("is_terminal")):
        return False
    if str(node.get("action") or "") != "submit":
        return False
    return not bool(node.get("boxed_letter"))


def drop_terminal_noise_nodes_from_tree(
    nodes_by_id: dict[str, dict[str, Any]],
    *,
    drop_error_terminal_nodes: bool,
    drop_invalid_submit_nodes: bool,
) -> tuple[dict[str, dict[str, Any]], list[str], list[str]]:
    dropped_error_ids: list[str] = []
    dropped_invalid_submit_ids: list[str] = []
    for node_id, node in nodes_by_id.items():
        if node_id == "root":
            continue
        if drop_error_terminal_nodes and is_error_terminal_node(node):
            dropped_error_ids.append(node_id)
            continue
        if drop_invalid_submit_nodes and is_invalid_submit_terminal_node(node):
            dropped_invalid_submit_ids.append(node_id)
    dropped_set = set(dropped_error_ids) | set(dropped_invalid_submit_ids)
    if not dropped_set:
        return nodes_by_id, [], []
    filtered: dict[str, dict[str, Any]] = {}
    for node_id, node in nodes_by_id.items():
        if node_id in dropped_set:
            continue
        updated = dict(node)
        updated["children_ids"] = [
            str(child_id)
            for child_id in node.get("children_ids", [])
            if str(child_id) not in dropped_set and str(child_id) in nodes_by_id
        ]
        filtered[node_id] = updated
    return filtered, dropped_error_ids, dropped_invalid_submit_ids



def _sample_child_ids(
    child_ids: list[str],
    expected_stats: dict[str, dict[str, Any]],
    *,
    keep_count: int,
    uniform_mix: float,
    temperature: float,
    rng: random.Random,
) -> list[str]:
    remaining = list(child_ids)
    selected: list[str] = []
    keep_count = min(max(0, keep_count), len(remaining))
    uniform_mix = min(max(float(uniform_mix), 0.0), 1.0)
    temperature = max(float(temperature), 1e-6)

    while remaining and len(selected) < keep_count:
        reward_weights: list[float] = []
        for child_id in remaining:
            expected_acc = max(0.0, float(expected_stats.get(child_id, {}).get("expected_acc", 0.0)))
            reward_weights.append(math.exp(expected_acc / temperature) if expected_acc > 0 else 1.0)
        reward_total = sum(reward_weights)
        n = len(remaining)
        weights: list[float] = []
        for reward_weight in reward_weights:
            reward_prob = (reward_weight / reward_total) if reward_total > 0 else (1.0 / n)
            weights.append((1.0 - uniform_mix) * reward_prob + uniform_mix * (1.0 / n))
        total = sum(weights)
        probs = [w / total for w in weights] if total > 0 else [1.0 / n] * n
        r = rng.random()
        cumulative = 0.0
        picked_index = len(remaining) - 1
        for idx, prob in enumerate(probs):
            cumulative += prob
            if r <= cumulative:
                picked_index = idx
                break
        selected.append(remaining.pop(picked_index))
    return selected



def _spread_child_ids(
    child_ids: list[str],
    expected_stats: dict[str, dict[str, Any]],
    *,
    keep_count: int,
) -> list[str]:
    ranked_child_ids = sorted(
        child_ids,
        key=lambda child_id: (
            -float(expected_stats.get(child_id, {}).get("expected_acc", 0.0)),
            -int(expected_stats.get(child_id, {}).get("correct_count", 0)),
            -int(expected_stats.get(child_id, {}).get("terminal_count", 0)),
            child_id,
        ),
    )
    keep_count = min(max(0, keep_count), len(ranked_child_ids))
    if keep_count == 0:
        return []
    if len(ranked_child_ids) < 4 or keep_count >= len(ranked_child_ids):
        return ranked_child_ids[:keep_count]

    selected: list[str] = [ranked_child_ids[0]]
    if keep_count > 1:
        selected.append(ranked_child_ids[-1])

    selected_set = set(selected)
    while len(selected) < keep_count:
        best_child_id: str | None = None
        best_score: tuple[float, float, float, float, str] | None = None
        selected_accs = [float(expected_stats.get(child_id, {}).get("expected_acc", 0.0)) for child_id in selected]
        for child_id in ranked_child_ids:
            if child_id in selected_set:
                continue
            child_acc = float(expected_stats.get(child_id, {}).get("expected_acc", 0.0))
            gaps = [abs(child_acc - acc) for acc in selected_accs]
            min_gap = min(gaps) if gaps else 0.0
            mean_gap = (sum(gaps) / len(gaps)) if gaps else 0.0
            score = (
                min_gap,
                mean_gap,
                child_acc,
                float(expected_stats.get(child_id, {}).get("terminal_count", 0)),
                child_id,
            )
            if best_score is None or score > best_score:
                best_score = score
                best_child_id = child_id
        if best_child_id is None:
            break
        selected.append(best_child_id)
        selected_set.add(best_child_id)

    selected_order = {child_id: idx for idx, child_id in enumerate(selected)}
    return sorted(
        selected,
        key=lambda child_id: (
            -float(expected_stats.get(child_id, {}).get("expected_acc", 0.0)),
            selected_order.get(child_id, len(selected_order)),
            child_id,
        ),
    )


def _dynamic_spread_keep_count(
    child_ids: list[str],
    expected_stats: dict[str, dict[str, Any]],
    *,
    min_keep: int,
    max_keep: int | None = None,
) -> tuple[int, float, float]:
    n = len(child_ids)
    min_keep = min(max(0, int(min_keep)), n)
    cap_keep = n if max_keep is None else min(max(int(max_keep), min_keep), n)
    if n <= min_keep:
        return n, 0.0, 0.0

    expected_accs = [float(expected_stats.get(child_id, {}).get("expected_acc", 0.0)) for child_id in child_ids]
    if not expected_accs:
        return min_keep, 0.0, 0.0

    mean_acc = sum(expected_accs) / len(expected_accs)
    variance = sum((value - mean_acc) ** 2 for value in expected_accs) / len(expected_accs)
    score = min(max(math.sqrt(variance / 0.25) if variance > 0 else 0.0, 0.0), 1.0)
    keep_count = min_keep + round((cap_keep - min_keep) * score)
    keep_count = min(max(min_keep, keep_count), cap_keep)
    return keep_count, variance, score


def prune_nodes(
    nodes_by_id: dict[str, dict[str, Any]],
    max_children: int,
    *,
    selection_mode: str = "topk",
    spread_max_children: int | None = None,
    sample_uniform_mix: float = 0.35,
    sample_temperature: float = 1.0,
    rng: random.Random | None = None,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], int]:
    expected_stats = descendant_reward_stats_by_node(nodes_by_id)
    keep_child_ids_by_parent: dict[str, list[str]] = {}
    truncation_rows: list[dict[str, Any]] = []

    for node_id, node in nodes_by_id.items():
        child_ids = [str(child_id) for child_id in node.get("children_ids", []) if str(child_id) in nodes_by_id]
        if selection_mode == "spread" and len(child_ids) < 4:
            keep_child_ids_by_parent[node_id] = child_ids
            continue
        if len(child_ids) <= max_children:
            keep_child_ids_by_parent[node_id] = child_ids
            continue

        ranked_child_ids = sorted(
            child_ids,
            key=lambda child_id: (
                -float(expected_stats.get(child_id, {}).get("expected_acc", 0.0)),
                -int(expected_stats.get(child_id, {}).get("correct_count", 0)),
                -int(expected_stats.get(child_id, {}).get("terminal_count", 0)),
                child_id,
            ),
        )
        effective_selection_mode = selection_mode
        dynamic_keep_count: int | None = None
        spread_variance: float | None = None
        spread_score: float | None = None
        if selection_mode == "sample":
            if rng is None:
                raise ValueError("rng is required when selection_mode=sample")
            kept = _sample_child_ids(
                ranked_child_ids,
                expected_stats,
                keep_count=max_children,
                uniform_mix=sample_uniform_mix,
                temperature=sample_temperature,
                rng=rng,
            )
            kept_set = set(kept)
            dropped = [child_id for child_id in ranked_child_ids if child_id not in kept_set]
        elif selection_mode == "spread":
            if len(ranked_child_ids) >= 4:
                dynamic_keep_count, spread_variance, spread_score = _dynamic_spread_keep_count(
                    ranked_child_ids,
                    expected_stats,
                    min_keep=max_children,
                    max_keep=spread_max_children,
                )
                kept = _spread_child_ids(
                    ranked_child_ids,
                    expected_stats,
                    keep_count=dynamic_keep_count,
                )
                effective_selection_mode = "spread_dynamic"
                kept_set = set(kept)
                dropped = [child_id for child_id in ranked_child_ids if child_id not in kept_set]
            else:
                effective_selection_mode = "topk_fallback"
                kept = ranked_child_ids[:max_children]
                dropped = ranked_child_ids[max_children:]
        else:
            kept = ranked_child_ids[:max_children]
            dropped = ranked_child_ids[max_children:]
        keep_child_ids_by_parent[node_id] = kept
        truncation_rows.append(
            {
                "node_id": node_id,
                "selection_mode": effective_selection_mode,
                "original_child_count": len(child_ids),
                "kept_child_count": len(kept),
                "dropped_child_count": len(dropped),
                "dynamic_keep_count": dynamic_keep_count,
                "spread_variance": spread_variance,
                "spread_score": spread_score,
                "kept_children": [
                    {
                        "node_id": child_id,
                        **expected_stats.get(child_id, {"expected_acc": 0.0, "correct_count": 0, "terminal_count": 0}),
                    }
                    for child_id in kept
                ],
                "dropped_children": [
                    {
                        "node_id": child_id,
                        **expected_stats.get(child_id, {"expected_acc": 0.0, "correct_count": 0, "terminal_count": 0}),
                    }
                    for child_id in dropped
                ],
            }
        )

    reachable_ids: set[str] = set()

    def walk(node_id: str) -> None:
        if node_id in reachable_ids or node_id not in nodes_by_id:
            return
        reachable_ids.add(node_id)
        for child_id in keep_child_ids_by_parent.get(node_id, []):
            walk(child_id)

    walk("root")
    pruned_nodes_by_id: dict[str, dict[str, Any]] = {}
    removed_node_count = 0
    for node_id, node in nodes_by_id.items():
        if node_id not in reachable_ids:
            removed_node_count += 1
            continue
        updated = dict(node)
        updated["children_ids"] = keep_child_ids_by_parent.get(node_id, [])
        pruned_nodes_by_id[node_id] = updated

    return pruned_nodes_by_id, truncation_rows, removed_node_count


def reconstruct_path(node: SearchNode, all_nodes: dict[str, SearchNode]) -> list[SearchNode]:
    path: list[SearchNode] = []
    current: SearchNode | None = node
    while current is not None:
        path.append(current)
        if current.parent_id is None:
            break
        current = all_nodes.get(current.parent_id)
    path.reverse()
    return path


def latest_delegate_confidence(path: list[SearchNode]) -> float:
    for node in reversed(path):
        if node.action == "delegate" and node.delegate_parse_ok and node.delegate_confidence is not None:
            return float(node.delegate_confidence)
    return 0.0


def normalize_metric(values: dict[str, float]) -> dict[str, float]:
    if not values:
        return {}
    minimum = min(values.values())
    maximum = max(values.values())
    if minimum == maximum:
        return {key: 0.0 for key in values}
    scale = maximum - minimum
    return {key: (value - minimum) / scale for key, value in values.items()}


def rank_nodes(nodes: list[SearchNode], all_nodes: dict[str, SearchNode]) -> list[SearchNode]:
    metrics = {
        node.node_id: {
            "latest_delegate_confidence": latest_delegate_confidence(reconstruct_path(node, all_nodes)),
            "trajectory_cost": sum(item.cost for item in reconstruct_path(node, all_nodes)),
            "trajectory_tokens": float(
                sum(item.input_tokens + item.output_tokens for item in reconstruct_path(node, all_nodes))
            ),
        }
        for node in nodes
    }
    conf_norm = normalize_metric({node_id: row["latest_delegate_confidence"] for node_id, row in metrics.items()})
    cost_norm = normalize_metric({node_id: row["trajectory_cost"] for node_id, row in metrics.items()})
    token_norm = normalize_metric({node_id: row["trajectory_tokens"] for node_id, row in metrics.items()})
    return sorted(
        nodes,
        key=lambda node: (
            -(
                conf_norm[node.node_id]
                - cost_norm[node.node_id]
                - token_norm[node.node_id]
            ),
            node.node_id,
        ),
    )


def build_leaf_trajectories(final_leaves: list[SearchNode], all_nodes: dict[str, SearchNode]) -> list[dict[str, Any]]:
    ranked_leaves = rank_nodes(final_leaves, all_nodes)
    trajectories: list[dict[str, Any]] = []
    metrics = {
        node.node_id: {
            "path": reconstruct_path(node, all_nodes),
        }
        for node in ranked_leaves
    }
    for index, leaf in enumerate(ranked_leaves, start=1):
        path = metrics[leaf.node_id]["path"]
        trajectories.append(
            {
                "trajectory_index": index,
                "leaf_node_id": leaf.node_id,
                "node_ids": [item.node_id for item in path],
                "depth": leaf.depth,
                "boxed_letter": leaf.boxed_letter,
                "latest_delegate_confidence": latest_delegate_confidence(path),
                "selection_score": None,
                "trajectory_cost": sum(item.cost for item in path),
                "trajectory_tokens": float(sum(item.input_tokens + item.output_tokens for item in path)),
                "final_answer_text": leaf.final_answer_text,
                "correct": leaf.is_correct,
                "actions": [item.action for item in path[1:]],
            }
        )

    if trajectories:
        score_inputs = {item["leaf_node_id"]: item for item in trajectories}
        conf_norm = normalize_metric(
            {
                leaf_id: float(item["latest_delegate_confidence"])
                for leaf_id, item in score_inputs.items()
            }
        )
        cost_norm = normalize_metric(
            {
                leaf_id: float(item["trajectory_cost"])
                for leaf_id, item in score_inputs.items()
            }
        )
        token_norm = normalize_metric(
            {
                leaf_id: float(item["trajectory_tokens"])
                for leaf_id, item in score_inputs.items()
            }
        )
        for item in trajectories:
            leaf_id = str(item["leaf_node_id"])
            item["selection_score"] = conf_norm[leaf_id] - cost_norm[leaf_id] - token_norm[leaf_id]

        trajectories.sort(key=lambda item: (-float(item["selection_score"]), str(item["leaf_node_id"])))
        for index, item in enumerate(trajectories, start=1):
            item["trajectory_index"] = index
    return trajectories


def majority_letter(trajectories: list[dict[str, Any]]) -> str | None:
    votes = Counter(str(item.get("boxed_letter")) for item in trajectories if item.get("boxed_letter"))
    if not votes:
        return None
    return sorted(votes.items(), key=lambda item: (-item[1], item[0]))[0][0]


def infer_stop_reason(result: dict[str, Any], latest: dict[str, Any], final_leaf_count: int, open_leaf_count: int) -> str:
    old_stop_reason = str(result.get("stop_reason") or "")
    if final_leaf_count == 0 and open_leaf_count == 0:
        return "no_reachable_leaves_after_prune"
    if old_stop_reason in {"leaf_acc_threshold_reached", "max_final_leaf_count_reached", "budget_exhausted_no_reopen_candidates"}:
        return old_stop_reason
    if open_leaf_count == 0:
        return "all_leaves_finished"
    return old_stop_reason or str(latest.get("final_summary", {}).get("stop_reason") or "completed")


def update_rounds(rounds: list[dict[str, Any]], reachable_ids: set[str], final_leaf_ids: set[str], failed_terminal_ids: set[str]) -> list[dict[str, Any]]:
    updated_rounds: list[dict[str, Any]] = []
    seen_final_leaf_ids: set[str] = set()
    seen_failed_terminal_ids: set[str] = set()
    for raw_round in rounds:
        round_row = dict(raw_round)
        created_node_ids = [str(node_id) for node_id in round_row.get("created_node_ids", []) if str(node_id) in reachable_ids]
        created_final_leaf_node_ids = [str(node_id) for node_id in round_row.get("created_final_leaf_node_ids", []) if str(node_id) in final_leaf_ids]
        created_failed_terminal_node_ids = [
            str(node_id) for node_id in round_row.get("created_failed_terminal_node_ids", []) if str(node_id) in failed_terminal_ids
        ]
        created_expandable_node_ids = [
            str(node_id) for node_id in round_row.get("created_expandable_node_ids", []) if str(node_id) in reachable_ids
        ]
        next_frontier_node_ids = [str(node_id) for node_id in round_row.get("next_frontier_node_ids", []) if str(node_id) in reachable_ids]
        selected_parent_ids = [str(node_id) for node_id in round_row.get("selected_parent_ids", []) if str(node_id) in reachable_ids]

        round_row["created_node_ids"] = created_node_ids
        round_row["created_final_leaf_node_ids"] = created_final_leaf_node_ids
        round_row["created_failed_terminal_node_ids"] = created_failed_terminal_node_ids
        round_row["created_expandable_node_ids"] = created_expandable_node_ids
        round_row["next_frontier_node_ids"] = next_frontier_node_ids
        round_row["selected_parent_ids"] = selected_parent_ids
        round_row["children_created"] = len(created_node_ids)
        round_row["selected_parent_count"] = len(selected_parent_ids)
        round_row["expanded_parent_count"] = len(selected_parent_ids)
        round_row["next_frontier_count_requested"] = len(next_frontier_node_ids)
        round_row["open_leaf_count_after"] = len(next_frontier_node_ids)
        seen_final_leaf_ids.update(created_final_leaf_node_ids)
        seen_failed_terminal_ids.update(created_failed_terminal_node_ids)
        round_row["final_leaf_count_after"] = len(seen_final_leaf_ids)
        round_row["failed_terminal_count_after"] = len(seen_failed_terminal_ids)

        updated_rounds.append(round_row)
    return updated_rounds


def rebuild_sample(
    sample_dir: Path,
    max_children: int,
    *,
    selection_mode: str = "topk",
    spread_max_children: int | None = None,
    sample_uniform_mix: float = 0.35,
    sample_temperature: float = 1.0,
    rng: random.Random | None = None,
    drop_error_terminal_nodes: bool = False,
    drop_invalid_submit_nodes: bool = False,
) -> dict[str, Any]:
    latest_path = sample_dir / "latest.json"
    result_path = sample_dir / "result.json"
    view_path = sample_dir / "view.json"
    nodes_jsonl_path = sample_dir / "nodes.jsonl"

    latest = load_json(latest_path)
    result = load_json(result_path) if result_path.exists() else None
    view = load_json(view_path) if view_path.exists() else None

    nodes_by_id = {}
    for item in latest.get("nodes", []):
        if not isinstance(item, dict):
            continue
        node_id = str(item.get("node_id") or "").strip()
        if node_id:
            nodes_by_id[node_id] = item
    if "root" not in nodes_by_id:
        raise ValueError(f"Missing root node in {latest_path}")

    dropped_error_node_ids: list[str] = []
    dropped_invalid_submit_node_ids: list[str] = []
    if drop_error_terminal_nodes or drop_invalid_submit_nodes:
        (
            nodes_by_id,
            dropped_error_node_ids,
            dropped_invalid_submit_node_ids,
        ) = drop_terminal_noise_nodes_from_tree(
            nodes_by_id,
            drop_error_terminal_nodes=drop_error_terminal_nodes,
            drop_invalid_submit_nodes=drop_invalid_submit_nodes,
        )

    pruned_nodes_by_id, truncation_rows, removed_node_count = prune_nodes(
        nodes_by_id,
        max_children=max_children,
        selection_mode=selection_mode,
        spread_max_children=spread_max_children,
        sample_uniform_mix=sample_uniform_mix,
        sample_temperature=sample_temperature,
        rng=rng,
    )
    removed_node_count += len(dropped_error_node_ids) + len(dropped_invalid_submit_node_ids)
    reachable_ids = set(pruned_nodes_by_id)

    all_nodes = {node_id: SearchNode.from_json(node) for node_id, node in pruned_nodes_by_id.items()}
    for node in all_nodes.values():
        node.children = [all_nodes[child_id] for child_id in pruned_nodes_by_id[node.node_id].get("children_ids", []) if child_id in all_nodes]

    open_frontier = [
        node
        for node in all_nodes.values()
        if node.node_id != "root" and node.action in {"root", "delegate"} and not node.is_terminal and not node.children
    ]
    final_leaves = [
        node
        for node in all_nodes.values()
        if node.node_id != "root" and node.is_terminal and node.action == "submit" and bool(node.boxed_letter)
    ]
    failed_terminal_nodes = [
        node
        for node in all_nodes.values()
        if node.node_id != "root" and node.is_terminal and not (node.action == "submit" and bool(node.boxed_letter))
    ]

    trajectories = build_leaf_trajectories(final_leaves, all_nodes)
    best_leaf = all_nodes.get(str(trajectories[0]["leaf_node_id"])) if trajectories else None
    majority_boxed_letter = majority_letter(trajectories)
    gold_answer_letter = str((result or {}).get("gold_answer_letter") or "")
    correct_leaf_count = sum(1 for node in final_leaves if node.is_correct)
    final_leaf_count = len(final_leaves)
    open_leaf_count = len(open_frontier)
    failed_terminal_count = len(failed_terminal_nodes)
    leaf_acc = (correct_leaf_count / final_leaf_count) if final_leaf_count > 0 else 0.0
    success = correct_leaf_count > 0
    majority_correct = bool(majority_boxed_letter) and majority_boxed_letter == gold_answer_letter
    best_leaf_conf = float(trajectories[0]["latest_delegate_confidence"]) if trajectories else None

    updated_rounds = update_rounds(
        [item for item in latest.get("rounds", []) if isinstance(item, dict)],
        reachable_ids=reachable_ids,
        final_leaf_ids={node.node_id for node in final_leaves},
        failed_terminal_ids={node.node_id for node in failed_terminal_nodes},
    )
    stop_reason = infer_stop_reason(result or {}, latest, final_leaf_count=final_leaf_count, open_leaf_count=open_leaf_count)

    latest["nodes"] = [all_nodes[node_id].to_json() for node_id in sorted(all_nodes.keys(), key=lambda item: (item != "root", item))]
    latest["final_trajectories"] = trajectories
    latest["final_leaf_count"] = final_leaf_count
    latest["failed_terminal_count"] = failed_terminal_count
    latest["open_leaf_count"] = open_leaf_count
    latest["open_frontier_node_ids"] = [node.node_id for node in open_frontier]
    latest["rounds"] = updated_rounds
    latest["prune_meta"] = {
        "max_children": max_children,
        "spread_max_children": spread_max_children,
        "selection_mode": selection_mode,
        "sample_uniform_mix": sample_uniform_mix,
        "sample_temperature": sample_temperature,
        "drop_error_terminal_nodes": bool(drop_error_terminal_nodes),
        "drop_invalid_submit_nodes": bool(drop_invalid_submit_nodes),
        "dropped_error_node_count": len(dropped_error_node_ids),
        "dropped_error_node_ids": dropped_error_node_ids,
        "dropped_invalid_submit_node_count": len(dropped_invalid_submit_node_ids),
        "dropped_invalid_submit_node_ids": dropped_invalid_submit_node_ids,
        "removed_node_count": removed_node_count,
        "truncated_parent_count": len(truncation_rows),
        "truncations": truncation_rows,
    }

    if isinstance(result, dict):
        result["final_leaf_count"] = final_leaf_count
        result["correct_leaf_count"] = correct_leaf_count
        result["open_leaf_count"] = open_leaf_count
        result["failed_terminal_count"] = failed_terminal_count
        result["success"] = success
        result["any_correct_leaf"] = success
        result["best_leaf_correct"] = bool(best_leaf and best_leaf.is_correct)
        result["majority_correct"] = majority_correct
        result["best_leaf_node_id"] = best_leaf.node_id if best_leaf is not None else None
        result["best_leaf_boxed_letter"] = best_leaf.boxed_letter if best_leaf is not None else None
        result["best_leaf_latest_delegate_confidence"] = best_leaf_conf
        result["majority_boxed_letter"] = majority_boxed_letter
        result["leaf_acc"] = leaf_acc
        result["stop_reason"] = stop_reason
        result["expansion_rounds_ran"] = len(updated_rounds)

    if isinstance(view, dict):
        view.setdefault("final_summary", {})
        view.setdefault("metrics", {})
        view["final_summary"].update(
            {
                "success": success,
                "any_correct_leaf": success,
                "best_leaf_correct": bool(best_leaf and best_leaf.is_correct),
                "majority_correct": majority_correct,
                "stop_reason": stop_reason,
                "best_leaf_node_id": best_leaf.node_id if best_leaf is not None else None,
                "best_leaf_boxed_letter": best_leaf.boxed_letter if best_leaf is not None else None,
                "best_leaf_latest_delegate_confidence": best_leaf_conf,
                "majority_boxed_letter": majority_boxed_letter,
            }
        )
        view["metrics"].update(
            {
                "final_leaf_count": final_leaf_count,
                "open_leaf_count": open_leaf_count,
                "correct_leaf_count": correct_leaf_count,
                "failed_terminal_count": failed_terminal_count,
                "expansion_rounds_ran": len(updated_rounds),
                "node_count": len(all_nodes),
            }
        )
        view["rounds"] = updated_rounds
        view["trajectories"] = trajectories
        view["open_frontier_node_ids"] = [node.node_id for node in open_frontier]
        view["final_leaf_node_ids"] = [node.node_id for node in final_leaves]
        view["failed_terminal_node_ids"] = [node.node_id for node in failed_terminal_nodes]

    write_json(latest_path, latest)
    if isinstance(result, dict):
        write_json(result_path, result)
    write_jsonl(nodes_jsonl_path, [all_nodes[node_id].to_json() for node_id in sorted(all_nodes.keys(), key=lambda item: (item != "root", item))])
    if isinstance(view, dict):
        write_json(view_path, view)

    summary_row = None
    if isinstance(result, dict):
        summary_row = {
            key: result.get(key)
            for key in [
                "task_id",
                "budget_limit",
                "budget_spent",
                "budget_exhausted",
                "stop_reason",
                "target_leaf_trajectories",
                "branching_factor",
                "leaf_expand_ratio",
                "frontier_limit",
                "sibling_pool_strategy",
                "path_max_steps",
                "final_leaf_count",
                "open_leaf_count",
                "failed_terminal_count",
                "correct_leaf_count",
                "success",
                "any_correct_leaf",
                "best_leaf_correct",
                "majority_correct",
                "best_leaf_node_id",
                "best_leaf_boxed_letter",
                "best_leaf_latest_delegate_confidence",
                "majority_boxed_letter",
                "gold_answer_letter",
                "total_tokens",
                "total_model_calls",
                "latency_seconds",
                "total_cost",
                "leaf_acc",
                "discipline",
                "question",
                "options",
                "orchestra_model",
                "status",
            ]
        }
    return {
        "summary_row": summary_row,
        "task_id": str((result or {}).get("task_id") or sample_dir.name),
        "removed_node_count": removed_node_count,
        "dropped_error_node_count": len(dropped_error_node_ids),
        "dropped_invalid_submit_node_count": len(dropped_invalid_submit_node_ids),
        "truncated_parent_count": len(truncation_rows),
        "has_result": isinstance(result, dict),
        "correct_leaf_count": correct_leaf_count,
    }


def rebuild_run_summary(run_dir: Path, scored_rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary_path = run_dir / "summary.json"
    summary = load_json(summary_path) if summary_path.exists() else {}
    sample_count = len(scored_rows)
    success_count = sum(1 for row in scored_rows if bool(row.get("success")))
    total_cost = sum(float(row.get("total_cost", 0.0) or 0.0) for row in scored_rows)
    total_tokens = sum(int(row.get("total_tokens", 0) or 0) for row in scored_rows)
    total_model_calls = sum(int(row.get("total_model_calls", 0) or 0) for row in scored_rows)
    avg_final_leaf_count = (
        sum(float(row.get("final_leaf_count", 0) or 0.0) for row in scored_rows) / sample_count if sample_count else 0.0
    )
    avg_expansion_rounds = (
        sum(float(row.get("expansion_rounds_ran", 0) or 0.0) for row in scored_rows) / sample_count if sample_count else 0.0
    )
    avg_total_cost = (total_cost / sample_count) if sample_count else 0.0

    summary.update(
        {
            "output_dir": str(run_dir),
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
        }
    )
    write_json(summary_path, summary)
    return summary


def main() -> int:
    args = parse_args()
    input_run_dir = args.input_run_dir.expanduser().resolve()
    output_run_dir = args.output_run_dir.expanduser().resolve()
    max_children = int(args.max_children)
    if max_children <= 0:
        raise ValueError("--max-children must be > 0")
    spread_max_children = int(args.spread_max_children)
    if spread_max_children <= 0:
        raise ValueError("--spread-max-children must be > 0")
    if spread_max_children < max_children:
        raise ValueError("--spread-max-children must be >= --max-children")
    if not input_run_dir.exists():
        raise FileNotFoundError(f"Input run dir does not exist: {input_run_dir}")

    ensure_output_dir(input_run_dir, output_run_dir, overwrite=args.overwrite)
    shutil.copytree(input_run_dir, output_run_dir, dirs_exist_ok=False)

    if not (0.0 <= float(args.sample_uniform_mix) <= 1.0):
        raise ValueError("--sample-uniform-mix must be between 0 and 1")
    if float(args.sample_temperature) <= 0:
        raise ValueError("--sample-temperature must be > 0")

    rng = random.Random(int(args.sample_seed))
    sample_dirs = iter_sample_dirs(output_run_dir)
    scored_rows: list[dict[str, Any]] = []
    prune_rows: list[dict[str, Any]] = []
    total_removed_node_count = 0
    total_truncated_parent_count = 0
    dropped_all_wrong_task_ids: list[str] = []
    for sample_dir in sample_dirs:
        sample_summary = rebuild_sample(
            sample_dir,
            max_children=max_children,
            selection_mode=str(args.selection_mode),
            spread_max_children=spread_max_children,
            sample_uniform_mix=float(args.sample_uniform_mix),
            sample_temperature=float(args.sample_temperature),
            rng=rng,
            drop_error_terminal_nodes=bool(args.drop_error_terminal_nodes),
                drop_invalid_submit_nodes=bool(args.drop_invalid_submit_nodes),
        )
        summary_row = sample_summary["summary_row"] if isinstance(sample_summary.get("summary_row"), dict) else None
        task_id = str(sample_summary["task_id"])
        is_all_wrong = int(sample_summary.get("correct_leaf_count") or 0) == 0
        dropped_all_wrong_tree = bool(args.drop_all_wrong_trees and is_all_wrong)
        if dropped_all_wrong_tree:
            shutil.rmtree(sample_dir)
            dropped_all_wrong_task_ids.append(task_id)
        elif isinstance(summary_row, dict):
            scored_rows.append(summary_row)
        prune_rows.append(
            {
                "task_id": task_id,
                "removed_node_count": sample_summary["removed_node_count"],
                "dropped_error_node_count": sample_summary["dropped_error_node_count"],
                "dropped_invalid_submit_node_count": sample_summary["dropped_invalid_submit_node_count"],
                "truncated_parent_count": sample_summary["truncated_parent_count"],
                "has_result": sample_summary["has_result"],
                "dropped_all_wrong_tree": dropped_all_wrong_tree,
            }
        )
        total_removed_node_count += int(sample_summary["removed_node_count"])
        total_truncated_parent_count += int(sample_summary["truncated_parent_count"])

    write_json(output_run_dir / "scored.json", scored_rows)
    rebuild_run_summary(output_run_dir, scored_rows)
    selected_tasks_path = output_run_dir / "selected_tasks.json"
    if selected_tasks_path.exists():
        write_json(selected_tasks_path, [str(row.get("task_id")) for row in scored_rows if row.get("task_id")])
    prune_summary = {
        "input_run_dir": str(input_run_dir),
        "output_run_dir": str(output_run_dir),
        "max_children": max_children,
        "spread_max_children": spread_max_children,
        "selection_mode": str(args.selection_mode),
        "sample_uniform_mix": float(args.sample_uniform_mix),
        "sample_temperature": float(args.sample_temperature),
        "sample_seed": int(args.sample_seed),
        "drop_error_terminal_nodes": bool(args.drop_error_terminal_nodes),
        "drop_invalid_submit_nodes": bool(args.drop_invalid_submit_nodes),
        "sample_count_before_drop": len(sample_dirs),
        "sample_count_after_drop": len(scored_rows),
        "drop_all_wrong_trees": bool(args.drop_all_wrong_trees),
        "dropped_all_wrong_task_ids": dropped_all_wrong_task_ids,
        "total_removed_node_count": total_removed_node_count,
        "total_truncated_parent_count": total_truncated_parent_count,
        "samples": prune_rows,
    }
    write_json(output_run_dir / "prune_children_summary.json", prune_summary)

    print(f"input_run_dir: {input_run_dir}")
    print(f"output_run_dir: {output_run_dir}")
    print(f"sample_count_before_drop: {len(sample_dirs)}")
    print(f"sample_count_after_drop: {len(scored_rows)}")
    print(f"max_children: {max_children}")
    print(f"spread_max_children: {spread_max_children}")
    print(f"selection_mode: {str(args.selection_mode)}")
    print(f"sample_uniform_mix: {float(args.sample_uniform_mix)}")
    print(f"sample_temperature: {float(args.sample_temperature)}")
    print(f"sample_seed: {int(args.sample_seed)}")
    print(f"drop_error_terminal_nodes: {bool(args.drop_error_terminal_nodes)}")
    print(f"drop_invalid_submit_nodes: {bool(args.drop_invalid_submit_nodes)}")
    print(f"drop_all_wrong_trees: {bool(args.drop_all_wrong_trees)}")
    print(f"dropped_all_wrong_tree_count: {len(dropped_all_wrong_task_ids)}")
    print(f"total_removed_node_count: {total_removed_node_count}")
    print(f"total_truncated_parent_count: {total_truncated_parent_count}")
    print(f"summary_path: {output_run_dir / 'prune_children_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
