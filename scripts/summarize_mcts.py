from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter
from pathlib import Path
from shutil import which
from statistics import mean
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize MCTS tree structures for a run directory")
    parser.add_argument(
        "path",
        nargs="?",
        help="Optional run_* directory, sample directory, or latest JSON file to inspect.",
    )
    parser.add_argument(
        "--run-dir",
        help="Run directory under results/mcts. Defaults to the latest run_* directory.",
    )
    parser.add_argument(
        "--top-trajectories",
        type=int,
        default=3,
        help="How many top-ranked trajectories to show for each tree.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON instead of text.",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Export Graphviz DOT for each tree and render SVG when `dot` is available.",
    )
    parser.add_argument(
        "--viz-dir",
        help="Directory to write visualization files into. Defaults to <run-dir>/tree_viz, or <sample-dir>/tree_viz for a single sample path.",
    )
    parser.add_argument(
        "--json-output-path",
        help="Optional file path to write the JSON payload. When set, JSON can be saved without printing to stdout.",
    )
    return parser.parse_args()


def resolve_target(path_arg: str | None, run_dir_arg: str | None) -> tuple[Path, list[Path] | None]:
    target_arg = run_dir_arg or path_arg
    if target_arg:
        target = Path(target_arg).expanduser().resolve()
        if target.is_file():
            if target.name != "latest.json":
                raise FileNotFoundError(f"Unsupported snapshot file: {target}")
            target = target.parent
        if not target.is_dir():
            raise FileNotFoundError(f"Directory not found: {target}")

        if (target / "samples").is_dir():
            return target, None

        if target.parent.name == "samples":
            run_dir = target.parent.parent
            if not run_dir.is_dir():
                raise FileNotFoundError(f"Run directory not found for sample directory: {target}")
            return run_dir, [target]

        raise FileNotFoundError(
            f"Expected a run_* directory, sample directory, or latest JSON file, got: {target}"
        )

    base_dirs = [
        Path("results/mcts_v2").resolve(),
        Path("results/mcts").resolve(),
    ]
    candidates: list[Path] = []
    for base_dir in base_dirs:
        if not base_dir.is_dir():
            continue
        candidates.extend(path for path in base_dir.glob("run_*") if path.is_dir())
    candidates = sorted(candidates, reverse=True)
    if not candidates:
        searched = ", ".join(str(path) for path in base_dirs)
        raise FileNotFoundError(f"No run_* directories found under any of: {searched}")
    return candidates[0], None


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_sample_payload(sample_dir: Path) -> tuple[str, dict[str, Any]] | None:
    latest_path = sample_dir / "latest.json"
    result_path = sample_dir / "result.json"
    view_path = sample_dir / "view.json"

    latest = load_json(latest_path) if latest_path.exists() else None
    result = load_json(result_path) if result_path.exists() else None
    view = load_json(view_path) if view_path.exists() else None

    if latest is not None:
        data = dict(latest)
        sources = ["latest.json"]
    elif view is not None:
        data = dict(view)
        sources = ["view.json"]
    elif result is not None:
        data = dict(result)
        sources = ["result.json"]
    else:
        return None

    if result is not None:
        sources.append("result.json")
        for key, value in result.items():
            data.setdefault(key, value)

    if view is not None:
        sources.append("view.json")
        if not data.get("trajectories") and view.get("trajectories"):
            data["trajectories"] = view["trajectories"]
        final_summary = dict(view.get("final_summary", {}) or {})
        metrics = dict(view.get("metrics", {}) or {})
        reference = dict(view.get("reference", {}) or {})
        task = dict(view.get("task", {}) or {})

        for key, value in final_summary.items():
            data.setdefault(key, value)
        for key, value in metrics.items():
            data.setdefault(key, value)
        if reference.get("gold_answer_letter") is not None:
            data.setdefault("gold_answer_letter", reference["gold_answer_letter"])
        if task.get("discipline") is not None:
            data.setdefault("discipline", task["discipline"])
        if task.get("question") is not None:
            data.setdefault("question", task["question"])
        if task.get("options") is not None:
            data.setdefault("options", task["options"])

    unique_sources = []
    for source in sources:
        if source not in unique_sources:
            unique_sources.append(source)
    return " + ".join(unique_sources), data


def discover_tree_payloads(run_dir: Path, sample_dirs: list[Path] | None = None) -> list[dict[str, Any]]:
    if sample_dirs is None:
        sample_dirs = sorted(path for path in (run_dir / "samples").iterdir() if path.is_dir())
    payloads: list[dict[str, Any]] = []
    for sample_dir in sample_dirs:
        merged = _load_sample_payload(sample_dir)
        if merged is None:
            continue
        source, data = merged
        payloads.append(
            {
                "sample_dir": sample_dir,
                "source": source,
                "data": data,
            }
        )
    return payloads


def infer_nodes_from_trajectories(data: dict[str, Any]) -> list[dict[str, Any]]:
    trajectories = list(data.get("trajectories", data.get("final_trajectories", [])) or [])
    node_map: dict[str, dict[str, Any]] = {
        "root": {
            "node_id": "root",
            "parent_id": None,
            "depth": 0,
            "children_ids": [],
            "boxed_letter": None,
            "confidence": None,
            "chosen_model": None,
            "is_correct": False,
        }
    }
    correct_leaf_ids = {
        str(traj.get("leaf_node_id"))
        for traj in trajectories
        if bool(traj.get("correct"))
    }
    for traj in trajectories:
        path = [str(node_id) for node_id in traj.get("node_ids", [])]
        for depth, node_id in enumerate(path):
            parent_id = path[depth - 1] if depth > 0 else None
            node = node_map.setdefault(
                node_id,
                {
                    "node_id": node_id,
                    "parent_id": parent_id,
                    "depth": depth,
                    "children_ids": [],
                    "boxed_letter": None,
                    "confidence": None,
                    "chosen_model": None,
                    "is_correct": False,
                },
            )
            node["depth"] = min(int(node.get("depth", depth)), depth)
            if depth > 0:
                node["parent_id"] = parent_id
                parent = node_map[path[depth - 1]]
                children = parent.setdefault("children_ids", [])
                if node_id not in children:
                    children.append(node_id)

        leaf_id = str(traj.get("leaf_node_id"))
        if leaf_id in node_map:
            node_map[leaf_id]["boxed_letter"] = traj.get("boxed_letter")
            node_map[leaf_id]["confidence"] = traj.get("confidence", traj.get("latest_delegate_confidence"))
            node_map[leaf_id]["is_correct"] = leaf_id in correct_leaf_ids

    return sorted(node_map.values(), key=lambda item: (int(item.get("depth", 0)), str(item.get("node_id"))))


def build_summary(payload: dict[str, Any], top_trajectories: int) -> dict[str, Any]:
    data = payload["data"]
    sample_dir = payload["sample_dir"]
    source = payload["source"]
    trajectories = list(data.get("trajectories", data.get("final_trajectories", [])) or [])
    rounds = list(data.get("rounds", []) or [])
    nodes = list(data.get("nodes", []) or [])

    if nodes:
        depth_counts = Counter(int(node.get("depth", 0)) for node in nodes)
        max_depth = max(depth_counts) if depth_counts else 0
        internal_nodes = sum(1 for node in nodes if node.get("children_ids"))
        nodes_total = len(nodes)
    else:
        unique_node_ids: set[str] = set()
        for traj in trajectories:
            unique_node_ids.update(str(node_id) for node_id in traj.get("node_ids", []))
        if unique_node_ids and "root" not in unique_node_ids:
            unique_node_ids.add("root")
        depth_counts = Counter(int(traj.get("depth", 0)) for traj in trajectories)
        max_depth = max((len(traj.get("node_ids", [])) - 1 for traj in trajectories), default=0)
        internal_nodes = None
        nodes_total = len(unique_node_ids)

    def round_leaf_text(round_item: dict[str, Any]) -> str:
        before = round_item.get("leaf_count_before")
        after = round_item.get("leaf_count_after")
        if before is not None or after is not None:
            return f"{before}->{after}"
        final_after = round_item.get("final_leaf_count_after")
        open_after = round_item.get("open_leaf_count_after")
        if final_after is not None or open_after is not None:
            return f"final={final_after}, open={open_after}"
        return "n/a"

    round_summaries = [
        {
            "round_index": int(round_item.get("round_index", 0)),
            "strategy": round_item.get("selection_strategy"),
            "selected": round_item.get("selected_parent_count"),
            "expanded": round_item.get("expanded_parent_count"),
            "children": round_item.get("children_created"),
            "leafs": round_leaf_text(round_item),
            "budget_spent": round_item.get("budget_spent"),
            "status": round_item.get("status", "completed"),
        }
        for round_item in rounds
    ]

    top_paths = []
    for traj in trajectories[: max(0, top_trajectories)]:
        top_paths.append(
            {
                "trajectory_index": traj.get("trajectory_index"),
                "leaf_node_id": traj.get("leaf_node_id"),
                "depth": traj.get("depth"),
                "boxed_letter": traj.get("boxed_letter"),
                "confidence": traj.get("confidence", traj.get("latest_delegate_confidence")),
                "correct": traj.get("correct"),
                "path": traj.get("node_ids", []),
            }
        )

    correct_leaf_count = int(
        data.get("correct_leaf_count", sum(1 for traj in trajectories if bool(traj.get("correct"))))
    )
    final_leaf_count = int(data.get("final_leaf_count", data.get("leaf_count", len(trajectories)) or 0))
    leaf_acc = (correct_leaf_count / final_leaf_count) if final_leaf_count else 0.0

    return {
        "task_id": data.get("task_id", sample_dir.name),
        "sample_dir": str(sample_dir),
        "source": source,
        "stop_reason": data.get("stop_reason", "in_progress"),
        "success": data.get("success"),
        "budget_spent": data.get("budget_spent"),
        "leaf_count": data.get("final_leaf_count", data.get("leaf_count", len(trajectories))),
        "correct_leaf_count": correct_leaf_count,
        "leaf_acc": leaf_acc,
        "round_count": len(rounds),
        "nodes_total": nodes_total,
        "internal_nodes": internal_nodes,
        "max_depth": max_depth,
        "depth_counts": dict(sorted(depth_counts.items())),
        "rounds": round_summaries,
        "top_trajectories": top_paths,
    }


def _escape_dot(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def build_dot_graph(payload: dict[str, Any]) -> str:
    data = payload["data"]
    task_id = str(data.get("task_id", payload["sample_dir"].name))
    nodes = list(data.get("nodes", []) or [])
    if not nodes:
        nodes = infer_nodes_from_trajectories(data)

    node_map = {str(node["node_id"]): node for node in nodes}
    if "root" not in node_map:
        node_map["root"] = {
            "node_id": "root",
            "parent_id": None,
            "depth": 0,
            "children_ids": [],
            "boxed_letter": None,
            "confidence": None,
            "chosen_model": None,
            "is_correct": False,
        }

    trajectories = list(data.get("trajectories", data.get("final_trajectories", [])) or [])
    trajectory_costs: dict[str, float] = {}
    for traj in trajectories:
        leaf_id = str(traj.get("leaf_node_id"))
        path = [str(node_id) for node_id in traj.get("node_ids", [])]
        trajectory_costs[leaf_id] = sum(float(node_map.get(node_id, {}).get("cost", 0.0) or 0.0) for node_id in path)
    correct_leaf_count = int(
        data.get("correct_leaf_count", sum(1 for traj in trajectories if bool(traj.get("correct"))))
    )
    final_leaf_count = int(data.get("final_leaf_count", data.get("leaf_count", len(trajectories)) or 0))
    leaf_acc = (correct_leaf_count / final_leaf_count) if final_leaf_count else 0.0

    title_lines = [
        task_id,
        f"leaf_acc={correct_leaf_count}/{final_leaf_count}={leaf_acc:.3f}",
    ]
    if "success" in data:
        title_lines.append(f"any_correct={bool(data.get('success'))}")
    if "majority_correct" in data:
        title_lines.append(f"majority_correct={bool(data.get('majority_correct'))}")

    lines = [
        "digraph MCTS {",
        '  graph [rankdir=TB, labelloc="t", labeljust="l", fontsize=18];',
        '  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=10, color="#475569"];',
        '  edge [color="#94a3b8", arrowsize=0.7];',
        f'  label="{_escape_dot(chr(10).join(title_lines))}";',
    ]

    for node_id, node in sorted(node_map.items(), key=lambda item: (int(item[1].get("depth", 0)), item[0])):
        children = list(node.get("children_ids", []) or [])
        is_leaf = node_id != "root" and not children
        boxed = node.get("boxed_letter")
        confidence = node.get("confidence")
        model = node.get("chosen_model")
        leaf_acc = 1.0 if bool(node.get("is_correct")) else 0.0
        trajectory_cost = trajectory_costs.get(node_id)
        label_parts = [node_id]
        if boxed:
            label_parts.append(f"boxed={boxed}")
        if is_leaf:
            label_parts.append(f"acc={leaf_acc:.2f}")
            if trajectory_cost is not None:
                label_parts.append(f"traj_cost=${trajectory_cost:.3f}")
        if confidence is not None:
            label_parts.append(f"conf={float(confidence):.2f}")
        if model:
            label_parts.append(str(model))
        label = _escape_dot("\n".join(label_parts))

        fill = "#e2e8f0"
        shape = "box"
        penwidth = "1.2"
        color = "#64748b"
        if node_id == "root":
            fill = "#cbd5e1"
            shape = "box"
        elif bool(node.get("is_correct")):
            fill = "#dcfce7"
            shape = "ellipse" if is_leaf else "box"
            color = "#16a34a"
        elif is_leaf:
            fill = "#fee2e2"
            shape = "ellipse"
            color = "#dc2626"

        lines.append(
            f'  "{_escape_dot(node_id)}" [label="{label}", shape={shape}, fillcolor="{fill}", color="{color}", penwidth={penwidth}];'
        )

    seen_edges: set[tuple[str, str]] = set()
    for node_id, node in node_map.items():
        parent_id = node.get("parent_id")
        if not parent_id:
            continue
        edge = (str(parent_id), node_id)
        if edge in seen_edges:
            continue
        seen_edges.add(edge)
        lines.append(f'  "{_escape_dot(edge[0])}" -> "{_escape_dot(edge[1])}" [color="#94a3b8", penwidth=1.0];')

    lines.append("}")
    return "\n".join(lines)


def export_visualizations(viz_dir: Path, payloads: list[dict[str, Any]]) -> list[dict[str, str]]:
    viz_dir.mkdir(parents=True, exist_ok=True)
    dot_bin = which("dot")
    exported: list[dict[str, str]] = []
    for payload in payloads:
        task_id = str(payload["data"].get("task_id", payload["sample_dir"].name)).replace("/", "_")
        dot_path = viz_dir / f"{task_id}.dot"
        svg_path = viz_dir / f"{task_id}.svg"
        dot_path.write_text(build_dot_graph(payload), encoding="utf-8")
        item = {"task_id": task_id, "dot": str(dot_path)}
        if dot_bin:
            subprocess.run([dot_bin, "-Tsvg", str(dot_path), "-o", str(svg_path)], check=True)
            item["svg"] = str(svg_path)
        exported.append(item)
    return exported


def build_tree_acc_distribution(
    summaries: list[dict[str, Any]], bucket_size: float = 0.1
) -> list[dict[str, Any]]:
    bucket_count = int(round(1.0 / bucket_size))
    counts: Counter[int] = Counter()

    for item in summaries:
        acc = float(item.get("leaf_acc", 0.0) or 0.0)
        bucket_index = min(bucket_count - 1, max(0, int(acc / bucket_size)))
        counts[bucket_index] += 1

    distribution = []
    total = len(summaries)
    for bucket_index in range(bucket_count):
        start = bucket_index * bucket_size
        end = min(1.0, start + bucket_size)
        bucket_label = (
            f"[{start:.1f}, {end:.1f}]"
            if bucket_index == bucket_count - 1
            else f"[{start:.1f}, {end:.1f})"
        )
        count = counts[bucket_index]
        distribution.append(
            {
                "bucket": bucket_label,
                "count": count,
                "ratio": (count / total) if total else 0.0,
            }
        )
    return distribution


def build_tree_acc_extremes(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(summaries)
    all_wrong_count = sum(1 for item in summaries if int(item.get("leaf_count", 0) or 0) > 0 and int(item["correct_leaf_count"]) == 0)
    all_correct_count = sum(
        1
        for item in summaries
        if int(item.get("leaf_count", 0) or 0) > 0 and int(item["correct_leaf_count"]) == int(item["leaf_count"])
    )
    return {
        "all_wrong_trees": {
            "count": all_wrong_count,
            "ratio": (all_wrong_count / total) if total else 0.0,
        },
        "all_correct_trees": {
            "count": all_correct_count,
            "ratio": (all_correct_count / total) if total else 0.0,
        },
    }


def format_tree_acc_distribution(
    tree_acc_distribution: list[dict[str, Any]], tree_acc_extremes: dict[str, Any]
) -> list[str]:
    lines = ["tree_acc_distribution:"]
    for bucket in tree_acc_distribution:
        lines.append(f"  {bucket['bucket']}: {bucket['count']} ({bucket['ratio']:.1%})")
    lines.append(
        "all_wrong_trees: {count} ({ratio:.1%})".format(
            count=tree_acc_extremes["all_wrong_trees"]["count"],
            ratio=tree_acc_extremes["all_wrong_trees"]["ratio"],
        )
    )
    lines.append(
        "all_correct_trees: {count} ({ratio:.1%})".format(
            count=tree_acc_extremes["all_correct_trees"]["count"],
            ratio=tree_acc_extremes["all_correct_trees"]["ratio"],
        )
    )
    return lines


def format_text(run_dir: Path, summaries: list[dict[str, Any]]) -> str:
    lines = [
        f"run_dir: {run_dir}",
        f"tree_count: {len(summaries)}",
    ]
    if summaries:
        lines.extend(
            [
                f"avg_rounds: {mean(item['round_count'] for item in summaries):.2f}",
                f"avg_nodes: {mean(item['nodes_total'] for item in summaries):.2f}",
                f"avg_leafs: {mean(item['leaf_count'] for item in summaries):.2f}",
                f"avg_budget_spent: {mean(float(item['budget_spent'] or 0.0) for item in summaries):.4f}",
            ]
        )

    # for item in summaries:
    #     lines.append("")
    #     lines.append(f"[{item['task_id']}] source={item['source']} stop={item['stop_reason']}")
        # lines.append(
        #     "  nodes={nodes} internal={internal} leafs={leafs} rounds={rounds} max_depth={depth}".format(
        #         nodes=item["nodes_total"],
        #         internal=item["internal_nodes"],
        #         leafs=item["leaf_count"],
        #         rounds=item["round_count"],
        #         depth=item["max_depth"],
        #     )
        # )
        # lines.append(
        #     "  acc={correct_leafs}/{leafs}={leaf_acc:.3f}".format(
        #         correct_leafs=item["correct_leaf_count"],
        #         leafs=item["leaf_count"],
        #         leaf_acc=float(item["leaf_acc"]),
        #     )
        # )
        # lines.append(f"  depth_counts={item['depth_counts']}")
        # lines.append("  rounds:")
        # for round_item in item["rounds"]:
        #     lines.append(
        #         "    r{idx}: {strategy} selected={selected} expanded={expanded} children={children} leafs={leafs} budget={budget:.4f} status={status}".format(
        #             idx=round_item["round_index"],
        #             strategy=round_item["strategy"],
        #             selected=round_item["selected"],
        #             expanded=round_item["expanded"],
        #             children=round_item["children"],
        #             leafs=round_item["leafs"],
        #             budget=float(round_item["budget_spent"] or 0.0),
        #             status=round_item["status"],
        #         )
        #     )
        # lines.append("  top_trajectories:")
        # for traj in item["top_trajectories"]:
        #     lines.append(
        #         "    #{idx}: leaf={leaf} depth={depth} boxed={boxed} conf={conf} correct={correct} path={path}".format(
        #             idx=traj["trajectory_index"],
        #             leaf=traj["leaf_node_id"],
        #             depth=traj["depth"],
        #             boxed=traj["boxed_letter"],
        #             conf=traj["confidence"],
        #             correct=traj["correct"],
        #             path=" -> ".join(str(node_id) for node_id in traj["path"]),
        #         )
        #     )
    return "\n".join(lines)


def collect_unfinished_task_numbers(payloads: list[dict[str, Any]]) -> list[str]:
    unfinished: list[str] = []
    for payload in payloads:
        sample_dir = payload["sample_dir"]
        if (sample_dir / "result.json").exists():
            continue
        task_id = str(payload["data"].get("task_id", sample_dir.name))
        suffix = task_id.rsplit("_", 1)[-1]
        if suffix.isdigit():
            unfinished.append(suffix)
    return unfinished


def _list_sample_dirs(run_dir: Path, sample_dirs: list[Path] | None) -> list[Path]:
    if sample_dirs is not None:
        return sorted(sample_dirs)
    samples_root = run_dir / "samples"
    if not samples_root.is_dir():
        return []
    return sorted(path for path in samples_root.iterdir() if path.is_dir())


def _iteration_dir_if_reasoning_or_expand_run(run_dir: Path) -> Path | None:
    """If run_dir is .../<iteration>/reasoning/<model> or .../<iteration>/expand/<model>, return iteration dir."""
    run_dir = run_dir.resolve()
    parent = run_dir.parent
    if parent.name not in ("reasoning", "expand"):
        return None
    return parent.parent


def _tree_runs_pipeline_scan(
    iteration_dir: Path,
    model_dir_name: str,
) -> tuple[int, int, list[dict[str, Any]]]:
    """Align with generate_mcts tree tqdm: a tree is done when expand sample has result.json.

    Per-tree layout: <iteration>/_tree_runs/<task_id>/expand/<model>/samples/<task_id>/result.json
    """
    tree_runs = (iteration_dir / "_tree_runs").resolve()
    if not tree_runs.is_dir():
        return 0, 0, []

    rows: list[dict[str, Any]] = []
    pending_n = 0
    total_n = 0

    for task_path in sorted(tree_runs.iterdir()):
        if not task_path.is_dir():
            continue
        tid = task_path.name
        total_n += 1
        expand_sample = task_path / "expand" / model_dir_name / "samples" / tid
        expand_result = expand_sample / "result.json"
        if expand_result.is_file():
            continue

        pending_n += 1
        reasoning_sample = task_path / "reasoning" / model_dir_name / "samples" / tid
        merged = _load_sample_payload(expand_sample) or _load_sample_payload(reasoning_sample)
        if merged is None:
            rows.append(
                {
                    "task_id": tid,
                    "sample_dir": str(expand_sample),
                    "status": "no_snapshot",
                    "leaves_closed_so_far": None,
                    "open_frontier_leaves": None,
                    "target_leaf_trajectories": None,
                    "budget_spent": None,
                    "stop_reason": None,
                }
            )
            continue
        _, data = merged
        closed = int(data.get("final_leaf_count", 0) or 0)
        open_leaf = int(data.get("open_leaf_count", 0) or 0)
        target = data.get("target_leaf_trajectories")
        budget = data.get("budget_spent")
        stop_reason = data.get("stop_reason")
        rows.append(
            {
                "task_id": str(data.get("task_id", tid)),
                "sample_dir": str(expand_sample),
                "status": "snapshot",
                "leaves_closed_so_far": closed,
                "open_frontier_leaves": open_leaf,
                "target_leaf_trajectories": target,
                "budget_spent": float(budget) if budget is not None else None,
                "stop_reason": stop_reason,
            }
        )

    return pending_n, total_n, rows


def _progress_from_tree_pipeline_summary(
    iteration_dir: Path,
    model_dir_name: str,
) -> tuple[list[dict[str, Any]], list[str]] | None:
    """Use generate_mcts's tree_pipeline_summary.json — same counters as the Trees tqdm bar.

    Expand ``result.json`` on disk can lag or differ across mounts; the summary file is updated
    when each future completes.
    """
    summary_path = iteration_dir / "_generated_configs" / "tree_pipeline_summary.json"
    if not summary_path.is_file():
        return None
    try:
        data = load_json(summary_path)
    except (OSError, json.JSONDecodeError):
        return None
    total = int(data.get("tree_count") or 0)
    completed = int(data.get("completed_tree_count") or 0)
    if total <= 0:
        return None
    pending = total - completed
    lines = [f"还有 {pending} 棵在跑（{completed}/{total} 已完成）"]

    completed_ids = {
        str(t.get("task_id"))
        for t in data.get("tasks", [])
        if isinstance(t, dict) and t.get("task_id") is not None
    }
    rows: list[dict[str, Any]] = []
    tree_runs = iteration_dir / "_tree_runs"
    if tree_runs.is_dir():
        for task_path in sorted(tree_runs.iterdir()):
            if not task_path.is_dir():
                continue
            tid = task_path.name
            if tid in completed_ids:
                continue
            expand_sample = task_path / "expand" / model_dir_name / "samples" / tid
            reasoning_sample = task_path / "reasoning" / model_dir_name / "samples" / tid
            merged = _load_sample_payload(expand_sample) or _load_sample_payload(reasoning_sample)
            if merged is None:
                rows.append(
                    {
                        "task_id": tid,
                        "sample_dir": str(expand_sample),
                        "status": "no_snapshot",
                        "leaves_closed_so_far": None,
                        "open_frontier_leaves": None,
                        "target_leaf_trajectories": None,
                        "budget_spent": None,
                        "stop_reason": None,
                        "progress_source": "tree_pipeline_summary.json",
                    }
                )
                continue
            _, pdata = merged
            _bs = pdata.get("budget_spent")
            rows.append(
                {
                    "task_id": str(pdata.get("task_id", tid)),
                    "sample_dir": str(expand_sample),
                    "status": "snapshot",
                    "leaves_closed_so_far": int(pdata.get("final_leaf_count", 0) or 0),
                    "open_frontier_leaves": int(pdata.get("open_leaf_count", 0) or 0),
                    "target_leaf_trajectories": pdata.get("target_leaf_trajectories"),
                    "budget_spent": float(_bs) if _bs is not None else None,
                    "stop_reason": pdata.get("stop_reason"),
                    "progress_source": "tree_pipeline_summary.json",
                }
            )

    return rows, lines


def build_in_progress_tree_report(
    run_dir: Path,
    sample_dirs: list[Path] | None,
    payloads: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Count trees still running.

    Under train-loop tree mode, prefer ``_generated_configs/tree_pipeline_summary.json`` — it
    matches the ``Trees`` tqdm bar. If that file is missing, fall back to scanning
    ``expand/.../result.json`` under ``_tree_runs/``.

    Legacy flat runs only have ``<run_dir>/samples/<task_id>/`` — then we count dirs there
    missing ``result.json``.
    """
    rows: list[dict[str, Any]] = []
    lines: list[str] = []

    if sample_dirs is not None and len(sample_dirs) == 1:
        return rows, lines

    iteration_dir = _iteration_dir_if_reasoning_or_expand_run(run_dir)
    if iteration_dir is not None:
        summary_progress = _progress_from_tree_pipeline_summary(iteration_dir, run_dir.name)
        if summary_progress is not None:
            return summary_progress

        pending_n, total_n, rows = _tree_runs_pipeline_scan(iteration_dir, run_dir.name)
        if total_n > 0:
            lines.append(
                f"还有 {pending_n} 棵在跑（{total_n - pending_n}/{total_n} 已完成；"
                f"无 tree_pipeline_summary.json，按磁盘 expand/result.json 估算）"
            )
            return rows, lines

    all_dirs = _list_sample_dirs(run_dir, sample_dirs)
    by_dir = {Path(p["sample_dir"]).resolve(): p for p in payloads}
    pending = [d for d in all_dirs if not (d / "result.json").exists()]
    if not pending:
        return rows, lines

    lines.append(f"还有 {len(pending)} 棵在跑（聚合 samples/ 下尚无 result.json）")

    for d in sorted(pending, key=lambda p: p.name):
        pl = by_dir.get(d.resolve())
        if pl is None:
            rows.append(
                {
                    "task_id": d.name,
                    "sample_dir": str(d),
                    "status": "no_snapshot",
                    "leaves_closed_so_far": None,
                    "open_frontier_leaves": None,
                    "target_leaf_trajectories": None,
                    "budget_spent": None,
                    "stop_reason": None,
                }
            )
            continue
        data = pl["data"]
        task_id = str(data.get("task_id", d.name))
        closed = int(data.get("final_leaf_count", 0) or 0)
        open_leaf = int(data.get("open_leaf_count", 0) or 0)
        target = data.get("target_leaf_trajectories")
        budget = data.get("budget_spent")
        stop_reason = data.get("stop_reason")
        rows.append(
            {
                "task_id": task_id,
                "sample_dir": str(d),
                "status": "snapshot",
                "leaves_closed_so_far": closed,
                "open_frontier_leaves": open_leaf,
                "target_leaf_trajectories": target,
                "budget_spent": float(budget) if budget is not None else None,
                "stop_reason": stop_reason,
                "source": pl.get("source"),
            }
        )

    return rows, lines


def main() -> int:
    args = parse_args()
    run_dir, sample_dirs = resolve_target(args.path, args.run_dir)
    payloads = discover_tree_payloads(run_dir, sample_dirs=sample_dirs)
    summaries = [build_summary(payload, top_trajectories=args.top_trajectories) for payload in payloads]
    tree_acc_distribution = build_tree_acc_distribution(summaries)
    tree_acc_extremes = build_tree_acc_extremes(summaries)
    exported_viz: list[dict[str, str]] = []
    if args.visualize:
        default_viz_dir = sample_dirs[0] / "tree_viz" if sample_dirs and len(sample_dirs) == 1 else (run_dir / "tree_viz")
        viz_dir = Path(args.viz_dir).expanduser().resolve() if args.viz_dir else default_viz_dir
        exported_viz = export_visualizations(viz_dir, payloads)

    in_progress_trees, in_progress_lines = build_in_progress_tree_report(run_dir, sample_dirs, payloads)
    unfinished_numbers = collect_unfinished_task_numbers(payloads)

    if args.json:
        json_payload = {
            "run_dir": str(run_dir),
            "tree_acc_distribution": tree_acc_distribution,
            "tree_acc_extremes": tree_acc_extremes,
            "trees": summaries,
            "visualizations": exported_viz,
            "unfinished_task_numbers": unfinished_numbers,
            "in_progress_trees": in_progress_trees,
        }
        if args.json_output_path:
            output_path = Path(args.json_output_path).expanduser().resolve()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                json.dumps(json_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        else:
            print(json.dumps(json_payload, ensure_ascii=False, indent=2))
    else:
        output_lines = format_tree_acc_distribution(tree_acc_distribution, tree_acc_extremes)
        if in_progress_lines:
            output_lines.append("")
            output_lines.extend(in_progress_lines)
        print("\n".join(output_lines))


if __name__ == "__main__":
    raise SystemExit(main())
