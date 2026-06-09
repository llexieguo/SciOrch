from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

from mcts.config import MCTSConfig
from mcts.runner import MCTSReasoningRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run independent MCTS reasoning experiments")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    return parser.parse_args()


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
    project_root = Path(__file__).resolve().parents[2]
    candidates = [
        Path.cwd() / ".env",
        config_path.parent / ".env",
        config_path.parent.parent / ".env",
        project_root / ".env",
    ]
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        _load_env_file(resolved)


async def _main() -> int:
    args = parse_args()
    config_path = Path(args.config).resolve()
    _load_env_candidates(config_path)

    config = MCTSConfig.load(config_path)
    runner = MCTSReasoningRunner(config)
    records, summary = await runner.run()

    print("=" * 60)
    print("MCTS Reasoning Run Complete")
    print(f"total_samples: {summary['total_samples']}")
    print(f"success_count: {summary['success_count']}")
    print(f"failure_count: {summary['failure_count']}")
    print(f"success_rate: {summary['success_rate']:.4f}")
    print(f"avg_expansion_rounds: {summary['avg_expansion_rounds']:.4f}")
    print(f"avg_final_leaf_count: {summary['avg_final_leaf_count']:.4f}")
    print(f"total_cost: {summary['total_cost']:.6f}")
    print(f"total_tokens: {summary['total_tokens']}")
    print(f"total_model_calls: {summary['total_model_calls']}")
    print(f"selected_task_ids: {json.dumps(summary['selected_task_ids'], ensure_ascii=False)}")
    print(f"dataset_name: {summary['dataset_name']}")
    print(f"dataset_split: {summary['dataset_split']}")
    print(f"sample_count: {summary['sample_count']}")
    print(f"sample_seed: {summary['sample_seed']}")
    print(f"output_dir: {summary['output_dir']}")
    print("=" * 60)

    failed = [item["task_id"] for item in records if not item["success"]]
    if failed:
        print(f"unsolved_task_ids: {json.dumps(failed, ensure_ascii=False)}")
    return 0


def main() -> int:
    return asyncio.run(_main())


if __name__ == "__main__":
    raise SystemExit(main())
