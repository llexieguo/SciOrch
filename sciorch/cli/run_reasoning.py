from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

from sciorch.config import OrchestratorConfig
from sciorch.runner.reasoning_runner import ReasoningRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SciOrch Reasoning pipeline")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing results instead of resuming")
    parser.add_argument("--resume", action="store_true", help="Resume from latest existing run directory")
    return parser.parse_args()


def _load_env_file(env_path: Path) -> None:
    """Load KEY=VALUE pairs into process env without overriding existing vars."""
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

        # Remove one pair of surrounding quotes.
        if len(value) >= 2 and (
            (value[0] == '"' and value[-1] == '"')
            or (value[0] == "'" and value[-1] == "'")
        ):
            value = value[1:-1]

        os.environ.setdefault(key, value)


def _load_env_candidates(config_path: Path) -> None:
    """Try common .env locations so gateway credentials/url are auto-loaded."""
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
    config = OrchestratorConfig.load(config_path)
    runner = ReasoningRunner(config)
    records, summary = await runner.run(resume=args.resume)

    # --- Per-dataset breakdown ---
    source_map = {}
    ds_path = config.dataset_name
    if ds_path:
        try:
            ds_data = json.loads(Path(ds_path).read_text(encoding="utf-8"))
            source_map = {item["id"]: item.get("source", "unknown") for item in ds_data}
        except Exception:
            pass

    def _infer_source(task_id: str) -> str:
        if task_id in source_map:
            return source_map[task_id]
        if task_id.startswith("SGI_Reasoning"):
            return "SGI"
        if "/" in task_id and len(task_id) < 20:
            return "SFE"
        return "SuperGPQA"

    # Infra-failed samples are excluded from the accuracy denominator.
    scored_records = [r for r in records if r.metadata.get("status") != "failed"]

    from collections import defaultdict
    per_ds: dict[str, list] = defaultdict(list)
    for r in scored_records:
        per_ds[_infer_source(r.task_id)].append(r)

    print("=" * 60)
    total_correct = sum(1 for r in scored_records if r.mca > 0)
    print("Per-dataset results:")
    for ds_name in sorted(per_ds.keys()):
        ds_records = per_ds[ds_name]
        ds_total = len(ds_records)
        ds_correct = sum(1 for r in ds_records if r.mca > 0)
        ds_acc = ds_correct / ds_total if ds_total else 0
        ds_ratio = ds_correct / total_correct if total_correct else 0
        ds_steps = sum(len(r.decision_steps) for r in ds_records) / ds_total if ds_total else 0
        print(f"  {ds_name:12s}  acc={ds_acc:.1%} ({ds_correct}/{ds_total})  ratio={ds_ratio:.1%}  avg_steps={ds_steps:.1f}")
    print(f"total_cost: ${summary.total_cost:.4f}")
    print(f"output_dir: {summary.output_dir}")
    print("=" * 60)
    return 0


def main() -> int:
    return asyncio.run(_main())


if __name__ == "__main__":
    raise SystemExit(main())
