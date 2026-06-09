#!/usr/bin/env python3
"""Canonical prompt-MAS runner for the MASLab-style baseline layout.

References:
- MASLab: https://github.com/MASWorks/MASLab
- CoMAS MASLab-style layout: https://github.com/xxyQwQ/CoMAS/tree/main/maslab

This runner is locally implemented for SciOrch. It wires together the
combined multimodal dataset loader, prompt-based MAS methods, OpenAI-compatible
API client, JSONL logging, and summary metric writing.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from maslab.evaluation import write_summary
from maslab.datasets.loader import load_combined_dataset
from maslab.methods.base import PromptMASConfig
from maslab.methods.llm_debate.llm_debate_main import LLMDebateBaseline
from maslab.methods.self_consistency.self_consistency_main import SelfConsistencyBaseline
from maslab.utils.formatting import format_query, normalize_answer_letter
from sciorch.llm.openai_compatible import OpenAICompatibleClient


DEFAULT_IMAGE_ROOT = PROJECT_ROOT / "maslab" / "datasets" / "data" / "images"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run prompt-based MAS baselines on combined JSON data.")
    parser.add_argument("--method", choices=["llm_debate", "self_consistency"], required=True)
    parser.add_argument("--input", required=True, type=Path, help="Path to combined JSON test data.")
    parser.add_argument("--output", type=Path, help="Output JSONL path. Defaults to output/baselines/<method>/<run_id>.jsonl.")
    parser.add_argument("--run-id", default=None, help="Optional stable run id for output files and logs.")
    parser.add_argument("--model", default="gpt-5.4", help="OpenAI-compatible model name.")
    parser.add_argument("--base-url", default=None, help="Override OpenAI-compatible base URL.")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY", help="Environment variable containing API key.")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit number of samples.")
    parser.add_argument("--sample-ids", default=None, help="Comma-separated exact sample ids to run, in the requested order.")
    parser.add_argument("--sample-method", choices=["head", "random"], default="head")
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--debate-agents", type=int, default=3)
    parser.add_argument("--debate-rounds", type=int, default=2)
    parser.add_argument("--self-consistency-n", type=int, default=5)
    parser.add_argument("--max-concurrency", type=int, default=1,
                        help="Max number of samples to process in parallel.")
    parser.add_argument(
        "--sleep-between-samples",
        type=float,
        default=0.0,
        help="Seconds to sleep after each sample, except after the final sample.",
    )
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--no-images", action="store_true", help="Do not attach local images to model calls.")
    parser.add_argument("--strict-images", action="store_true", help="Fail on truncated image files instead of loading them in tolerant mode.")
    parser.add_argument("--dry-run", action="store_true", help="Only load data and print prompt preview; no API calls.")
    return parser.parse_args()


def load_env_file(env_path: Path) -> None:
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
        value = value.strip().strip("'\"")
        if key:
            os.environ.setdefault(key, value)


def load_env_candidates(input_path: Path) -> None:
    candidates = [
        Path.cwd() / ".env",
        PROJECT_ROOT / ".env",
        input_path.parent / ".env",
        PROJECT_ROOT.parent / ".env",
    ]
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        load_env_file(resolved)


def parse_sample_ids(raw: str | None) -> list[str] | None:
    if raw is None:
        return None
    ids = [item.strip() for item in raw.split(",") if item.strip()]
    return ids or None


def select_samples(
    samples: list[dict[str, Any]],
    max_samples: int | None,
    method: str,
    seed: int,
    sample_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    if sample_ids:
        samples_by_id = {str(sample.get("id")): sample for sample in samples}
        missing = [sample_id for sample_id in sample_ids if sample_id not in samples_by_id]
        if missing:
            raise ValueError(f"Requested sample ids were not found: {missing}")
        selected = [samples_by_id[sample_id] for sample_id in sample_ids]
        return selected[:max_samples] if max_samples is not None else selected

    if max_samples is None or max_samples >= len(samples):
        return samples
    if max_samples <= 0:
        return []
    if method == "head":
        return samples[:max_samples]
    rng = random.Random(seed)
    indices = list(range(len(samples)))
    rng.shuffle(indices)
    return [samples[index] for index in indices[:max_samples]]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def safe_slug(value: str) -> str:
    slug = []
    for char in str(value):
        if char.isalnum() or char in {".", "-", "_"}:
            slug.append(char)
        else:
            slug.append("-")
    return "".join(slug).strip("-") or "run"


def make_run_id(args: argparse.Namespace, started_at: datetime) -> str:
    timestamp = started_at.strftime("%Y%m%d_%H%M%S")
    return f"{timestamp}_{safe_slug(args.method)}_{safe_slug(args.model)}"


def resolve_output_path(args: argparse.Namespace, run_id: str) -> Path:
    if args.output is not None:
        return args.output.expanduser().resolve()
    return (PROJECT_ROOT / "output" / "baselines" / args.method / f"{run_id}.jsonl").resolve()


def env_presence(args: argparse.Namespace) -> dict[str, Any]:
    api_key_names = [args.api_key_env, "OPENAI_API_KEY", "api_key", "API_KEY"]
    api_key_present = any(bool(os.getenv(name)) for name in dict.fromkeys(api_key_names))

    if args.base_url:
        base_url_source = "cli"
        base_url_present = True
    elif os.getenv("OPENAI_BASE_URL") or os.getenv("base_url") or os.getenv("BASE_URL"):
        base_url_source = "env"
        base_url_present = True
    else:
        base_url_source = None
        base_url_present = False

    return {
        "api_key_env": args.api_key_env,
        "api_key_present": api_key_present,
        "base_url_present": base_url_present,
        "base_url_source": base_url_source,
    }


def build_baseline(args: argparse.Namespace, client: OpenAICompatibleClient):
    config = PromptMASConfig(
        model=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        use_images=not args.no_images,
        tolerate_truncated_images=not args.strict_images,
    )
    if args.method == "llm_debate":
        return LLMDebateBaseline(
            client,
            config,
            agents_num=args.debate_agents,
            rounds_num=args.debate_rounds,
        )
    return SelfConsistencyBaseline(
        client,
        config,
        parallel_num=args.self_consistency_n,
    )


def print_dry_run(samples: list[dict[str, Any]], args: argparse.Namespace) -> None:
    print("=" * 80)
    print("Prompt MAS dry run")
    print(f"method: {args.method}")
    print(f"input_samples: {len(samples)}")
    print(f"model: {args.model}")
    print(f"use_images: {not args.no_images}")
    print(f"image_root: {args.image_root}")
    if not samples:
        print("No samples selected.")
        return
    sample = samples[0]
    options_len = len(sample.get("options") or [])
    print("-" * 80)
    print(f"id: {sample.get('id')}")
    print(f"source: {sample.get('source')}")
    print(f"subject: {sample.get('subject')}")
    print(f"gold_answer_raw: {sample.get('answer')}")
    print(f"gold_answer_letter: {normalize_answer_letter(sample.get('answer'), options_len)}")
    print(f"images: {len(sample.get('images') or [])}")
    print("-" * 80)
    print(format_query(sample))
    print("=" * 80)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def make_error_row(
    *,
    args: argparse.Namespace,
    run_id: str,
    sample: dict[str, Any],
    sample_index: int,
    exc: Exception,
    started_at: float,
) -> dict[str, Any]:
    latency_seconds = time.perf_counter() - started_at
    return {
        "run_id": run_id,
        "sample_index": sample_index,
        "sample_status": "error",
        "id": sample.get("id"),
        "task_id": sample.get("id"),
        "method": args.method,
        "model": args.model,
        "source": sample.get("source"),
        "subject": sample.get("subject"),
        "question": sample.get("question"),
        "options": sample.get("options"),
        "gold_answer_raw": sample.get("answer"),
        "final_boxed_letter": None,
        "parse_error": None,
        "mca": 0.0,
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "metrics": {
            "num_llm_calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "cost": 0.0,
            "latency_seconds": latency_seconds,
            "mca": 0.0,
        },
    }


async def run(args: argparse.Namespace) -> int:
    run_started_at = utc_now()
    run_start_perf = time.perf_counter()
    input_path = args.input.expanduser().resolve()
    load_env_candidates(input_path)
    samples = select_samples(
        load_combined_dataset(input_path),
        args.max_samples,
        args.sample_method,
        args.sample_seed,
        parse_sample_ids(args.sample_ids),
    )

    if args.dry_run:
        print_dry_run(samples, args)
        return 0

    run_id = args.run_id or make_run_id(args, run_started_at)
    output_path = resolve_output_path(args, run_id)
    image_root = args.image_root.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("", encoding="utf-8")

    client = OpenAICompatibleClient(
        base_url=args.base_url,
        api_key_env=args.api_key_env,
        default_temperature=args.temperature,
    )
    baseline = build_baseline(args, client)

    rows: list[dict[str, Any]] = [None] * len(samples)  # type: ignore[list-item]
    semaphore = asyncio.Semaphore(args.max_concurrency)
    write_lock = asyncio.Lock()
    completed = [0]  # mutable counter

    async def process_sample(index: int, sample: dict[str, Any]) -> None:
        async with semaphore:
            sid = sample.get("id")
            print(f"[{index}/{len(samples)}] {args.method} {sid}")
            sample_started_at = time.perf_counter()
            try:
                row = await baseline.run_sample(sample, image_root=image_root)
                row["run_id"] = run_id
                row["sample_index"] = index
                row["sample_status"] = "ok"
            except Exception as exc:
                row = make_error_row(
                    args=args,
                    run_id=run_id,
                    sample=sample,
                    sample_index=index,
                    exc=exc,
                    started_at=sample_started_at,
                )
                print(f"[error] {sid}: {type(exc).__name__}: {exc}")
            rows[index - 1] = row
            async with write_lock:
                append_jsonl_row(output_path, row)
                completed[0] += 1
                print(f"[done] {sid} ({completed[0]}/{len(samples)})")

    tasks = [process_sample(i, s) for i, s in enumerate(samples, start=1)]
    await asyncio.gather(*tasks)

    run_finished_at = utc_now()
    write_summary(
        output_path,
        rows,
        run_id=run_id,
        method=args.method,
        model=args.model,
        input_path=input_path,
        image_root=image_root,
        started_at=run_started_at,
        finished_at=run_finished_at,
        runtime_seconds=time.perf_counter() - run_start_perf,
        env=env_presence(args),
        config={
            "max_samples": args.max_samples,
            "sample_ids": parse_sample_ids(args.sample_ids),
            "sample_method": args.sample_method,
            "sample_seed": args.sample_seed,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "debate_agents": args.debate_agents,
            "debate_rounds": args.debate_rounds,
            "self_consistency_n": args.self_consistency_n,
            "sleep_between_samples": args.sleep_between_samples,
            "max_concurrency": args.max_concurrency,
            "use_images": not args.no_images,
            "tolerate_truncated_images": not args.strict_images,
        },
    )
    print(f"wrote: {output_path}")
    print(f"wrote: {output_path.with_suffix('.summary.json')}")
    return 0


def main() -> int:
    return asyncio.run(run(parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
