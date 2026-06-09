#!/usr/bin/env python3
"""
run_model_eval.py
=================
Run each candidate (sub-agent) model over a dataset and record which models
answer each task correctly vs. wrong. This produces the per-task "model pool"
signal that MCTS later reads (via scripts/build_model_pools.py) to know who can
solve which task.

The sub-agent API endpoint is read from the environment — the same variables the
configs reference (delegate_openai_base_url_env: base_url,
delegate_openai_api_key_env: api_key):

    export base_url="https://your-openai-compatible-endpoint/v1"
    export api_key="..."

Output: one JSON file per model under --output-dir (default data/model_evaluations/).
Each file is a list of rows holding `problem` + `prediction` (consumed by
scripts/build_model_pools.py) plus `gold_letter` / `predicted_letter` /
`is_correct` for quick inspection. Re-runs resume: already-answered tasks are kept.

Usage:
    # default: run configs/models.yaml on the training set -> data/model_evaluations/
    python scripts/run_model_eval.py

    # custom dataset / models / output / concurrency
    python scripts/run_model_eval.py \
        --dataset data/test_combined.json \
        --models configs/models.yaml \
        --output-dir data/model_evaluations \
        --concurrency 24
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sciorch.adapters.reasoning.dataset import _parse_answer_index
from sciorch.core.parsing import extract_unique_boxed_letter
from sciorch.llm.openai_compatible import OpenAICompatibleClient
from sciorch.model_list import resolve_model_list

SYSTEM_PROMPT = "You are a rigorous multimodal scientific reasoning assistant."


def _index_to_letter(idx: int | None) -> str | None:
    if idx is None or idx < 0:
        return None
    return chr(ord("A") + idx)


def _predicted_letter(text: str) -> str | None:
    letter, _ = extract_unique_boxed_letter(text or "")
    return letter.upper() if letter else None


def build_prompt(question: str, options: list[str]) -> str:
    lines = [str(question).strip()]
    opts = [str(o) for o in (options or []) if str(o).strip()]
    if opts:
        lines.append("")
        for i, opt in enumerate(opts):
            lines.append(f"{chr(ord('A') + i)}. {opt}")
        lines.append("")
        lines.append("Answer with the single correct option letter, formatted as \\boxed{LETTER}.")
    else:
        lines.append("")
        lines.append("Put your final answer in \\boxed{}.")
    return "\n".join(lines)


def load_samples(dataset: str, max_samples: int) -> list[dict]:
    path = Path(dataset)
    if not path.is_absolute():
        path = REPO_ROOT / dataset
    items = json.loads(path.read_text(encoding="utf-8"))
    samples: list[dict] = []
    for item in items:
        options = [str(o) for o in (item.get("options") or [])]
        gold_idx = _parse_answer_index(item.get("answer"), options)
        samples.append(
            {
                "task_id": str(item.get("id", len(samples))),
                "question": str(item.get("question", "")),
                "options": options,
                "images": list(item.get("images") or []),
                "source": item.get("source"),
                "subject": item.get("subject"),
                "gold_letter": _index_to_letter(gold_idx),
            }
        )
    if max_samples and max_samples > 0:
        samples = samples[:max_samples]
    return samples


async def eval_one_model(
    client: OpenAICompatibleClient,
    model: str,
    samples: list[dict],
    out_path: Path,
    *,
    concurrency: int,
    max_tokens: int,
    temperature: float,
) -> tuple[int, int]:
    """Run one model over all samples; write/refresh out_path; return (done, correct)."""
    results: dict[str, dict] = {}
    if out_path.exists():  # resume: keep prior rows
        try:
            for row in json.loads(out_path.read_text(encoding="utf-8")):
                if isinstance(row, dict) and row.get("task_id") is not None:
                    results[str(row["task_id"])] = row
        except Exception:
            results = {}

    pending = [s for s in samples if s["task_id"] not in results]
    semaphore = asyncio.Semaphore(concurrency)

    def flush() -> None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        ordered = [results[s["task_id"]] for s in samples if s["task_id"] in results]
        out_path.write_text(json.dumps(ordered, ensure_ascii=False, indent=2), encoding="utf-8")

    async def run_one(sample: dict) -> None:
        async with semaphore:
            prompt = build_prompt(sample["question"], sample["options"])
            error = None
            try:
                res = await client.ask(
                    model=model,
                    system_prompt=SYSTEM_PROMPT,
                    user_prompt=prompt,
                    images=sample["images"],
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                text = res.text
            except Exception as exc:  # noqa: BLE001 — record and continue
                text, error = "", repr(exc)
            pred = _predicted_letter(text)
            gold = sample["gold_letter"]
            row = {
                "task_id": sample["task_id"],
                "problem": sample["question"],   # build_model_pools matches on this
                "prediction": text,              # build_model_pools parses this
                "options": sample["options"],
                "predicted_letter": pred,
                "gold_letter": gold,
                "is_correct": pred is not None and pred == gold,
                "source": sample["source"],
                "subject": sample["subject"],
            }
            if error:
                row["error"] = error
            results[sample["task_id"]] = row
            flush()  # no await in this block -> atomic within the event loop

    if pending:
        await asyncio.gather(*(run_one(s) for s in pending))
    flush()
    ordered = [results[s["task_id"]] for s in samples if s["task_id"] in results]
    correct = sum(1 for r in ordered if r.get("is_correct"))
    return len(ordered), correct


def _parse_models(value: str) -> list[str]:
    if value.endswith((".yaml", ".yml")) and "," not in value:
        return resolve_model_list(value)
    return [m.strip() for m in value.split(",") if m.strip()]


def main() -> int:
    p = argparse.ArgumentParser(
        description="Run candidate models over a dataset to record per-task correct/wrong."
    )
    p.add_argument("--dataset", default="data/train_combined.json", help="local combined JSON")
    p.add_argument("--models", default="configs/models.yaml",
                   help="YAML list file (e.g. configs/models.yaml) or comma-separated model names")
    p.add_argument("--output-dir", default="data/model_evaluations")
    p.add_argument("--concurrency", type=int, default=24)
    p.add_argument("--max-samples", type=int, default=0, help="0 = all")
    p.add_argument("--max-tokens", type=int, default=4000)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--base-url", default=None, help="override env base_url")
    args = p.parse_args()

    models = _parse_models(args.models)
    if not models:
        raise SystemExit("no models to run (check --models)")

    samples = load_samples(args.dataset, args.max_samples)
    if not samples:
        raise SystemExit(f"no samples found in {args.dataset}")

    # Sub-agent endpoint from env: base_url / api_key (override base url via --base-url).
    client = OpenAICompatibleClient(
        base_url=args.base_url,
        api_key_env="api_key",
        default_temperature=args.temperature,
    )

    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / args.output_dir

    print(f"[run_model_eval] dataset={args.dataset} samples={len(samples)} "
          f"models={len(models)} -> {out_dir}")
    for i, model in enumerate(models, 1):
        out_path = out_dir / f"{model}.json"
        done, correct = asyncio.run(
            eval_one_model(
                client, model, samples, out_path,
                concurrency=args.concurrency,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
            )
        )
        acc = correct / done if done else 0.0
        print(f"  [{i}/{len(models)}] {model}: {done} tasks, {correct} correct "
              f"(acc={acc:.3f}) -> {out_path.name}")
    print("[done]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
