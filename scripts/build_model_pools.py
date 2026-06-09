#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCIORCH_ROOT = REPO_ROOT
if str(DEFAULT_SCIORCH_ROOT) not in sys.path:
    sys.path.insert(0, str(DEFAULT_SCIORCH_ROOT))

try:
    from sciorch.core.parsing import extract_unique_boxed_letter
except Exception:
    extract_unique_boxed_letter = None


ANSWER_PATTERNS = [
    re.compile(r"\\boxed\{\s*([A-Ja-j])\s*\}"),
    re.compile(r"[Ff]inal [Aa]nswer\s*[:：]?\s*([A-Ja-j])\b"),
    re.compile(r"[Tt]he answer is\s*[:：]?\s*([A-Ja-j])\b"),
    re.compile(r"[Aa]nswer\s*[:：]?\s*([A-Ja-j])\b"),
]


@dataclass
class TaskRecord:
    task_id: str
    question: str
    gold_answer_letter: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build per-task correct-model pools for an existing MCTS run by scanning SGI evaluation JSON files."
    )
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return payload


def normalize_question(text: str) -> str:
    text = str(text or "").lower()
    text = text.replace("<image><image>", " ")
    text = text.replace("<image> <image>", " ")
    text = text.replace("<image>", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_task_records(base_run_dir: Path) -> list[TaskRecord]:
    scored_path = base_run_dir / "scored.json"
    rows = json.loads(scored_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"Expected a list in {scored_path}")

    records: list[TaskRecord] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        task_id = str(row.get("task_id") or "").strip()
        question = str(row.get("question") or "").strip()
        gold = str(row.get("gold_answer_letter") or "").strip().upper()
        if not task_id or not question or not gold:
            continue
        records.append(TaskRecord(task_id=task_id, question=question, gold_answer_letter=gold))
    if not records:
        raise ValueError(f"No usable task rows found in {scored_path}")
    return records


def extract_prediction_letter(text: str) -> str | None:
    raw = str(text or "").strip()
    if not raw:
        return None
    if extract_unique_boxed_letter is not None:
        letter, _ = extract_unique_boxed_letter(raw)
        if letter:
            return letter.upper()
    for pattern in ANSWER_PATTERNS:
        match = pattern.search(raw)
        if match:
            return match.group(1).upper()
    return None


def extract_gold_letter(row: dict[str, Any]) -> str | None:
    formatted = row.get("formatted_a")
    if isinstance(formatted, str):
        formatted = formatted.strip().upper()
        if len(formatted) == 1 and formatted.isalpha():
            return formatted

    answer = row.get("answer")
    if isinstance(answer, int) and 0 <= answer < 26:
        return chr(ord("A") + answer)

    return None


def ensure_unique_question_map(tasks: list[TaskRecord]) -> dict[str, TaskRecord]:
    mapping: dict[str, TaskRecord] = {}
    duplicates: list[str] = []
    for task in tasks:
        key = normalize_question(task.question)
        if not key:
            continue
        if key in mapping and mapping[key].task_id != task.task_id:
            duplicates.append(task.task_id)
            continue
        mapping[key] = task
    if duplicates:
        raise ValueError(f"Found duplicated normalized questions in base run: {duplicates[:10]}")
    return mapping


def scan_evaluation_file(
    eval_path: Path,
    question_to_task: dict[str, TaskRecord],
    per_task_rows: dict[str, list[dict[str, Any]]],
    unmatched_models: list[str],
) -> None:
    model_name = eval_path.stem
    rows = json.loads(eval_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"Expected a list in {eval_path}")

    matched_any = False
    for row_index, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        key = normalize_question(row.get("problem", ""))
        task = question_to_task.get(key)
        if task is None:
            continue

        matched_any = True
        predicted_letter = extract_prediction_letter(row.get("prediction", ""))
        gold_letter = extract_gold_letter(row) or task.gold_answer_letter
        per_task_rows[task.task_id].append(
            {
                "model": model_name,
                "evaluation_row_index": row_index,
                "predicted_letter": predicted_letter,
                "gold_letter": gold_letter,
                "is_correct": predicted_letter == gold_letter if predicted_letter is not None else False,
            }
        )

    if not matched_any:
        unmatched_models.append(model_name)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_correct_model_pools_from_config(
    config_path: str | Path,
    *,
    verbose: bool = True,
) -> dict[str, Any]:
    config_path = Path(config_path).expanduser().resolve()
    config = load_yaml(config_path)

    base_run_dir = Path(config["base_run_dir"]).expanduser().resolve()
    evaluation_dir = Path(config["evaluation_dir"]).expanduser().resolve()
    output_dir = Path(config["pool_output_dir"]).expanduser().resolve()
    task_output_dir = output_dir / "tasks"

    tasks = load_task_records(base_run_dir)
    question_to_task = ensure_unique_question_map(tasks)
    per_task_rows: dict[str, list[dict[str, Any]]] = {task.task_id: [] for task in tasks}
    unmatched_models: list[str] = []

    eval_files = sorted(evaluation_dir.glob("*.json"))
    if not eval_files:
        raise ValueError(f"No evaluation JSON files found under {evaluation_dir}")

    for eval_path in eval_files:
        scan_evaluation_file(
            eval_path=eval_path,
            question_to_task=question_to_task,
            per_task_rows=per_task_rows,
            unmatched_models=unmatched_models,
        )

    task_payloads: list[dict[str, Any]] = []
    no_correct_model_task_ids: list[str] = []
    for task in tasks:
        model_rows = sorted(
            per_task_rows[task.task_id],
            key=lambda item: (not bool(item["is_correct"]), item["model"]),
        )
        correct_models = [row["model"] for row in model_rows if row["is_correct"]]
        if not correct_models:
            no_correct_model_task_ids.append(task.task_id)

        payload = {
            "task_id": task.task_id,
            "question": task.question,
            "gold_answer_letter": task.gold_answer_letter,
            "correct_models": correct_models,
            "correct_model_count": len(correct_models),
            "models_evaluated_count": len(model_rows),
            "model_rows": model_rows,
        }
        task_payloads.append(payload)
        write_json(task_output_dir / f"{task.task_id}.json", payload)

    summary = {
        "base_run_dir": str(base_run_dir),
        "evaluation_dir": str(evaluation_dir),
        "pool_output_dir": str(output_dir),
        "task_count": len(tasks),
        "evaluation_model_count": len(eval_files),
        "unmatched_models": unmatched_models,
        "task_ids_without_correct_models": no_correct_model_task_ids,
        "tasks": task_payloads,
    }
    write_json(output_dir / "summary.json", summary)
    write_json(output_dir / "task_ids_without_correct_models.json", no_correct_model_task_ids)

    if verbose:
        print(f"task_count: {len(tasks)}")
        print(f"evaluation_model_count: {len(eval_files)}")
        print(f"task_ids_without_correct_models_count: {len(no_correct_model_task_ids)}")
        print(f"summary_path: {output_dir / 'summary.json'}")
        print(f"task_output_dir: {task_output_dir}")
    return summary


def main() -> int:
    args = parse_args()
    build_correct_model_pools_from_config(args.config, verbose=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
