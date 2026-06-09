from __future__ import annotations

import re

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from sciorch.config import OrchestratorConfig
from sciorch.types import ReasoningSample


def _sample_head(samples: list[ReasoningSample], k: int) -> list[ReasoningSample]:
    return samples[:k]


def _sample_random(samples: list[ReasoningSample], k: int, seed: int) -> list[ReasoningSample]:
    rng = random.Random(seed)
    indices = list(range(len(samples)))
    rng.shuffle(indices)
    return [samples[idx] for idx in indices[:k]]


def _sample_stratified_discipline(samples: list[ReasoningSample], k: int, seed: int) -> list[ReasoningSample]:
    """Proportional stratified sampling by discipline with deterministic seed."""
    groups: dict[str, list[ReasoningSample]] = defaultdict(list)
    for sample in samples:
        groups[sample.discipline].append(sample)

    rng = random.Random(seed)
    for bucket in groups.values():
        rng.shuffle(bucket)

    total = len(samples)
    quotas: dict[str, int] = {}
    remainders: list[tuple[float, str]] = []
    allocated = 0

    for discipline, bucket in groups.items():
        exact = (k * len(bucket)) / total
        base = int(exact)
        quotas[discipline] = min(base, len(bucket))
        allocated += quotas[discipline]
        remainders.append((exact - base, discipline))

    remaining = k - allocated
    for _, discipline in sorted(remainders, key=lambda item: (-item[0], item[1])):
        if remaining <= 0:
            break
        if quotas[discipline] < len(groups[discipline]):
            quotas[discipline] += 1
            remaining -= 1

    if remaining > 0:
        for discipline, _ in sorted(
            ((name, len(bucket) - quotas[name]) for name, bucket in groups.items()),
            key=lambda item: item[1],
            reverse=True,
        ):
            if remaining <= 0:
                break
            capacity = len(groups[discipline]) - quotas[discipline]
            if capacity <= 0:
                continue
            add = min(capacity, remaining)
            quotas[discipline] += add
            remaining -= add

    selected: list[ReasoningSample] = []
    for discipline in sorted(groups.keys()):
        selected.extend(groups[discipline][: quotas[discipline]])

    rng.shuffle(selected)
    return selected[:k]


def _apply_sampling(config: OrchestratorConfig, samples: list[ReasoningSample]) -> list[ReasoningSample]:
    # None or a non-positive value (e.g. max_samples: 0) means "use all samples".
    if config.max_samples is None:
        return samples
    if config.max_samples <= 0:
        return samples
    if config.max_samples >= len(samples):
        return samples

    k = int(config.max_samples)
    if config.sample_method == "head":
        return _sample_head(samples, k)
    if config.sample_method == "random":
        return _sample_random(samples, k, config.sample_seed)
    if config.sample_method == "stratified_discipline":
        return _sample_stratified_discipline(samples, k, config.sample_seed)

    raise ValueError(f"Unsupported sample_method: {config.sample_method}")


def _extract_task_ids_from_json_value(value: Any) -> set[str]:
    task_ids: set[str] = set()
    if isinstance(value, dict):
        for key in ("task_id", "idx", "id"):
            item = value.get(key)
            if item is not None:
                task_ids.add(str(item))
        embedded = value.get("task_ids")
        if isinstance(embedded, list):
            task_ids.update(str(item) for item in embedded if item is not None)
    elif isinstance(value, list):
        if all(not isinstance(item, (dict, list)) for item in value):
            task_ids.update(str(item) for item in value if item is not None)
        else:
            for item in value:
                task_ids.update(_extract_task_ids_from_json_value(item))
    return task_ids


def _load_task_ids_from_file(path: Path) -> set[str]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        task_ids: set[str] = set()
        with path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                task_ids.update(_extract_task_ids_from_json_value(json.loads(line)))
        return task_ids

    if suffix == ".json":
        with path.open("r", encoding="utf-8") as handle:
            return _extract_task_ids_from_json_value(json.load(handle))

    task_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            task_ids.add(line)
    return task_ids


def _load_excluded_task_ids(path: Path | None) -> set[str]:
    if path is None:
        return set()
    if not path.exists():
        raise FileNotFoundError(f"exclude_task_ids_path does not exist: {path}")
    if path.is_file():
        return _load_task_ids_from_file(path)

    task_ids: set[str] = set()
    for candidate in sorted(path.rglob("*")):
        if not candidate.is_file():
            continue
        if candidate.suffix.lower() not in {".jsonl", ".json", ".txt"}:
            continue
        task_ids.update(_load_task_ids_from_file(candidate))
    return task_ids


def _rewrite_list_feature_type(node: Any) -> bool:
    """Rewrite legacy cached feature marker '_type: List' to 'Sequence'."""
    changed = False
    if isinstance(node, dict):
        if node.get("_type") == "List":
            node["_type"] = "Sequence"
            changed = True
        for value in node.values():
            if _rewrite_list_feature_type(value):
                changed = True
    elif isinstance(node, list):
        for value in node:
            if _rewrite_list_feature_type(value):
                changed = True
    return changed


def _patch_dataset_info_file(path: Path) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if not isinstance(payload, dict):
        return False
    changed = _rewrite_list_feature_type(payload)
    if not changed:
        return False
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return True


def _patch_cached_dataset_info(dataset_name: str) -> int:
    """
    Patch cached HuggingFace dataset_info.json files for old datasets versions.
    """
    cache_root = Path.home() / ".cache" / "huggingface" / "datasets"
    if not cache_root.exists():
        return 0
    repo_tail = dataset_name.split("/")[-1].strip().lower()
    patched = 0
    for path in cache_root.rglob("dataset_info.json"):
        parent = path.parent.name.lower()
        full_path = str(path).lower()
        if repo_tail and repo_tail not in parent and repo_tail not in full_path:
            continue
        if _patch_dataset_info_file(path):
            patched += 1
    return patched




def _parse_answer_index(answer, options: list) -> int:
    """Convert various answer formats to 0-based option index.

    Supports:
    - SGI: 0-based integer string ("5" -> 5)
    - SuperGPQA: single letter ("B" -> 1)
    - SFE: letter list string ("['D']" -> 3)
    """
    if answer is None:
        return 0
    ans = str(answer).strip()
    # 0-based integer (SGI): "5" -> 5
    if ans.isdigit():
        idx = int(ans)
        if 0 <= idx < len(options):
            return idx
        if 1 <= idx <= len(options):
            return idx - 1
        return 0
    # Single letter
    if len(ans) == 1 and ans.upper() in "ABCDEFGHIJ":
        return ord(ans.upper()) - ord("A")
    # SFE "['D']" style
    m = re.match(r"^\[?\s*['\"]?([A-Ja-j])['\"]?\s*\]?$", ans)
    if m:
        return ord(m.group(1).upper()) - ord("A")
    # Match against option text
    for i, opt in enumerate(options):
        if str(opt).strip() == ans:
            return i
    return 0


def load_reasoning_samples_from_local(config: OrchestratorConfig) -> "list[ReasoningSample]":
    """Load samples from a local JSON file (test_combined.json / train_combined.json)."""
    import json as _json
    path = Path(config.dataset_name)
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[3] / config.dataset_name
    with path.open("r", encoding="utf-8") as f:
        raw_items = _json.load(f)

    allow_disciplines = config.discipline_list()
    samples: list[ReasoningSample] = []
    for item in raw_items:
        discipline = str(item.get("subject") or item.get("discipline") or "unknown")
        if allow_disciplines and discipline not in allow_disciplines:
            continue
        options = [str(o) for o in item.get("options", [])]
        answer_index = _parse_answer_index(item.get("answer"), options)
        images = list(item.get("images") or [])
        sample = ReasoningSample(
            task_id=str(item.get("id", len(samples))),
            question=str(item.get("question", "")),
            options=options,
            answer_index=answer_index,
            steps=[],
            discipline=discipline,
            images=images,
        )
        samples.append(sample)

    excluded_task_ids = set(config.exclude_task_ids or [])
    if config.exclude_task_ids_path is not None:
        excluded_task_ids.update(_load_excluded_task_ids(config.exclude_task_ids_path))
    if excluded_task_ids:
        samples = [s for s in samples if s.task_id not in excluded_task_ids]
    if config.task_ids is not None:
        allowed = set(config.task_ids)
        samples = [s for s in samples if s.task_id in allowed]
        return samples
    return _apply_sampling(config, samples)


def load_reasoning_samples(config: OrchestratorConfig) -> list[ReasoningSample]:
    # Local JSON file support
    if config.dataset_name.endswith(".json"):
        return load_reasoning_samples_from_local(config)
    local_path = Path(config.dataset_name)
    if local_path.exists() and local_path.is_file():
        return load_reasoning_samples_from_local(config)

    try:
        from datasets import load_dataset
    except Exception as exc:
        raise RuntimeError("datasets package is required: pip install datasets") from exc

    try:
        raw_dataset = load_dataset(config.dataset_name, split=config.dataset_split)
    except ValueError as exc:
        msg = str(exc)
        if "Feature type 'List' not found" not in msg:
            raise
        patched = _patch_cached_dataset_info(config.dataset_name)
        if patched <= 0:
            raise RuntimeError(
                "Dataset cache format is incompatible with current datasets version "
                "(missing feature type 'List'), and no cache metadata could be patched. "
                "Try upgrading datasets, or refresh the local cache."
            ) from exc
        raw_dataset = load_dataset(config.dataset_name, split=config.dataset_split)
    allow_disciplines = config.discipline_list()

    samples: list[ReasoningSample] = []
    for item in raw_dataset:
        discipline = str(item.get("discipline", "unknown"))
        if allow_disciplines and discipline not in allow_disciplines:
            continue

        options = [str(option) for option in item.get("options", [])]
        answer_index = _parse_answer_index(item.get("answer", 0), options)
        steps = [str(step) for step in item.get("steps", [])]
        images = list(item.get("images", []))

        sample = ReasoningSample(
            task_id=str(item.get("idx", item.get("id", len(samples)))),
            question=str(item.get("question", "")),
            options=options,
            answer_index=answer_index,
            steps=steps,
            discipline=discipline,
            images=images,
        )
        samples.append(sample)

    excluded_task_ids = set(config.exclude_task_ids or [])
    if config.exclude_task_ids_path is not None:
        excluded_task_ids.update(_load_excluded_task_ids(config.exclude_task_ids_path))
    if excluded_task_ids:
        samples = [sample for sample in samples if sample.task_id not in excluded_task_ids]

    if config.task_ids is not None:
        allowed = set(config.task_ids)
        samples = [sample for sample in samples if sample.task_id in allowed]
        return samples

    return _apply_sampling(config, samples)
