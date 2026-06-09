from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def dump_json(path: Path, data: Any) -> None:
    _ensure_dir(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def export_reasoning_compatible_logs(
    output_dir: Path,
    model_name: str,
    discipline_repr: str,
    records: list[dict[str, Any]],
) -> Path:
    sanitized_model = model_name.replace("/", "_")
    target = (
        output_dir
        / "reasoning_compatible"
        / "experimental_reasoning"
        / "logs"
        / f"{sanitized_model}{discipline_repr}.json"
    )
    dump_json(target, records)
    return target
