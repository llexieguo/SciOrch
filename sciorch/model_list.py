"""Resolve a model-list config value.

A ``sub_models`` / ``candidate_models`` value may be either an inline YAML list
(backward compatible) or a string path to a YAML file holding the list (a bare
list, or a mapping with a ``models:`` key). This lets all configs share one
``configs/models.yaml`` instead of duplicating the pool.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def resolve_model_list(value: Any, *, config_path: Optional[Path | str] = None) -> list[str]:
    """Return a clean ``list[str]`` from an inline list or a path-to-file value.

    Relative paths are resolved against, in order: the current working dir, the
    repo root, then the referencing config file's directory. Returns ``[]`` for
    a None/empty value; raises FileNotFoundError if a path is given but missing.
    """
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, (str, Path)):
        ref = str(value).strip()
        if not ref:
            return []
        p = Path(ref)
        if p.is_absolute():
            candidates = [p]
        else:
            candidates = [Path.cwd() / ref, REPO_ROOT / ref]
            if config_path is not None:
                candidates.append(Path(config_path).resolve().parent / ref)
        for candidate in candidates:
            if candidate.is_file():
                with candidate.open("r", encoding="utf-8") as f:
                    loaded = yaml.safe_load(f)
                if isinstance(loaded, dict):
                    loaded = loaded.get("models")
                return [str(item).strip() for item in (loaded or []) if str(item).strip()]
        raise FileNotFoundError(
            f"model list file not found: {ref} (tried {[str(c) for c in candidates]})"
        )
    raise TypeError(f"model list must be a list or a file path, got {type(value).__name__}")
