from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from pathlib import Path
from typing import Any, Optional

import yaml

from sciorch.model_list import resolve_model_list

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class MCTSConfig:
    orchestra_model: str
    candidate_models: list[str]
    output_dir: Path
    main_use_images: bool = True
    orchestra_samples_per_prompt: int = 1
    instruction_similarity_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    instruction_similarity_batch_size: int = 32
    instruction_similarity_local_files_only: bool = False
    orchestra_enable_thinking: Optional[bool] = None
    delegate_enable_thinking: Optional[bool] = None
    orchestra_endpoint: str = "local"  # remote | local
    orchestra_local_base_url: Optional[str] = None
    orchestra_local_api_key: Optional[str] = None
    orchestra_local_api_key_env: str = "orchestra_api_key"
    orchestra_local_temperature: Optional[float] = None
    orchestra_openai_base_url: Optional[str] = None
    orchestra_openai_base_url_env: str = "orchestra_base_url"
    orchestra_openai_api_key_env: str = "orchestra_api_key"
    delegate_endpoint: str = "remote"  # remote | local
    delegate_local_base_url: Optional[str] = None
    delegate_local_api_key: Optional[str] = None
    delegate_local_api_key_env: str = "api_key"
    delegate_openai_base_url: Optional[str] = None
    delegate_openai_base_url_env: str = "base_url"
    delegate_openai_api_key_env: str = "api_key"
    dataset_name: str = "InternScience/SGI-Reasoning"
    dataset_split: str = "test"
    discipline: str | list[str] = "all"
    task_ids: list[str] | None = None
    exclude_task_ids: list[str] | None = None
    exclude_task_ids_path: Optional[Path] = None
    sample_count: int = 5
    sample_seed: int = 42
    tree_seed: int = 42
    resume: bool = True
    tree_budget_usd: float = 1.0
    target_leaf_trajectories: int | None = 20
    branching_factor: int = 2
    leaf_expand_ratio: float = 0.5
    frontier_limit: int | None = None
    sibling_pool_strategy: str = "random_partition"
    correct_model_pool_dir: Optional[Path] = None
    node_model_pool_size: int = 4
    node_max_steps: int = 4
    max_concurrency: int = 1
    show_progress: bool = True

    @staticmethod
    def _parse_bool(value: Any, default: bool) -> bool:
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"1", "true", "yes", "on"}:
                return True
            if lowered in {"0", "false", "no", "off"}:
                return False
        return bool(value)

    @classmethod
    def _parse_optional_bool(cls, value: Any) -> Optional[bool]:
        if value is None:
            return None
        return cls._parse_bool(value, default=False)

    @staticmethod
    def _resolve_path(path_value: str, config_path: Path) -> Path:
        path = Path(path_value)
        if path.is_absolute():
            return path
        return (PROJECT_ROOT / path).resolve()

    @classmethod
    def _resolve_optional_path(cls, path_value: Any, config_path: Path) -> Optional[Path]:
        if path_value is None:
            return None
        text = str(path_value).strip()
        if not text:
            return None
        return cls._resolve_path(text, config_path)

    @classmethod
    def load(cls, config_path: str | Path) -> "MCTSConfig":
        path = Path(config_path)
        with path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        candidate_models = resolve_model_list(raw.get("candidate_models"), config_path=path)
        if not candidate_models:
            raise ValueError("candidate_models must be a non-empty list or a path to a YAML list file")

        output_dir = cls._resolve_path(str(raw.get("output_dir", "results/mcts")), path)

        discipline = raw.get("discipline", "all")
        if isinstance(discipline, list):
            discipline = [str(item) for item in discipline]
        else:
            discipline = str(discipline)

        task_ids_raw = raw.get("task_ids", raw.get("task_id"))
        task_ids: list[str] | None = None
        if task_ids_raw is not None:
            if isinstance(task_ids_raw, list):
                task_ids = [str(item).strip() for item in task_ids_raw if str(item).strip()]
            elif isinstance(task_ids_raw, str):
                task_ids = [item.strip() for item in task_ids_raw.split(",") if item.strip()]
            else:
                raise ValueError("task_ids must be a string or a list of strings")
            if not task_ids:
                raise ValueError("task_ids must not be empty when provided")

        exclude_task_ids_raw = raw.get("exclude_task_ids")
        exclude_task_ids: list[str] | None = None
        if exclude_task_ids_raw is not None:
            if not isinstance(exclude_task_ids_raw, list):
                raise ValueError("exclude_task_ids must be a list when provided")
            exclude_task_ids = [str(item).strip() for item in exclude_task_ids_raw if str(item).strip()]
        exclude_task_ids_path = cls._resolve_optional_path(raw.get("exclude_task_ids_path"), path)

        orchestra_model = str(raw.get("orchestra_model", "qwen3-vl-8b")).strip()
        if not orchestra_model:
            raise ValueError("orchestra_model is required")

        orchestra_endpoint = str(raw.get("orchestra_endpoint", "local")).strip().lower()
        if orchestra_endpoint not in {"remote", "local"}:
            raise ValueError("orchestra_endpoint must be one of: remote, local")

        delegate_endpoint = str(raw.get("delegate_endpoint", raw.get("model_endpoint", "remote"))).strip().lower()
        if delegate_endpoint not in {"remote", "local"}:
            raise ValueError("delegate_endpoint must be one of: remote, local")

        config = cls(
            orchestra_model=orchestra_model,
            candidate_models=candidate_models,
            output_dir=output_dir,
            main_use_images=cls._parse_bool(raw.get("main_use_images", True), default=True),
            orchestra_samples_per_prompt=int(raw.get("orchestra_samples_per_prompt", 1)),
            instruction_similarity_model_name=str(
                raw.get("instruction_similarity_model_name", "sentence-transformers/all-MiniLM-L6-v2")
            ).strip(),
            instruction_similarity_batch_size=int(raw.get("instruction_similarity_batch_size", 32)),
            instruction_similarity_local_files_only=cls._parse_bool(
                raw.get("instruction_similarity_local_files_only", False),
                default=False,
            ),
            orchestra_enable_thinking=cls._parse_optional_bool(raw.get("orchestra_enable_thinking")),
            delegate_enable_thinking=cls._parse_optional_bool(raw.get("delegate_enable_thinking")),
            orchestra_endpoint=orchestra_endpoint,
            orchestra_local_base_url=raw.get("orchestra_local_base_url", raw.get("local_base_url")),
            orchestra_local_api_key=raw.get("orchestra_local_api_key", raw.get("local_api_key")),
            orchestra_local_api_key_env=str(
                raw.get("orchestra_local_api_key_env", raw.get("local_api_key_env", "orchestra_api_key"))
            ),
            orchestra_local_temperature=(
                float(raw["orchestra_local_temperature"])
                if raw.get("orchestra_local_temperature") is not None
                else (
                    float(os.environ["ORCHESTRA_LOCAL_TEMPERATURE"])
                    if os.getenv("ORCHESTRA_LOCAL_TEMPERATURE") not in {None, ""}
                    else None
                )
            ),
            orchestra_openai_base_url=raw.get("orchestra_openai_base_url", raw.get("openai_base_url")),
            orchestra_openai_base_url_env=str(
                raw.get("orchestra_openai_base_url_env", raw.get("openai_base_url_env", "orchestra_base_url"))
            ),
            orchestra_openai_api_key_env=str(
                raw.get("orchestra_openai_api_key_env", raw.get("openai_api_key_env", "orchestra_api_key"))
            ),
            delegate_endpoint=delegate_endpoint,
            delegate_local_base_url=raw.get("delegate_local_base_url", raw.get("local_base_url")),
            delegate_local_api_key=raw.get("delegate_local_api_key", raw.get("local_api_key")),
            delegate_local_api_key_env=str(
                raw.get("delegate_local_api_key_env", raw.get("local_api_key_env", "api_key"))
            ),
            delegate_openai_base_url=raw.get("delegate_openai_base_url", raw.get("openai_base_url")),
            delegate_openai_base_url_env=str(
                raw.get("delegate_openai_base_url_env", raw.get("openai_base_url_env", "base_url"))
            ),
            delegate_openai_api_key_env=str(
                raw.get("delegate_openai_api_key_env", raw.get("openai_api_key_env", "api_key"))
            ),
            dataset_name=str(raw.get("dataset_name", "InternScience/SGI-Reasoning")),
            dataset_split=str(raw.get("dataset_split", "test")),
            discipline=discipline,
            task_ids=task_ids,
            exclude_task_ids=exclude_task_ids,
            exclude_task_ids_path=exclude_task_ids_path,
            sample_count=int(raw.get("sample_count", raw.get("wrong_sample_count", 5))),
            sample_seed=int(raw.get("sample_seed", raw.get("wrong_sample_seed", 42))),
            tree_seed=int(raw.get("tree_seed", 42)),
            resume=cls._parse_bool(raw.get("resume", True), default=True),
            tree_budget_usd=float(raw.get("tree_budget_usd", 1.0)),
            target_leaf_trajectories=(
                20
                if "target_leaf_trajectories" not in raw
                else (
                    int(raw["target_leaf_trajectories"])
                    if raw.get("target_leaf_trajectories") is not None
                    else None
                )
            ),
            branching_factor=int(raw.get("branching_factor", 2)),
            leaf_expand_ratio=float(raw.get("leaf_expand_ratio", 0.5)),
            frontier_limit=(
                int(raw["frontier_limit"])
                if raw.get("frontier_limit") is not None
                else None
            ),
            sibling_pool_strategy=str(raw.get("sibling_pool_strategy", "random_partition")).strip().lower(),
            correct_model_pool_dir=cls._resolve_optional_path(raw.get("correct_model_pool_dir"), path),
            node_model_pool_size=int(raw.get("node_model_pool_size", 4)),
            node_max_steps=int(raw.get("node_max_steps", raw.get("node_max_attempts", 4))),
            max_concurrency=int(raw.get("max_concurrency", 1)),
            show_progress=cls._parse_bool(raw.get("show_progress", True), default=True),
        )
        effective_delegate_pool_size = len(config.candidate_models)
        if config.node_model_pool_size > effective_delegate_pool_size:
            raise ValueError("node_model_pool_size cannot exceed the effective delegate model pool length")
        if config.sample_count <= 0:
            raise ValueError("sample_count must be positive")
        if config.target_leaf_trajectories is not None and config.target_leaf_trajectories <= 0:
            raise ValueError("target_leaf_trajectories must be positive")
        if not 0.0 < config.leaf_expand_ratio <= 1.0:
            raise ValueError("leaf_expand_ratio must be in (0, 1]")
        if config.branching_factor <= 0:
            raise ValueError("branching_factor must be positive")
        if config.orchestra_samples_per_prompt <= 0:
            raise ValueError("orchestra_samples_per_prompt must be positive")
        if not config.instruction_similarity_model_name:
            raise ValueError("instruction_similarity_model_name must not be empty")
        if config.instruction_similarity_batch_size <= 0:
            raise ValueError("instruction_similarity_batch_size must be positive")
        if config.frontier_limit is not None and config.frontier_limit <= 0:
            raise ValueError("frontier_limit must be positive when provided")
        if config.sibling_pool_strategy not in {"sample", "random_partition", "correct_50_50"}:
            raise ValueError("sibling_pool_strategy must be one of: sample, random_partition, correct_50_50")
        if config.node_max_steps <= 0:
            raise ValueError("node_max_steps must be positive")
        return config

    def to_json(self) -> dict[str, Any]:
        data = asdict(self)
        data["output_dir"] = str(self.output_dir)
        return data
