from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Optional

import yaml

from sciorch.model_list import resolve_model_list

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class OrchestratorConfig:
    main_model: str
    sub_models: list[str]
    main_use_images: bool = True
    main_enable_thinking: Optional[bool] = None
    sub_enable_thinking: Optional[bool] = None
    judge_enable_thinking: Optional[bool] = None
    main_model_endpoint: str = "remote"  # remote | local
    main_local_base_url: Optional[str] = None
    main_local_api_key: Optional[str] = None
    main_local_api_key_env: str = "OPENAI_API_KEY"
    main_local_temperature: Optional[float] = None
    main_max_tokens: Optional[int] = None
    main_repetition_penalty: Optional[float] = None
    enable_scoring: bool = True
    judge_model: str = "o4-mini"
    max_steps: int = 4
    max_concurrency: int = 2
    discipline: str | list[str] = "all"
    dataset_split: str = "test"
    dataset_name: str = "InternScience/SGI-Reasoning"
    output_dir: Path = Path("/results")
    exclude_task_ids: list[str] | None = None
    exclude_task_ids_path: Optional[Path] = None
    task_ids: list[str] | None = None
    max_samples: Optional[int] = None
    sample_method: str = "head"  # head | random | stratified_discipline
    sample_seed: int = 42
    openai_base_url: Optional[str] = None
    openai_api_key_env: str = "OPENAI_API_KEY"
    openai_api_key: Optional[str] = None

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

    @classmethod
    def load(cls, config_path: str | Path) -> "OrchestratorConfig":
        config_path = Path(config_path)
        with config_path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        main_model = str(raw.get("main_model", "")).strip()
        if not main_model:
            raise ValueError("main_model is required")

        sub_models = resolve_model_list(raw.get("sub_models"), config_path=config_path)
        if not sub_models:
            raise ValueError("sub_models must be a non-empty list or a path to a YAML list file")

        output_dir = raw.get("output_dir", "/results")
        output_dir = cls._resolve_path(output_dir, config_path)
        exclude_task_ids_raw = raw.get("exclude_task_ids")
        exclude_task_ids = None
        if exclude_task_ids_raw is not None:
            if not isinstance(exclude_task_ids_raw, list):
                raise ValueError("exclude_task_ids must be a list when provided")
            exclude_task_ids = [str(item) for item in exclude_task_ids_raw if str(item).strip()]
        exclude_task_ids_path = cls._resolve_optional_path(raw.get("exclude_task_ids_path"), config_path)
        task_ids_raw = raw.get("task_ids")
        task_ids = None
        if task_ids_raw is not None:
            if not isinstance(task_ids_raw, list):
                raise ValueError("task_ids must be a list when provided")
            task_ids = [str(item) for item in task_ids_raw if str(item).strip()]

        discipline = raw.get("discipline", "all")
        if isinstance(discipline, list):
            discipline = [str(item) for item in discipline]
        else:
            discipline = str(discipline)

        sample_method = str(raw.get("sample_method", "head")).strip().lower()
        if sample_method not in {"head", "random", "stratified_discipline"}:
            raise ValueError("sample_method must be one of: head, random, stratified_discipline")

        main_model_endpoint = str(raw.get("main_model_endpoint", "remote")).strip().lower()
        if main_model_endpoint not in {"remote", "local"}:
            raise ValueError("main_model_endpoint must be one of: remote, local")

        config = cls(
            main_model=main_model,
            sub_models=sub_models,
            main_use_images=cls._parse_bool(raw.get("main_use_images", True), default=True),
            main_enable_thinking=cls._parse_optional_bool(raw.get("main_enable_thinking")),
            sub_enable_thinking=cls._parse_optional_bool(raw.get("sub_enable_thinking")),
            judge_enable_thinking=cls._parse_optional_bool(raw.get("judge_enable_thinking")),
            main_model_endpoint=main_model_endpoint,
            main_local_base_url=raw.get("main_local_base_url"),
            main_local_api_key=raw.get("main_local_api_key"),
            main_local_api_key_env=str(raw.get("main_local_api_key_env", "OPENAI_API_KEY")),
            main_local_temperature=(
                float(raw["main_local_temperature"])
                if raw.get("main_local_temperature") is not None
                else (
                    float(os.environ["MAIN_LOCAL_TEMPERATURE"])
                    if os.getenv("MAIN_LOCAL_TEMPERATURE") not in {None, ""}
                    else None
                )
            ),
            main_max_tokens=int(raw["main_max_tokens"]) if raw.get("main_max_tokens") is not None else None,
            main_repetition_penalty=float(raw["main_repetition_penalty"]) if raw.get("main_repetition_penalty") is not None else None,
            enable_scoring=cls._parse_bool(raw.get("enable_scoring", True), default=True),
            judge_model=str(raw.get("judge_model", "o4-mini")),
            max_steps=int(raw.get("max_steps", raw.get("max_attempts", 4))),
            max_concurrency=int(raw.get("max_concurrency", 2)),
            discipline=discipline,
            dataset_split=str(raw.get("dataset_split", "test")),
            dataset_name=str(raw.get("dataset_name", "InternScience/SGI-Reasoning")),
            output_dir=output_dir,
            exclude_task_ids=exclude_task_ids,
            exclude_task_ids_path=exclude_task_ids_path,
            task_ids=task_ids,
            max_samples=int(raw["max_samples"]) if raw.get("max_samples") is not None else None,
            sample_method=sample_method,
            sample_seed=int(raw.get("sample_seed", 42)),
            openai_base_url=raw.get("openai_base_url"),
            openai_api_key_env=str(raw.get("openai_api_key_env", "OPENAI_API_KEY")),
            openai_api_key=raw.get("openai_api_key"),
        )
        if config.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if config.max_concurrency <= 0:
            raise ValueError("max_concurrency must be positive")
        return config

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

    def discipline_list(self) -> list[str] | None:
        if isinstance(self.discipline, str) and self.discipline.lower() == "all":
            return None
        if isinstance(self.discipline, list):
            return self.discipline
        return [self.discipline]

    def discipline_repr(self) -> str:
        if isinstance(self.discipline, str):
            if self.discipline.lower() == "all":
                return "['all']"
            return str([self.discipline])
        return str(self.discipline)
