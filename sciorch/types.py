from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ReasoningSample:
    task_id: str
    question: str
    options: list[str]
    answer_index: int
    steps: list[str]
    discipline: str
    images: list[Any]


@dataclass
class DelegateRequest:
    task_id: str
    question: str
    options: list[str]
    images: list[Any]
    model: str
    instruction: str
    task_type: Optional[str] = None
    prior_attempts: list["AttemptRecord"] = field(default_factory=list)


@dataclass
class DelegateResult:
    raw_answer_text: str
    answer: Optional[str]
    confidence: Optional[float]
    reasoning_summary: str
    parse_ok: bool
    error: Optional[str]
    cost: float
    thinking: str = ""
    system_prompt: Optional[str] = None
    user_prompt: Optional[str] = None
    parsed_payload: dict[str, Any] = field(default_factory=dict)
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class SubmitResult:
    final_answer_text: str
    final_boxed_letter: Optional[str]
    done: bool
    reason: str
    step_count: int


@dataclass
class MainAction:
    action: str  # delegate_task | submit
    reasoning: str
    thinking: str = ""
    model: Optional[str] = None
    instruction: Optional[str] = None
    task_type: Optional[str] = None
    submit_reason: Optional[str] = None
    final_answer: Optional[str] = None
    final_boxed_letter: Optional[str] = None
    system_prompt: Optional[str] = None
    user_prompt: Optional[str] = None
    parsed_payload: dict[str, Any] = field(default_factory=dict)
    raw_response: Optional[str] = None
    cost: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class AttemptRecord:
    attempt_index: int
    model: str
    instruction: str
    delegate_result: DelegateResult
    main_reasoning: str = ""


@dataclass
class ReasoningRunRecord:
    task_id: str
    discipline: str
    question: str
    options: list[str]
    gold_answer_index: int
    gold_answer_letter: str
    reference_steps: list[str]
    final_answer_text: str
    final_boxed_letter: Optional[str]
    mca: float
    rv: float
    total_cost: float
    main_tokens: int = 0
    sub_tokens: int = 0
    total_tokens: int = 0
    latency_seconds: float = 0.0
    models_used: list[str] = field(default_factory=list)
    model_usage: dict[str, int] = field(default_factory=dict)
    attempts: list[AttemptRecord] = field(default_factory=list)
    decision_steps: list[dict[str, Any]] = field(default_factory=list)
    raw_calls: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunSummary:
    total_samples: int
    avg_mca: float
    avg_rv: float
    avg_steps: float
    total_cost: float
    output_dir: str
    total_main_tokens: int = 0
    total_sub_tokens: int = 0
    total_tokens: int = 0
    total_latency_seconds: float = 0.0
    avg_latency_seconds: float = 0.0
    run_wall_time_seconds: float = 0.0
    models_used: list[str] = field(default_factory=list)
    model_usage: dict[str, int] = field(default_factory=dict)
    sample_method: str = "head"
    sample_seed: int = 42
    sampled_task_ids: list[str] = field(default_factory=list)
