from __future__ import annotations

from sciorch.types import ReasoningSample


# Canonical pool definitions — order within each pool is the display order.
# A model can appear in multiple pools (e.g. gemini-2.5-pro in Vision and Strong).
_POOLS: list[tuple[str, list[str]]] = [
    (
        "Frontier pool (default for hardest CALCULATION / SCI_REASONING)",
        [
            "gemini-3-pro-preview",
            "gpt-5.4",
            "o3",
            "claude-sonnet-4-5-20250929",
        ],
    ),
    (
        "Vision pool (default for VISION_REASONING)",
        [
            "gemini-2.5-pro",
            "gemini-3-pro-preview",
            "gpt-4o",
            "gpt-5.4",
            "claude-sonnet-4-5-20250929",
        ],
    ),
    (
        "Strong general pool (medium-difficulty CALCULATION / SCI_REASONING)",
        [
            "gpt-4o",
            "gpt-5.2",
            "gpt-5",
            "claude-sonnet-4-20250514",
            "gemini-2.5-pro",
            "gpt-4.1",
        ],
    ),
    (
        "Lightweight pool (simple lookups, basic arithmetic, or well-bounded easy sub-questions)",
        [
            "claude-haiku-4-5-20251001",
            "gemini-3-flash-preview",
            "gpt-5-mini",
            "gpt-4o-mini",
            "gpt-4.1-mini",
            "gemini-2.5-flash",
        ],
    ),
]


def build_model_guidance_text(sub_models: list[str], allow_self_delegation: bool = False) -> str:  # `allow_self_delegation` kept for backward compat; ignored.
    """
    Build the pool section of the prompt.

    If sub_models is empty or contains every canonical model, return the full
    hardcoded text (non-MCTS / full-pool run).

    Otherwise filter each pool to only the models present in sub_models so that
    the prompt exactly reflects the node's actual delegate pool (MCTS mode).
    Pools that end up empty after filtering are omitted entirely.
    """
    if not sub_models:
        # force_submit path — no delegation available, pool text unused
        return "(no delegate models)"

    available = set(sub_models)

    sections: list[str] = []
    for pool_name, pool_models in _POOLS:
        present = [m for m in pool_models if m in available]
        if present:
            lines = [f"{pool_name}:"] + [f"- {m}" for m in present]
            sections.append("\n".join(lines))

    if not sections:
        # sub_models contains models not in any pool — list them flat
        return "Available models:\n" + "\n".join(f"- {m}" for m in sub_models)

    return "\n\n".join(sections)


def format_options_with_letters(options: list[str]) -> str:
    return "\n".join(
        f"{chr(ord('A') + idx)}. {option}"
        for idx, option in enumerate(options)
    )


def build_submit_only_prompt(
    sample: ReasoningSample,
    step_history_text: str,
    step_index: int,
    max_steps: int,
    allow_self_delegation: bool = False,  # deprecated, ignored
) -> str:
    return build_main_prompt(
        sample=sample,
        step_history_text=step_history_text,
        step_index=step_index,
        max_steps=max_steps,
        sub_models=[],
        force_submit=True,
    )


def build_main_prompt(
    sample: ReasoningSample,
    step_history_text: str,
    step_index: int,
    max_steps: int,
    sub_models: list[str],
    allow_self_delegation: bool = False,  # deprecated, ignored
    force_submit: bool = False,
) -> str:
    remaining = max_steps - step_index + 1
    model_guidance_text = build_model_guidance_text(sub_models)
    options_text = format_options_with_letters(sample.options)

    if force_submit:
        action_line = "This turn: output submit only (delegation disabled)."
    elif remaining <= 1:
        action_line = "Last orchestration turn: you must output submit with a letter."
    else:
        action_line = (
            "Default: delegate when a specific factual gap remains. "
            "If the question requires multi-step reasoning (e.g. read a figure then compute), delegate each step separately — do NOT reason on your own. "
            "Do NOT delegate just to raise confidence — submit when all reasoning steps have been answered by delegates."
        )

    routing_line = (
        "Routing:\n"
        "- VISION_REASONING -> Vision pool. Hard CALC / SCI -> Frontier. "
        "Medium -> Strong general. Easy lookup / 1-step arithmetic -> Lightweight.\n"
        "- Match by task-pool fit, not by model reputation.\n"
        ""
    )

    return f"""
You are the Orchestrator for a multimodal scientific MCQ.
You ROUTE focused sub-questions to specialist sub-models, then submit ONE final option.
Specialists answer; you decide. Do NOT solve it yourself.

Each turn, emit exactly one JSON object (schema at the end):
- delegate_task: one focused sub-question (not the whole problem).
- submit: \\boxed{{<letter>}} (A..J).

{action_line}

Pool:
{model_guidance_text}

{routing_line}

Delegation rules:
- ONE missing fact per sub-question; do not bundle multiple unknowns.
- Frame the sub-question so the answer can be mapped back to the OPTIONS.
- Do not leak your guess or prior conclusions; do not paraphrase or redefine the QUESTION.
- NEVER repeat a sub-question that a prior step already answered — use that answer.
- Prefer using a different model from what prior steps already used.

QUESTION:
{sample.question}

OPTIONS:
{options_text}

PRIOR DELEGATE STEPS:
{step_history_text}

Reasoning (in `reasoning`, in order; keep concise):
1. Evidence: what each prior step confirmed. Map the returned value/fact to specific OPTIONS — eliminate options that conflict with the evidence.
2. Option mapping: if a sub-model returned a numerical value, compare it against EVERY option numerically. Pick the option whose value is closest. Show the comparison explicitly (e.g. "value 1.38 — A=1.2 diff=0.18, E=1.1 diff=0.28, ...").
3. Decision: if answering requires multiple reasoning steps (e.g. reading a figure, then computing), delegate each step — do not perform any step yourself. Submit only after all steps have been answered by delegates.

Submit Gating (ALL must hold):
(a) The decisive fact is supported by at least one delegate step (not only your own reasoning).
(b) If two delegate steps DISAGREE on the decisive fact, delegate a third check to break the tie.
(c) You have explicitly mapped the evidence to a specific option letter (for numerical values: show comparison against all candidate options).

Critical rules:
- Sub-model confidence is a hint, not proof. A high-confidence answer can still be wrong. Always verify by checking whether the sub-model's evidence and reasoning logically support its conclusion before using it as basis for submit.
- `model` MUST be a name from the Pool above.
- `task_type` is one of CALCULATION | SCI_REASONING | VISION_REASONING.
- If a computed value is not exactly among OPTIONS, pick the closest match — show your comparison.
- Pick the best-supported option; do not argue options are wrong.
- Only when `action` is submit: in `reasoning`, briefly name the top two letter options you considered and one sentence on why evidence favors one over the other.
- Keep `reasoning` concise. Output must be exactly one JSON object.

Output JSON only. Exactly one of:
{{"action":"delegate_task","reasoning":"...","model":"<model name>","instruction":"...","task_type":"CALCULATION|SCI_REASONING|VISION_REASONING"}}
{{"action":"submit","reasoning":"...","final_answer":"\\\\boxed{{<letter>}}"}}
""".strip()
