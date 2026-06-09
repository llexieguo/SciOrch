from __future__ import annotations


def build_node_instruction(
    *,
    is_last_step: bool = False,
    is_budget_exhausted: bool = False,
) -> str:
    constraints = []
    if is_last_step:
        constraints.append("- this is the final allowed step: you must submit a final answer now, do not delegate further")
    if is_budget_exhausted:
        constraints.append("- budget is exhausted: you must submit a final answer now, do not delegate further")

    task = (
        "Task:\n"
        "- produce exactly one next orchestra step for this branch\n"
        "- explore a meaningfully different verification direction from prior steps when possible\n"
        "- if earlier steps are weak or misleading, explicitly overturn them\n"
        "- if evidence is not yet enough, ask one focused delegate question that seeks raw facts, calculations, or mechanism-level evidence\n"
        "- if evidence is already sufficient, submit the final option yourself"
    )

    if constraints:
        return "Constraints:\n" + "\n".join(constraints) + "\n" + task
    return task
