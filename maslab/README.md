# MASLab-Style Baseline Layout

This directory contains the MASLab/CoMAS-style structure used to organize
prompt-based multi-agent baselines in SciOrch.

The implementation is not copied from MASLab. The local code reimplements the
LLM Debate and Self-Consistency flows for SciOrch's combined multimodal
benchmark format, OpenAI-compatible API client, image loading, JSONL logging,
image warning tracking, cost tracking, and per-source metrics.

References:

- MASLab: https://github.com/MASWorks/MASLab
- CoMAS MASLab layout: https://github.com/xxyQwQ/CoMAS/tree/main/maslab

## Mapping

| MASLab/CoMAS concept | SciOrch implementation |
| --- | --- |
| `maslab/methods/llm_debate` | `maslab.methods.llm_debate.llm_debate_main.LLMDebateBaseline` |
| `maslab/methods/self_consistency` | `maslab.methods.self_consistency.self_consistency_main.SelfConsistencyBaseline` |
| `maslab/inference.py` | canonical prompt-MAS baseline runner |
| `maslab/evaluation.py` | summary metrics and JSON summary writer |
| `maslab/datasets` | `maslab/datasets/data/test_combined.json` plus local images |
| `maslab/utils` | formatting, image loading, and call serialization helpers |

The legacy entry point `scripts/run_baseline.py` remains as a thin
compatibility wrapper around `maslab.inference`.
The legacy aggregate module `maslab.methods.prompt_mas` remains for recovery and
old imports, but the implementation is split into method-specific modules.

## Running Baselines

Use the project runner as the canonical entry point:

```bash
python -u scripts/run_baseline.py \
  --method llm_debate \
  --input maslab/datasets/data/test_combined.json \
  --model gpt-4o
```

For self-consistency under token-per-minute limits, use
`--sleep-between-samples` to throttle full runs:

```bash
python -u scripts/run_baseline.py \
  --method self_consistency \
  --input maslab/datasets/data/test_combined.json \
  --model gpt-4o \
  --self-consistency-n 5 \
  --sleep-between-samples 30
```

Outputs are written under `output/baselines/` by default.
