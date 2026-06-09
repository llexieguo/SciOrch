# Utilities

Shared prompt-based MAS utilities are kept in canonical modules instead of
being duplicated across individual methods.

Current modules:

- prompt formatting, answer parsing, and image loading:
  `maslab.utils.formatting` and `maslab.utils.images`
- summary metrics and JSON summary writing:
  `maslab.evaluation`
- OpenAI-compatible multimodal API calls and token/cost accounting:
  `sciorch.llm.openai_compatible`
