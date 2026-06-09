from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence


@dataclass
class MiniLMInstructionSimilarity:
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size: int = 32
    local_files_only: bool = False
    _torch: Any = field(init=False, default=None, repr=False)
    _tokenizer: Any = field(init=False, default=None, repr=False)
    _model: Any = field(init=False, default=None, repr=False)
    _device: str | None = field(init=False, default=None, repr=False)
    _functional: Any = field(init=False, default=None, repr=False)

    def _ensure_loaded(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return
        try:
            import torch
            import torch.nn.functional as F
            from transformers import AutoModel, AutoTokenizer
        except Exception as exc:  # pragma: no cover - dependency availability depends on runtime env
            raise RuntimeError(
                "Instruction diversity sampling requires torch and transformers. "
                "Install them to use sentence-transformers/all-MiniLM-L6-v2."
            ) from exc

        self._torch = torch
        self._functional = F
        # Keep instruction-similarity encoding on CPU so it does not contend with
        # the main inference workloads for GPU memory/compute.
        self._device = "cpu"
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            local_files_only=self.local_files_only,
        )
        self._model = AutoModel.from_pretrained(
            self.model_name,
            local_files_only=self.local_files_only,
        ).to(self._device)
        self._model.eval()

    def _mean_pool(self, last_hidden_state: Any, attention_mask: Any) -> Any:
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        masked = last_hidden_state * mask
        summed = self._torch.sum(masked, dim=1)
        denom = self._torch.clamp(mask.sum(dim=1), min=1e-9)
        return summed / denom

    def encode(self, texts: Sequence[str]) -> Any:
        self._ensure_loaded()
        if not texts:
            return self._torch.empty((0, 0))

        embeddings: list[Any] = []
        with self._torch.no_grad():
            for start in range(0, len(texts), self.batch_size):
                batch = [str(text) for text in texts[start : start + self.batch_size]]
                encoded = self._tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    max_length=256,
                )
                encoded = {
                    key: value.to(self._device)
                    for key, value in encoded.items()
                }
                outputs = self._model(**encoded)
                pooled = self._mean_pool(outputs.last_hidden_state, encoded["attention_mask"])
                normalized = self._functional.normalize(pooled, p=2, dim=1)
                embeddings.append(normalized.cpu())
        return self._torch.cat(embeddings, dim=0)

    def similarity_matrix(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        embeddings = self.encode(texts)
        matrix = embeddings @ embeddings.T
        return [
            [float(value) for value in row]
            for row in matrix.tolist()
        ]
