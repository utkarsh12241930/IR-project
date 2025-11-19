import math
import re
from collections import Counter
from typing import Iterable, List

import numpy as np
import torch
from torch import nn


_TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)


class HardnessFeatureExtractor:
    """
    Lightweight heuristic scorer that approximates linguistic hardness for queries.
    The score is normalized to [0, 1] by combining token length, character entropy,
    punctuation density, and code-mixing indicators. The heuristics are intentionally
    simple so they can run inside HF dataset map functions without torch.
    """

    def __init__(
        self,
        max_token_reference: int = 24,
        entropy_reference: float = 4.0,
        punctuation_reference: float = 0.25,
    ):
        self.max_token_reference = max_token_reference
        self.entropy_reference = entropy_reference
        self.punctuation_reference = punctuation_reference

    def __call__(self, queries: Iterable[str]) -> List[float]:
        if isinstance(queries, str):
            queries = [queries]
        return [self._score_single(query) for query in queries]

    def _score_single(self, query: str) -> float:
        if not isinstance(query, str):
            query = str(query)

        tokens = _TOKEN_PATTERN.findall(query.lower())
        token_count = len(tokens)
        token_norm = min(token_count / self.max_token_reference, 1.0)

        punctuation_ratio = self._punct_ratio(query)
        punctuation_norm = min(punctuation_ratio / self.punctuation_reference, 1.0)

        code_mix_ratio = self._code_mix_ratio(tokens)

        entropy = self._char_entropy(query)
        entropy_norm = min(entropy / self.entropy_reference, 1.0)

        score = (
            0.35 * token_norm
            + 0.25 * entropy_norm
            + 0.2 * punctuation_norm
            + 0.2 * code_mix_ratio
        )
        return float(max(0.0, min(score, 1.0)))

    @staticmethod
    def _punct_ratio(text: str) -> float:
        if not text:
            return 0.0
        punct_count = sum(1 for ch in text if not ch.isalnum() and not ch.isspace())
        return punct_count / max(len(text), 1)

    @staticmethod
    def _code_mix_ratio(tokens: List[str]) -> float:
        if not tokens:
            return 0.0
        latin = sum(1 for tok in tokens if tok.encode("ascii", "ignore"))
        return 1.0 - (latin / len(tokens))

    @staticmethod
    def _char_entropy(text: str) -> float:
        if not text:
            return 0.0
        counts = Counter(text)
        total = float(len(text))
        entropy = 0.0
        for count in counts.values():
            p = count / total
            entropy -= p * math.log(p + 1e-12, 2)
        return entropy


class HardnessPredictor(nn.Module):
    """
    Small MLP that predicts hardness directly from DPR query embeddings.
    The predictor learns to approximate the heuristic target produced by
    HardnessFeatureExtractor but can also adapt during finetuning.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        bottleneck = max(32, hidden_size // 4)
        self.layers = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, bottleneck),
            nn.ReLU(),
            nn.Linear(bottleneck, 1),
        )

    def forward(self, query_embeddings: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.layers(query_embeddings)).squeeze(-1)


def normalize_hardness_array(values: Iterable[float]) -> np.ndarray:
    """
    Utility helper to convert an iterable of floats to a numpy array that
    is safe to store inside a HuggingFace dataset column.
    """
    if isinstance(values, np.ndarray):
        return values.astype(np.float32)
    return np.asarray(list(values), dtype=np.float32)

