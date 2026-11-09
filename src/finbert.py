"""FinBERT inference on Apple Silicon MPS.

Scores a list of texts and returns, per text, the three class probabilities
(positive, negative, neutral) and a VADER-compatible compound score computed
as `P(pos) - P(neg)` in the range [-1, +1].

Why that compound: the dissertation pipeline downstream (daily aggregation,
LSTM feature table) consumes a single scalar sentiment per text. Making the
FinBERT output slot into the same shape means Model D can reuse the existing
training infrastructure without changes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_NAME = "ProsusAI/finbert"
MAX_LENGTH = 128

LABEL_ORDER = ["positive", "negative", "neutral"]


@dataclass
class FinBERTScorer:
    device: str = "mps"
    batch_size: int = 64

    def __post_init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        self.model.eval()
        self.model.to(self.device)

        id2label = {int(k): v.lower() for k, v in self.model.config.id2label.items()}
        self._label_cols = [id2label[i] for i in range(len(id2label))]
        self._pos_idx = self._label_cols.index("positive")
        self._neg_idx = self._label_cols.index("negative")
        self._neu_idx = self._label_cols.index("neutral")

    @torch.inference_mode()
    def score_batch(self, texts: list[str]) -> np.ndarray:
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        logits = self.model(**enc).logits
        probs = torch.softmax(logits, dim=-1)
        return probs.cpu().numpy()

    def iter_scores(
        self, texts: Iterable[str]
    ) -> Iterator[tuple[int, float, float, float, float]]:
        """Yield (row_index, pos, neg, neu, compound) for each text.

        row_index is the position in the input iterable so callers can
        attach scores to the original dataframe row.
        """
        buf: list[str] = []
        offsets: list[int] = []
        for i, t in enumerate(texts):
            buf.append(t)
            offsets.append(i)
            if len(buf) >= self.batch_size:
                yield from self._flush(buf, offsets)
                buf, offsets = [], []
        if buf:
            yield from self._flush(buf, offsets)

    def _flush(
        self, texts: list[str], offsets: list[int]
    ) -> Iterator[tuple[int, float, float, float, float]]:
        probs = self.score_batch(texts)
        for row_i, p in zip(offsets, probs):
            pos = float(p[self._pos_idx])
            neg = float(p[self._neg_idx])
            neu = float(p[self._neu_idx])
            compound = pos - neg
            yield (row_i, pos, neg, neu, compound)
