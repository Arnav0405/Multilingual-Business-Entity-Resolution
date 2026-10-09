"""
Demo serving: checkpoint loading, E5 embedding, cosine top-K search, and a
char n-gram TF-IDF scorer for the "what lexical matching would score" contrast.

Reuses the exact project formatting rules (src/encoder.py + src/dataset.py):
    * `query: ` prefix on the query/anchor side, `passage: ` on the corpus side
    * CLS pooling + L2 normalization (handled inside E5Encoder)
    * modelD  = raw name,        max_length 64
    * modelADDR = "Business name: <name> | Address: <addr>", max_length 128, fillna('')

Nothing here mutates the training/eval pipeline; this module is read-only
against src/ and baselines/.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# --- sys.path bootstrap (mirrors training/train_contrastive.py) --------------
REPO_ROOT = Path(__file__).resolve().parents[3]
BER = REPO_ROOT / "code" / "business_entity_resolution"
sys.path.insert(0, str(BER / "src"))
sys.path.insert(0, str(BER / "baselines"))

from encoder import E5Encoder, QUERY_PREFIX, PASSAGE_PREFIX  # noqa: E402
from evaluate import script_tags  # noqa: E402  (re-exported for callers)

# --- constants ---------------------------------------------------------------
CHECKPOINT_D = BER / "results" / "modelD_epoch3"
CHECKPOINT_ADDR = BER / "results" / "modelADDR_epoch3"

# Model reference thresholds from the report (used in tests / app notes).
THRESHOLD_D = 0.701
THRESHOLD_ADDR = 0.614

# Text variants (must match training exactly).
VARIANT_NAME = "name"
VARIANT_NAME_ADDRESS = "name_address"
MAX_LEN_NAME = 64
MAX_LEN_ADDR = 128

__all__ = [
    "REPO_ROOT", "BER", "CHECKPOINT_D", "CHECKPOINT_ADDR",
    "QUERY_PREFIX", "PASSAGE_PREFIX",
    "THRESHOLD_D", "THRESHOLD_ADDR",
    "VARIANT_NAME", "VARIANT_NAME_ADDRESS", "MAX_LEN_NAME", "MAX_LEN_ADDR",
    "load_encoder", "encode_texts", "search", "TfidfCharScorer", "script_tags",
]


def load_encoder(checkpoint_dir: Path, max_length: int) -> E5Encoder:
    """Load a fine-tuned checkpoint directory as an E5Encoder (fp16 on CUDA)."""
    return E5Encoder(model_name=str(checkpoint_dir), max_length=max_length)


def encode_texts(encoder: E5Encoder, texts: list[str], prefix: str,
                 batch_size: int = 256) -> np.ndarray:
    """Embed texts with the given E5 prefix -> [N, 384] float32, L2-normalized."""
    if not texts:
        return np.empty((0, 384), dtype=np.float32)
    return encoder.encode(texts, prefix, batch_size=batch_size)


def search(query_emb: np.ndarray, corpus_emb: np.ndarray, k: int):
    """Top-k cosine search. Both inputs must already be L2-normalized.

    Returns (idx, scores) as 1-D arrays of length min(k, N). Accepts a single
    query vector of shape (384,) or (1, 384).
    """
    q = np.asarray(query_emb, dtype=np.float32)
    if q.ndim == 2:
        q = q[0]
    sims = corpus_emb @ q                      # [N] cosine (both unit-norm)
    k = int(min(k, sims.shape[0]))
    order = np.argsort(-sims, kind="mergesort")[:k]
    return order, sims[order]


class TfidfCharScorer:
    """char_wb 2-4 gram TF-IDF cosine — the headline lexical baseline family.

    Fit once on the corpus names, then score any (query, name) pair the same
    elementwise-product way baselines/tfidf.py::cross_scores does.
    """

    def __init__(self):
        self._vec = None

    def fit(self, names: list[str]) -> "TfidfCharScorer":
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._vec = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(2, 4),
            sublinear_tf=True, min_df=2,
        )
        self._vec.fit([n if isinstance(n, str) else "" for n in names])
        return self

    def score(self, a: str, b: str) -> float:
        if self._vec is None:
            raise RuntimeError("TfidfCharScorer.fit() must be called first")
        va = self._vec.transform([a if isinstance(a, str) else ""])
        vb = self._vec.transform([b if isinstance(b, str) else ""])
        prod = float(va.multiply(vb).sum())
        na = float(np.sqrt(va.multiply(va).sum()))
        nb = float(np.sqrt(vb.multiply(vb).sum()))
        denom = na * nb
        return prod / denom if denom > 0 else 0.0
