"""
E5 scoring/encoding helpers shared by local eval + Colab notebooks.

Design: encode once to a matrix (fp16 on GPU), pairwise cosine via matmul.
`query:` prefix for the S1 side, `passage:` for the candidate side per
the model card; this module centralizes the strings so both sides agree.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

MODEL_NAME = "intfloat/multilingual-e5-small"
QUERY_PREFIX = "query: "
PASSAGE_PREFIX = "passage: "


class E5Encoder:
    def __init__(
        self,
        model_name: str = MODEL_NAME,
        device: str | None = None,
        dtype: torch.dtype | None = None,
        max_length: int = 64,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype or (torch.float16 if "cuda" in self.device else torch.float32)
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()
        if "cuda" in self.device:
            self.model = self.model.half()

    @torch.no_grad()
    def encode(self, texts: list[str], prefix: str, batch_size: int = 256) -> np.ndarray:
        """L2-normalized mean-pooled... actually CLS per E5 convention -> [N, 384] fp32 np array."""
        out = np.empty((len(texts), 384), dtype=np.float32)
        model = self.model
        tok = self.tokenizer
        for i in range(0, len(texts), batch_size):
            batch = [prefix + t for t in texts[i:i + batch_size]]
            enc = tok(
                batch, padding=True, truncation=True,
                max_length=self.max_length, return_tensors="pt",
            ).to(self.device)
            hidden = model(**enc).last_hidden_state
            emb = hidden[:, 0]  # E5 uses CLS with the prefix scheme
            emb = F.normalize(emb.float(), dim=-1)
            out[i:i + len(batch)] = emb.cpu().numpy().astype(np.float32)
        return out


def pairwise_cosine(a: np.ndarray, b: np.ndarray, batch_size: int = 200_000) -> np.ndarray:
    """Row-wise cosine for aligned rows a[i] vs b[i]."""
    assert a.shape == b.shape
    ta = torch.from_numpy(a)
    tb = torch.from_numpy(b)
    out = torch.empty(len(a), dtype=torch.float32)
    for i in range(0, len(a), batch_size):
        out[i:i + batch_size] = (ta[i:i + batch_size] * tb[i:i + batch_size]).sum(-1)
    return out.numpy()


def cross_cosine_matrix(q: np.ndarray, p: np.ndarray, batch: int = 512) -> np.ndarray:
    """q [N,384] @ p.T -> [N, M] cosine (both must already be L2-normalized)."""
    tq = torch.from_numpy(q)
    tp = torch.from_numpy(p).T
    chunks = []
    for i in range(0, len(tq), batch):
        chunks.append(tq[i:i + batch] @ tp)
    return torch.cat(chunks).numpy()
