"""
Contrastive losses -- handoff sections 7-8.

MultipleNegativesRankingLoss when possibile comes from
sentence-transformers; this module implements the same objective
explicitly (used by the Colab notebook's pure-transformers loop where
installing sentence-transformers on the T4 is slower than a 30-line loss).

MNRL with additional explicit negatives:
    scores = anchor @ [pos_1..pos_B, neg_1..neg_K]
    loss   = cross_entropy(labels = index of the positive)
In-batch negatives are inherent (all other positives in the batch act as
negatives); optional stored hard/easy negatives are appended per-sample.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def mnrl(
    anchor_emb: torch.Tensor,          # [B, D] L2-normalized
    positive_emb: torch.Tensor,        # [B, D] L2-normalized
    extra_neg_emb: torch.Tensor | None = None,  # [B, K, D] optionally appended per-sample
    scale: float = 20.0,
) -> torch.Tensor:
    """Cross-entropy over cosine-similarity logits; in-batch + optional explicit negatives."""
    sim = anchor_emb @ positive_emb.T * scale              # [B, B]
    if extra_neg_emb is not None:
        B, K, D = extra_neg_emb.shape
        neg = torch.einsum("bd,bkd->bk", anchor_emb, extra_neg_emb) * scale  # [B, K]
        sim = torch.cat([sim, neg], dim=1)                  # [B, B + K]
    labels = torch.arange(sim.shape[0], device=sim.device)
    return F.cross_entropy(sim, labels)


def soft_target_ce_from_scores(
    scores: torch.Tensor,   # [B, B] cosine sims
    tau: float = 0.05,
) -> torch.Tensor:
    """Symmetric CLIP-style loss (kept for ablation; default training uses mnrl)."""
    labels = torch.arange(scores.shape[0], device=scores.device)
    return 0.5 * (
        F.cross_entropy(scores / tau, labels) + F.cross_entropy(scores.T / tau, labels)
    )
