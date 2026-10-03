"""
Shared evaluation module -- handoff sections 10-14.

Consumes (pairs_df, scores) where scores are aligned by integer position
to the pairs dataframe exactly as loaded by `load_split_pairs`, and
produces:

  * binary pair metrics: precision / recall / F1 / F0.5 at a threshold
  * threshold tuning (max F0.5 on validation)
  * retrieval metrics over each S1's candidate pool: Recall@1/5/10, MRR
  * slices: overall / same-script / cross-script / per-Indic-script

Script detection works on the raw strings (no normalization anywhere).

Score-notebook contract (Colab artifacts): a parquet/TSV with columns
    split, row_idx, score
where row_idx is the integer position of the pair in the split's TSV
as read by `load_split_pairs`.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
SUBSET = REPO_ROOT / "data" / "entity_resolution_subset"
RESULTS_DIR = REPO_ROOT / "code" / "business_entity_resolution" / "results"

INDIC_RANGES = {
    "Devanagari": (0x0900, 0x097F),
    "Bengali":    (0x0980, 0x09FF),
    "Gurmukhi":   (0x0A00, 0x0A7F),
    "Gujarati":   (0x0A80, 0x0AFF),
    "Oriya":      (0x0B00, 0x0B7F),
    "Tamil":      (0x0B80, 0x0BFF),
    "Telugu":     (0x0C00, 0x0C7F),
    "Kannada":    (0x0C80, 0x0CFF),
    "Malayalam":  (0x0D00, 0x0D7F),
}

_SCRIPT_RES = {
    name: re.compile(f"[{chr(lo)}-{chr(hi)}]")
    for name, (lo, hi) in INDIC_RANGES.items()
}

_cache: dict[str, frozenset] = {}


def script_tags(s: object) -> frozenset:
    """Scripts present in a raw string. 'Latin' when no Indic range hits."""
    if not isinstance(s, str) or not s:
        return frozenset({"<empty>"})
    hit = _cache.get(s)
    if hit is not None:
        return hit
    tags = {name for name, rx in _SCRIPT_RES.items() if rx.search(s)}
    if not tags:
        tags = {"Latin"}
    out = frozenset(tags)
    _cache[s] = out
    return out


def pair_script_relation(df: pd.DataFrame) -> pd.Series:
    """'cross_script' / 'same_script' per pair, from raw names."""
    def rel(a, b):
        return "cross_script" if script_tags(a) != script_tags(b) else "same_script"
    return pd.Series(
        [rel(a, b) for a, b in zip(df["s1_name"], df["candidate_name"])],
        index=df.index,
    )


def candidate_indic_scripts(df: pd.DataFrame) -> pd.Series:
    """For each pair, the tuple of Indic scripts in the CANDIDATE name."""
    def tags(s):
        return tuple(sorted(set(script_tags(s)) & set(INDIC_RANGES)))
    return pd.Series([tags(s) for s in df["candidate_name"]], index=df.index)


# ============================================================
# Loading
# ============================================================

def load_split_pairs(split: str) -> pd.DataFrame:
    """Read {split}_pairs.tsv. Row order defines the score alignment."""
    return pd.read_csv(SUBSET / f"{split}_pairs.tsv", sep="\t")


def align_scores(ref_len: int, score_df: pd.DataFrame) -> np.ndarray:
    """Expand a (split, row_idx, score) score file into a dense vector."""
    out = np.full(ref_len, np.nan, dtype=np.float64)
    r = score_df["row_idx"].to_numpy()
    if r.min() < 0 or r.max() >= ref_len:
        raise ValueError("score row_idx outside split range")
    if len(set(r)) != len(r):
        raise ValueError("duplicate row_idx in score file")
    out[r] = score_df["score"].to_numpy(dtype=np.float64)
    if np.isnan(out).any():
        raise ValueError(f"{int(np.isnan(out).sum())} pairs have no score")
    return out


# ============================================================
# Binary metrics
# ============================================================

def binary_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    pred = scores >= threshold
    y = y_true.astype(bool)
    tp = int((pred & y).sum())
    fp = int((pred & ~y).sum())
    fn = int((~pred & y).sum())
    tn = int((~pred & ~y).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    f05 = 1.25 * p * r / (0.25 * p + r) if 0.25 * p + r else 0.0
    return {
        "threshold": threshold, "precision": p, "recall": r,
        "f1": f1, "f0.5": f05,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def best_threshold_f05(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Threshold maximizing F0.5 on the given (validation) scores.

    Grid = one candidate per cutoff k (0..n): 'classify the top-k scores
    positive'. O(n log n): one sort + cumulative counts.
    """
    order = np.argsort(-scores, kind="mergesort")
    s_sorted = scores[order]
    y_sorted = y_true[order].astype(bool)
    n = len(s_sorted)
    total_pos = int(y_sorted.sum())

    # k = number of items classified positive (0..n).
    # tp(k) = cum_tp[k-1] (with tp(0)=0); fp(k) = k - tp(k); fn = total_pos - tp(k).
    cum_tp = np.concatenate([[0], np.cumsum(y_sorted)])          # length n+1
    k = np.arange(n + 1)
    tp = cum_tp
    fp = k - tp
    fn = total_pos - tp

    p = np.where(k > 0, tp / np.maximum(k, 1), 0.0)
    r = np.where(total_pos > 0, tp / max(total_pos, 1), 0.0)
    denom = 0.25 * p + r
    f05 = np.where(denom > 0, 1.25 * p * r / np.where(denom > 0, denom, 1.0), 0.0)

    k_best = int(np.argmax(f05))

    # threshold that classifies exactly the top-k items positive:
    # s_sorted is descending, so any value in (s_sorted[k], s_sorted[k-1]]
    # works for 1 <= k <= n-1; k=0 -> above max; k=n -> below min.
    if k_best == 0:
        return float(s_sorted[0] + 1.0)
    if k_best == n:
        return float(s_sorted[-1] - 1.0)
    return float((s_sorted[k_best - 1] + s_sorted[k_best]) / 2.0)


# ============================================================
# Retrieval metrics (handoff 10-11)
# ============================================================

def retrieval_metrics(
    df: pd.DataFrame,
    scores: np.ndarray,
    ks: tuple[int, ...] = (1, 5, 10),
) -> dict:
    """
    Rank each S1's candidate pool (its pairs in this split) by score.
    Recall@k  = fraction of S1s (with >= 1 true match in the pool) that
                retrieve at least one true match in the top k.
    MRR       = mean of 1 / rank(first true match); S1s with >= 1 true
                match only (zero-match S1s would have rank = undefined).
    """
    d = df[["s1_id", "label"]].copy()
    d["score"] = scores
    d["row"] = np.arange(len(d))

    n_s1_with_pos = 0
    hits = {k: 0 for k in ks}
    rr_sum = 0.0

    for _, g in d.groupby("s1_id", sort=False):
        g = g.sort_values("score", ascending=False, kind="mergesort")
        y = g["label"].to_numpy().astype(bool)
        if not y.any():
            continue
        n_s1_with_pos += 1
        first_hit = int(np.argmax(y))         # 0-based rank of first true match
        for k in ks:
            if first_hit < k:
                hits[k] += 1
        rr_sum += 1.0 / (first_hit + 1)

    return {
        "s1_with_true_match": n_s1_with_pos,
        "s1_without_true_match": int(d["s1_id"].nunique() - n_s1_with_pos),
        **{f"recall@{k}": (hits[k] / n_s1_with_pos if n_s1_with_pos else 0.0) for k in ks},
        "mrr": (rr_sum / n_s1_with_pos if n_s1_with_pos else 0.0),
    }


# ============================================================
# Sliced reporting (handoff 13-14)
# ============================================================

def slice_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    return binary_metrics(y_true, scores, threshold)


def sliced_report(
    df: pd.DataFrame,
    scores: np.ndarray,
    threshold: float,
) -> dict:
    """Overall / same-script / cross-script + per-Indic-script breakdown."""
    y = df["label"].to_numpy()
    rel = pair_script_relation(df).to_numpy()

    same = rel == "same_script"
    cross = rel == "cross_script"

    out = {
        "overall": slice_metrics(y, scores, threshold),
        "same_script": slice_metrics(y[same], scores[same], threshold),
        "cross_script": slice_metrics(y[cross], scores[cross], threshold),
    }

    cand_scripts = candidate_indic_scripts(df[cross])
    per_script: dict[str, dict] = {}
    for scripts in sorted(set(cand_scripts)):
        if not scripts:
            continue
        mask = np.zeros(len(df), dtype=bool)
        idx = cand_scripts.index[cand_scripts.map(lambda t: t == scripts)]
        mask[idx] = True
        if mask.any():
            key = "+".join(scripts) if scripts else "reverse_cross_Llatin"
            per_script[key] = slice_metrics(y[cross & mask], scores[cross & mask], threshold)
    out["cross_script_by_script"] = per_script
    return out


# ============================================================
# Export: dedicated cross-script test subset (handoff 13)
# ============================================================

def export_test_cross_script(overwrite: bool = False) -> Path | None:
    out = SUBSET / "test_pairs_cross_script.tsv"
    if out.exists() and not overwrite:
        return out
    df = load_split_pairs("test")
    rel = pair_script_relation(df)
    df[rel.values == "cross_script"].to_csv(out, sep="\t", index=False)
    return out


def save_results(name: str, payload: dict) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    p = RESULTS_DIR / f"{name}.json"
    p.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return p
