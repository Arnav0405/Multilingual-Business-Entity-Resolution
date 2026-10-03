"""
Dataset construction for contrastive training -- handoff section 6.

Splits the subset TSVs into anchor / positive / negative triples from the
RAW strings. No normalization, no transliteration: the model must learn
the cross-script relationship itself.

Text variants (handoff section 16 is phase 10):
    name          -> "business_name"
    name_address  -> "Business name: ...| Address: ..." raw concatenation
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
SUBSET = REPO_ROOT / "data" / "entity_resolution_subset"


def apply_text_variant(df: pd.DataFrame, side_cols: tuple[str, str], variant: str) -> pd.Series:
    """ Return the raw text for each row given the variant. NaN coerced to ''. """
    a, b = side_cols
    clean = lambda c: df[c].fillna('').astype(str)
    if variant == "name":
        return clean(a)
    if variant == "name_address":
        return "Business name: " + clean(a) + " | Address: " + clean(b)
    raise ValueError(f"unknown variant {variant}")


@dataclass(frozen=True)
class Triple:
    anchor: str
    positive: str
    negative: str | None
    negative_kind: str  # "hard" | "easy" | "none"


def load_pairs(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    df["row_idx"] = df.index
    return df


def pairs_to_triples(df: pd.DataFrame, text_variant: str = "name") -> list[Triple]:
    """Group by s1_id: each positive row contributes (anchor, positive),
    plus at most one negative (hard preferred, then easy) as an explicit
    third column for later MNRL-with-negatives runs."""
    s1_names = (
        df[["s1_id", "s1_name", "s1_address"]]
        .drop_duplicates("s1_id")
        .set_index("s1_id")
    )
    triples: list[Triple] = []
    for s1_id, g in df.groupby("s1_id", sort=False):
        pos = g[g["label"] == 1]
        neg = g[g["label"] == 0]
        hard = neg[neg["pair_type"] == "hard_negative"]
        easy = neg[neg["pair_type"] == "easy_negative"]
        neg_row = hard.iloc[0] if len(hard) else (easy.iloc[0] if len(easy) else None)
        for _, prow in pos.iterrows():
            strings = apply_text_variant(
                pd.DataFrame([prow]), ("s1_name", "s1_address"), text_variant
            ).iloc[0]
            pstring = apply_text_variant(
                pd.DataFrame([prow]), ("candidate_name", "candidate_address"), text_variant
            ).iloc[0]
            if neg_row is not None:
                nstring = apply_text_variant(
                    pd.DataFrame([neg_row]), ("candidate_name", "candidate_address"), text_variant
                ).iloc[0]
                nkind = "hard" if neg_row["pair_type"] == "hard_negative" else "easy"
            else:
                nstring, nkind = None, "none"
            triples.append(
                Triple(
                    anchor=str(strings),
                    positive=str(pstring),
                    negative=None if nstring is None else str(nstring),
                    negative_kind=nkind,
                )
            )
    return triples


def encode_columns(df: pd.DataFrame, text_variant: str) -> tuple[pd.Series, pd.Series]:
    """Left/right raw text series used for pair scoring (eval path)."""
    left = apply_text_variant(df, ("s1_name", "s1_address"), text_variant)
    right = apply_text_variant(df, ("candidate_name", "candidate_address"), text_variant)
    return left, right
