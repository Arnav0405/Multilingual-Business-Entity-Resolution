"""
Business Entity Resolution - data loader + first-look EDA.

Problem statement (inferred from the data; README.MD is currently EMPTY):
  Three sources each list real-world businesses:
      train_source{1,2,3}.tsv -> entity_id, business_name, business_address, country
      (source1 ids prefixed "S1-", source2 "S2-", source3 "S3-")
  train_ground_truth.tsv maps every source1 entity to the entity_ids in
  sources 2/3 that refer to the SAME real-world business:
      source1_entity_id \t matched_entity_ids   (comma-separated)

  Task: entity resolution / record linkage - decide which records across
  the three sources point to the same business, despite differences in
  name spelling, transliteration (e.g. Hindi names in source2), address
  formatting, and missing fields.

Run:  python df.py            (from anywhere; paths resolve via this file)
"""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

THIS_FILE = Path(__file__).resolve()
# src/ -> business_entity_resolution/ -> code/ -> <repo root>
REPO_ROOT = THIS_FILE.parents[3]
DATA_DIR = REPO_ROOT / "data" / "student_resource" / "dataset" / "train"

SOURCE_FILES = {
    "source1": "train_source1.tsv",
    "source2": "train_source2.tsv",
    "source3": "train_source3.tsv",
}
GROUND_TRUTH_FILE = "train_ground_truth.tsv"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_tsv(name: str, directory: Path | str | None = None) -> pd.DataFrame:
    """Load one train TSV. Everything stays as plain strings: entity ids are
    identifiers (never numbers) and empty address/name cells are '' not NaN."""
    path = Path(directory) / name if directory else DATA_DIR / name
    return pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        na_filter=False,
        quoting=csv.QUOTE_NONE,
        engine="c",
    )


def load_all(directory: Path | str | None = None) -> dict[str, pd.DataFrame]:
    """Load the three source listings + the ground-truth matches."""
    data = {key: load_tsv(fname, directory) for key, fname in SOURCE_FILES.items()}
    data["ground_truth"] = load_tsv(GROUND_TRUTH_FILE, directory)
    return data


def parse_ground_truth(gt: pd.DataFrame) -> dict[str, list[str]]:
    """Map source1_entity_id -> list of matched entity_ids (S2-/S3- prefixed)."""
    out: dict[str, list[str]] = {}
    for s1_id, matches in zip(gt["source1_entity_id"], gt["matched_entity_ids"]):
        out[s1_id] = [m for m in str(matches).split(",") if m]
    return out


# ---------------------------------------------------------------------------
# EDA
# ---------------------------------------------------------------------------

def eda_source(df: pd.DataFrame, label: str) -> None:
    print(f"\n--- {label} ---")
    print(f"rows={len(df):,}  cols={list(df.columns)}  "
          f"mem={df.memory_usage(deep=True).sum() / 1e6:.0f} MB")

    for col in df.columns:
        empty = int((df[col] == "").sum())
        print(f"  {col:<17} empty={empty:>10,} ({empty / len(df):5.1%})  "
              f"unique={df[col].nunique():,}")

    dup_ids = int(df["entity_id"].duplicated().sum())
    print(f"  duplicate entity_id rows: {dup_ids:,}")

    if "country" in df.columns:
        top = df["country"].value_counts().head(5)
        print("  top countries: " + ", ".join(f"{c}={n:,}" for c, n in top.items()))

    print("  sample rows:")
    for _, row in df.head(2).iterrows():
        print(f"    {row['entity_id']} | {row['business_name'][:45]!r} "
              f"| {row['business_address'][:45]!r} | {row['country']}")


def eda_ground_truth(gt: pd.DataFrame) -> None:
    print(f"\n--- ground truth ---")
    print(f"rows={len(gt):,}")

    parsed = parse_ground_truth(gt)
    n_matches = pd.Series([len(v) for v in parsed.values()])
    print(f"source1 entities: {len(parsed):,}")
    print(f"  with zero matches : {(n_matches == 0).sum():,}")
    print(f"  matches per entity: min={n_matches.min()} median={n_matches.median():.0f} "
          f"mean={n_matches.mean():.2f} max={n_matches.max()}")

    # Which sources do the matches come from?
    has_s2 = sum(any(m.startswith("S2-") for m in v) for v in parsed.values())
    has_s3 = sum(any(m.startswith("S3-") for m in v) for v in parsed.values())
    both = sum(any(m.startswith("S2-") for m in v) and any(m.startswith("S3-") for m in v)
               for v in parsed.values())
    print(f"  entities with >=1 S2 match: {has_s2:,}   >=1 S3 match: {has_s3:,}   both: {both:,}")
    print(f"  total positive pairs (S1,S2/3): {int(n_matches.sum()):,}")

    # a worked example: pick an entity with matches and show the raw records
    for s1_id, matches in parsed.items():
        if matches:
            print(f"  example group {s1_id} -> {matches[:4]}")
            break


# ---------------------------------------------------------------------------

def main() -> None:
    print(f"Data dir: {DATA_DIR}")
    data = load_all()

    for key in ("source1", "source2", "source3"):
        eda_source(data[key], key)
    eda_ground_truth(data["ground_truth"])

    print("\nWhat the task is: given the three source listings, predict which")
    print("records refer to the same business (train ground truth shows the")
    print("S1 <-> S2/S3 matches to learn from; same format expected at inference).")


if __name__ == "__main__":
    main()
