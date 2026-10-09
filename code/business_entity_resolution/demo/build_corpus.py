"""
Demo corpus builder (run once, offline).

Builds a fresh ~5k-record S2/S3 catalog of real planted duplicate clusters,
disjoint from the existing 10k train/val/test subset, then precomputes the
two corpus embedding matrices:

    corpus_addr.npy  modelADDR, "Business name: <n> | Address: <a>", max_len 128
    corpus_name.npy  modelD,   raw name,                             max_len 64

Ground truth here is the *inversion* of train_ground_truth.tsv: each S1 group
id denotes one real-world business, and its S2/S3 members are the duplicates.
An S2/S3 id that appears under two S1 groups is deduped first-wins so no
record_id ever occurs twice (review-focus #3).

Produces demo_corpus.tsv with columns:
    entity_id, record_id, business_name, business_address, country, source, script
The tsv row order is the embedding row order (they are aligned).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# --- sys.path bootstrap (mirrors training/train_contrastive.py) --------------
REPO_ROOT = Path(__file__).resolve().parents[3]
BER = REPO_ROOT / "code" / "business_entity_resolution"
sys.path.insert(0, str(BER / "src"))
sys.path.insert(0, str(BER / "baselines"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset import apply_text_variant  # noqa: E402
from evaluate import script_tags  # noqa: E402
import serving  # noqa: E402

DEMO_DIR = Path(__file__).resolve().parent
DATA = REPO_ROOT / "data"
SUBSET = DATA / "entity_resolution_subset"
S2_PATH = DATA / "train_source2.tsv"
S3_PATH = DATA / "train_source3.tsv"
GT_PATH = DATA / "train_ground_truth.tsv"

# Selection knobs. TARGET_CLUSTERS is tuned so the realised corpus lands near
# the ~5k-record target (real clusters average ~3.5-4 S2/S3 members each).
TARGET_CLUSTERS = 1500
MIN_RECORDS = 2
MAX_RECORDS = 6
SEED = 42
CROSS_SCRIPT_OVERSAMPLE = 3.0

# DuckDB tuning (mirror src/creat_subset.py, 15 GB laptop)
DUCKDB_THREADS = 16
DUCKDB_MEMORY_LIMIT = "8GB"

INDIC_TAGS = {
    "Devanagari", "Tamil", "Telugu", "Bengali", "Gujarati",
    "Kannada", "Malayalam", "Gurmukhi", "Oriya",
}


# ============================================================
# Ground truth → long (record_id, entity_id) table
# ============================================================

def load_ground_truth() -> pd.DataFrame:
    """Columns: source1_entity_id, matched_entity_ids (raw comma-joined ids)."""
    gt = pd.read_csv(
        GT_PATH, sep="\t", dtype=str, keep_default_na=False,
        usecols=["source1_entity_id", "matched_entity_ids"],
    )
    return gt


def explode_matches(gt: pd.DataFrame) -> pd.DataFrame:
    """Split matched_entity_ids into one row per S2/S3 id.

    Columns: record_id (S2/S3 id), entity_id (S1 group id). record_id is
    deduped first-wins when an id appears under multiple S1 groups, so the
    result never repeats a record_id.
    """
    long = gt.assign(
        record_id=gt["matched_entity_ids"].astype(str).str.split(",")
    ).explode("record_id")
    long["record_id"] = long["record_id"].astype(str).str.strip()
    long = long[long["record_id"] != ""]
    long = long[["record_id", "source1_entity_id"]].rename(
        columns={"source1_entity_id": "entity_id"}
    )
    long = long.drop_duplicates(subset="record_id", keep="first")
    return long[["record_id", "entity_id"]].reset_index(drop=True)


# ============================================================
# Existing subset ids (to exclude → demo is unseen data)
# ============================================================

def collect_subset_ids() -> tuple[set[str], set[str]]:
    """(s1_ids, candidate_ids) drawn from the 10k subset files."""
    s1_ids: set[str] = set()
    cand_ids: set[str] = set()
    for split in ("train", "val", "test"):
        s1 = pd.read_csv(SUBSET / f"{split}_s1.tsv", sep="\t", dtype=str,
                         usecols=["entity_id"])
        s1_ids |= set(s1["entity_id"].dropna())
        pairs = pd.read_csv(SUBSET / f"{split}_pairs.tsv", sep="\t", dtype=str,
                            usecols=["candidate_id"])
        cand_ids |= set(pairs["candidate_id"].dropna())
    return s1_ids, cand_ids


# ============================================================
# DuckDB record lookup (S2/S3 are 1 GB combined)
# ============================================================

def primary_script(name: object) -> str:
    """Primary Indic script of a name, else 'Latin'."""
    tags = set(script_tags(name)) & INDIC_TAGS
    return sorted(tags)[0] if tags else "Latin"


def _duck_connect(s2_path: Path = S2_PATH, s3_path: Path = S3_PATH):
    import duckdb

    con = duckdb.connect()
    con.execute(f"PRAGMA threads={DUCKDB_THREADS}")
    con.execute(f"PRAGMA memory_limit='{DUCKDB_MEMORY_LIMIT}'")
    for name, path in (("s2", s2_path), ("s3", s3_path)):
        con.execute(f"""
            CREATE OR REPLACE VIEW {name} AS
            SELECT * FROM read_csv(
                '{path.as_posix()}',
                delim='\\t', header=true, quote='"', escape='"', nullstr=''
            )
        """)
    return con


def fetch_records(con, record_ids) -> pd.DataFrame:
    """Raw records for the given S2/S3 ids.

    Columns: entity_id, business_name, business_address, country, source.
    Deduped first-wins on entity_id.
    """
    ids = [str(i) for i in record_ids]
    s2_ids = [i for i in ids if i.startswith("S2-")]
    s3_ids = [i for i in ids if i.startswith("S3-")]
    frames = []
    for src, view, idlist in (("S2", "s2", s2_ids), ("S3", "s3", s3_ids)):
        if not idlist:
            continue
        con.register("tmp_ids", pd.DataFrame({"entity_id": idlist}))
        q = (f"SELECT entity_id, business_name, business_address, country "
             f"FROM {view} WHERE entity_id IN (SELECT entity_id FROM tmp_ids)")
        df = con.execute(q).df()
        df["source"] = src
        frames.append(df)
    if not frames:
        return pd.DataFrame(
            columns=["entity_id", "business_name", "business_address", "country", "source"]
        )
    out = pd.concat(frames, ignore_index=True)
    return out.drop_duplicates(subset="entity_id", keep="first").reset_index(drop=True)


# ============================================================
# Cluster selection
# ============================================================

def select_clusters(
    matches: pd.DataFrame,
    excluded_s1: set[str],
    excluded_rec: set[str],
    con=None,
) -> pd.DataFrame:
    """Filter + sample clusters. Returns a long (entity_id, record_id) table.

    1. drop excluded ids,
    2. keep groups with record count in [MIN_RECORDS, MAX_RECORDS],
    3. favour cross-script groups (oversampled CROSS_SCRIPT_OVERSAMPLE : 1),
       sample TARGET_CLUSTERS groups with seed SEED.
    """
    m = matches[
        ~matches["record_id"].isin(excluded_rec)
        & ~matches["entity_id"].isin(excluded_s1)
    ].copy()
    counts = m.groupby("entity_id")["record_id"].transform("size")
    m = m[(counts >= MIN_RECORDS) & (counts <= MAX_RECORDS)].reset_index(drop=True)
    if m.empty:
        return m[["entity_id", "record_id"]]

    own_con = con is None
    if own_con:
        con = _duck_connect()
    try:
        recs = fetch_records(con, m["record_id"].unique())
    finally:
        if own_con:
            con.close()

    recs["script"] = recs["business_name"].map(primary_script)
    scr = dict(zip(recs["entity_id"], recs["script"]))
    m["script"] = m["record_id"].map(scr)

    group_scripts = m.groupby("entity_id")["script"].nunique()
    cross = (group_scripts > 1)
    all_ids = cross.index.to_numpy()
    is_cross = cross.to_numpy()

    weights = np.where(is_cross, CROSS_SCRIPT_OVERSAMPLE, 1.0)
    weights = weights / weights.sum()
    n = int(min(TARGET_CLUSTERS, len(all_ids)))
    rng = np.random.default_rng(SEED)
    chosen = set(rng.choice(all_ids, size=n, replace=False, p=weights))

    out = m[m["entity_id"].isin(chosen)]
    return out[["entity_id", "record_id"]].reset_index(drop=True)


def build_records(clusters: pd.DataFrame, s2_path: Path, s3_path: Path) -> pd.DataFrame:
    """Materialize the chosen clusters' records from S2/S3.

    Columns: entity_id, record_id, business_name, business_address, country,
    source, script.
    """
    con = _duck_connect(s2_path, s3_path)
    try:
        recs = fetch_records(con, clusters["record_id"].unique())
    finally:
        con.close()
    recs["script"] = recs["business_name"].map(primary_script)

    cl = clusters.rename(columns={"entity_id": "group_id"})
    recs = recs.rename(columns={"entity_id": "record_id"})
    out = cl.merge(recs, on="record_id", how="inner")
    out = out.rename(columns={"group_id": "entity_id"})
    cols = ["entity_id", "record_id", "business_name", "business_address",
            "country", "source", "script"]
    return out[cols].reset_index(drop=True)


# ============================================================
# Main
# ============================================================

def main() -> None:
    t0 = time.time()
    print("[1/5] loading ground truth ...", flush=True)
    gt = load_ground_truth()
    matches = explode_matches(gt)
    print(f"      {len(matches):,} unique S2/S3 matches across "
          f"{gt['source1_entity_id'].nunique():,} S1 groups")

    print("[2/5] collecting existing-subset ids ...", flush=True)
    s1_ids, cand_ids = collect_subset_ids()
    print(f"      exclude {len(s1_ids):,} S1 ids, {len(cand_ids):,} candidate ids")

    print("[3/5] selecting clusters (cross-script favoured) ...", flush=True)
    clusters = select_clusters(matches, s1_ids, cand_ids)
    print(f"      {clusters['entity_id'].nunique():,} clusters, "
          f"{len(clusters):,} records")

    print("[4/5] materializing records from S2/S3 ...", flush=True)
    records = build_records(clusters, S2_PATH, S3_PATH)
    records = records.dropna(subset=["business_name"]).reset_index(drop=True)
    if records.empty:
        raise SystemExit("no records materialized — check the id prefixes / paths")
    print(f"      {len(records):,} records; scripts: "
          f"{records['script'].value_counts().to_dict()}")

    out_tsv = DEMO_DIR / "demo_corpus.tsv"
    records.to_csv(out_tsv, sep="\t", index=False, encoding="utf-8")
    print(f"      wrote {out_tsv.name}")

    print("[5/5] embedding corpus (modelADDR then modelD) ...", flush=True)
    import torch

    enc_addr = serving.load_encoder(serving.CHECKPOINT_ADDR, serving.MAX_LEN_ADDR)
    addr_text = apply_text_variant(
        records, ("business_name", "business_address"), "name_address"
    ).tolist()
    addr_emb = serving.encode_texts(enc_addr, addr_text, serving.PASSAGE_PREFIX, batch_size=64)
    np.save(DEMO_DIR / "corpus_addr.npy", addr_emb)
    print(f"      wrote corpus_addr.npy {addr_emb.shape}")
    del enc_addr
    torch.cuda.empty_cache()

    enc_name = serving.load_encoder(serving.CHECKPOINT_D, serving.MAX_LEN_NAME)
    name_text = records["business_name"].fillna("").astype(str).tolist()
    name_emb = serving.encode_texts(enc_name, name_text, serving.PASSAGE_PREFIX, batch_size=256)
    np.save(DEMO_DIR / "corpus_name.npy", name_emb)
    print(f"      wrote corpus_name.npy {name_emb.shape}")

    print(f"done in {time.time() - t0:,.0f}s")


if __name__ == "__main__":
    main()
