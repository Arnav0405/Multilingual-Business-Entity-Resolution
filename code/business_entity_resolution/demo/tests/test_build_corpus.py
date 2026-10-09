import sys
from pathlib import Path

import pandas as pd
import pytest

DEMO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEMO))
import build_corpus  # noqa: E402


# --- pure-function tests (no 1 GB files) ------------------------------------

def test_explode_matches_columns():
    gt = pd.DataFrame([{"source1_entity_id": "S1-a", "matched_entity_ids": "S2-x,S3-y"}])
    long = build_corpus.explode_matches(gt)
    assert list(long.columns) == ["record_id", "entity_id"]
    assert set(long["record_id"]) == {"S2-x", "S3-y"}
    assert (long["entity_id"] == "S1-a").all()


def test_no_duplicate_record_ids():
    gt = pd.DataFrame([
        {"source1_entity_id": "S1-a", "matched_entity_ids": "S2-x,S3-y"},
        {"source1_entity_id": "S1-b", "matched_entity_ids": "S2-x,S3-z"},
    ])
    long = build_corpus.explode_matches(gt)
    # an S2/S3 id mapped under two S1s must be dropped or first-wins — never twice
    assert long["record_id"].is_unique


def test_explode_handles_empty_matches():
    gt = pd.DataFrame([
        {"source1_entity_id": "S1-a", "matched_entity_ids": ""},
        {"source1_entity_id": "S1-b", "matched_entity_ids": "S2-x"},
    ])
    long = build_corpus.explode_matches(gt)
    assert set(long["record_id"]) == {"S2-x"}


# --- corpus-artifact tests (need the generated demo_corpus.tsv) -------------

CORPUS = DEMO / "demo_corpus.tsv"


@pytest.mark.slow
@pytest.mark.skipif(not CORPUS.exists(), reason="demo_corpus.tsv not built")
def test_corpus_disjoint_from_subset():
    corpus = pd.read_csv(CORPUS, sep="\t", dtype=str)
    s1_ids, cand_ids = build_corpus.collect_subset_ids()
    assert set(corpus["entity_id"]) & s1_ids == set()
    assert set(corpus["record_id"]) & cand_ids == set()


@pytest.mark.slow
@pytest.mark.skipif(not CORPUS.exists(), reason="demo_corpus.tsv not built")
def test_scripts_populated():
    corpus = pd.read_csv(CORPUS, sep="\t", dtype=str)
    assert corpus["script"].notna().all()
    assert len(set(corpus["script"]) & build_corpus.INDIC_TAGS) >= 1
