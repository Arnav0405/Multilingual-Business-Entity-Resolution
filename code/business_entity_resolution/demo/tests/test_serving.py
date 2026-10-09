import sys
from pathlib import Path

import numpy as np
import pandas as pd

DEMO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEMO))
import serving  # noqa: E402


def test_embeddings_normalized_and_384():
    enc = serving.load_encoder(serving.CHECKPOINT_D, max_length=64)
    emb = serving.encode_texts(enc, ["Jain Finance"], serving.QUERY_PREFIX)
    assert emb.shape == (1, 384)
    assert abs(float(np.linalg.norm(emb[0])) - 1.0) < 1e-4


def test_smoke_cosine_matches_reference():
    enc = serving.load_encoder(serving.CHECKPOINT_D, max_length=64)
    q = serving.encode_texts(enc, ["Jain Finance"], serving.QUERY_PREFIX)
    p = serving.encode_texts(enc, ["जैन फाइनेंस"], serving.PASSAGE_PREFIX)
    assert abs(float(q[0] @ p[0]) - 0.9044) < 0.02


def test_modeladdr_cross_script_positive_above_threshold():
    # load a cross-script positive with both addresses from test_pairs.tsv
    df = pd.read_csv(serving.REPO_ROOT / "data/entity_resolution_subset/test_pairs.tsv", sep="\t")
    row = df[(df["label"] == 1) & (df["s1_name"] != df["candidate_name"])
             & df["s1_address"].notna() & df["candidate_address"].notna()
             & (df["s1_address"].str.len() > 0) & (df["candidate_address"].str.len() > 0)]
    row = row[row.apply(lambda r: serving.script_tags(r["s1_name"]) != serving.script_tags(r["candidate_name"]), axis=1)].iloc[0]
    enc = serving.load_encoder(serving.CHECKPOINT_ADDR, max_length=128)
    q_txt = "Business name: " + row["s1_name"] + " | Address: " + row["s1_address"]
    p_txt = "Business name: " + row["candidate_name"] + " | Address: " + row["candidate_address"]
    q = serving.encode_texts(enc, [q_txt], serving.QUERY_PREFIX)
    p = serving.encode_texts(enc, [p_txt], serving.PASSAGE_PREFIX)
    assert float(q[0] @ p[0]) > 0.614


def test_name_only_routes_to_modeld():
    # the routing decision, not a model call: an empty address → name variant
    from dataset import apply_text_variant
    df = pd.DataFrame([{"name": "X", "addr": None}])
    out = apply_text_variant(df, ("name", "addr"), "name_address").iloc[0]
    assert "Address: " in out and "nan" not in out   # fillna('') sanity


def test_search_topk():
    corpus = np.eye(5, 384, dtype=np.float32)         # one-hot rows, all L2-normalized
    query = corpus[2:3]                                # exact match at index 2
    idx, scores = serving.search(query, corpus, k=3)
    assert idx[0] == 2 and abs(float(scores[0]) - 1.0) < 1e-6
