"""
Small local tests for the notebook + training modules (run before any
Colab session). Each test mirrors exactly one notebook logic block so a
failure pinpoints the cell to fix.

    cd code/business_entity_resolution
    .venv/Scripts/python.exe -m pytest training/test_notebook_logic.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE.parent / "baselines"))

from dataset import apply_text_variant, load_pairs  # noqa: E402
from losses import mnrl  # noqa: E402
import evaluate  # noqa: E402

TRAIN_PAIRS = evaluate.SUBSET / "train_pairs.tsv"


# ------------------------------------------------------------
# notebook data-upload cell + data shape sanity
# ------------------------------------------------------------

def test_subset_tsvs_present_and_readable():
    for split in ("train", "val", "test"):
        df = load_pairs(evaluate.SUBSET / f"{split}_pairs.tsv")
        assert {"s1_id", "s1_name", "candidate_name", "label", "pair_type", "row_idx"}.issubset(df.columns)
        assert len(df) > 0


def test_labels_binary_and_pair_type_consistent():
    df = load_pairs(TRAIN_PAIRS)
    assert set(df["label"].unique()) == {0, 1}
    assert ((df["pair_type"] == "positive") == (df["label"] == 1)).all()


# ------------------------------------------------------------
# score-book contract (notebook output -> local eval)
# ------------------------------------------------------------

def test_score_book_alignment_round_trip():
    df = load_pairs(evaluate.SUBSET / "test_pairs.tsv")
    scores = np.random.RandomState(0).rand(len(df)).astype(np.float32)
    book = pd.DataFrame({"split": "test", "row_idx": df["row_idx"], "score": scores})
    dense = evaluate.align_scores(len(df), book)
    assert np.allclose(dense, scores)


def test_align_scores_rejects_duplicates_and_gaps():
    df = load_pairs(evaluate.SUBSET / "val_pairs.tsv")
    n = len(df)
    good = pd.DataFrame({"split": "val", "row_idx": np.arange(n), "score": np.zeros(n)})
    evaluate.align_scores(n, good)

    dup = pd.DataFrame({"split": "val", "row_idx": [0, 0], "score": [0.0, 0.1]})
    with pytest.raises(ValueError):
        evaluate.align_scores(n, dup)

    short = good.iloc[:-1]
    with pytest.raises(ValueError):
        evaluate.align_scores(n, short)


# ------------------------------------------------------------
# mirrors notebook helpers: script tags, variants, enc prefix
# ------------------------------------------------------------

def test_script_tags_match_evaluate_module():
    pairs = [
        ("Silver Solutions Private Limited", {"Latin"}),
        ("सिल्वर सॉल्यूशंस प्राइवेट लिमिटेड", {"Devanagari"}),
        ("ఆల్ఫా ఇంటర్నేషనల్ ప్రైవేట్ లిమిటెడ్", {"Telugu"}),
        ("ஜெயா கேபிடல்", {"Tamil"}),
    ]
    for s, expect in pairs:
        tags = {t for t in evaluate.script_tags(s) if t in {"Latin", "Devanagari", "Telugu", "Tamil"}}
        assert tags == expect, (s, tags, expect)


def test_apply_variant_name_and_name_address():
    df = pd.DataFrame([{"s1_name": "Jain Healthcare", "s1_address": "B-73 Sector 6 Noida"}])
    assert apply_text_variant(df, ("s1_name", "s1_address"), "name").iloc[0] == "Jain Healthcare"
    v = apply_text_variant(df, ("s1_name", "s1_address"), "name_address").iloc[0]
    assert v == "Business name: Jain Healthcare | Address: B-73 Sector 6 Noida"
    with pytest.raises(ValueError):
        apply_text_variant(df, ("s1_name", "s1_address"), "bogus")


# ------------------------------------------------------------
# mirrors notebook training path: dataset, collate, loss
# ------------------------------------------------------------

def test_group_pairs_dataset_respects_labels():
    from train_contrastive import GroupPairs
    df = load_pairs(TRAIN_PAIRS).head(400)
    ds = GroupPairs(df, max_hard=4)
    assert len(ds) > 0
    item = ds[0]
    assert set(item) == {"anchor", "positives", "hards"}
    assert item["anchor"] and len(item["positives"]) >= 1
    # hards must come only from label=0 hard_negative rows (a truncated
    # frame may legitimately give a group zero hards)
    if item["hards"]:
        sub = df[df["s1_id"] == df[df["s1_name"] == item["anchor"]]["s1_id"].iloc[0]]
        rows = sub[sub["candidate_name"].isin(item["hards"])]
        assert (rows["label"] == 0).all(), rows


def test_collate_shapes_match():
    import torch
    from transformers import AutoTokenizer
    from train_contrastive import collate
    tok = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-small")
    batch = [
        {"anchor": "Silver Solutions Private Limited", "positives": ["सिल्वर सॉल्यूशंस प्राइवेट लिमिटेड", "Silver Sol Pvt Ltd"], "hards": ["Silver Solutions Trading Private Limited", "Silver Trading Co"]},
        {"anchor": "Alpha International Pvt Ltd", "positives": ["ఆల్ఫా ఇంటర్నేషనల్ ప్రైవేట్ లిమిటెడ్"], "hards": []},
    ]
    out = collate(batch, tok, max_pos=2, max_extra_neg=4)
    # anchors: 2 + 1 positives = 3 anchor rows, 3 positive rows, 2 hard rows
    assert out["anchor"]["input_ids"].shape[0] == 3
    assert out["positive"]["input_ids"].shape[0] == 3
    assert out["hard"]["input_ids"].shape[0] == 2
    assert out["anchor"]["input_ids"].shape[1] <= 64
    # prefix scheme actually visible to the tokenizer (XLM-R adds <s>...</s>)
    assert "query:" in tok.decode(out["anchor"]["input_ids"][0])


def test_collate_no_hard_negatives_phase_C():
    import torch
    from transformers import AutoTokenizer
    from train_contrastive import collate
    tok = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-small")
    batch = [{"anchor": "A", "positives": ["B"], "hards": ["C"]}]
    out = collate(batch, tok, max_pos=2, max_extra_neg=0)
    assert out["hard"] is None
    assert out["positive"]["input_ids"].shape[0] == 1


def test_mnrl_decreases_with_better_alignment():
    import torch
    torch.manual_seed(0)
    B, D = 8, 16
    a = torch.nn.functional.normalize(torch.randn(B, D), dim=-1)
    # identical positives: loss should be minimal-ish but > 0 because of in-batch masking
    loss_same = mnrl(a, a.clone())
    # shuffled positives: loss should be strictly higher
    perm = torch.randperm(B)
    loss_shuffled = mnrl(a, a[perm])
    assert loss_shuffled > loss_same


def test_mnrl_with_explicit_negatives_changes_loss():
    import torch
    from losses import mnrl as mnrl_fn
    torch.manual_seed(0)
    B, D, K = 8, 16, 2
    a = torch.nn.functional.normalize(torch.randn(B, D), dim=-1)
    p = torch.nn.functional.normalize(torch.randn(B, D), dim=-1)
    neg = torch.nn.functional.normalize(torch.randn(B, K, D), dim=-1)
    loss_no = mnrl_fn(a, p, extra_neg_emb=None)
    loss_neg = mnrl_fn(a, p, extra_neg_emb=neg)
    assert loss_neg > loss_no


def test_mnrl_gradient_flows_through_extra_negatives():
    import torch
    from losses import mnrl as mnrl_fn
    torch.manual_seed(0)
    a_leaf = torch.randn(4, 8, requires_grad=True)
    p_leaf = torch.randn(4, 8, requires_grad=True)
    neg_leaf = torch.randn(4, 2, 8, requires_grad=True)
    a = torch.nn.functional.normalize(a_leaf, dim=-1)
    p = torch.nn.functional.normalize(p_leaf, dim=-1)
    neg = torch.nn.functional.normalize(neg_leaf, dim=-1)
    loss = mnrl_fn(a, p, extra_neg_emb=neg)
    loss.backward()
    for t in (a_leaf, p_leaf, neg_leaf):
        assert t.grad is not None and t.grad.abs().sum() > 0


# ------------------------------------------------------------
# notebook MNRL cell is identical math to src/losses.py
# ------------------------------------------------------------

def test_notebook_mnrl_matches_src_loss():
    import torch
    torch.manual_seed(1)
    a = torch.nn.functional.normalize(torch.randn(5, 10), dim=-1)
    p = torch.nn.functional.normalize(torch.randn(5, 10), dim=-1)
    extraneg = torch.nn.functional.normalize(torch.randn(5, 2, 10), dim=-1)
    l_src = mnrl(a, p, extra_neg_emb=extraneg)
    # notebook version (from the ipynb source, inline)
    sim = a @ p.T * 20.0
    neg = torch.einsum("bd,bkd->bk", a, extraneg) * 20.0
    sim = torch.cat([sim, neg], dim=1)
    labels = torch.arange(sim.shape[0])
    l_nb = torch.nn.functional.cross_entropy(sim, labels)
    assert torch.allclose(l_src, l_nb, atol=1e-6)


# ------------------------------------------------------------
# threshold tuning + slices (local eval path, notebook end-to-end)
# ------------------------------------------------------------

def test_best_threshold_f05_on_perfect_scores():
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.8, 0.9])
    thr = evaluate.best_threshold_f05(y, s)
    m = evaluate.binary_metrics(y, s, thr)
    assert m["f0.5"] == 1.0 and m["precision"] == 1.0 and m["recall"] == 1.0


def test_sliced_report_keys_present():
    df = load_pairs(evaluate.SUBSET / "test_pairs.tsv").head(500)
    s = np.random.RandomState(1).rand(len(df))
    rep = evaluate.sliced_report(df, s, threshold=0.5)
    assert {"overall", "same_script", "cross_script"} <= set(rep)
    for part in ("overall", "same_script", "cross_script"):
        assert {"precision", "recall", "f1", "f0.5"} <= set(rep[part])


def test_retrieval_metrics_perfect_and_random():
    df = load_pairs(evaluate.SUBSET / "test_pairs.tsv").head(50)
    # perfect: positives 1.0, negatives 0.0
    y = df["label"].to_numpy().astype(bool)
    s_perfect = np.where(y, 1.0, 0.0).astype(np.float64)
    r = evaluate.retrieval_metrics(df, s_perfect)
    assert r["recall@1"] == 1.0
    assert r["mrr"] == 1.0
    s_rand = np.random.RandomState(2).rand(len(df))
    r2 = evaluate.retrieval_metrics(df, s_rand)
    assert 0 <= r2["mrr"] <= 1
