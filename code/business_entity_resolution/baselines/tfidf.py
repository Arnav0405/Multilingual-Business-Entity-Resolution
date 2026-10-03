"""
Model A baseline: TF-IDF char + word n-grams -> cosine -> threshold.

Handoff section 4. No normalization, no preprocessing of any kind:
strings go to the vectorizer exactly as they appear in the TSVs.

Char n-grams capture cross-script subword overlap; word n-grams capture
lexical identity. Vectorizer variant is a flag so the ablation is one
run each:

    --analyzer char   (char_wb 2-4)
    --analyzer word   (word 1-2)
    --analyzer both   (FeatureUnion of the two, default)

Outputs a score book (results/tfidf_<variant>_scores.parquet) with
    split, row_idx, score
consumable by baselines/report.py, plus results/tfidf_<variant>.json.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import hstack, vstack
from sklearn.feature_extraction.text import TfidfVectorizer

import evaluate  # sibling module

REPO_ROOT = Path(__file__).resolve().parents[2]
SPLITS = ("train", "val", "test")


def build_pairs(split: str) -> pd.DataFrame:
    df = evaluate.load_split_pairs(split)
    if split == "train":
        # dedupe identical (s1_name, candidate_name) text pairs:
        # scores are per STRING PAIR, and training a threshold does not
        # need the 69k duplicates this subset carries.
        df = df.drop_duplicates(subset=["s1_name", "candidate_name"]).reset_index(drop=True)
    df["row_idx"] = df.index
    return df


class BothVectorizer:
    """Char-WB 2-4 + word 1-2 concatenation, fitted once."""

    def fit(self, corpus):
        self.char = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2, sublinear_tf=True, max_features=400_000)
        self.word = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=400_000)
        self.char.fit(corpus)
        self.word.fit(corpus)
        return self

    def transform(self, texts):
        return hstack([self.char.transform(texts), self.word.transform(texts)])


def cross_scores(vec, a_texts, b_texts, batch=100_000) -> np.ndarray:
    """cosine(TFIDF(a), TFIDF(b)) for aligned a/b lists."""
    va = vec.transform(a_texts)
    vb = vec.transform(b_texts)
    # row-wise cosine via elementwise products in batches
    out = np.empty(len(a_texts), dtype=np.float32)
    for i in range(0, len(a_texts), batch):
        pa = va[i:i + batch]
        pb = vb[i:i + batch]
        na = np.sqrt(pa.multiply(pa).sum(axis=1)).A.ravel()
        nb = np.sqrt(pb.multiply(pb).sum(axis=1)).A.ravel()
        denom = na * nb
        prod = pa.multiply(pb).sum(axis=1).A.ravel()
        out[i:i + batch] = np.divide(prod, denom, out=np.zeros_like(prod), where=denom > 0)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["char", "word", "both"], default="both")
    args = ap.parse_args()

    t0 = time.time()
    data = {s: build_pairs(s) for s in SPLITS}
    train_df, val_df, test_df = data["train"], data["val"], data["test"]

    fit_texts = [t for s in ("train", "val") for t in (
        data[s]["s1_name"].tolist() + data[s]["candidate_name"].tolist()
    )]
    print(f"[{args.variant}] fitting on {len(fit_texts)} name strings ...")
    model = BothVectorizer() if args.variant == "both" else (
        TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2, sublinear_tf=True)
        if args.variant == "char" else
        TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    )
    model.fit(fit_texts)

    score_frames = []
    for split, df in data.items():
        s = cross_scores(model, df["s1_name"].tolist(), df["candidate_name"].tolist())
        score_frames.append(pd.DataFrame({"split": split, "row_idx": df["row_idx"], "score": s}))
        print(f"  {split}: {len(df)} pairs scored in elapsed {time.time()-t0:.0f}s")

    all_scores = pd.concat(score_frames, ignore_index=True)
    out_scores = REPO_ROOT / "code" / "business_entity_resolution" / "results"
    out_scores.mkdir(parents=True, exist_ok=True)
    score_path = out_scores / f"tfidf_{args.variant}_scores.parquet"
    all_scores.to_parquet(score_path, index=False)

    y_val = val_df["label"].to_numpy()
    s_val = all_scores[all_scores["split"] == "val"]["score"].to_numpy()
    thr = evaluate.best_threshold_f05(y_val, s_val)

    y_test = test_df["label"].to_numpy()
    s_test = all_scores[all_scores["split"] == "test"]["score"].to_numpy()
    report = evaluate.sliced_report(test_df, s_test, thr)
    retrieval = evaluate.retrieval_metrics(test_df, s_test)
    val_report = evaluate.sliced_report(val_df, s_val, thr)

    payload = {
        "model": f"tfidf_{args.variant}",
        "threshold_val_f05": float(thr),
        "val": val_report,
        "test": report,
        "retrieval": retrieval,
        "n_train_pairs_scored": int(len(train_df)),
        "elapsed_s": time.time() - t0,
    }
    out_json = out_scores / f"tfidf_{args.variant}.json"
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps({
        "test_overall": report["overall"],
        "test_same": report["same_script"],
        "test_cross": report["cross_script"],
        "retrieval": retrieval,
    }, indent=2))
    print("saved:", score_path.name, out_json.name)


if __name__ == "__main__":
    main()
