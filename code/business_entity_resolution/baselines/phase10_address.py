"""
Phase 10: A' (name+address) vs D (name-only) controlled comparison.

The naive A'-vs-D table is fine for the both-addresses rows but confounded
by the ~3.5% of pairs where either side's address is empty (A' sees
'Address: ' empty; D never saw addresses at all). This script computes:

  overall            -- standard shared eval (A' scores, D scores)
  addr_ok slice      -- pairs where BOTH sides have a non-empty address
  addr_missing slice -- the complement

for binary pair metrics, plus the retrieval view per slice. Deltas D -> A'
per slice answer "does contextual address info improve resolution?".
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import evaluate  # noqa: E402

RESULTS = evaluate.RESULTS_DIR


def addr_slice_masks(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    s1_ok = df["s1_address"].fillna("").astype(str).str.strip().str.len() > 0
    cand_ok = df["candidate_address"].fillna("").astype(str).str.strip().str.len() > 0
    both = (s1_ok & cand_ok).to_numpy()
    return both, ~both


def main() -> None:
    a_name = "modelA_name_address"
    d_name = "modelD"

    a_book = pd.read_parquet(RESULTS / f"{a_name}_scores.parquet")
    d_book = pd.read_parquet(RESULTS / f"{d_name}_scores.parquet")

    out: dict = {}
    for split in ("val", "test"):
        df = evaluate.load_split_pairs(split)
        a_s = evaluate.align_scores(len(df), a_book[a_book["split"] == split][["row_idx", "score"]])
        d_s = evaluate.align_scores(len(df), d_book[d_book["split"] == split][["row_idx", "score"]])
        y = df["label"].to_numpy()

        # A' uses its own val-tuned threshold; D also uses its own val one.
        # Tune both thresholds HERE on the same val rows (already the case:
        # report.py used these same books), then apply to test per model.
        if split == "val":
            out["thresholds"] = {
                a_name: float(evaluate.best_threshold_f05(y, a_s)),
                d_name: float(evaluate.best_threshold_f05(y, d_s)),
            }
        both, missing = addr_slice_masks(df)
        thr_a = out["thresholds"][a_name]
        thr_d = out["thresholds"][d_name]

        out[split] = {
            "overall": {
                a_name: evaluate.binary_metrics(y, a_s, thr_a),
                d_name: evaluate.binary_metrics(y, d_s, thr_d),
            },
            "addr_both": {
                a_name: evaluate.binary_metrics(y[both], a_s[both], thr_a),
                d_name: evaluate.binary_metrics(y[both], d_s[both], thr_d),
                "n_pos": int(y[both].sum()),
            },
            "addr_missing": {
                a_name: evaluate.binary_metrics(y[missing], a_s[missing], thr_a),
                d_name: evaluate.binary_metrics(y[missing], d_s[missing], thr_d),
                "n_pos": int(y[missing].sum()),
            },
        }
        out[split]["addr_both_retrieval"] = {
            a_name: evaluate.retrieval_metrics(df[both], a_s[both]),
            d_name: evaluate.retrieval_metrics(df[both], d_s[both]),
        }
        # script relation slice, overall test only
        if split == "test":
            rel = evaluate.pair_script_relation(df).to_numpy()
            out["test"]["cross_script"] = {
                a_name: evaluate.binary_metrics(y[rel == "cross_script"], a_s[rel == "cross_script"], thr_a),
                d_name: evaluate.binary_metrics(y[rel == "cross_script"], d_s[rel == "cross_script"], thr_d),
            }

    p = RESULTS / "phase10_address_experiment.json"
    p.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---- print ----
    def row(tag, model, m):
        return f"| {tag} | {model} | {m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} | {m['f0.5']:.4f} |"

    print("## A' (name+address) vs D (name-only), same pairs, each at its own val-tuned threshold")
    print("thresholds:", out["thresholds"])
    for split in ("val", "test"):
        print(f"\n### {split}")
        print("| Slice | Model | Precision | Recall | F1 | F0.5 |")
        print("|---|---|---:|---:|---:|---:|")
        for tag in ("overall", "addr_both", "addr_missing"):
            n_pos = out[split][tag].get("n_pos")
            for model in (d_name, a_name):
                m = out[split][tag][model]
                label = f"{tag} (n_pos={n_pos})" if n_pos is not None else tag
                print(row(label, model, m))
        if split == "test":
            print("\ncross-script slice (test):")
            for model in (d_name, a_name):
                print(row("cross_script", model, out["test"]["cross_script"][model]))
    print("\nsaved ->", p.name)


if __name__ == "__main__":
    main()
