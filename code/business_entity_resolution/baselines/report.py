"""
Local evaluation of a Colab-produced score book -> results/<model>.json
+ a merged central table across all models run so far.

Usage:
    .venv/Scripts/python.exe baselines/report.py --model  modelB
    .venv/Scripts/python.exe baselines/report.py           # re-merge all

Score book: results/<model>_scores.parquet with (split, row_idx, score),
row_idx aligned to load_split_pairs order. Train pair rows are allowed to
be absent (TF-IDF deduped its train scoring) but if present must align.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import evaluate  # noqa: E402

RESULTS = evaluate.RESULTS_DIR


def eval_one_model(model: str) -> dict:
    score_path = RESULTS / f"{model}_scores.parquet"
    if not score_path.exists():
        raise FileNotFoundError(score_path)
    book = pd.read_parquet(score_path)

    payload: dict = {"model": model}
    for split in ("train", "val", "test"):
        sub = book[book["split"] == split]
        if not len(sub):
            continue
        df = evaluate.load_split_pairs(split)
        scores = evaluate.align_scores(len(df), sub[["row_idx", "score"]])

        if split == "val":
            thr = evaluate.best_threshold_f05(df["label"].to_numpy(), scores)
            payload["threshold_val_f05"] = float(thr)
            payload["val"] = evaluate.sliced_report(df, scores, thr)
        elif split == "test":
            thr = payload.get("threshold_val_f05")
            if thr is None:
                # modelB book was produced without a train section: tune
                # threshold on val within this same branch first
                val_df = evaluate.load_split_pairs("val")
                val_sub = book[book["split"] == "val"]
                val_scores = evaluate.align_scores(
                    len(val_df), val_sub[["row_idx", "score"]]
                )
                thr = evaluate.best_threshold_f05(val_df["label"].to_numpy(), val_scores)
                payload["threshold_val_f05"] = float(thr)
                payload["val"] = evaluate.sliced_report(val_df, val_scores, thr)
            payload["test"] = evaluate.sliced_report(df, scores, thr)
            payload["retrieval"] = evaluate.retrieval_metrics(df, scores)
        else:  # train, informational
            payload["train_pairs_scored"] = int(len(sub))

    return payload


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="re-evaluate one parquet, e.g. modelB")
    ap.add_argument("--out-name", default=None, help="override results/<name>.json")
    args = ap.parse_args()

    if args.model:
        payload = eval_one_model(args.model)
        name = args.out_name or args.model
        out = evaluate.save_results(name, payload)
        print(json.dumps(payload["test"]["overall"], indent=2))
        print(json.dumps(payload["test"]["cross_script"], indent=2))
        print(json.dumps(payload["retrieval"], indent=2))
        print("->", out)
        return

    # merge all model reports into the central table (handoff 13-14)
    rows = []
    for p in sorted(RESULTS.glob("*.json")):
        if p.name in ("env_smoke.json",):
            continue
        data = json.loads(p.read_text(encoding="utf-8"))
        if "model" not in data or "test" not in data:
            continue
        base = {
            "model": data["model"],
            "overall_f05": data["test"]["overall"]["f0.5"],
            "same_script_f05": data["test"]["same_script"]["f0.5"],
            "cross_script_f05": data["test"]["cross_script"]["f0.5"],
            "cross_recall": data["test"]["cross_script"]["recall"],
            "recall@1": data.get("retrieval", {}).get("recall@1"),
            "mrr": data.get("retrieval", {}).get("mrr"),
        }
        per_script = data["test"].get("cross_script_by_script", {})
        for script, m in per_script.items():
            base[f"{script}_recall"] = m["recall"]
        rows.append(base)

    table = pd.DataFrame(rows)
    cols = [c for c in table.columns if c != "model"]
    table = table[["model"] + cols].sort_values("cross_script_f05")
    out = evaluate.save_results("central_table", {"rows": table.to_dict("records")})
    print(table.to_string(index=False))
    print("->", out)


if __name__ == "__main__":
    main()
