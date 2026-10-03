"""
Phase 8: per-script cross-model matrix + dedicated cross-script table.

Produces:
  results/per_script_matrix.json  -- {script: {model: {n_pos, recall, f05}}}
  results/cross_script_table.json -- dedicated cross-script slice per model
Prints both as markdown-ready tables.

All models must have been evaluated first (report.py --model <m>).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import evaluate  # noqa: E402

RESULTS = evaluate.RESULTS_DIR

# Canonical script order for the handoff section 14 table
SCRIPT_ORDER = [
    "Devanagari", "Bengali", "Gujarati", "Gurmukhi", "Kannada",
    "Malayalam", "Oriya", "Tamil", "Telugu",
]

MODEL_ORDER = [
    "tfidf_char", "tfidf_word", "tfidf_both",
    "modelB", "modelC", "modelD", "modelA_name_address",
    "modelB_name_address", "modelC_addrres",
]


def main() -> None:
    # ---- per-script matrix ----------------------------------------------
    reports: dict[str, dict] = {}
    for p in sorted(RESULTS.glob("*.json")):
        if p.name in ("env_smoke.json", "central_table.json",
                      "per_script_matrix.json", "cross_script_table.json"):
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        if "model" in d and "test" in d:
            reports[d["model"]] = d

    if not reports:
        raise SystemExit("no model reports found -- run report.py --model <m> first")

    per_script: dict[str, dict[str, dict]] = {}
    md_lines = ["| Script | n | " + " | ".join(
        f"{m} R@10 / F0.5" for m in reports)]  # simplistic header, recomputed below
    matrix_rows = []
    for script in SCRIPT_ORDER:
        row = {"script": script}
        n_pos = None
        for model, rep in reports.items():
            entry = rep["test"].get("cross_script_by_script", {}).get(script)
            if entry is None:
                row[model] = None
                continue
            n = entry["tp"] + entry["fn"]
            n_pos = n if n_pos is None else n_pos  # n is same across models
            row[model] = {"n": n, "recall": entry["recall"], "f05": entry["f0.5"]}
        per_script[script] = {m: row[m] for m in reports if row[m] is not None}
        matrix_rows.append(row)

    # n per script from any model (they all share the same slice definition)
    n_by_script = {}
    for script in SCRIPT_ORDER:
        for model, rep in reports.items():
            e = rep["test"]["cross_script_by_script"].get(script)
            if e:
                n_by_script[script] = e["tp"] + e["fn"]
                break

    # ---- markdown tables --------------------------------------------------
    models = [m for m in MODEL_ORDER if m in reports] + \
             [m for m in reports if m not in MODEL_ORDER]

    out1 = ["## Per-script cross-script performance (test split)"]
    out1.append("")
    out1.append("| Script | n | " + " | ".join(models) + " |")
    out1.append("|---|---:|" + "|".join(["---"] * len(models)) + "|")
    for script in SCRIPT_ORDER:
        cells = []
        for m in models:
            e = per_script[script].get(m)
            if e is None:
                cells.append("--")
            else:
                cells.append(f"{e['recall']:.2f} / {e['f05']:.2f}")
        out1.append(f"| {script} | {n_by_script.get(script, 0)} | " + " | ".join(cells) + " |")

    # ---- dedicated cross-script table -------------------------------------
    out2 = ["", "## Dedicated cross-script slice (test split, threshold = each model's val-tuned thr)"]
    out2.append("")
    out2.append("| Model | Threshold | n pos | Precision | Recall | F1 | F0.5 |")
    out2.append("|---|---:|---:|---:|---:|---:|---:|")
    for m in models:
        rep = reports[m]["test"]["cross_script"]
        n = rep["tp"] + rep["fn"]
        out2.append(
            f"| {m} | {reports[m].get('threshold_val_f05', float('nan')):.3f} | {n} | "
            f"{rep['precision']:.3f} | {rep['recall']:.3f} | {rep['f1']:.3f} | {rep['f0.5']:.3f} |"
        )

    md = "\n".join(out1 + out2)
    (RESULTS / "phase8_tables.md").write_text(md, encoding="utf-8")
    (RESULTS / "per_script_matrix.json").write_text(
        json.dumps({"scripts": per_script, "n_by_script": n_by_script},
                   indent=2, ensure_ascii=False),
        encoding="utf-8")
    (RESULTS / "cross_script_table.json").write_text(
        json.dumps({m: reports[m]["test"]["cross_script"] for m in models},
                   indent=2, ensure_ascii=False),
        encoding="utf-8")

    print(md)
    print("\nsaved: phase8_tables.md, per_script_matrix.json, cross_script_table.json")


if __name__ == "__main__":
    main()
