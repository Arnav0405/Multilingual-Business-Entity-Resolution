"""
Subset validator -- handoff step 3 ("Do this before training anything").

Verifies data/entity_resolution_subset against the raw files:

  1. required columns exist in every file
  2. train/val/test S1 IDs don't overlap
  3. positives have label=1, negatives label=0 (pair_type <-> label consistent)
  4. every positive pair exists in the raw ground truth
     (and no negative pair exists in the ground truth)
  5. no S1 leakage across pair splits; splits exactly tile selected_s1
     (+ informational: candidate S2/S3 overlap across splits)
  6. cross-script positives exist in every split (per-script counts)
  7. no unexpected nulls / empty names; s1_name matches the S1 tables
  8. counts match subset_statistics.json

Writes data/entity_resolution_subset/validation_report.json.
Exits non-zero on any failed check.

No normalization anywhere: script detection works on the raw strings.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

import duckdb
import pandas as pd

# ============================================================
# PATHS
# ============================================================

REPO_ROOT = Path(__file__).resolve().parents[3]   # scripts/ -> ... -> repo root
SUBSET = REPO_ROOT / "data" / "entity_resolution_subset"
RAW = REPO_ROOT / "data"

PAIR_FILES = {
    split: SUBSET / f"{split}_pairs.tsv" for split in ("train", "val", "test")
}
S1_FILES = {
    split: SUBSET / f"{split}_s1.tsv" for split in ("train", "val", "test")
}
SELECTED = SUBSET / "selected_s1.tsv"
STATS_JSON = SUBSET / "subset_statistics.json"
GT_PATH = RAW / "train_ground_truth.tsv"
REPORT_PATH = SUBSET / "validation_report.json"

PAIR_COLS = [
    "s1_id", "s1_name", "s1_address", "s1_country",
    "candidate_id", "candidate_name", "candidate_address", "candidate_country",
    "label", "pair_type",
]
S1_COLS = [
    "entity_id", "business_name", "business_address", "country",
    "match_count", "indic_match_count", "entity_type",
]

# ============================================================
# Script detection on raw strings (no normalization)
# ============================================================

INDIC_RANGES = {
    "Devanagari": (0x0900, 0x097F),
    "Bengali":    (0x0980, 0x09FF),
    "Gurmukhi":   (0x0A00, 0x0A7F),
    "Gujarati":   (0x0A80, 0x0AFF),
    "Oriya":      (0x0B00, 0x0B7F),
    "Tamil":      (0x0B80, 0x0BFF),
    "Telugu":     (0x0C00, 0x0C7F),
    "Kannada":    (0x0C80, 0x0CFF),
    "Malayalam":  (0x0D00, 0x0D7F),
}

# Literal characters inside the character class (same trick as the
# subset builder: no backslash-u escapes needed).
_SCRIPT_RES = {
    name: re.compile(f"[{chr(lo)}-{chr(hi)}]")
    for name, (lo, hi) in INDIC_RANGES.items()
}

_tags_cache: dict[str, frozenset] = {}


def script_tags(s: object) -> frozenset:
    """Set of scripts present in a raw string. Latin if no Indic range hits."""
    if not isinstance(s, str) or not s:
        return frozenset({"<empty>"})
    cached = _tags_cache.get(s)
    if cached is not None:
        return cached
    tags = {name for name, rx in _SCRIPT_RES.items() if rx.search(s)}
    if not tags:
        tags = {"Latin"}
    out = frozenset(tags)
    _tags_cache[s] = out
    return out


def indic_scripts(s: object) -> set[str]:
    """Subset of script_tags that are Indic scripts (may be empty)."""
    return set(script_tags(s)) & set(INDIC_RANGES)


def is_cross_script(a: object, b: object) -> bool:
    return script_tags(a) != script_tags(b)


# ============================================================
# Reporting helpers
# ============================================================

REPORT: dict = {}
FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    REPORT[name] = {"passed": bool(ok), "detail": detail}
    status = "PASS" if ok else "FAIL"
    print(f"{status}  {name}" + (f"  -- {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def info(name: str, detail: str) -> None:
    REPORT[name] = {"passed": True, "detail": detail, "info_only": True}
    print(f"INFO  {name}  -- {detail}")


def load_pairs(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", dtype={"label": "int8"})


# ============================================================
# Main
# ============================================================

def main() -> None:
    print("=" * 70)
    print("SUBSET VALIDATION:", SUBSET.name)
    print("=" * 70)

    pairs = {split: load_pairs(path) for split, path in PAIR_FILES.items()}
    s1_tabs = {
        split: pd.read_csv(path, sep="\t") for split, path in S1_FILES.items()
    }
    selected = pd.read_csv(SELECTED, sep="\t")
    stats = json.loads(STATS_JSON.read_text(encoding="utf-8"))

    # ---- 1. required columns -------------------------------------------
    for split, df in pairs.items():
        missing = [c for c in PAIR_COLS if c not in df.columns]
        check(f"columns[{split}_pairs]", not missing, f"missing={missing}")
    for split, df in s1_tabs.items():
        missing = [c for c in S1_COLS if c not in df.columns]
        check(f"columns[{split}_s1]", not missing, f"missing={missing}")

    # ---- 2. S1 ID overlap across splits --------------------------------
    id_sets = {split: set(df["entity_id"]) for split, df in s1_tabs.items()}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        inter = id_sets[a] & id_sets[b]
        check(f"s1_id_no_overlap[{a},{b}]", len(inter) == 0, f"overlap={len(inter)}")

    # ---- 3. label <-> pair_type consistency ----------------------------
    for split, df in pairs.items():
        pos_ok = bool(((df["pair_type"] == "positive") == (df["label"] == 1)).all())
        neg_types = set(df.loc[df["label"] == 0, "pair_type"]) <= {"hard_negative", "easy_negative"}
        check(f"label_pair_type_consistent[{split}]", pos_ok and neg_types)

    # ---- 4. pairs vs raw ground truth (DuckDB, equi-joins) -------------
    con = duckdb.connect()
    con.execute("SET memory_limit='6GB'")
    con.execute("SET threads=4")
    con.execute(f"SET temp_directory='{(SUBSET / 'duckdb_tmp').as_posix()}'")

    con.execute(
        f"""
        CREATE OR REPLACE VIEW gt AS
        SELECT DISTINCT
            source1_entity_id AS s1,
            replace(replace(trim(t.cand), chr(13), ''), chr(10), '') AS cand
        FROM read_csv('{GT_PATH.as_posix()}', sep='\t', header=true, all_varchar=true),
             unnest(string_split(matched_entity_ids, ',')) AS t(cand)
        WHERE t.cand <> ''
        """
    )

    for split, df in pairs.items():
        pos = df[df["label"] == 1][["s1_id", "candidate_id"]]
        neg = df[df["label"] == 0][["s1_id", "candidate_id"]]
        con.register("_pos", pos)
        con.register("_neg", neg)

        pos_in_gt = con.execute(
            "SELECT count(*) FROM _pos p SEMI JOIN gt g "
            "ON p.s1_id = g.s1 AND p.candidate_id = g.cand"
        ).fetchone()[0]
        check(
            f"positives_in_ground_truth[{split}]",
            pos_in_gt == len(pos),
            f"{pos_in_gt}/{len(pos)} found",
        )

        neg_in_gt = con.execute(
            "SELECT count(*) FROM _neg p SEMI JOIN gt g "
            "ON p.s1_id = g.s1 AND p.candidate_id = g.cand"
        ).fetchone()[0]
        check(
            f"negatives_not_in_ground_truth[{split}]",
            neg_in_gt == 0,
            f"{neg_in_gt} negatives are actually GT matches",
        )
        con.unregister("_pos")
        con.unregister("_neg")

    con.close()

    # ---- 5. leakage / split tiling --------------------------------------
    pair_ids = {split: set(df["s1_id"]) for split, df in pairs.items()}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        inter = pair_ids[a] & pair_ids[b]
        check(f"pair_split_no_s1_leakage[{a},{b}]", len(inter) == 0, f"overlap={len(inter)}")

    union_ids = set()
    for split in pairs:
        union_ids |= pair_ids[split]
    selected_ids = set(selected["entity_id"])
    check(
        "splits_tile_selected_s1",
        union_ids == selected_ids,
        f"union={len(union_ids)}, selected={len(selected_ids)}, "
        f"missing={len(selected_ids - union_ids)}, extra={len(union_ids - selected_ids)}",
    )

    # pairs belong to their own split's s1 table
    for split, df in pairs.items():
        own = id_sets[split]
        stray = set(df["s1_id"]) - own
        check(f"pairs_match_own_s1_table[{split}]", len(stray) == 0, f"stray={len(stray)}")

    # candidate-side overlap across splits (informational only: the
    # resolution targets are S1 entities, but report it for transparency)
    cand_sets = {split: set(df["candidate_id"]) for split, df in pairs.items()}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        info(
            f"candidate_overlap[{a},{b}]",
            f"{len(cand_sets[a] & cand_sets[b])} shared candidate IDs (informational)",
        )

    # ---- 6. cross-script positives + per-script breakdown ----------------
    per_split_stats: dict[str, dict] = {}
    for split, df in pairs.items():
        s1_tags = df["s1_name"].map(script_tags)
        cand_tags = df["candidate_name"].map(script_tags)
        cross_mask = s1_tags != cand_tags
        pos_mask = df["label"] == 1
        neg_mask = df["label"] == 0

        cross_pos = df[cross_mask & pos_mask]
        script_counts: Counter[str] = Counter()
        names = cross_pos["candidate_name"].tolist()
        for name in names:
            script_counts.update(indic_scripts(name) or {"Latin"})

        per_split_stats[split] = {
            "s1_entities": int(df["s1_id"].nunique()),
            "positive_pairs": int(pos_mask.sum()),
            "negative_pairs": int(neg_mask.sum()),
            "hard_negative_pairs": int((df.loc[neg_mask, "pair_type"] == "hard_negative").sum()),
            "easy_negative_pairs": int((df.loc[neg_mask, "pair_type"] == "easy_negative").sum()),
            "cross_script_positives": int(len(cross_pos := df[cross_mask & pos_mask])),
            "cross_script_negatives": int((cross_mask & neg_mask).sum()),
            "cross_script_candidate_scripts": dict(
                sorted(script_counts.items(), key=lambda kv: -kv[1])
            ),
        }

    total_cross = sum(v["cross_script_positives"] for v in per_split_stats.values())
    check("cross_script_positives_exist", total_cross > 0, f"total={total_cross}")
    for split, v in per_split_stats.items():
        print(
            f"      {split}: S1={v['s1_entities']}, pos={v['positive_pairs']}, "
            f"neg={v['negative_pairs']}, cross-script pos={v['cross_script_positives']}"
        )

    # ---- 7. nulls / empty names / alias consistency -----------------------
    for split, df in pairs.items():
        bad = sum(
            int(df[c].isna().sum()) + int((df[c].astype(str).str.strip() == "").sum())
            for c in ("s1_id", "s1_name", "s1_country", "candidate_id", "candidate_name", "candidate_country")
        )
        check(f"no_null_or_empty_required_fields[{split}]", bad == 0, f"bad={bad}")
        empty_addr = int(
            df["candidate_address"].isna().sum()
            + (df["candidate_address"].astype(str).str.strip() == "").sum()
        )
        name = df["s1_name"]
        empty_s1 = int(name.isna().sum() + (name.astype(str).str.strip() == "").sum())
        info(
            f"empty_addresses[{split}]",
            f"s1={empty_s1}, candidate={empty_addr}",
        )

        mism = df[["s1_id", "s1_name"]].drop_duplicates().merge(
            s1_tabs[split][["entity_id", "business_name"]],
            left_on="s1_id", right_on="entity_id", how="left",
        )
        mismatched = int((mism["s1_name"] != mism["business_name"]).sum())
        check(f"pair_names_match_s1_tables[{split}]", mismatched == 0, f"mismatches={mismatched}")

    bad_prefix = sum(
        int((~df["candidate_id"].str.match(r"^S[23]-")).sum()) for df in pairs.values()
    )
    check("candidate_id_format", bad_prefix == 0, f"bad={bad_prefix}")

    # ---- 8. consistency with subset_statistics.json ----------------------
    expected = {
        "train_s1": stats["train_s1"], "val_s1": stats["val_s1"], "test_s1": stats["test_s1"],
        "train_pairs": stats["train_pairs"], "val_pairs": stats["val_pairs"],
        "test_pairs": stats["test_pairs"],
        "total_positive_pairs": stats["total_positive_pairs"],
        "total_negative_pairs": stats["total_negative_pairs"],
        "hard_negatives": stats["hard_negatives"],
        "easy_negatives": stats["easy_negatives"],
        "cross_script_positive_pairs": stats["cross_script_positive_pairs"],
    }
    observed = {
        "train_s1": len(s1_tabs["train"]), "val_s1": len(s1_tabs["val"]),
        "test_s1": len(s1_tabs["test"]),
        "train_pairs": len(pairs["train"]), "val_pairs": len(pairs["val"]),
        "test_pairs": len(pairs["test"]),
        "total_positive_pairs": sum(v["positive_pairs"] for v in per_split_stats.values()),
        "total_negative_pairs": sum(v["negative_pairs"] for v in per_split_stats.values()),
        "hard_negatives": sum(v["hard_negative_pairs"] for v in per_split_stats.values()),
        "easy_negatives": sum(v["easy_negative_pairs"] for v in per_split_stats.values()),
        "cross_script_positive_pairs": total_cross,
    }
    diffs = {k: (expected[k], observed[k]) for k in expected if expected[k] != observed[k]}
    check("statistics_json_consistent", not diffs,
          "" if not diffs else f"(expected, observed): {diffs}")

    # ---- final report -----------------------------------------------------
    REPORT["per_split_stats"] = per_split_stats
    REPORT["overall"] = {
        "all_passed": not FAILURES,
        "n_failed": len(FAILURES),
        "failed_checks": FAILURES,
    }
    REPORT_PATH.write_text(json.dumps(REPORT, indent=2, ensure_ascii=False), encoding="utf-8")
    print("=" * 70)
    if FAILURES:
        print("FAILED:", ", ".join(FAILURES))
        sys.exit(1)
    print("ALL CHECKS PASSED ->", REPORT_PATH.name)


if __name__ == "__main__":
    main()
