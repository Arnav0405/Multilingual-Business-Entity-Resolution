"""
Entity-resolution subset builder (v2).

Builds a stratified subset of S1 entities with train/val/test splits,
positive pairs, hard negatives and easy negatives.

Design notes vs v1:

- All heavy relational work is executed by DuckDB. pandas is only used
  for the stratified sampling steps and final assembly.

- Every negative-generation query uses PURE EQUI-JOINS over precomputed
  key tables (gt_pairs / s2_col / s3_col / cand_keys). No correlated
  NOT-EXISTS subqueries: DuckDB 1.5.x executes those as re-scan + filter
  joins that can take hours on this dataset; the equi-join form is
  fast and streams.

- Unicode Indic detection uses a real character class (literal Indic
  characters in a regex range), not backslash-u escapes: DuckDB 1.5.6
  rejects backslash-u escape sequences inside SQL string literals
  ("invalid escape sequence").

- Work is staged in a persistent DuckDB file (with CHECKPOINTs between
  heavy steps) so an interrupted run loses at most the current step.

- Threads/memory are set for a 15 GB laptop; DuckDB spills intermediate
  state to disk in data/entity_resolution_subset/duckdb_tmp instead of
  exhausting RAM.
"""

from pathlib import Path
import time
import duckdb
import pandas as pd
import json
import random


# ============================================================
# CONFIG
# ============================================================

THIS_FILE = Path(__file__).resolve()
# src/ -> business_entity_resolution/ -> code/ -> <repo root>
REPO_ROOT = THIS_FILE.parents[3]
DATA = REPO_ROOT / "data"


S1_PATH = DATA / "train_source1.tsv"
S2_PATH = DATA / "train_source2.tsv"
S3_PATH = DATA / "train_source3.tsv"
GT_PATH = DATA / "train_ground_truth.tsv"

OUT = DATA / "entity_resolution_subset"
OUT.mkdir(parents=True, exist_ok=True)

EXAMPLES = OUT / "examples"
EXAMPLES.mkdir(exist_ok=True)

SEED = 42

VERSION_TAG = "v2-equijoin"

# Persistent staging database (rebuilt by this script; safe to delete
# afterwards — every table is reproducible from the source TSVs).
DB_PATH = OUT / "subset_staging.duckdb"
TEMP_DIR = OUT / "duckdb_tmp"
TEMP_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# Target number of S1 entities
# ------------------------------------------------------------

TOTAL_S1 = 10_000

# Approximate composition
TARGET_US = 3_500
TARGET_INDIA = 6_500

# Within India, explicitly reserve a substantial amount
# for cross-script entities.
TARGET_INDIA_CROSS_SCRIPT = 2_000

# Zero-match entities are important because the model must
# learn that some S1 entities have NO corresponding record.
TARGET_ZERO_MATCH = 1_500

# Split is performed at S1 entity level.
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15

# Negative generation
HARD_NEG_PER_S1 = 8
EASY_NEG_PER_S1 = 2
MAX_NAME_LENGTH_DIFF = 8

# Machine tuning: 8C/16T Ryzen 8845HS, 15.3 GB RAM.
DUCKDB_THREADS = 16
DUCKDB_MEMORY_LIMIT = "8GB"


# ============================================================
# DUCKDB
# ============================================================

print("=" * 80)
print("INITIALISING DUCKDB")
print("=" * 80)

TOTAL_START = time.time()

con = duckdb.connect(str(DB_PATH))
con.execute(f"PRAGMA threads={DUCKDB_THREADS}")
con.execute(f"PRAGMA memory_limit='{DUCKDB_MEMORY_LIMIT}'")
con.execute(f"PRAGMA temp_directory='{TEMP_DIR.as_posix()}'")

for name, path in [
    ("s1", S1_PATH),
    ("s2", S2_PATH),
    ("s3", S3_PATH),
    ("gt", GT_PATH),
]:
    con.execute(f"""
        CREATE OR REPLACE VIEW {name} AS
        SELECT *
        FROM read_csv(
            '{path.as_posix()}',
            delim='\\t',
            header=true,
            quote='"',
            escape='"',
            nullstr=''
        )
    """)


def step_start(label):
    print("\n[STEP] " + label, flush=True)
    return time.time()


def step_done(label, t0):
    dt = time.time() - t0
    print(f"[OK]   {label}  ({dt:,.0f}s)", flush=True)


def table_count(name):
    try:
        return con.execute(
            f"SELECT COUNT(*) FROM {name}"
        ).fetchone()[0]
    except Exception as e:
        print(f"  (count of {name} failed: {e})")
        return -1


# ============================================================
# 1. STAGING A: GT PAIRS
# ============================================================
#
# One row per (s1_id, candidate_id) ground-truth pair, materialised
# once and reused by every later step (profiling, positives,
# negatives).

t0 = step_start("staging.gt_pairs")

con.execute("""
    CREATE OR REPLACE TABLE gt_pairs AS
    SELECT
        gt.source1_entity_id AS s1_id,
        TRIM(t.matched_id) AS candidate_id
    FROM gt,
    UNNEST(
        string_split(COALESCE(gt.matched_entity_ids, ''), ',')
    ) AS t(matched_id)
    WHERE TRIM(t.matched_id) <> ''
""")

con.execute(
    "CREATE INDEX IF NOT EXISTS idx_gt_pairs_s1 ON gt_pairs(s1_id)"
)
con.execute("CHECKPOINT")

print("gt_pairs rows:", table_count("gt_pairs"))
step_done("staging.gt_pairs", t0)


# ============================================================
# 2. BUILD S1 ENTITY PROFILE
# ============================================================

print("\n" + "=" * 80)
print("BUILDING S1 ENTITY PROFILE")
print("=" * 80)

# We identify cross-script matches using Indic Unicode ranges.
#
# The observed data is particularly interesting because S1 names
# are Latin while many matching Indian S2/S3 names are in
# Devanagari, Telugu, Tamil, Kannada, Bengali, Gujarati, etc.
#
# We therefore classify an S1 entity as cross-script if ANY of
# its known positive S2/S3 names contains an Indic character.

profile_query = """
WITH all_matches AS (

    SELECT
        p.s1_id,
        s2.business_name AS matched_name

    FROM gt_pairs p
    JOIN s2 ON p.candidate_id = s2.entity_id

    WHERE p.candidate_id LIKE 'S2-%'

    UNION ALL

    SELECT
        p.s1_id,
        s3.business_name AS matched_name

    FROM gt_pairs p
    JOIN s3 ON p.candidate_id = s3.entity_id

    WHERE p.candidate_id LIKE 'S3-%'
),

match_features AS (

    SELECT
        s1_id,

        COUNT(*) AS match_count,

        SUM(
            CASE
                WHEN regexp_matches(
                    COALESCE(matched_name, ''),
                    '[ऀ-ॿ]'
                )
                THEN 1
                ELSE 0
            END
        ) AS indic_match_count

    FROM all_matches

    GROUP BY s1_id
)

SELECT

    s1.entity_id,
    s1.business_name,
    s1.business_address,
    s1.country,

    COALESCE(m.match_count, 0) AS match_count,

    COALESCE(m.indic_match_count, 0) AS indic_match_count,

    CASE
        WHEN COALESCE(m.match_count, 0) = 0
        THEN 'zero'

        WHEN COALESCE(m.indic_match_count, 0) > 0
        THEN 'cross_script'

        WHEN COALESCE(m.match_count, 0) = 1
        THEN 'one_match'

        WHEN COALESCE(m.match_count, 0) BETWEEN 2 AND 3
        THEN 'two_three_matches'

        WHEN COALESCE(m.match_count, 0) BETWEEN 4 AND 6
        THEN 'four_six_matches'

        ELSE 'seven_plus_matches'
    END AS entity_type

FROM s1

LEFT JOIN match_features m
    ON s1.entity_id = m.s1_id
"""


print("Executing entity profiling query...", flush=True)

t0 = step_start("profile")

profile = con.execute(profile_query).df()

print(f"\nTotal S1 entities: {len(profile):,}")

print("\nEntity types:")
print(
    profile["entity_type"]
    .value_counts()
    .to_string()
)

print("\nCountries:")
print(
    profile["country"]
    .value_counts()
    .to_string()
)

step_done("profile", t0)


# ============================================================
# 3. CHECK CROSS-SCRIPT DISTRIBUTION
# ============================================================

print("\n" + "=" * 80)
print("CROSS-SCRIPT DISTRIBUTION")
print("=" * 80)

cross_by_country = pd.crosstab(
    profile["country"],
    profile["entity_type"]
)

print(cross_by_country.to_string())


# ============================================================
# 4. SELECT ENTITIES
# ============================================================

print("\n" + "=" * 80)
print("SELECTING SUBSET")
print("=" * 80)


def sample_group(df, n):
    """
    Deterministic random sampling.
    """
    n = min(n, len(df))

    if n == 0:
        return df.iloc[0:0].copy()

    return df.sample(
        n=n,
        random_state=SEED
    )


selected_parts = []


# ------------------------------------------------------------
# INDIA — CROSS SCRIPT
# ------------------------------------------------------------

india_cross = profile[
    (profile["country"] == "India") &
    (profile["entity_type"] == "cross_script")
]

india_cross_selected = sample_group(
    india_cross,
    TARGET_INDIA_CROSS_SCRIPT
)

selected_parts.append(india_cross_selected)

print(
    f"India cross-script: "
    f"{len(india_cross_selected):,}"
)


# ------------------------------------------------------------
# INDIA — ZERO MATCH
# ------------------------------------------------------------

india_zero = profile[
    (profile["country"] == "India") &
    (profile["entity_type"] == "zero")
]

india_zero_selected = sample_group(
    india_zero,
    min(TARGET_ZERO_MATCH, TARGET_INDIA // 4)
)

selected_parts.append(india_zero_selected)

print(
    f"India zero-match: "
    f"{len(india_zero_selected):,}"
)


# ------------------------------------------------------------
# INDIA — OTHER ENTITIES
# ------------------------------------------------------------

already_selected = set(
    pd.concat(selected_parts)["entity_id"]
)

india_remaining = profile[
    (profile["country"] == "India") &
    (~profile["entity_id"].isin(already_selected))
]

remaining_india = TARGET_INDIA - sum(
    len(x) for x in selected_parts
)

india_other = sample_group(
    india_remaining,
    remaining_india
)

selected_parts.append(india_other)

print(
    f"India other: "
    f"{len(india_other):,}"
)


# ------------------------------------------------------------
# US
# ------------------------------------------------------------

already_selected = set(
    pd.concat(selected_parts)["entity_id"]
)

us = profile[
    (profile["country"] == "US") &
    (~profile["entity_id"].isin(already_selected))
]

us_selected = sample_group(
    us,
    TARGET_US
)

selected_parts.append(us_selected)

print(
    f"US: "
    f"{len(us_selected):,}"
)


# ------------------------------------------------------------
# COMBINE
# ------------------------------------------------------------

selected = pd.concat(
    selected_parts,
    ignore_index=True
)

# If we selected slightly fewer than requested because some
# stratum was smaller than expected, fill from remaining S1.

if len(selected) < TOTAL_S1:

    remaining_ids = set(profile["entity_id"]) - set(
        selected["entity_id"]
    )

    remaining = profile[
        profile["entity_id"].isin(remaining_ids)
    ]

    additional = sample_group(
        remaining,
        TOTAL_S1 - len(selected)
    )

    selected = pd.concat(
        [selected, additional],
        ignore_index=True
    )


# Ensure exactly TOTAL_S1 if possible.

selected = selected.drop_duplicates(
    subset=["entity_id"]
)

if len(selected) > TOTAL_S1:
    selected = sample_group(
        selected,
        TOTAL_S1
    )


print(
    f"\nFinal selected S1 entities: "
    f"{len(selected):,}"
)

print("\nFinal country distribution:")
print(
    selected["country"]
    .value_counts()
    .to_string()
)

print("\nFinal entity-type distribution:")
print(
    selected["entity_type"]
    .value_counts()
    .to_string()
)


# ============================================================
# 5. SAVE SELECTED S1
# ============================================================

selected.to_csv(
    OUT / "selected_s1.tsv",
    sep="\t",
    index=False
)


# Register selected S1 IDs in DuckDB (used by all pair queries).

con.register(
    "selected_s1_df",
    selected[["entity_id"]]
)


# ============================================================
# 6. ENTITY-LEVEL TRAIN / VAL / TEST SPLIT
# ============================================================

print("\n" + "=" * 80)
print("CREATING ENTITY-LEVEL SPLIT")
print("=" * 80)

# Stratify by country + entity type.
#
# This prevents, for example, all cross-script examples from
# accidentally ending up in the training set.

selected["split_group"] = (
    selected["country"].astype(str)
    + "__"
    + selected["entity_type"].astype(str)
)

train_parts = []
val_parts = []
test_parts = []

for group, df in selected.groupby("split_group"):

    df = df.sample(
        frac=1,
        random_state=SEED
    )

    n = len(df)

    n_train = int(n * TRAIN_FRAC)
    n_val = int(n * VAL_FRAC)

    train_parts.append(
        df.iloc[:n_train]
    )

    val_parts.append(
        df.iloc[n_train:n_train + n_val]
    )

    test_parts.append(
        df.iloc[n_train + n_val:]
    )


train_s1 = pd.concat(
    train_parts,
    ignore_index=True
)

val_s1 = pd.concat(
    val_parts,
    ignore_index=True
)

test_s1 = pd.concat(
    test_parts,
    ignore_index=True
)


# Remove helper column before saving.

for df in [train_s1, val_s1, test_s1]:

    if "split_group" in df.columns:
        df.drop(
            columns=["split_group"],
            inplace=True
        )


print(
    f"Train S1: {len(train_s1):,}"
)

print(
    f"Validation S1: {len(val_s1):,}"
)

print(
    f"Test S1: {len(test_s1):,}"
)


# Save.

train_s1.to_csv(
    OUT / "train_s1.tsv",
    sep="\t",
    index=False
)

val_s1.to_csv(
    OUT / "val_s1.tsv",
    sep="\t",
    index=False
)

test_s1.to_csv(
    OUT / "test_s1.tsv",
    sep="\t",
    index=False
)


# ============================================================
# 7. STAGING B: PAIR/KEY TABLES FOR PAIR AND NEGATIVE GENERATION
# ============================================================
#
# sel_m     : gt pairs whose S1 is a selected entity
# s2_col    : every S2 record + was_positive flag (joinable copy)
# s3_col    : every S3 record + was_positive flag (joinable copy)
# cand_keys : (candidate_id, name_len, country) for every candidate
#
# Everything later is a pure equi-join over these tables. The
# was_positive flag replaces the old runtime "is this candidate a
# known positive for some selected S1" NOT-EXISTS test.

t0 = step_start("staging.sel_m / col tables / cand_keys")

con.execute("""
    CREATE OR REPLACE TABLE sel_m AS
    SELECT g.s1_id, g.candidate_id
    FROM gt_pairs g
    SEMI JOIN selected_s1_df s ON g.s1_id = s.entity_id
""")

con.execute("""
    CREATE OR REPLACE TABLE s2_col AS
    SELECT
        s2.entity_id AS candidate_id,
        s2.business_name AS candidate_name,
        s2.business_address AS candidate_address,
        s2.country AS candidate_country,
        (m.candidate_id IS NOT NULL) AS was_positive
    FROM s2
    LEFT JOIN (
        SELECT DISTINCT candidate_id FROM sel_m
    ) m
        ON m.candidate_id = s2.entity_id
""")

con.execute("""
    CREATE OR REPLACE TABLE s3_col AS
    SELECT
        s3.entity_id AS candidate_id,
        s3.business_name AS candidate_name,
        s3.business_address AS candidate_address,
        s3.country AS candidate_country,
        (m.candidate_id IS NOT NULL) AS row_was_positive
    FROM s3
    LEFT JOIN (
        SELECT DISTINCT candidate_id FROM sel_m
    ) m
        ON m.candidate_id = s3.entity_id
""")

con.execute("""
    CREATE OR REPLACE TABLE cand_keys AS
    SELECT
        candidate_id,
        candidate_country,
        was_positive AS row_was_positive,
        LENGTH(COALESCE(candidate_name, '')) AS name_len
    FROM s2_col
    UNION ALL
    SELECT
        candidate_id,
        candidate_country,
        row_was_positive,
        LENGTH(COALESCE(candidate_name, ''))
    FROM s3_col
    WHERE candidate_id IS NOT NULL AND candidate_id <> ''
""")

con.execute("CHECKPOINT")

print("sel_m rows:    ", table_count("sel_m"))
print("s2_col rows:   ", table_count("s2_col"))
print("s3_col rows:   ", table_count("s3_col"))
print("cand_keys rows:", table_count("cand_keys"))
step_done("staging.sel_m / col tables / cand_keys", t0)


# Normalise the flag name (the s3 branch created it as
# row_was_positive); keep one canonical column everywhere.
con.execute("ALTER TABLE s3_col RENAME COLUMN row_was_positive TO was_positive")
con.execute("ALTER TABLE cand_keys RENAME COLUMN row_was_positive TO was_positive")


# ============================================================
# 8. BUILD POSITIVE PAIRS
# ============================================================

print("\n" + "=" * 80)
print("BUILDING POSITIVE PAIRS")
print("=" * 80)

t0 = step_start("positive_pairs")

positive_query = """
SELECT

    m.s1_id,
    s1.business_name AS s1_name,
    s1.business_address AS s1_address,
    s1.country AS s1_country,

    m.candidate_id,

    COALESCE(c2.candidate_name, c3.candidate_name) AS candidate_name,
    COALESCE(c2.candidate_address, c3.candidate_address)
        AS candidate_address,
    COALESCE(c2.candidate_country, c3.candidate_country)
        AS candidate_country,

    1 AS label,
    'positive' AS pair_type

FROM sel_m m

JOIN s1
    ON m.s1_id = s1.entity_id

LEFT JOIN s2_col c2
    ON m.candidate_id = c2.candidate_id
    AND m.candidate_id LIKE 'S2-%'

LEFT JOIN s3_col c3
    ON m.candidate_id = c3.candidate_id
    AND m.candidate_id LIKE 'S3-%'
"""

positive_pairs = con.execute(
    positive_query
).df()

print(
    f"Positive pairs: "
    f"{len(positive_pairs):,}"
)

step_done("positive_pairs", t0)


# ============================================================
# 9. BUILD HARD NEGATIVES
# ============================================================

print("\n" + "=" * 80)
print("BUILDING HARD NEGATIVES")
print("=" * 80)

# Strategy:
#
# For each selected S1, take candidate records that are:
#
#   - from the same country
#   - known positive records for another selected S1
#   - reasonably similar in raw name length
#
# We deliberately DON'T normalize names here.
#
# This is dataset construction, not the model itself.
#
# The eventual Transformer sees the raw business names.

# ------------------------------------------------------------
# Stage the universe WITHOUT name fields, then attach them after
# the marking join. Length filtering runs on stored integers, and
# the true-positive exclusion is a single ANTI JOIN (equi-join).
# ------------------------------------------------------------

t0 = step_start("hard_negatives (universe)")

con.execute(f"""
    CREATE OR REPLACE TABLE hard_neg_universe AS
    SELECT
        s1.entity_id AS s1_id,
        s1.business_name AS s1_name,
        LENGTH(COALESCE(s1.business_name, '')) AS s1_name_len,
        c.candidate_id,
        c.candidate_country
    FROM selected_s1_df selected

    JOIN s1
        ON selected.entity_id = s1.entity_id

    JOIN cand_keys c
        ON c.candidate_country = s1.country

    WHERE
        c.was_positive = TRUE

      AND ABS(
            c.name_len
            - LENGTH(COALESCE(s1.business_name, ''))
          ) <= {MAX_NAME_LENGTH_DIFF}

      -- Drop (s1, candidate) pairs that are ground-truth positives
      -- for that S1 (a true match must never be labelled 0 for
      -- that S1). Expressed via a NOT IN over gt_pairs instead of
      -- an ANTI JOIN after a WHERE clause (DuckDB only accepts
      -- ANTI JOIN directly after the FROM chain).
      AND (s1.entity_id, c.candidate_id) NOT IN (
            SELECT g.s1_id, g.candidate_id FROM gt_pairs g
          )
""")

print(
    "hard_neg_universe rows:",
    table_count("hard_neg_universe"),
)
step_done("hard_negatives (universe)", t0)


# ------------------------------------------------------------
# Attach candidate names/addresses/countries and rank per S1.
# Sorting by hash gives a deterministic "random" pick per S1.
# ------------------------------------------------------------

t0 = step_start("hard_negatives (rank + names)")

hard_negatives = con.execute(f"""
    WITH ranked AS (
        SELECT
            u.s1_id,
            s1.business_name AS s1_name,
            s1.business_address AS s1_address,
            s1.country AS s1_country,
            u.candidate_id,
            COALESCE(c2.candidate_name, c3.candidate_name)
                AS candidate_name,
            COALESCE(c2.candidate_address, c3.candidate_address)
                AS candidate_address,
            COALESCE(c2.candidate_country, c3.candidate_country)
                AS candidate_country,
            ROW_NUMBER() OVER (
                PARTITION BY u.s1_id
                ORDER BY hash(
                    u.s1_id || '|' || u.candidate_id
                )
            ) AS rn
        FROM hard_neg_universe u

        JOIN s1
            ON u.s1_id = s1.entity_id

        LEFT JOIN s2_col c2
            ON u.candidate_id = c2.candidate_id
            AND u.candidate_id LIKE 'S2-%'

        LEFT JOIN s3_col c3
            ON u.candidate_id = c3.candidate_id
            AND u.candidate_id LIKE 'S3-%'

        WHERE COALESCE(c2.candidate_name, c3.candidate_name) IS NOT NULL
          AND TRIM(COALESCE(c2.candidate_name, c3.candidate_name)) <> ''
    )
    SELECT
        s1_id,
        s1_name,
        s1_address,
        s1_country,
        candidate_id,
        candidate_name,
        candidate_address,
        candidate_country,
        0 AS label,
        'hard_negative' AS pair_type
    FROM ranked
    WHERE rn <= {HARD_NEG_PER_S1}
""").df()

print(
    f"Hard negatives generated: "
    f"{len(hard_negatives):,}"
)

step_done("hard_negatives (rank + names)", t0)


# ============================================================
# 10. ADD EASY NEGATIVES
# ============================================================

print("\nGenerating easy negatives...")

t0 = step_start("easy_negatives")

easy_negative_query = f"""
WITH ranked AS (
    SELECT
        s1.entity_id AS s1_id,
        s1.business_name AS s1_name,
        s1.business_address AS s1_address,
        s1.country AS s1_country,
        c.candidate_id,
        c.candidate_name,
        c.candidate_address,
        c.candidate_country,
        ROW_NUMBER() OVER (
            PARTITION BY s1.entity_id
            ORDER BY hash(
                'easy|' || s1.entity_id || '|' || c.candidate_id
            )
        ) AS rn
    FROM selected_s1_df selected

    JOIN s1
        ON selected.entity_id = s1.entity_id

    JOIN s2_col c
        ON c.candidate_country = s1.country

    ANTI JOIN gt_pairs g
        ON g.s1_id = s1.entity_id
        AND g.candidate_id = c.candidate_id

    WHERE c.candidate_name IS NOT NULL
      AND TRIM(c.candidate_name) <> ''

    UNION ALL

    SELECT
        s1.entity_id,
        s1.business_name,
        s1.business_address,
        s1.country,
        c.candidate_id,
        c.candidate_name,
        c.candidate_address,
        c.candidate_country,
        ROW_NUMBER() OVER (
            PARTITION BY s1.entity_id
            ORDER BY hash(
                'easy|' || s1.entity_id || '|' || c.candidate_id
            )
        ) AS rn
    FROM selected_s1_df selected

    JOIN s1
        ON selected.entity_id = s1.entity_id

    JOIN s3_col c
        ON c.candidate_country = s1.country

    ANTI JOIN gt_pairs g
        ON g.s1_id = s1.entity_id
        AND g.candidate_id = c.candidate_id

    WHERE c.candidate_name IS NOT NULL
      AND TRIM(c.candidate_name) <> ''
)
SELECT
    s1_id,
    s1_name,
    s1_address,
    s1_country,
    candidate_id,
    candidate_name,
    candidate_address,
    candidate_country,
    0 AS label,
    'easy_negative' AS pair_type
FROM ranked
WHERE rn <= {EASY_NEG_PER_S1}
"""

easy_negatives = con.execute(
    easy_negative_query
).df()

print(
    f"Easy negatives generated: "
    f"{len(easy_negatives):,}"
)

step_done("easy_negatives", t0)


con.execute("CHECKPOINT")


# ============================================================
# 11. COMBINE PAIRS
# ============================================================

pairs = pd.concat(
    [
        positive_pairs,
        hard_negatives,
        easy_negatives
    ],
    ignore_index=True
)

pairs = pairs.drop_duplicates(
    subset=["s1_id", "candidate_id"]
)

print("\n" + "=" * 80)
print("FINAL PAIR DATASET")
print("=" * 80)

print(
    f"Total pairs: "
    f"{len(pairs):,}"
)

print("\nLabels:")
print(
    pairs["label"]
    .value_counts()
    .to_string()
)

print("\nPair types:")
print(
    pairs["pair_type"]
    .value_counts()
    .to_string()
)


# ============================================================
# 12. ASSIGN SPLIT BASED ON S1
# ============================================================

train_ids = set(
    train_s1["entity_id"]
)

val_ids = set(
    val_s1["entity_id"]
)

test_ids = set(
    test_s1["entity_id"]
)

assert train_ids.isdisjoint(val_ids)
assert train_ids.isdisjoint(test_ids)
assert val_ids.isdisjoint(test_ids)


def assign_split(s1_id):

    if s1_id in train_ids:
        return "train"

    if s1_id in val_ids:
        return "val"

    if s1_id in test_ids:
        return "test"

    raise ValueError(
        f"Unknown S1 ID: {s1_id}"
    )


pairs["split"] = pairs["s1_id"].apply(
    assign_split
)


# ============================================================
# 13. SAVE PAIRS
# ============================================================

train_pairs = pairs[
    pairs["split"] == "train"
].drop(columns=["split"])

val_pairs = pairs[
    pairs["split"] == "val"
].drop(columns=["split"])

test_pairs = pairs[
    pairs["split"] == "test"
].drop(columns=["split"])


train_pairs.to_csv(
    OUT / "train_pairs.tsv",
    sep="\t",
    index=False
)

val_pairs.to_csv(
    OUT / "val_pairs.tsv",
    sep="\t",
    index=False
)

test_pairs.to_csv(
    OUT / "test_pairs.tsv",
    sep="\t",
    index=False
)


# ============================================================
# 14. CROSS-SCRIPT EXAMPLES
# ============================================================

print("\n" + "=" * 80)
print("SAVING CROSS-SCRIPT EXAMPLES")
print("=" * 80)

# Use Python Unicode detection here because this is only being
# applied to the relatively small positive-pair dataframe.

def has_indic(text):

    if not isinstance(text, str):
        return False

    return any(
        0x0900 <= ord(ch) <= 0x0DFF
        for ch in text
    )


cross_script = positive_pairs[
    positive_pairs["candidate_name"].apply(
        has_indic
    )
].copy()

print(
    f"Cross-script positive pairs: "
    f"{len(cross_script):,}"
)

cross_script[
    [
        "s1_id",
        "s1_name",
        "candidate_id",
        "candidate_name",
        "s1_country"
    ]
].head(500).to_csv(
    EXAMPLES / "cross_script_pairs.tsv",
    sep="\t",
    index=False
)


# ============================================================
# 15. FINAL STATISTICS
# ============================================================

statistics = {

    "version": VERSION_TAG,
    "seed": SEED,

    "total_s1": len(selected),

    "train_s1": len(train_s1),
    "val_s1": len(val_s1),
    "test_s1": len(test_s1),

    "train_pairs": len(train_pairs),
    "val_pairs": len(val_pairs),
    "test_pairs": len(test_pairs),

    "total_positive_pairs": int(
        (pairs["label"] == 1).sum()
    ),

    "total_negative_pairs": int(
        (pairs["label"] == 0).sum()
    ),

    "hard_negatives": int(
        (pairs["pair_type"] == "hard_negative").sum()
    ),

    "easy_negatives": int(
        (pairs["pair_type"] == "easy_negative").sum()
    ),

    "cross_script_positive_pairs": int(
        len(cross_script)
    ),

    "country_distribution": (
        selected["country"]
        .value_counts()
        .to_dict()
    ),

    "entity_type_distribution": (
        selected["entity_type"]
        .value_counts()
        .to_dict()
    ),

    "train_country_distribution": (
        train_s1["country"]
        .value_counts()
        .to_dict()
    ),

    "val_country_distribution": (
        val_s1["country"]
        .value_counts()
        .to_dict()
    ),

    "test_country_distribution": (
        test_s1["country"]
        .value_counts()
        .to_dict()
    )
}


with open(
    OUT / "subset_statistics.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        statistics,
        f,
        indent=2,
        ensure_ascii=False
    )


# ============================================================
# 16. SANITY CHECKS
# ============================================================

print("\n" + "=" * 80)
print("SANITY CHECKS")
print("=" * 80)

# No S1 leakage.

assert len(
    train_ids & val_ids
) == 0

assert len(
    train_ids & test_ids
) == 0

assert len(
    val_ids & test_ids
) == 0

# Every pair belongs to exactly one split.

assert (
    len(train_pairs)
    + len(val_pairs)
    + len(test_pairs)
    == len(pairs)
)

# Every positive pair belongs to a selected S1.

assert set(
    positive_pairs["s1_id"]
).issubset(
    set(selected["entity_id"])
)

# No (s1, candidate) pair may be both labelled 1 and 0.

assert (
    pairs.groupby(["s1_id", "candidate_id"])["label"]
    .nunique()
    .max()
    == 1
)

print("✓ No S1 entity leakage")
print("✓ All positive pairs belong to selected entities")
print("✓ Every pair assigned to exactly one split")
print("✓ No conflicting pair labels")


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("DONE")
print("=" * 80)

total_min = (time.time() - TOTAL_START) / 60

print(f"""
Output:

{OUT}

S1:
    train_s1.tsv
    val_s1.tsv
    test_s1.tsv

Training data:
    train_pairs.tsv
    val_pairs.tsv
    test_pairs.tsv

Analysis:
    subset_statistics.json
    examples/cross_script_pairs.tsv

Final counts:

    S1 entities : {len(selected):,}
    Train       : {len(train_s1):,}
    Validation  : {len(val_s1):,}
    Test        : {len(test_s1):,}

    Total pairs : {len(pairs):,}
    Positives   : {(pairs["label"] == 1).sum():,}
    Negatives   : {(pairs["label"] == 0).sum():,}

    Cross-script positives:
        {len(cross_script):,}

Runtime: {total_min:.1f} min
""")
