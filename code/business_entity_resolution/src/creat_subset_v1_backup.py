from pathlib import Path
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


# ============================================================
# DUCKDB
# ============================================================

print("=" * 80)
print("INITIALISING DUCKDB")
print("=" * 80)

con = duckdb.connect()
con.execute("PRAGMA threads=4")

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


# ============================================================
# 1. BUILD S1 ENTITY PROFILE
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
WITH gt_pairs AS (

    SELECT
        source1_entity_id AS s1_id,
        TRIM(matched_id) AS matched_id

    FROM gt,
    UNNEST(
        string_split(
            COALESCE(matched_entity_ids, ''),
            ','
        )
    ) AS t(matched_id)

    WHERE TRIM(matched_id) <> ''
),

all_matches AS (

    SELECT
        p.s1_id,
        p.matched_id,

        CASE
            WHEN p.matched_id LIKE 'S2-%'
            THEN s2.business_name
            ELSE s3.business_name
        END AS matched_name

    FROM gt_pairs p

    LEFT JOIN s2
        ON p.matched_id = s2.entity_id

    LEFT JOIN s3
        ON p.matched_id = s3.entity_id
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


print("Executing entity profiling query...")

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


# ============================================================
# 2. CHECK CROSS-SCRIPT DISTRIBUTION
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
# 3. SELECT ENTITIES
# ============================================================

print("\n" + "=" * 80)
print("SELECTING SUBSET")
print("=" * 80)

rng = random.Random(SEED)


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
# 4. SAVE SELECTED S1
# ============================================================

selected.to_csv(
    OUT / "selected_s1.tsv",
    sep="\t",
    index=False
)


# ============================================================
# 5. ENTITY-LEVEL TRAIN / VAL / TEST SPLIT
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
# 6. CREATE GROUND-TRUTH POSITIVE PAIRS
# ============================================================

print("\n" + "=" * 80)
print("BUILDING POSITIVE PAIRS")
print("=" * 80)

# Register selected S1 IDs in DuckDB.

con.register(
    "selected_s1_df",
    selected[["entity_id"]]
)

positive_query = """
WITH pairs AS (

    SELECT
        gt.source1_entity_id AS s1_id,
        TRIM(matched_id) AS candidate_id

    FROM gt,

    UNNEST(
        string_split(
            COALESCE(gt.matched_entity_ids, ''),
            ','
        )
    ) AS t(matched_id)

    WHERE TRIM(matched_id) <> ''

)

SELECT

    p.s1_id,
    s1.business_name AS s1_name,
    s1.business_address AS s1_address,
    s1.country AS s1_country,

    p.candidate_id,

    CASE
        WHEN p.candidate_id LIKE 'S2-%'
        THEN s2.business_name
        ELSE s3.business_name
    END AS candidate_name,

    CASE
        WHEN p.candidate_id LIKE 'S2-%'
        THEN s2.business_address
        ELSE s3.business_address
    END AS candidate_address,

    CASE
        WHEN p.candidate_id LIKE 'S2-%'
        THEN s2.country
        ELSE s3.country
    END AS candidate_country,

    1 AS label,
    'positive' AS pair_type

FROM pairs p

JOIN selected_s1_df selected
    ON p.s1_id = selected.entity_id

JOIN s1
    ON p.s1_id = s1.entity_id

LEFT JOIN s2
    ON p.candidate_id = s2.entity_id

LEFT JOIN s3
    ON p.candidate_id = s3.entity_id
"""

positive_pairs = con.execute(
    positive_query
).df()

print(
    f"Positive pairs: "
    f"{len(positive_pairs):,}"
)


# ============================================================
# 7. GET ALL CANDIDATE RECORDS THAT PARTICIPATE IN POSITIVES
# ============================================================

print("\nCollecting candidate pool...")

positive_candidate_ids = set(
    positive_pairs["candidate_id"]
)

# We will create hard negatives from records that are known
# matches for OTHER S1 entities.
#
# This is much better than generating completely random
# unrelated negatives.

con.register(
    "positive_candidate_ids_df",
    pd.DataFrame({
        "candidate_id": list(
            positive_candidate_ids
        )
    })
)


# ============================================================
# 8. BUILD HARD NEGATIVES
# ============================================================

print("\n" + "=" * 80)
print("BUILDING HARD NEGATIVES")
print("=" * 80)

# Strategy:
#
# For each selected S1, take candidate records that are:
#
#   - from the same country
#   - known positive records for another S1
#   - reasonably similar in raw name length
#
# We deliberately DON'T normalize names here.
#
# This is dataset construction, not the model itself.
#
# The eventual Transformer sees the raw business names.

negative_query = """
WITH selected_positive_pairs AS (

    SELECT
        p.s1_id,
        p.candidate_id
    FROM (
        SELECT
            gt.source1_entity_id AS s1_id,
            TRIM(matched_id) AS candidate_id
        FROM gt,
        UNNEST(
            string_split(
                COALESCE(gt.matched_entity_ids, ''),
                ','
            )
        ) AS t(matched_id)
        WHERE TRIM(matched_id) <> ''
    ) p

    JOIN selected_s1_df selected
        ON p.s1_id = selected.entity_id
),

selected_positive_records AS (

    SELECT DISTINCT
        p.candidate_id,

        CASE
            WHEN p.candidate_id LIKE 'S2-%'
            THEN s2.business_name
            ELSE s3.business_name
        END AS candidate_name,

        CASE
            WHEN p.candidate_id LIKE 'S2-%'
            THEN s2.country
            ELSE s3.country
        END AS candidate_country

    FROM selected_positive_pairs p

    LEFT JOIN s2
        ON p.candidate_id = s2.entity_id

    LEFT JOIN s3
        ON p.candidate_id = s3.entity_id
),

candidate_source AS (

    SELECT
        entity_id AS candidate_id,
        business_name AS candidate_name,
        business_address AS candidate_address,
        country AS candidate_country
    FROM s2

    UNION ALL

    SELECT
        entity_id AS candidate_id,
        business_name AS candidate_name,
        business_address AS candidate_address,
        country AS candidate_country
    FROM s3
),

positive_lookup AS (

    SELECT
        s1_id,
        candidate_id
    FROM selected_positive_pairs
)

SELECT

    s1.entity_id AS s1_id,
    s1.business_name AS s1_name,
    s1.business_address AS s1_address,
    s1.country AS s1_country,

    c.candidate_id,
    c.candidate_name,
    c.candidate_address,
    c.candidate_country,

    0 AS label,
    'hard_negative' AS pair_type

FROM selected_s1_df selected

JOIN s1
    ON selected.entity_id = s1.entity_id

JOIN candidate_source c
    ON c.candidate_country = s1.country

WHERE

    -- Similar raw name length.
    ABS(
        LENGTH(COALESCE(c.candidate_name, ''))
        -
        LENGTH(COALESCE(s1.business_name, ''))
    ) <= 8

    -- Avoid generating the true positive pair.
    AND NOT EXISTS (
        SELECT 1
        FROM positive_lookup p
        WHERE
            p.s1_id = s1.entity_id
            AND p.candidate_id = c.candidate_id
    )

    -- Avoid empty names.
    AND c.candidate_name IS NOT NULL
    AND TRIM(c.candidate_name) <> ''

QUALIFY ROW_NUMBER() OVER (
    PARTITION BY s1.entity_id
    ORDER BY hash(
        s1.entity_id || '|' || c.candidate_id
    )
) <= 8
"""

print(
    "Generating up to 8 hard negatives per S1..."
)

hard_negatives = con.execute(
    negative_query
).df()

print(
    f"Hard negatives generated: "
    f"{len(hard_negatives):,}"
)


# ============================================================
# 9. ADD EASY NEGATIVES
# ============================================================

print("\nGenerating easy negatives...")

easy_negative_query = """
WITH positive_lookup AS (

    SELECT
        gt.source1_entity_id AS s1_id,
        TRIM(matched_id) AS candidate_id

    FROM gt,

    UNNEST(
        string_split(
            COALESCE(gt.matched_entity_ids, ''),
            ','
        )
    ) AS t(matched_id)

    WHERE TRIM(matched_id) <> ''

),

candidate_source AS (

    SELECT
        entity_id AS candidate_id,
        business_name AS candidate_name,
        business_address AS candidate_address,
        country AS candidate_country
    FROM s2

    UNION ALL

    SELECT
        entity_id AS candidate_id,
        business_name AS candidate_name,
        business_address AS candidate_address,
        country AS candidate_country
    FROM s3
)

SELECT

    s1.entity_id AS s1_id,
    s1.business_name AS s1_name,
    s1.business_address AS s1_address,
    s1.country AS s1_country,

    c.candidate_id,
    c.candidate_name,
    c.candidate_address,
    c.candidate_country,

    0 AS label,
    'easy_negative' AS pair_type

FROM selected_s1_df selected

JOIN s1
    ON selected.entity_id = s1.entity_id

JOIN candidate_source c
    ON c.candidate_country = s1.country

WHERE

    c.candidate_name IS NOT NULL
    AND TRIM(c.candidate_name) <> ''

    AND NOT EXISTS (
        SELECT 1
        FROM positive_lookup p
        WHERE
            p.s1_id = s1.entity_id
            AND p.candidate_id = c.candidate_id
    )

QUALIFY ROW_NUMBER() OVER (
    PARTITION BY s1.entity_id
    ORDER BY hash(
        'easy|' || s1.entity_id || '|' || c.candidate_id
    )
) <= 2
"""

easy_negatives = con.execute(
    easy_negative_query
).df()

print(
    f"Easy negatives generated: "
    f"{len(easy_negatives):,}"
)


# ============================================================
# 10. COMBINE PAIRS
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
# 11. ASSIGN SPLIT BASED ON S1
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
# 12. SAVE PAIRS
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
# 13. CROSS-SCRIPT EXAMPLES
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
# 14. FINAL STATISTICS
# ============================================================

statistics = {

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
# 15. SANITY CHECKS
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

print("✓ No S1 entity leakage")
print("✓ All positive pairs belong to selected entities")
print("✓ Every pair assigned to exactly one split")


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("DONE")
print("=" * 80)

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
""")
