from pathlib import Path
import unicodedata
import re
import json
import duckdb
import pandas as pd


# ============================================================
# CONFIG
# ============================================================
THIS_FILE = Path(__file__).resolve()
# src/ -> business_entity_resolution/ -> code/ -> <repo root>
REPO_ROOT = THIS_FILE.parents[3]
DATA = REPO_ROOT / "data"

S1 = DATA / "train_source1.tsv"
S2 = DATA / "train_source2.tsv"
S3 = DATA / "train_source3.tsv"
GT = DATA / "train_ground_truth.tsv"

OUTPUT = DATA / "profiling"
OUTPUT.mkdir(exist_ok=True)

SAMPLE_POSITIVE_PAIRS = 50_000
SAMPLE_NAMES = 50_000

# Use a fixed seed for reproducibility
SEED = 42


# ============================================================
# SCRIPT DETECTION
# ============================================================

# These ranges are intentionally broad. We want to identify
# the writing system, not the exact Unicode block.

SCRIPT_RANGES = {
    "Devanagari": [
        (0x0900, 0x097F),
    ],
    "Bengali": [
        (0x0980, 0x09FF),
    ],
    "Gurmukhi": [
        (0x0A00, 0x0A7F),
    ],
    "Gujarati": [
        (0x0A80, 0x0AFF),
    ],
    "Oriya": [
        (0x0B00, 0x0B7F),
    ],
    "Tamil": [
        (0x0B80, 0x0BFF),
    ],
    "Telugu": [
        (0x0C00, 0x0C7F),
    ],
    "Kannada": [
        (0x0C80, 0x0CFF),
    ],
    "Malayalam": [
        (0x0D00, 0x0D7F),
    ],
    "Thai": [
        (0x0E00, 0x0E7F),
    ],
    "Arabic": [
        (0x0600, 0x06FF),
        (0x0750, 0x077F),
        (0x08A0, 0x08FF),
    ],
    "Cyrillic": [
        (0x0400, 0x04FF),
    ],
    "Greek": [
        (0x0370, 0x03FF),
    ],
    "Hebrew": [
        (0x0590, 0x05FF),
    ],
    "Han": [
        (0x4E00, 0x9FFF),
    ],
    "Hiragana": [
        (0x3040, 0x309F),
    ],
    "Katakana": [
        (0x30A0, 0x30FF),
    ],
}


def char_script(ch):
    cp = ord(ch)

    for script, ranges in SCRIPT_RANGES.items():
        for lo, hi in ranges:
            if lo <= cp <= hi:
                return script

    # Basic Latin + Latin extended
    if (
        0x0041 <= cp <= 0x005A
        or 0x0061 <= cp <= 0x007A
        or 0x00C0 <= cp <= 0x024F
        or 0x1E00 <= cp <= 0x1EFF
    ):
        return "Latin"

    return None


def detect_scripts(text):
    """
    Return all scripts represented by actual alphabetic characters.
    Ignore punctuation, numbers, spaces, etc.
    """

    if not text:
        return set()

    scripts = set()

    for ch in str(text):
        script = char_script(ch)

        if script is not None:
            scripts.add(script)

    return scripts


def primary_script(text):
    scripts = detect_scripts(text)

    if not scripts:
        return "None"

    if len(scripts) == 1:
        return next(iter(scripts))

    return "Mixed"


def script_signature(text):
    """
    More detailed representation.

    Example:
        'Ram Marketing' -> 'Latin'
        'राम मार्केटिंग' -> 'Devanagari+Latin'
    """

    scripts = sorted(detect_scripts(text))

    if not scripts:
        return "None"

    return "+".join(scripts)


# ============================================================
# DUCKDB SETUP
# ============================================================

print("=" * 80)
print("INITIALISING DUCKDB")
print("=" * 80)

con = duckdb.connect()

# Give DuckDB reasonable parallelism.
con.execute("PRAGMA threads=4")

# Create views over the TSV files.
# Explicit delimiter avoids CSV auto-detection mistakes.
con.execute(f"""
    CREATE OR REPLACE VIEW s1 AS
    SELECT *
    FROM read_csv(
        '{S1.as_posix()}',
        delim='\\t',
        header=true,
        quote='"',
        escape='"',
        nullstr=''
    )
""")

con.execute(f"""
    CREATE OR REPLACE VIEW s2 AS
    SELECT *
    FROM read_csv(
        '{S2.as_posix()}',
        delim='\\t',
        header=true,
        quote='"',
        escape='"',
        nullstr=''
    )
""")

con.execute(f"""
    CREATE OR REPLACE VIEW s3 AS
    SELECT *
    FROM read_csv(
        '{S3.as_posix()}',
        delim='\\t',
        header=true,
        quote='"',
        escape='"',
        nullstr=''
    )
""")

con.execute(f"""
    CREATE OR REPLACE VIEW gt AS
    SELECT *
    FROM read_csv(
        '{GT.as_posix()}',
        delim='\\t',
        header=true,
        quote='"',
        escape='"',
        nullstr=''
    )
""")


# ============================================================
# BASIC DATASET PROFILE
# ============================================================

print("\n" + "=" * 80)
print("1. BASIC DATASET PROFILE")
print("=" * 80)

for name, table in [
    ("SOURCE1", "s1"),
    ("SOURCE2", "s2"),
    ("SOURCE3", "s3"),
]:
    print(f"\n--- {name} ---")

    result = con.execute(f"""
        SELECT
            COUNT(*) AS rows,
            COUNT(DISTINCT entity_id) AS unique_entity_ids,
            COUNT(DISTINCT business_name) AS unique_business_names,
            COUNT(DISTINCT business_address) AS unique_addresses,
            COUNT(*) FILTER (
                WHERE business_name IS NULL OR business_name = ''
            ) AS empty_names,
            COUNT(*) FILTER (
                WHERE business_address IS NULL OR business_address = ''
            ) AS empty_addresses
        FROM {table}
    """).fetchone()

    print(f"rows               : {result[0]:,}")
    print(f"unique entity_id   : {result[1]:,}")
    print(f"unique names       : {result[2]:,}")
    print(f"unique addresses   : {result[3]:,}")
    print(f"empty names        : {result[4]:,}")
    print(f"empty addresses    : {result[5]:,}")

    countries = con.execute(f"""
        SELECT country, COUNT(*) AS count
        FROM {table}
        GROUP BY country
        ORDER BY count DESC
    """).df()

    print("\nCountries:")
    print(countries.to_string(index=False))


# ============================================================
# GROUND TRUTH PROFILE
# ============================================================

print("\n" + "=" * 80)
print("2. GROUND TRUTH PROFILE")
print("=" * 80)

gt_stats = con.execute("""
    SELECT
        COUNT(*) AS s1_entities,
        COUNT(*) FILTER (
            WHERE matched_entity_ids IS NULL
               OR TRIM(matched_entity_ids) = ''
        ) AS zero_matches
    FROM gt
""").fetchone()

print(f"S1 entities           : {gt_stats[0]:,}")
print(f"S1 entities with 0    : {gt_stats[1]:,}")
print(
    f"S1 entities with >=1  : "
    f"{gt_stats[0] - gt_stats[1]:,}"
)


# ============================================================
# MATCH COUNT DISTRIBUTION
# ============================================================

print("\n" + "=" * 80)
print("3. MATCH COUNT DISTRIBUTION")
print("=" * 80)

match_dist = con.execute("""
    SELECT
        CASE
            WHEN matched_entity_ids IS NULL
              OR TRIM(matched_entity_ids) = ''
            THEN 0
            ELSE array_length(
                string_split(matched_entity_ids, ',')
            )
        END AS match_count,
        COUNT(*) AS s1_entities
    FROM gt
    GROUP BY match_count
    ORDER BY match_count
""").df()

print(match_dist.to_string(index=False))

match_dist.to_csv(
    OUTPUT / "match_count_distribution.csv",
    index=False
)


# ============================================================
# MATCH COUNT SUMMARY
# ============================================================

summary = con.execute("""
    SELECT
        MIN(match_count),
        MEDIAN(match_count),
        AVG(match_count),
        MAX(match_count)
    FROM (
        SELECT
            CASE
                WHEN matched_entity_ids IS NULL
                  OR TRIM(matched_entity_ids) = ''
                THEN 0
                ELSE array_length(
                    string_split(matched_entity_ids, ',')
                )
            END AS match_count
        FROM gt
    )
""").fetchone()

print("\nMatch statistics:")
print(f"minimum : {summary[0]}")
print(f"median  : {summary[1]:.2f}")
print(f"mean    : {summary[2]:.2f}")
print(f"maximum : {summary[3]}")


# ============================================================
# S2 VS S3 MATCH DISTRIBUTION
# ============================================================

print("\n" + "=" * 80)
print("4. S2 / S3 MATCH DISTRIBUTION")
print("=" * 80)

s2s3 = con.execute("""
    SELECT
        SUM(
            CASE
                WHEN matched_entity_ids IS NOT NULL
                 AND regexp_matches(matched_entity_ids, '(^|,)S2-')
                THEN 1 ELSE 0
            END
        ) AS entities_with_s2,

        SUM(
            CASE
                WHEN matched_entity_ids IS NOT NULL
                 AND regexp_matches(matched_entity_ids, '(^|,)S3-')
                THEN 1 ELSE 0
            END
        ) AS entities_with_s3,

        SUM(
            CASE
                WHEN matched_entity_ids IS NOT NULL
                 AND regexp_matches(matched_entity_ids, '(^|,)S2-')
                 AND regexp_matches(matched_entity_ids, '(^|,)S3-')
                THEN 1 ELSE 0
            END
        ) AS entities_with_both
    FROM gt
""").fetchone()

print(f"With S2 match       : {s2s3[0]:,}")
print(f"With S3 match       : {s2s3[1]:,}")
print(f"With both           : {s2s3[2]:,}")


# ============================================================
# SAMPLE S1 ENTITIES
# ============================================================

print("\n" + "=" * 80)
print("5. STRATIFIED S1 SAMPLE")
print("=" * 80)

# We want a useful sample, not just random rows.
#
# Approximately:
#   US     : 10k
#   India  : 20k
#
# And spread across different match-count groups.

sample_s1 = con.execute("""
    WITH base AS (
        SELECT
            s1.*,
            CASE
                WHEN gt.matched_entity_ids IS NULL
                  OR TRIM(gt.matched_entity_ids) = ''
                THEN 0
                ELSE array_length(
                    string_split(gt.matched_entity_ids, ',')
                )
            END AS match_count
        FROM s1
        JOIN gt
          ON s1.entity_id = gt.source1_entity_id
    ),

    ranked AS (
        SELECT
            *,
            ROW_NUMBER() OVER (
                PARTITION BY
                    country,
                    CASE
                        WHEN match_count = 0 THEN 'zero'
                        WHEN match_count = 1 THEN 'one'
                        WHEN match_count BETWEEN 2 AND 3 THEN 'two_three'
                        WHEN match_count BETWEEN 4 AND 6 THEN 'four_six'
                        ELSE 'seven_plus'
                    END
                ORDER BY hash(entity_id)
            ) AS rn
        FROM base
    )

    SELECT
        entity_id,
        business_name,
        business_address,
        country,
        match_count
    FROM ranked
    WHERE
        (
            country = 'US' AND rn <= 3500
        )
        OR
        (
            country = 'India' AND rn <= 6500
        )
""").df()

print(f"Sampled S1 entities: {len(sample_s1):,}")

print("\nCountry distribution:")
print(
    sample_s1["country"]
    .value_counts()
    .to_string()
)

print("\nMatch-count distribution:")
print(
    sample_s1["match_count"]
    .value_counts()
    .sort_index()
    .to_string()
)

sample_s1.to_csv(
    OUTPUT / "sample_s1.csv",
    index=False
)


# ============================================================
# SCRIPT DISTRIBUTION ON S1
# ============================================================

print("\n" + "=" * 80)
print("6. BUSINESS NAME SCRIPT DISTRIBUTION")
print("=" * 80)

sample_s1["name_script"] = sample_s1["business_name"].apply(
    primary_script
)

sample_s1["name_script_signature"] = sample_s1["business_name"].apply(
    script_signature
)

print("\nBy country:")
print(
    pd.crosstab(
        sample_s1["country"],
        sample_s1["name_script"],
        margins=True
    ).to_string()
)

print("\nOverall:")
print(
    sample_s1["name_script"]
    .value_counts()
    .to_string()
)

print("\nDetailed script combinations:")
print(
    sample_s1["name_script_signature"]
    .value_counts()
    .head(30)
    .to_string()
)

sample_s1.to_csv(
    OUTPUT / "sample_s1_with_scripts.csv",
    index=False
)


# ============================================================
# SAMPLE POSITIVE PAIRS
# ============================================================

print("\n" + "=" * 80)
print("7. SAMPLING POSITIVE MATCH PAIRS")
print("=" * 80)

# Convert the comma-separated ground truth into individual pairs.
#
# We sample AFTER creating the pairs so that the sample represents
# actual matching relationships.

positive_pairs = con.execute(f"""
    WITH pairs AS (
        SELECT
            gt.source1_entity_id AS s1_id,
            TRIM(matched_id) AS matched_id
        FROM gt,
        UNNEST(string_split(gt.matched_entity_ids, ',')) AS t(matched_id)
        WHERE
            gt.matched_entity_ids IS NOT NULL
            AND TRIM(gt.matched_entity_ids) <> ''
    ),

    sampled AS (
        SELECT *
        FROM pairs
        USING SAMPLE reservoir({SAMPLE_POSITIVE_PAIRS} ROWS)
        REPEATABLE({SEED})
    )

    SELECT *
    FROM sampled
""").df()

print(f"Positive pairs sampled: {len(positive_pairs):,}")

# Save the pair IDs.
positive_pairs.to_csv(
    OUTPUT / "sample_positive_pair_ids.csv",
    index=False
)


# ============================================================
# LOAD TEXT FOR POSITIVE PAIRS
# ============================================================

print("\nJoining positive pairs to source records...")

positive_pairs = con.register(
    "positive_pairs_df",
    positive_pairs
)

positive_data = con.execute("""
    SELECT
        p.s1_id,

        s1.business_name AS s1_name,
        s1.business_address AS s1_address,
        s1.country AS s1_country,

        p.matched_id,

        CASE
            WHEN p.matched_id LIKE 'S2-%'
            THEN s2.business_name
            ELSE s3.business_name
        END AS matched_name,

        CASE
            WHEN p.matched_id LIKE 'S2-%'
            THEN s2.business_address
            ELSE s3.business_address
        END AS matched_address,

        CASE
            WHEN p.matched_id LIKE 'S2-%'
            THEN s2.country
            ELSE s3.country
        END AS matched_country

    FROM positive_pairs_df p

    JOIN s1
      ON p.s1_id = s1.entity_id

    LEFT JOIN s2
      ON p.matched_id = s2.entity_id

    LEFT JOIN s3
      ON p.matched_id = s3.entity_id
""").df()

print(f"Positive pairs with text: {len(positive_data):,}")


# ============================================================
# POSITIVE PAIR ANALYSIS
# ============================================================

positive_data["s1_script"] = positive_data["s1_name"].apply(
    primary_script
)

positive_data["matched_script"] = positive_data["matched_name"].apply(
    primary_script
)

positive_data["s1_scripts"] = positive_data["s1_name"].apply(
    script_signature
)

positive_data["matched_scripts"] = positive_data["matched_name"].apply(
    script_signature
)


def classify_script_relation(row):

    a = detect_scripts(row["s1_name"])
    b = detect_scripts(row["matched_name"])

    if not a or not b:
        return "unknown"

    if a == b:
        if len(a) == 1:
            return "same_script"
        return "same_mixed_script"

    # Any overlap but not identical
    if a & b:
        return "partially_shared_script"

    return "cross_script"


positive_data["script_relation"] = positive_data.apply(
    classify_script_relation,
    axis=1
)

positive_data["s1_name_len"] = (
    positive_data["s1_name"]
    .fillna("")
    .str.len()
)

positive_data["matched_name_len"] = (
    positive_data["matched_name"]
    .fillna("")
    .str.len()
)

positive_data["name_length_ratio"] = (
    positive_data["matched_name_len"]
    /
    positive_data["s1_name_len"].clip(lower=1)
)


# ============================================================
# POSITIVE PAIR STATISTICS
# ============================================================

print("\n" + "=" * 80)
print("8. POSITIVE PAIR SCRIPT RELATIONSHIP")
print("=" * 80)

print(
    positive_data["script_relation"]
    .value_counts(normalize=False)
    .to_string()
)

print("\nPercentages:")
print(
    (
        positive_data["script_relation"]
        .value_counts(normalize=True)
        * 100
    )
    .round(2)
    .to_string()
)


print("\nBy country:")
print(
    pd.crosstab(
        positive_data["s1_country"],
        positive_data["script_relation"],
        normalize="index"
    ).round(3).to_string()
)


# ============================================================
# NAME LENGTH STATISTICS
# ============================================================

print("\n" + "=" * 80)
print("9. POSITIVE NAME LENGTH STATISTICS")
print("=" * 80)

print(
    positive_data[
        ["s1_name_len", "matched_name_len", "name_length_ratio"]
    ].describe().round(2).to_string()
)


# ============================================================
# CROSS-SCRIPT EXAMPLES
# ============================================================

print("\n" + "=" * 80)
print("10. CROSS-SCRIPT POSITIVE EXAMPLES")
print("=" * 80)

cross = positive_data[
    positive_data["script_relation"] == "cross_script"
].copy()

print(f"Cross-script pairs in sample: {len(cross):,}")

if len(cross) > 0:

    examples = cross[
        [
            "s1_id",
            "s1_name",
            "matched_id",
            "matched_name",
            "s1_scripts",
            "matched_scripts",
            "s1_country"
        ]
    ].head(100)

    print(
        examples.to_string(index=False)
    )

    examples.to_csv(
        OUTPUT / "cross_script_examples.csv",
        index=False
    )


# ============================================================
# SCRIPT PAIR MATRIX
# ============================================================

print("\n" + "=" * 80)
print("11. SCRIPT PAIR MATRIX")
print("=" * 80)

script_matrix = pd.crosstab(
    positive_data["s1_scripts"],
    positive_data["matched_scripts"]
)

print(
    script_matrix.head(30).to_string()
)

script_matrix.to_csv(
    OUTPUT / "positive_script_matrix.csv"
)


# ============================================================
# SUMMARY JSON
# ============================================================

summary_output = {
    "sample_s1_entities": int(len(sample_s1)),
    "sample_positive_pairs": int(len(positive_data)),
    "cross_script_positive_pairs": int(
        (positive_data["script_relation"] == "cross_script").sum()
    ),
    "same_script_positive_pairs": int(
        (positive_data["script_relation"] == "same_script").sum()
    ),
    "countries": (
        sample_s1["country"]
        .value_counts()
        .to_dict()
    ),
    "s1_scripts": (
        sample_s1["name_script"]
        .value_counts()
        .to_dict()
    ),
    "positive_script_relations": (
        positive_data["script_relation"]
        .value_counts()
        .to_dict()
    ),
}

with open(
    OUTPUT / "profiling_summary.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        summary_output,
        f,
        indent=2,
        ensure_ascii=False
    )


# ============================================================
# DONE
# ============================================================

print("\n" + "=" * 80)
print("DONE")
print("=" * 80)

print(f"""
Output directory:

    {OUTPUT}

Important files:

    match_count_distribution.csv
    sample_s1.csv
    sample_s1_with_scripts.csv
    sample_positive_pair_ids.csv
    cross_script_examples.csv
    positive_script_matrix.csv
    profiling_summary.json
""")
