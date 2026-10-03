"""
Business Entity Resolution - data loader + first-look EDA.

Problem statement (inferred from the data; README.MD is currently EMPTY):
  Three sources each list real-world businesses:
      train_source{1,2,3}.tsv -> entity_id, business_name, business_address, country
      (source1 ids prefixed "S1-", source2 "S2-", source3 "S3-")
  train_ground_truth.tsv maps every source1 entity to the entity_ids in
  sources 2/3 that refer to the SAME real-world business:
      source1_entity_id \t matched_entity_ids   (comma-separated)

  Task: entity resolution / record linkage - decide which records across
  the three sources point to the same business, despite differences in
  name spelling, transliteration (e.g. Hindi names in source2), address
  formatting, and missing fields.

Run:  python df.py            (from anywhere; paths resolve via this file)
"""

from __future__ import annotations

import bisect
import csv
from collections import Counter
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

THIS_FILE = Path(__file__).resolve()
# src/ -> business_entity_resolution/ -> code/ -> <repo root>
REPO_ROOT = THIS_FILE.parents[3]
DATA_DIR = REPO_ROOT / "data"

SOURCE_FILES = {
    "source1": "train_source1.tsv",
    "source2": "train_source2.tsv",
    "source3": "train_source3.tsv",
}
GROUND_TRUTH_FILE = "train_ground_truth.tsv"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_tsv(name: str, directory: Path | str | None = None) -> pd.DataFrame:
    """Load one train TSV. Everything stays as plain strings: entity ids are
    identifiers (never numbers) and empty address/name cells are '' not NaN."""
    path = Path(directory) / name if directory else DATA_DIR / name
    return pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        na_filter=False,
        quoting=csv.QUOTE_NONE,
        engine="c",
    )


def load_all(directory: Path | str | None = None) -> dict[str, pd.DataFrame]:
    """Load the three source listings + the ground-truth matches."""
    data = {key: load_tsv(fname, directory) for key, fname in SOURCE_FILES.items()}
    data["ground_truth"] = load_tsv(GROUND_TRUTH_FILE, directory)
    return data


def parse_ground_truth(gt: pd.DataFrame) -> dict[str, list[str]]:
    """Map source1_entity_id -> list of matched entity_ids (S2-/S3- prefixed)."""
    out: dict[str, list[str]] = {}
    for s1_id, matches in zip(gt["source1_entity_id"], gt["matched_entity_ids"]):
        out[s1_id] = [m for m in str(matches).split(",") if m]
    return out


# ---------------------------------------------------------------------------
# EDA
# ---------------------------------------------------------------------------

def eda_source(df: pd.DataFrame, label: str) -> None:
    print(f"\n--- {label} ---")
    print(f"rows={len(df):,}  cols={list(df.columns)}  "
          f"mem={df.memory_usage(deep=True).sum() / 1e6:.0f} MB")

    for col in df.columns:
        empty = int((df[col] == "").sum())
        print(f"  {col:<17} empty={empty:>10,} ({empty / len(df):5.1%})  "
              f"unique={df[col].nunique():,}")

    dup_ids = int(df["entity_id"].duplicated().sum())
    print(f"  duplicate entity_id rows: {dup_ids:,}")

    if "country" in df.columns:
        top = df["country"].value_counts().head(5)
        print("  top countries: " + ", ".join(f"{c}={n:,}" for c, n in top.items()))

    print("  sample rows:")
    for _, row in df.head(2).iterrows():
        print(f"    {row['entity_id']} | {row['business_name'][:45]!r} "
              f"| {row['business_address'][:45]!r} | {row['country']}")


def eda_ground_truth(gt: pd.DataFrame) -> None:
    print(f"\n--- ground truth ---")
    print(f"rows={len(gt):,}")

    parsed = parse_ground_truth(gt)
    n_matches = pd.Series([len(v) for v in parsed.values()])
    print(f"source1 entities: {len(parsed):,}")
    print(f"  with zero matches : {(n_matches == 0).sum():,}")
    print(f"  matches per entity: min={n_matches.min()} median={n_matches.median():.0f} "
          f"mean={n_matches.mean():.2f} max={n_matches.max()}")

    # Which sources do the matches come from?
    has_s2 = sum(any(m.startswith("S2-") for m in v) for v in parsed.values())
    has_s3 = sum(any(m.startswith("S3-") for m in v) for v in parsed.values())
    both = sum(any(m.startswith("S2-") for m in v) and any(m.startswith("S3-") for m in v)
               for v in parsed.values())
    print(f"  entities with >=1 S2 match: {has_s2:,}   >=1 S3 match: {has_s3:,}   both: {both:,}")
    print(f"  total positive pairs (S1,S2/3): {int(n_matches.sum()):,}")

    # a worked example: pick an entity with matches and show the raw records
    for s1_id, matches in parsed.items():
        if matches:
            print(f"  example group {s1_id} -> {matches[:4]}")
            break


# ---------------------------------------------------------------------------
# Languages / scripts
# ---------------------------------------------------------------------------
#
# Two-level detection, cheapest first:
#   1. SCRIPT  - the Unicode block the text is written in. Deterministic,
#                dependency-free, and for this dataset it answers most of the
#                question: source1 is pure Latin/English, sources 2/3 mix Latin
#                with nine Indian scripts.
#   2. LANGUAGE - inside the Latin block (which is >90% of rows) the script
#                tells you nothing, so we ask langdetect. For non-Latin scripts
#                the block IS the language signal (Devanagari -> Hindi, ...),
#                so langdetect is not consulted.
#
# langdetect is optional: `uv pip install --python .venv/Scripts/python.exe
# langdetect`. Without it, Latin rows are reported as "Latin (undetermined)"
# and the script breakdown still works.

try:  # optional dependency - the rest of this file must run without it
    from langdetect import DetectorFactory, detect as _langdetect
    DetectorFactory.seed = 0  # otherwise results differ run to run
    _HAS_LANGDETECT = True
except Exception:  # pragma: no cover - depends on the environment
    _langdetect = None
    _HAS_LANGDETECT = False

# Non-overlapping Unicode blocks -> script name. Ordered/range-sorted below so
# a codepoint lookup is a single bisect.
_SCRIPT_RANGES: tuple[tuple[int, int, str], ...] = tuple(sorted((
    (0x0041, 0x005A, "Latin"), (0x0061, 0x007A, "Latin"),   # A-Z a-z
    (0x00C0, 0x024F, "Latin"), (0x1E00, 0x1EFF, "Latin"),   # Latin-1/suppl, ext-B
    (0x0370, 0x03FF, "Greek"), (0x1F00, 0x1FFF, "Greek"),
    (0x0400, 0x052F, "Cyrillic"),
    (0x0530, 0x058F, "Armenian"),
    (0x0590, 0x05FF, "Hebrew"),
    (0x0600, 0x06FF, "Arabic"),
    (0x0700, 0x074F, "Syriac"),
    (0x0900, 0x097F, "Devanagari"),
    (0x0980, 0x09FF, "Bengali"),
    (0x0A00, 0x0A7F, "Gurmukhi"),
    (0x0A80, 0x0AFF, "Gujarati"),
    (0x0B00, 0x0B7F, "Oriya"),
    (0x0B80, 0x0BFF, "Tamil"),
    (0x0C00, 0x0C7F, "Telugu"),
    (0x0C80, 0x0CFF, "Kannada"),
    (0x0D00, 0x0D7F, "Malayalam"),
    (0x0E00, 0x0E7F, "Thai"),
    (0x10A0, 0x10FF, "Georgian"), (0x2D00, 0x2D2F, "Georgian"),
    (0x1100, 0x11FF, "Hangul"), (0x3130, 0x318F, "Hangul"),
    (0xAC00, 0xD7AF, "Hangul"),
    (0x30A0, 0x30FF, "Katakana"), (0xFF66, 0xFF9D, "Katakana"),
    (0x3400, 0x4DBF, "Han"), (0x4E00, 0x9FFF, "Han"), (0xF900, 0xFAFF, "Han"),
)))
_SCRIPT_STARTS = [r[0] for r in _SCRIPT_RANGES]

# Script -> the language that block is written in. Cyrillic/Han are reported as
# their dominant language; this dataset is US/India so those blocks are ~noise.
_SCRIPT_LANGUAGE = {
    "Devanagari": "Hindi", "Bengali": "Bengali", "Gurmukhi": "Punjabi",
    "Gujarati": "Gujarati", "Oriya": "Odia", "Tamil": "Tamil",
    "Telugu": "Telugu", "Kannada": "Kannada", "Malayalam": "Malayalam",
    "Thai": "Thai", "Greek": "Greek", "Hebrew": "Hebrew", "Arabic": "Arabic",
    "Syriac": "Syriac", "Armenian": "Armenian", "Georgian": "Georgian",
    "Cyrillic": "Russian", "Han": "Chinese", "Hangul": "Korean",
    "Katakana": "Japanese",
}

# ISO-639-1 -> readable name, for the langdetect half of the report.
_LANG_NAMES = {
    "en": "English", "fr": "French", "de": "German", "es": "Spanish",
    "it": "Italian", "pt": "Portuguese", "nl": "Dutch", "ro": "Romanian",
    "ru": "Russian", "uk": "Ukrainian", "pl": "Polish", "tr": "Turkish",
    "vi": "Vietnamese", "id": "Indonesian", "sv": "Swedish", "da": "Danish",
    "no": "Norwegian", "fi": "Finnish", "cs": "Czech", "hu": "Hungarian",
    "el": "Greek", "he": "Hebrew", "ar": "Arabic", "fa": "Persian",
    "hi": "Hindi", "bn": "Bengali", "ta": "Tamil", "te": "Telugu",
    "kn": "Kannada", "gu": "Gujarati", "mr": "Marathi", "pa": "Punjabi",
    "ml": "Malayalam", "or": "Odia", "ne": "Nepali", "si": "Sinhala",
    "th": "Thai", "ja": "Japanese", "ko": "Korean", "zh": "Chinese",
    "af": "Afrikaans", "ca": "Catalan", "hr": "Croatian", "sl": "Slovenian",
    "bg": "Bulgarian", "lt": "Lithuanian", "lv": "Latvian", "et": "Estonian",
    "sq": "Albanian", "is": "Icelandic", "cy": "Welsh", "eu": "Basque",
    "gl": "Galician", "sk": "Slovak", "sr": "Serbian", "ms": "Malay",
    "la": "Latin", "eo": "Esperanto", "tl": "Tagalog", "sw": "Swahili",
}


def _script_of_char(ch: str) -> str | None:
    """Unicode script (block name) of one character, or None if it carries no
    script signal (digits, punctuation, whitespace, ASCII symbols)."""
    cp = ord(ch)
    if 0x41 <= cp <= 0x5A or 0x61 <= cp <= 0x7A:  # fast path: A-Z a-z
        return "Latin"
    i = bisect.bisect_right(_SCRIPT_STARTS, cp) - 1
    if i >= 0:
        lo, hi, name = _SCRIPT_RANGES[i]
        if lo <= cp <= hi:
            return name
    return None


def detect_script(text: str) -> str:
    """Dominant script of `text` - the block owning the most letters.

    A mixed string (e.g. an English name with a Devanagari street) is reported
    as its majority script, with ties broken alphabetically so the result is
    deterministic. Returns "Empty" for blank cells and "Undetermined" when the
    text has letters in no known block.
    """
    if not text or not text.strip():
        return "Empty"
    counts: dict[str, int] = {}
    for ch in text:
        name = _script_of_char(ch)
        if name is not None:
            counts[name] = counts.get(name, 0) + 1
    if not counts:
        return "Undetermined"
    return min(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0]


def detect_language(text: str, script: str | None = None) -> str:
    """Language of `text`.

    Non-Latin scripts name their own language. Latin script - which is where
    every transliteration and every US business name lives - needs a real
    detector, so langdetect decides; without langdetect the answer degrades to
    "Latin (undetermined)" rather than guessing.
    """
    script = script or detect_script(text)
    if script != "Latin":
        return _SCRIPT_LANGUAGE.get(script, script)
    if not _HAS_LANGDETECT:
        return "Latin (undetermined)"
    text = text.strip()
    if not text:
        return "Empty"
    try:
        assert _langdetect is not None  # narrowed by _HAS_LANGDETECT
        code = _langdetect(text)
    except Exception:  # LangDetectException on empty/featureless input
        return "undetermined"
    return _LANG_NAMES.get(code, code)


def collect_languages(directory: Path | str | None = None, *,
                      stride: int = 25, chunk_size: int = 200_000,
                      max_chunks: int | None = None,
                      lang_sample: int = 2_000) -> dict:
    """Stream the three train sources and tally script / language per column.

    The files are ~1.3 GB and ~12.5M rows, so this reads them in chunks and
    takes every `stride`-th row of each chunk - the counts below are a
    SAMPLE, not exact totals, and every printed total says so. Pass stride=1
    (and expect minutes) for a full pass.
    """
    per_source: dict[str, dict] = {}
    for key, fname in SOURCE_FILES.items():
        path = Path(directory) / fname if directory else DATA_DIR / fname
        scripts = {"business_name": Counter(), "business_address": Counter()}
        langs = {"business_name": Counter(), "business_address": Counter()}
        by_country: Counter = Counter()
        scanned = 0
        for n, chunk in enumerate(pd.read_csv(
                path, sep="\t", dtype=str, keep_default_na=False, na_filter=False,
                quoting=csv.QUOTE_NONE, engine="c",
                usecols=["business_name", "business_address", "country"],
                chunksize=chunk_size)):
            if max_chunks is not None and n >= max_chunks:
                break
            sample = chunk.iloc[::stride]
            scanned += len(sample)
            # script is cheap: run it on the whole strided sample
            for col in ("business_name", "business_address"):
                for text, script in zip(sample[col], map(detect_script, sample[col])):
                    scripts[col][script] += 1
            # language detection is the expensive half: cap it per source/column
            for col in ("business_name", "business_address"):
                head = sample[col].head(lang_sample)
                langs[col].update(map(detect_language, head))
            by_country.update(zip(sample["country"],
                                  map(detect_language, sample["business_name"])))
        per_source[key] = {"file": fname, "sampled_rows": scanned,
                           "scripts": scripts, "languages": langs,
                           "country_languages": by_country}
    return per_source


def eda_languages(directory: Path | str | None = None, *,
                  stride: int = 25, chunk_size: int = 200_000,
                  max_chunks: int | None = None,
                  lang_sample: int = 2_000) -> None:
    """Print every language found in the three train source files.

    Per source: the script breakdown for name and address, the languages
    langdetect assigns (Latin rows only get a real verdict), the languages per
    country, and a final all-sources summary of the distinct languages.

    CAVEAT on the language list: a business name is 2-4 tokens, and langdetect
    needs a paragraph to be statistically sound, so the Latin rows over-fire -
    "Estonian"/"Catalan"/"so" on US business names is detector noise, not
    real language. The SCRIPT breakdown is the trustworthy half; the language
    list is trustworthy for the non-Latin rows (Hindi, Tamil, ...) and only
    indicative for Latin ones.
    """
    print(f"Data dir: {DATA_DIR if directory is None else Path(directory)}")
    print(f"langdetect installed: {_HAS_LANGDETECT}"
          f"   (script detection always runs; stride={stride}, "
          f"lang_sample={lang_sample:,} per source/column)")

    data = collect_languages(directory, stride=stride, chunk_size=chunk_size,
                             max_chunks=max_chunks, lang_sample=lang_sample)

    for key in ("source1", "source2", "source3"):
        info = data[key]
        n = max(info["sampled_rows"], 1)
        print(f"\n--- {key} ({info['file']}) ---")
        print(f"sampled {info['sampled_rows']:,} rows (1 in {stride})")

        for col in ("business_name", "business_address"):
            counts = info["scripts"][col]
            if not counts:
                continue
            print(f"  script of {col}:")
            for script, c in counts.most_common():
                print(f"    {script:<12} {c:>9,}  {c / n:6.2%}")
            top = ", ".join(f"{lang} {c:,}"
                            for lang, c in info["languages"][col].most_common(5))
            print(f"  languages of {col} (top 5): {top or 'n/a'}")

        countries: dict[str, int] = {}
        for (country, lang), c in info["country_languages"].items():
            countries[country] = countries.get(country, 0) + c
        for country, c in sorted(countries.items(), key=lambda kv: -kv[1]):
            names = ", ".join(f"{lang} {cnt:,}" for (co, lang), cnt
                              in info["country_languages"].most_common()
                              if co == country)
            print(f"  country {country!r:<8} ({c:,} names): {names}")

    # cross-source summary: the distinct languages in the train files
    totals: Counter = Counter()
    for info in data.values():
        for col in ("business_name", "business_address"):
            totals.update(info["languages"][col])
    print(f"\n--- all sources: {len(totals)} distinct languages ---")
    for lang, c in totals.most_common():
        print(f"  {lang:<24} {c:>9,}")
    if not _HAS_LANGDETECT:
        print("  (Latin-script rows are unattributed - `pip install langdetect` "
              "to resolve them into English/French/...)")


# ---------------------------------------------------------------------------

def main() -> None:
    print(f"Data dir: {DATA_DIR}")
    data = load_all()

    for key in ("source1", "source2", "source3"):
        eda_source(data[key], key)
    eda_ground_truth(data["ground_truth"])
    eda_languages()

    print("\nWhat the task is: given the three source listings, predict which")
    print("records refer to the same business (train ground truth shows the")
    print("S1 <-> S2/S3 matches to learn from; same format expected at inference).")


if __name__ == "__main__":
    main()
