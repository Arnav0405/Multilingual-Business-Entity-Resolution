"""
Pre-process the Business Entity Resolution data.

Pipeline (per spec):
  1. Drop rows with any empty field in the three source listings.
     Exception: the ground-truth file is NEVER dropped — an empty
     `source1_entity_id` there is replaced with the literal token `<empty>`
     (empty `matched_entity_ids` get the same token).
  2. String normalization: unicode NFKC fold, lowercase, strip accents,
     collapse punctuation/whitespace, drop legal-entity suffix noise
     ("ltd", "llc", "pvt", ...). Keeps Devanagari (Hindi) characters so
     source2's Indian names survive normalization.
  3. Character n-grams (n=3, full-string fallback for very short strings)
     over normalized name and address.
  4. Canopy clustering blocking: each record is assigned to canopies defined
     by a small family of cheap blocking keys (name prefix+length bucket,
     token prefixes, sampled n-gram buckets, character-order skeleton).
     NO country blocking — the test data contains countries not in train.
  5. Everything is written as ONE csv file with preprocessed fields, the
     n-grams, canopy ids, and (for source1 rows) the ground-truth matches.

Run:  python pre-process.py               # train sources + ground truth
      python pre-process.py --split test  # test sources (no gt merge)
Output: <repo>/output/preprocessed_train.csv (or _test.csv)
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
import unicodedata
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[3]
DATA_DIR = REPO_ROOT / "data" / "student_resource" / "dataset"
OUTPUT_DIR = REPO_ROOT / "output"

SOURCE_FILES = {
    "S1": "train_source1.tsv",
    "S2": "train_source2.tsv",
    "S3": "train_source3.tsv",
}
GROUND_TRUTH_FILE = "train_ground_truth.tsv"

EMPTY_TOKEN = "<empty>"

# ---------------------------------------------------------------------------
# String normalization
# ---------------------------------------------------------------------------

# Legal / corporate suffixes and filler words stripped from names.
_SUFFIXES = {
    "ltd", "limited", "llc", "llp", "lp", "inc", "incorporated", "corp",
    "corporation", "co", "company", "pvt", "private", "pcl", "plc",
    "gmbh", "srl", "sa", "ag", "bv", "nv", "pty",
}

_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
_WS_RE = re.compile(r"\s+")
# NOTE: matras are category Mc, which [^\w\s] kills. They are semantic in
# Devanagari, so normalize_text must keep every mark that is not an ASCII
# (Latin) accent. See the mark-preservation step inside normalize_text.

_PLACEHOLDER_ADDRESSES = {"na", "n a", "n", "none", "null", "--", "-"}


def normalize_text(text: str) -> str:
    """NFKC fold, lowercase, strip accents/diacritics, kill punctuation,
    collapse whitespace. Preserves word characters of any script (so Devanagari
    and other non-Latin names are kept, not emptied)."""
    if not text or text == EMPTY_TOKEN:
        return ""
    s = unicodedata.normalize("NFKC", text).lower()
    # decompose, then strip accents from Latin letters only — removing
    # combining marks from Devanagari would destroy matras
    s = unicodedata.normalize("NFD", s)
    chars = []
    for ch in s:
        # Latin diacritics live exclusively in U+0300-036F — drop those.
        # Devanagari matras/virama are elsewhere (Mc/Mn but not in that
        # range) and stay, so Hindi names are untouched.
        if 0x0300 <= ord(ch) <= 0x036F:
            continue
        cat = unicodedata.category(ch)
        if cat == "Cn" or ord(ch) < 0x20:  # unassigned / control junk
            continue
        chars.append(ch)
    s = "".join(chars)
    # recompose so accented-Latin bases are normal single chars again
    s = unicodedata.normalize("NFC", s)
    # punctuation removal — but matras/Halant are category Mc/Mn, not P*, and
    # must survive: they spell Devanagari words
    def _is_punct(ch: str) -> bool:
        cat = unicodedata.category(ch)
        if cat in ("Mc", "Mn", "Me"):
            return False
        return not (ch.isalnum() or ch.isspace() or cat.startswith("N"))
    s = "".join(" " if _is_punct(ch) else ch for ch in s)
    return _WS_RE.sub(" ", s).strip()


def normalize_name(text: str) -> str:
    """normalize_text + drop legal-entity suffix tokens."""
    norm = normalize_text(text)
    if not norm:
        return ""
    toks = [t for t in norm.split() if t not in _SUFFIXES]
    # keep at least one token so the record is never fully emptied
    return " ".join(toks) if toks else norm


def normalize_address(text: str) -> str:
    """normalize_text + map placeholder addresses to '' (handled downstream)."""
    norm = normalize_text(text)
    if not norm or norm in _PLACEHOLDER_ADDRESSES:
        return ""
    return norm


# ---------------------------------------------------------------------------
# Character n-grams
# ---------------------------------------------------------------------------

NGRAM_SIZE = 3
FALLBACK_NGRAM_SIZE = 2   # for very short strings
MAX_NGRAMS = 24           # cap per string to keep the csv bounded


def char_ngrams(text: str, n: int = NGRAM_SIZE,
                max_ngrams: int = MAX_NGRAMS) -> list[str]:
    """Character n-grams over the space-padded string (padded form is the
    record-linkage standard: 'ab cd' -> ' ab','abc','bc ',' cd')."""
    if not text:
        return []
    padded = f" {text} "
    if len(padded) < 2 * n - 1:
        n = FALLBACK_NGRAM_SIZE
    grams = [padded[i: i + n] for i in range(len(padded) - n + 1)]
    seen: set[str] = set()
    out: list[str] = []
    for g in grams:
        if g not in seen:
            seen.add(g)
            out.append(g)
        if len(out) >= max_ngrams:
            break
    return out


# ---------------------------------------------------------------------------
# Phonetic keys
# ---------------------------------------------------------------------------

try:
    import jellyfish

    def _metaphone_key(word: str) -> str:
        return jellyfish.metaphone(word) or ""

    def _soundex_key(word: str) -> str:
        try:
            return jellyfish.soundex(word) or ""
        except Exception:
            return ""
except ImportError:  # stdlib fallback so the pipeline never breaks
    def _metaphone_key(word: str) -> str:
        return _stdlib_soundex(word)

    def _soundex_key(word: str) -> str:
        return _stdlib_soundex(word)


def _stdlib_soundex(word: str) -> str:
    """Classic American Soundex — used only if jellyfish is unavailable."""
    if not word:
        return ""
    codes = {**{c: "1" for c in "bfpv"}, **{c: "2" for c in "cgjkqsxz"},
             "d": "3", "t": "3", "l": "4", "m": "5",
             "n": "5", "r": "6"}
    w = word.upper()
    first = w[0]
    digits = []
    prev_code = None
    for ch in w[1:]:
        code = codes.get(ch)
        if code is None:
            prev_code = None if ch in "hw" else prev_code
            continue
        if code != prev_code:
            digits.append(code)
        prev_code = code
    return (first + "".join(digits) + "000")[:4]

##
## Phonetic-key design note: metaphone alone conflates many French/English
## names; we combine metaphone (primary) + soundex (secondary) and bucket by
## leading chars so canopy families stay cheap. Both operate ONLY on
## ASCII-romanized text — jellyfish is not unicode-script aware.
##


def phonetic_key(name_norm: str) -> str:
    """phonetic skeleton of a normalized name: metaphone of the first token,
    plus soundex of the last (distinguishing) token. Empty when no tokens.
    Primary key for the name_phonetic column."""
    toks = name_norm.split() if name_norm else []
    if not toks:
        return ""
    first_mp = _metaphone_key(toks[0])
    last_sdx = _soundex_key(toks[-1]) if len(toks) > 1 else ""
    return (first_mp + ("_" + last_sdx if last_sdx else "")).strip("_")


def phonetic_buckets(name_norm: str) -> set[str]:
    """ALL phonetic buckets of a name: {metaphone, soundex} x {first token,
    last token}. Two variant spellings of the same name usually share at
    least one bucket (metaphone catches vowel drift, soundex catches
    initial-consonant-cluster differences like Sh-/S-)."""
    toks = name_norm.split() if name_norm else []
    buckets: set[str] = set()
    if not toks:
        return buckets
    mp_first = _metaphone_key(toks[0])
    sdx_first = _soundex_key(toks[0])
    mp_last = _metaphone_key(toks[-1])
    sdx_last = _soundex_key(toks[-1])
    for k in (mp_first, sdx_first, mp_last, sdx_last):
        if k:
            buckets.add(k)
    return buckets


def phonetic_variants(name_norm: str) -> list[str]:
    """Additional blocking variants: metaphone of the LAST token and the
    concatenated name skeleton — catches swapped first/last token orders
    and pooled multi-token name spellings."""
    toks = name_norm.split() if name_norm else []
    variants: list[str] = []
    if len(toks) >= 2:
        mp_last = _metaphone_key(toks[-1])
        if mp_last:
            variants.append(mp_last)
    if toks:
        skel = "".join(_metaphone_key(t) for t in toks[:3])
        if skel:
            variants.append(skel[:10])
    return variants


def _canopy_id(tag: str, key: str) -> str:
    """Stable short canopy id: <tag><first 10 hex of md5(tag|key)>."""
    digest = hashlib.md5(f"{tag}|{key}".encode("utf-8")).hexdigest()[:10]
    return f"{tag}{digest}"


def canopy_keys(name_norm: str, address_norm: str) -> list[str]:
    """Generate canopy blocking keys for one record.

    Canopy-key families (name-first — the ground truth matches are driven by
    name similarity; address only assists when the name is bare):
      N1  name-prefix(4) + length bucket
      N2  first-two-chars of first two tokens
      N3  first char of first token + first 3 chars of last token
      N4  sampled name 3-gram buckets
      N5  character-order skeleton (collapses transpositions/reorders)
      N7  phonetic keys (metaphone/soundex of romanized name) — catches
          same-sound different-spelling pairs across French/English/Indic
      N6  address-assisted key, only when the name has <=1 token
      A1  address-prefix key when the record has NO usable name
    No country key: test data contains countries unseen in train on purpose.
    """
    keys: list[str] = []
    toks = name_norm.split() if name_norm else []

    if toks:
        first = toks[0]
        len_bucket = min(len(name_norm) // 6, 8)
        keys.append(_canopy_id("N1", f"{first[:4]}#{len_bucket}"))
        if len(toks) >= 2:
            keys.append(_canopy_id("N2", f"{first[:2]}_{toks[1][:2]}"))
            keys.append(_canopy_id("N3", f"{first[:1]}_{toks[-1][:3]}"))
        if len(toks) <= 1 and address_norm:
            keys.append(_canopy_id(
                "N6", f"{first[:1]}_{address_norm.split()[0][:3]}"))
        grams = char_ngrams(name_norm, NGRAM_SIZE, MAX_NGRAMS)
        for g in grams[::4][:5]:
            keys.append(_canopy_id("N4", g))
        skeleton = "".join(dict.fromkeys(name_norm.replace(" ", "")))[:12]
        if skeleton:
            keys.append(_canopy_id("N5", f"{skeleton[:2]}{len(skeleton)}"))
        # N7: phonetic canopies — only meaningful for romanized tokens, so
        # skip tokens containing non-ASCII chars (jellyfish treats any
        # non-ASCII as unknown and would emit empty/garbage keys)
        ascii_toks = [t for t in toks if t.isascii()]
        if ascii_toks:
            for bucket in phonetic_buckets(" ".join(ascii_toks)):
                keys.append(_canopy_id("N7", bucket))
    elif address_norm:
        atoks = address_norm.split()
        keys.append(_canopy_id(
            "A1", f"{atoks[0][:4]}#{min(len(address_norm) // 6, 8)}"))
    return keys


# ---------------------------------------------------------------------------
# Loading + cleaning
# ---------------------------------------------------------------------------

def load_tsv(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        na_filter=False,
        quoting=csv.QUOTE_NONE,
        engine="c",
    )


def clean_source(df: pd.DataFrame, source_tag: str) -> pd.DataFrame:
    """Drop rows with an empty name or address; rename to raw columns."""
    before = len(df)
    mask = (df["business_name"].str.strip() == "") | \
           (df["business_address"].str.strip() == "")
    df = df[~mask].copy()
    df = df.rename(columns={
        "entity_id": "entity_id",
        "business_name": "name_raw",
        "business_address": "address_raw",
    })
    df["source"] = source_tag
    for col in ("entity_id", "name_raw", "address_raw", "country"):
        df[col] = df[col].str.strip()
    df = df.drop_duplicates(subset=["entity_id"])
    print(f"  source {source_tag}: kept {len(df):,} / {before:,} rows "
          f"(dropped {before - len(df):,})")
    return df


def load_ground_truth_df(gt: pd.DataFrame) -> pd.DataFrame:
    """Ground truth rows are NEVER dropped. Empty ids are tokenized
    to <empty> instead."""
    before = len(gt)
    gt["source1_entity_id"] = gt["source1_entity_id"].str.strip()
    gt["matched_entity_ids"] = gt["matched_entity_ids"].str.strip()
    gt["source1_entity_id"] = gt["source1_entity_id"].replace("", EMPTY_TOKEN)
    gt["matched_entity_ids"] = gt["matched_entity_ids"].replace("", EMPTY_TOKEN)
    n_tok = int((gt["source1_entity_id"] == EMPTY_TOKEN).sum())
    print(f"  ground truth: kept all {before:,} rows "
          f"(tokenized {n_tok:,} empty source1 ids to {EMPTY_TOKEN!r})")
    return gt


def load_ground_truth(path: Path) -> pd.DataFrame:
    return load_ground_truth_df(load_tsv(path))


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def preprocess(sources: dict[str, pd.DataFrame],
               gt: pd.DataFrame | None) -> pd.DataFrame:
    """Clean + normalize + n-grams + canopies; return ONE combined frame.

    Output columns:
      entity_id, source, name_raw, address_raw, name_norm, address_norm,
      name_ngrams, address_ngrams, name_ngram_count, address_ngram_count,
      name_phonetic, canopy_ids, country, ground_truth
    """
    frames = []
    for tag, df in sources.items():
        print(f"Processing source {tag}...")
        df = clean_source(df, tag)
        df["name_norm"] = df["name_raw"].map(normalize_name)
        df["address_norm"] = df["address_raw"].map(normalize_address)

        name_grams = [char_ngrams(s) for s in df["name_norm"]]
        addr_grams = [char_ngrams(s) for s in df["address_norm"]]
        df["name_ngrams"] = ["|".join(g) for g in name_grams]
        df["address_ngrams"] = ["|".join(g) for g in addr_grams]
        df["name_ngram_count"] = [len(g) for g in name_grams]
        df["address_ngram_count"] = [len(g) for g in addr_grams]

        canopy_keys_list = [canopy_keys(n, a)
                            for n, a in zip(df["name_norm"], df["address_norm"])]
        df["canopy_ids"] = [" ".join(k) for k in canopy_keys_list]
        # phonetic key column: metaphone skeleton of ASCII tokens only
        df["name_phonetic"] = [
            phonetic_key(" ".join(t for t in s.split() if t.isascii()))
            for s in df["name_norm"]
        ]
        frames.append(df[[
            "entity_id", "source", "name_raw", "address_raw", "name_norm",
            "address_norm", "name_ngrams", "address_ngrams",
            "name_ngram_count", "address_ngram_count", "name_phonetic",
            "canopy_ids", "country"]])

    combined = pd.concat(frames, ignore_index=True)

    # Merge ground truth so the final csv is self-contained; only source1
    # rows carry matches. Empty gt ids were already tokenized to <empty>,
    # and those tokens are never joined against real entity ids.
    if gt is not None:
        gt_map = dict(zip(gt["source1_entity_id"], gt["matched_entity_ids"]))
        s1_mask = combined["source"] == "S1"
        combined["ground_truth"] = ""
        combined.loc[s1_mask, "ground_truth"] = \
            combined.loc[s1_mask, "entity_id"].map(gt_map).fillna("")
    else:
        combined["ground_truth"] = ""

    cols = ["entity_id", "source", "name_raw", "address_raw", "name_norm",
            "address_norm", "name_ngrams", "address_ngrams",
            "name_ngram_count", "address_ngram_count", "name_phonetic",
            "canopy_ids", "country", "ground_truth"]
    return combined[cols]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["train", "test"], default="train",
                        help="preprocess the train or test sources")
    parser.add_argument("--out", default=None,
                        help="output csv path "
                             "(default output/preprocessed_<split>.csv)")
    args = parser.parse_args()

    split_dir = DATA_DIR / args.split
    src = dict(SOURCE_FILES)
    gt_path: Path | None = None
    if args.split == "test":
        src = {k: f.replace("train_", "test_") for k, f in src.items()}
    else:
        gt_path = split_dir / GROUND_TRUTH_FILE

    print(f"Data dir: {split_dir}")
    sources = {tag: load_tsv(split_dir / fname) for tag, fname in src.items()}
    gt = load_ground_truth(gt_path) if gt_path is not None else None

    combined = preprocess(sources, gt)

    out_path = Path(args.out) if args.out else \
        OUTPUT_DIR / f"preprocessed_{args.split}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing {out_path} ...")
    combined.to_csv(out_path, index=False, encoding="utf-8")
    print(f"rows={len(combined):,}  cols={list(combined.columns)}")

    s1 = combined[combined["source"] == "S1"]
    has_grams = (s1["name_ngram_count"] > 0).mean()
    print(f"verification: source1 rows={len(s1):,}  "
          f"with n-grams={has_grams:.1%}  "
          f"gt cell non-empty={(s1['ground_truth'] != '').sum():,}")
    print("Done.")


if __name__ == "__main__":
    main()
