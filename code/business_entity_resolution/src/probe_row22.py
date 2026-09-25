# -*- coding: utf-8 -*-
"""Probe row 22 of train_source2.tsv through the full preprocess pipeline."""
import sys, csv
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import importlib.util
spec = importlib.util.spec_from_file_location(
    "pp", str(Path(__file__).resolve().parent / "pre-process.py"))
pp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pp)

import pandas as pd

path = Path(r"C:\Users\Arnav Waghdhare\Desktop\Arnav20\Coding\Python\amazon_ml_2026"
            r"\data\student_resource\dataset\train\train_source2.tsv")
# row indexing: header + line numbers; grab lines 20-24 (1-based incl header)
df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False,
                 na_filter=False, quoting=csv.QUOTE_NONE, engine="c",
                 skiprows=range(1, 21), nrows=4)
print("rows fetched:")
for _, r in df.iterrows():
    print(" ", r["entity_id"], "|", repr(r["business_name"])[:70])

row = df[df["entity_id"].str.contains("22", na=False)]
# safer: take the Tamil-looking row (3rd fetched row, original line 22)
tamil_row = None
for _, r in df.iterrows():
    # find the Tamil-script row (Tamil block U+0B80-U+0BFF)
    if any(0x0B80 <= ord(ch) <= 0x0BFF for ch in r["business_name"]):
        tamil_row = r
        break
if tamil_row is None:
    tamil_row = df.iloc[2]
print("\n=== testing row:", tamil_row["entity_id"], "===")
print("raw name:", repr(tamil_row["business_name"]))
print("raw addr:", repr(tamil_row["business_address"]))

df1 = pd.DataFrame([tamil_row.to_dict()])
clean = pp.clean_source(df1, "S2")
print("\nsurvives clean_source:", len(clean) == 1)

name_norm = pp.normalize_name(tamil_row["business_name"])
addr_norm = pp.normalize_address(tamil_row["business_address"])
print("name_norm:", repr(name_norm))
print("addr_norm:", repr(addr_norm))

grams = pp.char_ngrams(name_norm)
print("n-gram count:", len(grams), "first 5:", grams[:5])

pk_ascii = " ".join(t for t in name_norm.split() if t.isascii())
print("ascii tokens for phonetic:", repr(pk_ascii))
print("phonetic key (name_phonetic col value):",
      repr(pp.phonetic_key(pk_ascii)))

keys = pp.canopy_keys(name_norm, addr_norm)
print("canopy keys:", len(keys))
from collections import Counter
fams = Counter(k[:2] for k in keys)
print("by family:", dict(fams))

print("full preprocess on this single row:")
# and confirm a Tamil-transliterated variant matches it through N7
ppmm = pp
combined = pp.preprocess({"S2": df1}, None)
print(combined[["entity_id", "name_norm", "name_phonetic",
                "name_ngram_count", "canopy_ids"]].iloc[0].to_string())

# Hypothetical romanized twin: if this row later gets transliterated to
# 'global business private limited', does it share an N7 canopy with a
# French-style misspelling 'globale business prive limitted'?
tamil_roman = "global business private limited"
french_ish = "globale business prive limited"
b1 = pp.phonetic_buckets(tamil_roman)
b2 = pp.phonetic_buckets(french_ish)
print("\nromanized-twin N7 overlap:", b1 & b2)
k1 = pp.canopy_keys(tamil_roman, "")
k2 = pp.canopy_keys(french_ish, "")
print("shared canopies:", len(set(k1) & set(k2)), "->",
      sorted(set(k1) & set(k2))[:3])
print("probe done")
