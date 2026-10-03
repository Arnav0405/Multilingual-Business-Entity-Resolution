# Business Entity Resolution — Multilingual & Cross-Script

Academic NLP study built from the Amazon ML Challenge 2026 dataset: **can a multilingual Transformer learn to identify the same business entity when its name appears in a different script, transliteration, abbreviation, or noisy form?**

**Headline result:** cross-script F0.5 **0.113 (TF-IDF) → 0.997 (contrastive E5 + address)**, with **zero cross-script false positives** on the test split.

---

## TL;DR results (test split, 19,558 pairs)

| Model | Overall F0.5 | Same-script F0.5 | Cross-script F0.5 | Cross recall | Recall@1 / MRR |
|---|---:|---:|---:|---:|---:|
| TF-IDF char (2–4g) | 0.929 | 0.974 | 0.113 | 2.5% | 0.940 / 0.949 |
| TF-IDF word (1–2g) | 0.897 | 0.941 | 0.205 | 5.0% | 0.939 / 0.954 |
| TF-IDF both | 0.922 | 0.966 | 0.178 | 4.2% | 0.941 / 0.950 |
| E5 pretrained (B) | 0.963 | 0.971 | 0.932 | 86.8% | 0.999 / 0.9996 |
| E5 + contrastive (C) | 0.990 | 0.988 | 0.995 | 99.1% | 1.000 / 1.000 |
| E5 + hard negs (D) | 0.989 | 0.988 | 0.995 | 99.5% | 1.000 / 1.000 |
| A′ = D + address concat | **0.997** | 0.996 | **0.999** | 99.7% | 1.000 / 1.000 |

A′ caveat: on rows where an address is missing (n=173 positives), the concat hurts — D 0.995 vs A′ 0.988. Conditional model selection on field presence is the practical recipe.

---

## Repository layout

```
nlp_ca/
├── plan.md                                    # phase checklist + ticked boxes + central results table
├── report.md                                  # full academic report (phase 11)
├── data/entity_resolution_subset/             # subset TSVs + statistics + validation report
└── code/business_entity_resolution/
    ├── src/
    │   ├── creat_subset.py                    # builds the stratified 10,000-S1 subset (v2 equijoin, seed 42)
    │   ├── dataset.py                         # raw-text pair/triple loading, name / name_address variants
    │   ├── encoder.py                         # E5 wrapper (query:/passage: prefixes, CLS pooling)
    │   └── losses.py                          # MNRL / InfoNCE + optional per-anchor hard negatives
    ├── scripts/
    │   ├── env_smoke.py                       # GPU + model smoke test → results/env_smoke.json
    │   └── validate_subset.py                 # 31 integrity checks → validation_report.json
    ├── baselines/
    │   ├── tfidf.py                           # Model A: TF-IDF (char/word/both variants)
    │   ├── evaluate.py                        # shared metrics: P/R/F1/F0.5, retrieval, per-script slices
    │   ├── report.py                          # evaluates score books → results/<model>.json
    │   ├── phase8_tables.py                   # results/phase8_tables.md
    │   ├── phase10_address.py                 # address-presence controlled comparison
    │   └── make_figures.py                    # all 7 report figures → results/figures/
    ├── training/
    │   ├── build_notebook.py                  # generates the Colab notebook
    │   ├── entity_resolution_colab.ipynb      # PHASE-flagged: "B" | "C" | "D" | "ADDR"
    │   ├── train_contrastive.py               # local trainer (same logic as notebook)
    │   └── test_notebook_logic.py             # 18 pytest tests of notebook logic
    └── results/                               # metrics JSONs, score parquets, checkpoints, figures/
```

## Pipeline (reproduction notes)

Everything assumes the venv at `code/business_entity_resolution/.venv` (Python 3.13, `torch` cu130). All local python runs use `-X utf8`.

### 1. Build the subset

```bash
cd code/business_entity_resolution/src
../.venv/Scripts/python.exe -X utf8 creat_subset.py
# → data/entity_resolution_subset/{train,val,test}_{s1,pairs}.tsv + subset_statistics.json
```

### 2. Validate it

```bash
cd code/business_entity_resolution/scripts
../.venv/Scripts/python.exe -X utf8 validate_subset.py
# → all 31 checks pass → data/entity_resolution_subset/validation_report.json
```

### 3. TF-IDF baseline

```bash
cd code/business_entity_resolution/baselines
../.venv/Scripts/python.exe -X utf8 tfidf.py --variant char
../.venv/Scripts/python.exe -X utf8 tfidf.py --variant word
../.venv/Scripts/python.exe -X utf8 tfidf.py --variant both
# → results/tfidf_*.json + *_scores.parquet
```

### 4. Neural models (Colab T4, free tier)

Generate and open `training/entity_resolution_colab.ipynb`, set `PHASE = "B" | "C" | "D" | "ADDR"`, upload the three `{train,val,test}_pairs.tsv` files (~31 MB) into `/content/data`, run all cells.

- **B** — encodes all splits with pretrained `intfloat/multilingual-e5-small`, scoring cosines → `modelB_scores.parquet`
- **C** — 3-epoch contrastive fine-tune (MNRL, batch 64), then score → `modelC_scores.parquet` + `modelC_checkpoint.zip`
- **D** — same C + explicit per-anchor hard negatives → `modelD_scores.parquet` + checkpoint
- **ADDR** — name+address variant (max_length 128, batch 32) → `modelA_name_address_scores.parquet` + checkpoint

Download the parquet (and checkpoint zip for C/D/ADDR) back into local `results/`.

### 5. Local evaluation

```bash
cd code/business_entity_resolution/baselines
../.venv/Scripts/python.exe -X utf8 report.py --model modelB
../.venv/Scripts/python.exe -X utf8 report.py --model modelC
../.venv/Scripts/python.exe -X utf8 report.py --model modelD
../.venv/Scripts/python.exe -X utf8 report.py --model modelA_name_address
../.venv/Scripts/python.exe -X utf8 phase8_tables.py      # results/phase8_tables.md
../.venv/Scripts/python.exe -X utf8 phase10_address.py    # results/phase10_address_experiment.json
```

Every model is scored at its own **val-tuned F0.5 threshold**, carried frozen to the test split; the central-table comparison additionally fixes one shared threshold across all models.

### 6. Figures

```bash
../.venv/Scripts/python.exe -X utf8 make_figures.py
# → results/figures/fig{1,2,3,4,5,7,8}*.png
```

### 7. Tests

```bash
cd code/business_entity_resolution/training
../.venv/Scripts/python.exe -X utf8 -m pytest test_notebook_logic.py -q
# → 18 passed
```

## Key implementation notes (things that bit us)

- **No pre-normalisation.** Script tagging uses Unicode ranges on raw strings; nothing is lowercased/stemmed/transliterated before the encoder.
- **fp32 master weights + `torch.autocast(fp16)` around encoder forwards only.** End-to-end fp16 weight training diverged to all-NaN checkpoints on the T4 within a few steps (AdamW + OneCycle, lr 3e-5). NaN guard per step; checkpoint NaN-scan before scoring.
- **Hard negatives are aligned per anchor row** (uniform K=4, pool-filled with replacement) — not per group. Group-level collection produces ragged tensors that break `h.view(a.shape[0], K, -1)` reshaping.
- **`fillna('')` before the address concatenation.** 3,824 train rows have an empty candidate address; without coercion the string silently becomes `"nan"` and the model learns a spurious missing-address identity.
- **Batch 64 fits the T4 in fp16 autocast; batch 96 does not.**

## Dataset sizes for sanity checks

- 10,000 S1 (6,995 train / 1,495 val / 1,510 test), seed 42
- 129,761 total pairs = 29,761 positives + 80,000 hard negatives + 20,000 easy negatives
- 6,044 cross-script positives (4,290 train / 870 val / 884 test)
- Test cross-script positives: **884**

Full narrative, per-script tables, error analysis, and figures are in [`report.md`](../../report.md).
