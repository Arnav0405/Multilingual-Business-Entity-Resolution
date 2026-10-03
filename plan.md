# Multilingual Cross-Script Business Entity Resolution — Execution Plan

Source: ChatGPT handoff (academic project). Track progress here — tick a box only when its
verification artifact actually exists.

**Research question:** Can a multilingual Transformer learn to identify the same business
entity when its name appears in different scripts, transliterations, abbreviations, and noisy
textual forms?

---
---

## Phase 0 — Environment

Goal: reproducible runtime locally (RTX 4060) and on Colab (T4); model download verified.

- [x] Existing venv (`code/business_entity_resolution/.venv`, Python 3.13) has torch 2.9.0+cu130, duckdb, pandas, numpy, tqdm, pyarrow (`python -c "import torch; print(torch.__version__, torch.cuda.is_available())"` → `2.9.0+cu130 True`)
- [x] Add missing deps: scikit-learn 1.9.1, scipy 1.18.1 (via `uv pip install --python code/business_entity_resolution/.venv/Scripts/python.exe`)
- [ ] `transformers` + `sentencepiece` installed in the venv (needed to build the score book from exported embeddings without re-encoding)
- [ ] `intfloat/multilingual-e5-small` downloaded + smoke test: encode 2 strings locally, cosine sanity (Latin≈Latin high, Latin vs Devanagari computed and recorded in `results/env_smoke.json`)
- [x] Colab T4 noted as the training/compute target for Models B/C/D notebooks

## Phase 1 — Dataset

- [x] `build_subset.py` equivalent run (`creat_subset.py`, v2-equijoin) → `data/entity_resolution_subset/` (train/val/test `*_s1.tsv`, `*_pairs.tsv`, `subset_statistics.json`, `examples/cross_script_pairs.tsv`)
- [x] `subset_statistics.json` inspected — 10,000 S1 (6,995/1,495/1,510), 29,761 positives (6,044 cross-script), 80k hard + 20k easy negatives, India 6,500 / US 3,500
- [x] `selected_s1.tsv` save-after-resume bug fixed in `creat_subset.py` (section 5b) and file regenerated from authoritative splits (old file was a fresh-sample artifact, 9,888/10,000 mismatched IDs)

## Phase 2 — Subset validation

- [x] `code/business_entity_resolution/scripts/validate_subset.py` — 8 handoff checks + counts
- [x] All checks pass → `data/entity_resolution_subset/validation_report.json`

## Phase 3 — Shared evaluation module

- [x] `baselines/evaluate.py` — script-tags on raw strings (9 Indic ranges), binary P/R/F1/F0.5, F0.5-optimal threshold tuning, retrieval (Recall@1/5/10, MRR), slices Overall / Same / Cross / per-script
- [x] Score-book contract fixed: `results/<model>_scores.parquet` with `split,row_idx,score`; row_idx = position in split's TSV as read; exported `data/entity_resolution_subset/test_pairs_cross_script.tsv`

## Phase 4 — Model A: TF-IDF baseline

- [x] `baselines/tfidf.py` run for all 3 variants (char_wb 2–4 / word 1–2 / both)
- [x] Reports: `results/tfidf_char.json`, `results/tfidf_word.json`, `results/tfidf_both.json` (+ `_scores.parquet` each)
- Test overall F0.5: char **0.974**, word 0.897, both 0.922 | Cross-script recall: 2.5% / 5.0% / 4.2% — same-script ceiling ≈0.97, cross-script broken, as intended

## Phase 5 — Model B: Pretrained multilingual E5, no fine-tuning (Colab T4)

- [ ] `training/notebooks/entity_resolution_colab.ipynb` written: shared header (data upload/mount, score-book helper), then flag-gated phases B / C / D
- [ ] Model B run: encode all S1 + candidate names with `intfloat/multilingual-e5-small` (fp16, `query:`/`passage:` prefixes), cosine
- [ ] Scores exported as `modelB_scores.parquet` → local `baselines/report.py` evaluation (Colab gets no raw data download back from us)
- [ ] `results/modelB.json` produced: binary at val-tuned threshold + retrieval metrics, Overall / Same / Cross

## Phase 6 — Model C: Contrastive fine-tuning (Colab T4)

- [ ] `src/dataset.py` — anchor/positive/negative triples straight from subset TSVs, no normalization
- [ ] `losses.py` — MultipleNegativesRankingLoss (in-batch + explicit negatives mix)
- [ ] `training/train_contrastive.py` — `sentence-transformers`, batch size capped for T4 15 GB, checkpointing after each epoch
- [ ] Train run (Model C: positives + in-batch negatives only) → `modelC/model.safetensors` downloaded locally
- [ ] Local eval on RTX 4060 → `results/modelC.json` (same report shape as Model B)

## Phase 7 — Model D: Contrastive + hard negatives (Colab T4)

- [ ] Same trainer, `--hard-negatives` flag: anchors batched with stored hard negatives alongside in-batch
- [ ] Train → download → local eval → `results/modelD.json`

## Phase 8 — Full results table + cross-script/per-script eval

- [ ] `baselines/report.py` — merges all `results/*.json` (and `*_scores.parquet` re-runs at a fixed comparison threshold) into the central table with Overall / Same / Cross F0.5 columns
- [ ] Per-script table (Devanagari/Bengali/Gujarati/Gurmukhi/Kannada/Malayalam/Oriya/Tamil/Telugu): n pairs × {TF-IDF, E5, Contrastive E5}
- [ ] `test_cross_script.tsv` dedicated eval run for every model

## Phase 9 — Ablations

- [ ] Ablation 1: pretrained E5 vs fine-tuned (B vs C)
- [ ] Ablation 2: fine-tune vs fine-tune + hard negatives (C vs D)
- [ ] Ablation 3: same-script vs cross-script slices (already built into evaluate.py — just tabulated)
- [ ] Written into the table + short prose in the report

## Phase 10 — Address experiment (optional, after name-only works)

- [ ] Experiment A vs B: name-only vs `Business name: ...\nAddress: ...` raw concatenation, same trainer, no handcrafted address features
- [ ] Scores + report in the same shape

## Phase 11 — Report

- [ ] Populate `Documentation_template.md` (or an academic copy) with methodology + Phase 8/9 tables + error analysis (sampled FP/FN per slice) + per-script findings
- [ ] README.MD in `code/business_entity_resolution/` filled (currently empty)
- [ ] Reproduction notes: how to re-run subset build → validate → baselines → Colab → local eval

---
---

## Explicitly excluded (handoff §19)

LightGBM, handcrafted features, old normalization pipeline, FAISS infra, 10M candidate
generation, leaderboard pipeline, DoRA — none of these get built.

## Notes

- Row-order dependence: any re-scoring must use the same `load_split_pairs` order
  (train dedup on name-pairs happens in `tfidf.py` — val/test are untouched).
- `subset_staging.duckdb` (4.6 GB) is deletable; everything reproducible from source TSVs.
- Cross-script definition at eval time = set-of-Unicode-script difference between raw S1
  name and raw candidate name (evaluates transliteration only, not Latin-noise variants).
