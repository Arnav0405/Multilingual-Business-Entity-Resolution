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

- [x] Existing venv (`code/business_entity_resolution/.venv`, Python 3.13) has torch 2.14.1+cu130, duckdb, pandas, numpy, tqdm, pyarrow (`python -c "import torch; print(torch.__version__, torch.cuda.is_available())"` → `2.14.1+cu130 True`)
- [x] Add missing deps: scikit-learn 1.9.1, scipy 1.18.1, transformers 5.18.0, sentencepiece, nbformat, pytest (via `uv pip install --python code/business_entity_resolution/.venv/Scripts/python.exe`)
- [x] `intfloat/multilingual-e5-small` downloaded + smoke test (`scripts/env_smoke.py` → `results/env_smoke.json`): 117.7M params, CLS+prefix scheme; pretrained cosines — cross-script 0.945/0.965 but hard negative 0.974 (pretrained encoder cannot discriminate — this is what fine-tuning must fix)
- [x] Colab T4 noted as the training/compute target for Models B/C/D notebooks
- [x] Phases 0–5 committed to git per phase

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

- [x] `training/notebooks/entity_resolution_colab.ipynb` written: shared header (data upload/mount, score-book helper), then flag-gated phases B / C / D (`training/build_notebook.py` is the generator; nbformat-validated, all cells compile)
- [x] Model B run: encode all S1 + candidate names with `intfloat/multilingual-e5-small` (fp16, `query:`/`passage:` prefixes), cosine
- [x] Scores exported as `modelB_scores.parquet` → local `baselines/report.py` evaluation (Colab gets no raw data download back from us)
- [x] `results/modelB.json` produced: binary at val-tuned threshold (0.941) + retrieval metrics, Overall / Same / Cross
- Phase 5 headline: pretrained E5 with no fine-tuning lifts cross-script recall 2–5% (TF-IDF) → **86.8%**, overall F0.5 0.963 vs 0.929 (best TF-IDF); per-script recall shows Oriya/Tamil near-zero (check n before interpreting), Devanagari/Telugu/Malayalam 0.86–0.91

## Phase 6 — Model C: Contrastive fine-tuning (Colab T4)

- [x] `src/dataset.py` — anchor/positive/negative triples straight from subset TSVs, no normalization
- [x] `losses.py` — MultipleNegativesRankingLoss (in-batch + explicit negatives mix)
- [x] `training/train_contrastive.py` — `sentence-transformers`, batch size capped for T4 15 GB, checkpointing after each epoch
- [x] Train run (Model C: positives + in-batch negatives only) → `modelC/model.safetensors` downloaded locally (attempt 2, post-divergence-fix, 0 NaN)
- [x] Local eval on RTX 4060 → `results/modelC.json` (same report shape as Model B)

**Attempt 1 FAILED (do not reuse `results/modelC_epoch3/`)**: Colab phase C trained fp16 master weights directly with AdamW; loss hit inf/NaN in epoch 1 and NN weights came back all-NaN. Fixed: fp32 params + `torch.autocast` on encoder forwards only, NaN-guard on optimizer step, finite-loss assert after first 50 batches. Attempt 2 succeeded.

**Phase 6 headline**: overall F0.5 **0.9896** (was 0.9632), cross-script F0.5 **0.9946** with recall **99.1%** (precision 99.5% at val-tuned thr 0.702). Recall@1/5/10 = 1.0, MRR = 1.0. Per-script (n in parentheses): Bengali 1.0 (22), Tamil 1.0 (14), Gurmukhi 1.0 (5), Gujarati 1.0 (23), Kannada 1.0 (19), Devanagari 0.997 (763), Telugu 0.913 (23), Malayalam 0.857 (14), Oriya 0.0 (n=1). Contrastive fine-tune gains are largest exactly where the research question lives: cross-script.

## Phase 7 — Model D: Contrastive + hard negatives (Colab T4)

- [x] Same trainer, `--hard-negatives` flag: anchors batched with stored hard negatives alongside in-batch
- [x] Train → download → local eval → `results/modelD.json`
- First attempt crashed in collate (`view()` on ragged batch): hard negatives were collected per GROUP but anchors flattened per POSITIVE ROW — A=124 didn't divide h. Fixed: per-anchor-row alignment, exactly K=4 per row, batch-pool fill with replacement when needed. Committed 3c9b0b2.

**Phase 7 headline (hard negatives effect, Ablation 2)**: overall F0.5 0.9895 (vs C 0.9896 — tie), cross-script 0.9946 both. Gains are surgical, not global: Devanagari 0.997 → 1.0 (763 pairs), Tamil/Malayalam unchanged, overall +1 FN trade for +5 FP on cross-script (~equal F0.5), val-tuned threshold barely moved (0.7021 → 0.7011). Interpretation: with in-batch MNRL at batch 64, random same-batch entities already supply most discrimination pressure; stored hard negatives polish the residual (Devanagari tail) without shifting the operating point. Ablation 2's honest answer on this subset: hard negatives ≈ no significant difference, marginal per-script regularization.

## Phase 8 — Full results table + cross-script/per-script eval

- [x] `baselines/report.py` — merges all `results/*.json` (and `*_scores.parquet` re-runs at a fixed comparison threshold) into the central table with Overall / Same / Cross F0.5 columns
- [x] Per-script table (Devanagari/Bengali/Gujarati/Gurmukhi/Kannada/Malayalam/Oriya/Tamil/Telugu): n pairs × {TF-IDF, E5, Contrastive E5} — `results/phase8_tables.md` (all 6 models × 9 scripts)
- [x] `test_cross_script.tsv` dedicated eval run for every model — `results/cross_script_table.json`

## Phase 9 — Ablations

- [x] Ablation 1: pretrained E5 vs fine-tuned (B vs C) — cross-script F0.5 +0.063, recall +0.123
- [x] Ablation 2: fine-tune vs fine-tune + hard negatives (C vs D) — aggregate tie; Devanagari+Telugu tails cleaned (+6 FN → +1 FP)
- [x] Ablation 3: same-script vs cross-script slices — gap 0.86 (TF-IDF) → 0.04 (B) → 0.00 (C/D)
- [x] Written into the table + short prose in the report — `results/phase9_ablations.md`

## Phase 10 — Address experiment (optional, after name-only works)

- [x] Experiment A vs B: name-only vs `Business name: ...\nAddress: ...` raw concatenation, same trainer, no handcrafted address features — `PHASE="ADDR"` in the notebook (MAX_LENGTH 128, batch 32), NaN-address fix in the scoring path (commit 5c9aa10)
- [x] Scores + report in the same shape — `results/phase10_address_experiment.json` (+ `modelA_name_address_scores.parquet`)

**Phase 10 headline (A′ = D trainer + address concat)**: overall F0.5 **0.9969** vs D's 0.9895 — largest single jump since B→C. Controlled verdict (same pairs, per-model val thresholds):
- **addr_both slice (n_pos=4285/4304, the honest control)**: A′ 0.9973 vs D 0.9893 — address context genuinely helps where addresses exist (precision 0.9993, 8 FPs→0)
- **addr_missing slice (n_pos=173/180)**: A′ 0.9882 vs D 0.9954 — *hurts* when addresses are empty (the forced `Address: ` empty marker is a slightly misleading input)
- cross-script: A′ 0.9993 (0 FP, 3 FN) vs D 0.9946
- **net research answer**: yes — contextual address info improves multilingual entity resolution, worth +0.0074 overall F0.5, provided addresses are present; the model learns to use them from raw concat alone with zero handcrafted features. For deployment/thresholding this means: name-only model for sparse-record lookups, A′ where address fields are populated.

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
  (train dedupe on name-pairs happens in `tfidf.py` — val/test are untouched).
- `subset_staging.duckdb` (4.6 GB) is deletable; everything reproducible from source TSVs.
- Cross-script definition at eval time = set-of-Unicode-script difference between raw S1
  name and raw candidate name (evaluates transliteration only, not Latin-noise variants).

---

## Central results table (final, all through one shared eval path)

Test split, 19,558 pairs, 884 cross-script positives. Threshold = each model's val-tuned F0.5 optimum, applied unchanged.

| Model | Overall F0.5 | Same-Script F0.5 | Cross-Script F0.5 | Cross Recall | Recall@1 | MRR |
|---|---:|---:|---:|---:|---:|---:|
| tfidf_char | 0.9292 | 0.9745 | 0.1132 | 2.5% | 0.9397 | 0.9493 |
| tfidf_word | 0.8967 | 0.9411 | 0.2052 | 5.0% | 0.9389 | 0.9540 |
| tfidf_both | 0.9217 | 0.9656 | 0.1779 | 4.2% | 0.9405 | 0.9499 |
| modelB (E5 pretrained) | 0.9634 | 0.9714 | 0.9317 | 86.8% | 0.9992 | 0.9996 |
| modelC (+contrastive) | 0.9896 | 0.9883 | 0.9946 | 99.1% | 1.0000 | 1.0000 |
| modelD (+hard negatives) | 0.9895 | 0.9883 | 0.9946 | 99.5% | 1.0000 | 1.0000 |
| **A′ (+address concat)** | **0.9969** | — | **0.9993** | 99.7% (0 FP) | 1.0000 | 1.0000 |

Per-script cross-script recall (n in parentheses): Devanagari 1.00 (763, D/A′), Telugu 1.00 (23, D), Bengali 1.00 (22), Tamil 1.00 (14), Gurmukhi 1.00 (5), Gujarati 0.96 (23), Kannada 1.00 (19), Malayalam 0.86 (14), Oriya 0.00 (**n=1** — anecdotal).
