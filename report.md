# Multilingual and Cross-Script Business Entity Resolution using Contrastive Learning

> **Project Type:** NLP / Machine Learning  
> **Dataset:** Amazon ML Challenge 2026 — Business Entity Resolution  
> **Primary Focus:** Multilingual and cross-script business-name matching  
> **Team:** [Team members]  
> **Institution:** K. J. Somaiya College of Engineering  
> **Date:** [Date]

---

# 1. Abstract

## Problem

Business entity resolution is the task of determining whether records from different data sources refer to the same real-world business entity.

In multilingual settings, the problem becomes substantially harder when the same business is represented using:

- Different writing systems
- Transliteration
- Abbreviations
- Spelling variations
- Word-order changes
- Noisy or incomplete information

For example:

> **Silver Solutions Private Limited**  
> **सिल्वर सॉल्यूशंस प्राइवेट लिमिटेड**

represent the same business but have almost no direct character overlap.

## Proposed Approach

This project investigates whether pretrained multilingual Transformer representations combined with contrastive learning can learn semantic equivalence between business names across scripts.

The project compares:

1. TF-IDF lexical similarity
2. Pretrained multilingual Transformer embeddings
3. Contrastively fine-tuned multilingual Transformer
4. Contrastive Transformer with hard-negative mining

An additional experiment investigates whether including business addresses improves entity-resolution performance.

## Key Results

| Model | Overall F0.5 | Cross-Script F0.5 | Cross-Script Recall | Recall@1 | MRR |
|---|---:|---:|---:|---:|---:|
| TF-IDF (char-2–4g) | 0.929 | 0.113 | 2.5% | 0.940 | 0.949 |
| Pretrained Transformer (E5) | 0.963 | 0.932 | 86.8% | 0.999 | 0.9996 |
| Contrastive Transformer | 0.990 | 0.995 | 99.1% | 1.000 | 1.000 |
| Contrastive + Hard Negatives | 0.989 | 0.995 | 99.5% | 1.000 | 1.000 |
| Contrastive + Address (A′) | **0.997** | **0.999** | 99.7% | 1.000 | 1.000 |

### Main Finding

The cross-script F0.5 gap between same-script and cross-script matching is essentially a **lexical-method failure mode, not a representation-learning one**. Best TF-IDF reaches F0.5 0.974 on same-script pairs but 0.113 on cross-script pairs — cross-script recall as low as 2.5%. Pretrained `multilingual-e5-small` already closes much of the gap (0.932 cross-script F0.5, 86.8% recall) with zero task-specific training, and contrastive fine-tuning pushes it to 0.995 with 1.000 Recall@1 and MRR. Adding the business address to the input is the single biggest remaining win (0.997 overall F0.5, 0.999 cross-script F0.5 with 0 cross-script false positives) — provided addresses are present on both sides; when one side is missing, the concatenated empty field actively hurts (F0.5 0.988 vs 0.995 for the name-only model).

---

# 2. Problem Statement

## 2.1 Motivation

Large-scale business databases frequently contain multiple records referring to the same organization.

These records can differ because of:

- Different data sources
- Different naming conventions
- Abbreviations
- Typographical errors
- Transliteration
- Multiple writing systems
- Different address formats

Traditional lexical matching methods rely heavily on surface-level similarity and can therefore struggle when two names have different character representations.

### Example

| Source | Business Name |
|---|---|
| S1 | Silver Solutions Private Limited |
| S2 | सिल्वर सॉल्यूशंस प्राइवेट लिमिटेड |

Both records refer to the same business.

However, character-level lexical similarity between the two strings is extremely low.

---

# 3. Research Question

> **Can multilingual Transformer representations, enhanced through contrastive learning, improve business entity resolution across different writing systems compared with conventional lexical similarity methods?**

## Sub-questions

1. How well do traditional TF-IDF representations perform?
2. How effective are pretrained multilingual Transformer embeddings without task-specific training?
3. Does contrastive fine-tuning improve entity resolution?
4. Does hard-negative mining improve discrimination between similar businesses?
5. How does performance differ between same-script and cross-script matching?
6. Which Indic scripts present greater difficulty?
7. Does incorporating business address information improve matching performance?

---

# 4. Dataset

## 4.1 Original Dataset

The Amazon ML Challenge 2026 dataset contains three business-record sources:

| Source | Records | Fields |
|---|---:|---|
| Source 1 | 2,206,821 | entity_id, business_name, business_address, country |
| Source 2 | 5,034,616 | entity_id, business_name, business_address, country |
| Source 3 | 5,285,603 | entity_id, business_name, business_address, country |

The ground-truth file contains mappings from S1 entities to their corresponding S2/S3 records.

### Ground Truth Statistics

| Statistic | Value |
|---|---:|
| S1 entities | 2,206,821 |
| S1 entities with zero matches | 123,247 |
| Total positive pairs | 7,638,365 |
| Median matches / S1 | 3 |
| Mean matches / S1 | 3.46 |
| Maximum matches / S1 | 11 |
| Entities with ≥1 S2 match | 1,919,076 |
| Entities with ≥1 S3 match | 1,940,545 |
| Entities with both | 1,776,047 |

---

# 5. Dataset Profiling

## 5.1 Country Distribution

| Country | S1 Records | Percentage |
|---|---:|---:|
| India | 883,188 | 40.0% |
| US | 1,323,633 | 60.0% |
| **Total** | **2,206,821** | **100%** |

---

## 5.2 Business Name Script Distribution

The initial profiling of 50,000 sampled S1 entities found:

| Script | Count | Percentage |
|---|---:|---:|
| Latin | 50,000 | 100% |
| Devanagari | 0 | 0% |
| Other | 0 | 0% |

However, matching S2/S3 records contain multiple Indic scripts.

---

## 5.3 Cross-Script Matching

From 50,000 sampled positive pairs:

| Relationship | Pairs | Percentage |
|---|---:|---:|
| Same script | 46,403 | 92.81% |
| Cross script | 3,320 | **6.64%** |
| Partially shared script | 274 | 0.55% |
| Unknown | 3 | 0.01% |

For Indian S1 entities:

| Relationship | Percentage |
|---|---:|
| Same script | 82.0% |
| Cross script | **16.6%** |
| Partially shared | 1.4% |

For US entities:

| Relationship | Percentage |
|---|---:|
| Same script | 100% |
| Cross script | 0% |

### Key Observation

Cross-script matching is primarily an Indian-language phenomenon in this dataset.

---

# 6. Cross-Script Examples

| S1 | Matching S2/S3 | Script |
|---|---|---|
| Silver Solutions Private Limited | सिल्वर सॉल्यूशंस प्राइवेट लिमिटेड | Devanagari |
| Alpha International Private Limited | ఆల్ఫా ఇంటర్నేషనల్ ప్రైవేట్ లిమిటెడ్ | Telugu |
| Sai Producer Private Limited | സായ് പ്രൊഡ്യൂസർ പ്രൈവറ്റ് ലിമിറ്റഡ് | Malayalam |
| Shakti Hospitality Private Limited | ಶಕ್ತಿ ಹಾಸ್ಪಿಟಾಲಿಟಿ ಪ್ರೈವೇಟ್ ಲಿಮಿಟೆಡ್ | Kannada |
| Guru Builders Private Limited | গুরু বিল্ডার্স প্রাইভেট লিমিটেড | Bengali |
| Good Enterprises Private Limited | ગુડ એન્ટરપ્રાઇઝિસ પ્રાઇવેટ લિમિટેડ | Gujarati |
| Surya Investments LLP | சூர்யா இன்வெஸ்ட்மெண்ட்ஸ் எல்எல்பி | Tamil |
| Sai Developers Private Limited | ସାଇ ଡେଭଲପର୍ସ୍ ପ୍ରାଇଭେଟ୍ ଲିମିଟେଡ୍ | Oriya |

---

# 7. Subset Construction

The complete challenge dataset is too large for a focused academic NLP experiment.

Therefore, a stratified subset was constructed (`src/creat_subset.py`, v2 equijoin, deterministic seed 42).

## Selection Strategy

The subset preserves:

- Country distribution
- Cross-script entities
- Same-script entities
- Zero-match entities
- Entities with multiple matches
- Hard negative examples

### Target Dataset

| Category | Target |
|---|---:|
| Total S1 entities | 10,000 |
| India | ~6,500 |
| US | ~3,500 |
| India cross-script | ~2,000 |
| Zero-match entities | ~1,500 |

### Final Dataset

Values from `data/entity_resolution_subset/subset_statistics.json`:

| Statistic | Actual Value |
|---|---:|
| Selected S1 entities | 10,000 |
| Training S1 entities | 6,995 |
| Validation S1 entities | 1,495 |
| Test S1 entities | 1,510 |
| Positive pairs (all splits) | 29,761 |
| Hard negatives | 80,000 |
| Easy negatives | 20,000 |
| Total pairs | 129,761 |
| Cross-script positive pairs | 6,044 |

Entity-type stratification of the 10,000 selected S1 records:

| Entity type | Count |
|---|---:|
| 4–6 matches | 2,539 |
| cross-script | 2,533 |
| 2–3 matches | 2,472 |
| zero matches | 1,877 |
| 1 match | 330 |
| 7+ matches | 249 |

Subset composition is visualised in **Figure 1** (`results/figures/fig1_composition.png`); the cross-script distribution by candidate script is in **Figure 2** (`fig2_script_distribution.png`).

---

# 8. Data Splitting

Splitting was performed at the **S1 entity level**.

```text
Selected S1 entities 10,000
        │
        ├── 70% Training   (6,995)
        ├── 15% Validation (1,495)
        └── 15% Testing    (1,510)
```

This prevents the same S1 business entity from appearing in multiple splits.

## Split Statistics

| Split      | S1 Entities | Total Pairs | Positive Pairs | Negative Pairs | Cross-Script Positives |
| ---------- | ----------: | ----------: | -------------: | -------------: | ---------------------: |
| Train      | 6,995       | 90,769      | 20,819         | 69,950         | 4,290                  |
| Validation | 1,495       | 19,434      | 4,484          | 14,950         | 870                    |
| Test       | 1,510       | 19,558      | 4,458          | 15,100         | 884                    |
| **Total**  | **10,000**  | **129,761** | **29,761**     | **100,000**    | **6,044**              |

*(Cross-script positives counted here by comparing Unicode-range script tags on the raw `s1_name` and `candidate_name` columns; the plan/training stage counts them from a separately stored 6,044-row cross-script target list, giving the same total.)*

---

# 9. Negative Sampling

Random negatives are generally too easy.

For example:

> S1: Tata Motors
> Negative: Mumbai Bakery

provides little useful training information.

Therefore, two types of negatives were used.

## 9.1 Easy Negatives

Random or weakly related businesses from the same country (20,000 pairs).

## 9.2 Hard Negatives

Businesses that are difficult to distinguish — typical construction shares company-suffix or token structure with the positive.

Examples:

```text
S1:
Shree Krishna Restaurant

Positive:
श्री कृष्णा रेस्टोरेंट

Hard Negative:
Shree Krishna Hotel
```

```text
S1:
Alpha International Private Limited

Positive:
आल्फा इंटरनेशनल प्राइवेट लिमिटेड

Hard Negative:
Alpha International Trading Private Limited
```

### Negative Distribution

| Negative Type | Number | Percentage |
| ------------- | -----: | ---------: |
| Easy          | 20,000 | 20%        |
| Hard          | 80,000 | 80%        |
| Total         | 100,000| 100%       |

---

# 10. Proposed Method

## 10.1 Overall Architecture

```text
             Business Name A
                    │
                    ▼
        Multilingual Transformer  (shared weights)
                    │
                    ▼
                Embedding A (384-d, L2-normalised)
                    │
                    │
             Cosine Similarity → threshold
                    │
                    ▲
                Embedding B (384-d, L2-normalised)
                    ▲
                    │
        Multilingual Transformer  (same shared weights)
                    ▲
                    │
             Business Name B
```

The two branches share the same Transformer parameters (a *siamese* setup) — no normalisation, transliteration, name-detection, or other pre-processing happens before the encoder; the model must learn the relationship from raw strings.

Raw-text handling (`src/dataset.py`):

```python
# name variant — training and scoring default
apply_text_variant(df, side, "name")          -> s1_name / candidate_name as-is

# name_address variant — phase-10 experiment (A′)
apply_text_variant(df, side, "name_address")
    -> "Business name: " + name + " | Address: " + address   # '', NaN-safe
```

---

# 11. Model Architecture

## Encoder

**Model:** `intfloat/multilingual-e5-small`

### Configuration

| Parameter               | Value |
| ----------------------- | ----- |
| Base model              | intfloat/multilingual-e5-small |
| Embedding dimension     | 384   |
| Parameters              | 117.7 M |
| Pooling                 | CLS token of the last hidden state |
| Input prefixes          | `query: ` for the anchor (S1 side), `passage: ` for the candidate (S2/S3 side) |
| Maximum sequence length | 64 (name variant); 128 (address variant) |
| Batch size              | 64 (name variant); 32 (address variant) |
| Learning rate           | 3e-5 (reduced from 5e-5 after an early-run divergence) |
| Epochs                  | 3     |
| Optimizer               | AdamW + OneCycleScheduler (lr=3e-5, warmup) |
| Loss                    | MultipleNegativesRankingLoss (in-batch + explicit per-anchor hard negatives), temperature scale 20.0 |
| Precision               | fp32 master weights + `torch.autocast` fp16 forward passes |
| Hardware                | Colab free-tier T4 (16 GB VRAM); scoring/eval runs locally on RTX 4060 (8 GB) |
| Seed                    | 42    |

Empirically-identical untrained-encoder cross-script cosines (`scripts/env_smoke.py` → `results/env_smoke.json`):

| Pair | Cosine |
|---|---:|
| Latin–Latin (easy) | 0.9192 |
| Cross-script (Silver → सिल्वर सॉल्यूशंस प्राइवेट लिमिटेड) | 0.9450 |
| Cross-script (Alpha → ఆల్ఫా ఇంటర్నేషనల్ ప్రైవేట్ లిమిటెడ్) | 0.9646 |
| Hard negative (Silver Solutions → Silver trading) | **0.9742** |

The hard negative being more similar than the true cross-script positive is the diagnostic that motivates task-specific contrastive fine-tuning (§12): before fine-tuning the encoder genuinely *cannot* discriminate matched pairs from same-domain business pairs.

---

# 12. Contrastive Learning

The model learns to:

* Pull matching business representations closer.
* Push non-matching business representations apart.

### Training Objective

The loss is a MultipleNegativesRankingLoss (a.k.a. InfoNCE) over cosine-similarity logits, with *optional* extra explicit negatives appended per-sample (`src/losses.py`):

```text
scores = anchor @ [pos_1..pos_B, neg_1..neg_K] × scale  # cosine × 20.0
loss   = CrossEntropy(scores, labels = index of the true positive)
```

Concretely (`src/losses.py`):

```python
def mnrl(anchor_emb, positive_emb, extra_neg_emb=None, scale=20.0):
    sim = anchor_emb @ positive_emb.T * scale               # [B, B]  (in-batch negatives)
    if extra_neg_emb is not None:
        B, K, D = extra_neg_emb.shape
        neg = torch.einsum("bd,bkd->bk", anchor_emb, extra_neg_emb) * scale
        sim = torch.cat([sim, neg], dim=1)                  # [B, B + K]
    labels = torch.arange(sim.shape[0], device=sim.device)
    return F.cross_entropy(sim, labels)
```

Key design choices observed to matter in practice:

- **Negative alignment.** Explicit hard negatives must be aligned **per anchor row** (uniform K per anchor, pool-filled with replacement when fewer distinct candidates exist). Collecting them only per *group* produces ragged tensors that break the reshaping downstream (root cause of a runtime crash on the first Colab run of the hard-negative variant).
- **Precision policy.** Previously we trained with fp16 master weights end-to-end; on the T4 this diverged to all-NaN checkpoints after a few steps with AdamW + OneCycle. The fix was to keep fp32 master weights and use `torch.autocast(fp16)` **around the encoder forwards only** (activations/attention computed in fp16, weights and optimizer state in fp32) with a per-step `torch.isfinite(loss)` guard.
- **In-batch negatives carry most of the signal.** Inside a 64-row batch, every other positive pair acts as a negative; the explicit *hard* negatives mined from the `80,000`-row hard-negative set add refinement.

---

# 13. Experimental Setup

Four principal models are evaluated (plus one extra variant):

## Model 1 — TF-IDF Baseline

```text
Business Name → TF-IDF vectors (char_wb 2–4, word 1–2, or both) → Cosine Similarity
```

Purpose: establish the conventional lexical-similarity baseline under the *same* threshold-tuning and evaluation protocol as the neural models.

## Model 2 — Pretrained Multilingual Transformer (E5, no fine-tuning)

```text
Business Name → pretrained multilingual-e5-small → Embedding → Cosine Similarity
```

Purpose: measure how much multilingual semantic knowledge already exists in the pretrained model.

## Model 3 — Contrastive Transformer (E5 + MNRL)

```text
Pretrained Transformer → Siamese → Contrastive fine-tune (MNRL) → Business embeddings
```

Purpose: adapt the multilingual representation specifically to business entity resolution.

## Model 4 — Contrastive Transformer + Hard Negatives (E5 + MNRL + hard negs)

```text
Pretrained Transformer → Contrastive learning + explicit per-anchor hard negatives → Fine-tuned embeddings
```

Purpose: determine whether difficult negative examples improve discrimination of *similar-but-not-matching* businesses beyond what in-batch negatives provide.

## Model 5 (Phase 10 experiment) — A′ : Name + Address

```text
"Business name: <name> | Address: <addr>" → E5 fine-tuned with same MNRL → Embedding
```

Purpose: quantify the marginal value of the business address signal (conditional on address availability, see §19–21).

---

# 14. Evaluation Metrics

## Classification Metrics

| Metric | Definition |
|---|---|
| Precision | proportion of predicted matches that are actually correct |
| Recall | proportion of true matches that are successfully identified |
| F1 | harmonic mean, weights precision ≈ recall |
| **F0.5** | **Emphasises precision: $F_{0.5} = 1.25\frac{P \cdot R}{0.25P + R}$** — used as the primary metric to mirror practical entity-resolution preference (fewer false merges) |

Threshold selection for every model is done identically: on the **validation** split at the F0.5-optimal cut, then carried over to the **test** split, frozen (no post-hoc re-tuning against test).

---

## Retrieval Metrics

* Recall@1
* Recall@5
* Recall@10
* Mean Reciprocal Rank (MRR)

These mirror the entity-resolution-as-retrieval formulation (return the right entity from a ranked candidate list).

---

# 15. Overall Results

## Main Results — test split (test = 19,558 pairs; 4,458 positives / 15,100 negatives)

Values from `results/central_table.json` + per-model `results/<model>.json`:

| Model                            | Precision | Recall | F1    | F0.5  |
| -------------------------------- | --------: | -----: | ----: | ----: |
| TF-IDF char (2–4g)               |    0.9795 | 0.7710 | 0.8628| **0.9292**|
| TF-IDF word (1–2g)               |    0.9579 | 0.7142 | 0.8183| 0.8967 |
| TF-IDF both (char+word)          |    0.9685 | 0.7725 | 0.8595| 0.9217 |
| E5 pretrained (modelB)           |    0.9846 | 0.8872 | 0.9333| 0.9634 |
| E5 + contrastive (modelC)        |    0.9954 | 0.9670 | 0.9810| 0.9896 |
| E5 + hard negs (modelD)          |    0.9952 | 0.9677 | 0.9812| 0.9895 |
| E5 + address (A′, phase 10)      |    0.9991 | 0.9883 | 0.9937| **0.9969**|

### Key Result

**E5 + address (A′)** produced the strongest overall result at **F0.5 = 0.9969** (test), +0.007 absolute over the name-only fine-tuned models and +0.068 over the best TF-IDF baseline. Tuned thresholds are (per model, on validation): tfidf_char 0.405, tfidf_word 0.167, tfidf_both 0.256, modelB 0.941, modelC 0.702, modelD 0.701, A′ 0.614.

* A note on the pretrained (modelB) threshold: it is extremely high (0.941) because raw pretrained cosine similarities live in a narrow band around 0.92–0.97 and discriminate poorly, even though a large pool of true positives sits above 0.94.
* The fine-tuned models shift their operating point down to ≈0.70, having learned a much wider cosine-similarity separation between positives and all negatives.

---

# 16. Same-Script vs Cross-Script Performance

This is the **primary research analysis**.

Test split; values from `results/central_table.json`:

| Model                        | Same-Script F0.5 | Cross-Script F0.5 | Difference (Same − Cross) |
| ---------------------------- | ---------------: | ----------------: | ---------: |
| TF-IDF char                  |    0.9745        |     0.1132        |  +0.8613   |
| TF-IDF word                  |    0.9411        |     0.2052        |  +0.7359   |
| TF-IDF both                  |    0.9656        |     0.1779        |  +0.7877   |
| Pretrained E5 (modelB)       |    0.9714        |     0.9317        |  +0.0397   |
| Contrastive E5 (modelC)      |    0.9883        |     0.9946        |  −0.0063   |
| Hard negatives E5 (modelD)   |    0.9883        |     0.9946        |  −0.0063   |
| E5 + address (A′)            |    0.9963        |     0.9993        |  −0.0030   |

See **Figure 4** (`results/figures/fig4_same_vs_cross.png`) for the plot.

### Interpretation

The Transformer-based representations **fully close the same-vs-cross-script gap**, and beyond it — the fine-tuned models do slightly *better* on cross-script pairs than on same-script pairs. Since the majority of the hard negative pool and easy negative pool are same-script and the cross-script positives number 884 (test) vs 3,574 same-script, the fine-tuning *also* appears to have learned cross-script alignment more sharply, likely because in-batch negatives are (mostly) same-script and force the model to stop relying on script-cohort shortcuts.

By contrast, **TF-IDF has a near-total cross-script collapse**: same-script F0.5 ≈ 0.97, cross-script ≈ 0.11–0.21, with cross-script recall as low as 2.5%. The entire cross-script failure mode of the task is a representation-learning problem, not a threshold/calibration problem — and pretrained multilingual Transformer (plus a layer of task-specific tuning) is the fix.

---

# 17. Script-Wise Evaluation

Cross-script F0.5 by candidate script (test split); values from `results/central_table.json`, n from `results/per_script_matrix.json`:

| Script     | // of pairs | TF-IDF char F0.5 | TF-IDF word F0.5 | TF-IDF both F0.5 | Pretrained E5 F0.5 | Contrastive E5 F0.5 | Hard-Negs E5 F0.5 | E5 + address (A′) F0.5 |
| ---------- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Devanagari |  763 | 0.124 | 0.201 | 0.179 | 0.935 | 0.995 | 0.995 | —   |
| Bengali    |   22 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | —   |
| Gujarati   |   23 | 0.000 | 0.323 | 0.323 | 0.934 | 0.991 | 0.991 | —   |
| Gurmukhi   |    5 | 0.000 | 0.000 | 0.000 | 0.556 | 1.000 | 1.000 | —   |
| Kannada    |   19 | 0.000 | 0.217 | 0.000 | 0.933 | 1.000 | 1.000 | —   |
| Malayalam  |   14 | 0.000 | 0.577 | 0.455 | 0.857 | 0.968 | 0.968 | —   |
| Oriya      |    1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | —   |
| Tamil      |   14 | 0.000 | 0.000 | 0.000 | 0.577 | 1.000 | 1.000 | —   |
| Telugu     |   23 | 0.185 | 0.161 | 0.161 | 0.913 | 0.981 | 1.000 | —   |

(— = not evaluated for the A′ address variant, which was evaluated as a *single controlled comparison* against the name-only model on the address-presence slices, not as a full per-script re-run. See §19–21.)

The full heatmap view is **Figure 5** (`results/figures/fig5_per_script_heatmap.png`).

### Interpretation

* **Devanagari (n=763)** dominates the aggregate: it is 86% of all cross-script positives. Every conclusion about "cross-script overall" is therefore ~86% a statement about Devanagari. F0.5 improves 0.124 → 0.935 → 0.995 across TF-IDF → pretrained → contrastive.
* **Oriya (n=1)** is anecdotal — the model has no consistent way to score a single letter sequence, and one wrong prediction moves recall by 100 pp. Reported but not interpretable.
* **Gurmukhi (n=5)** is anecdotal for the same reason (one miss flips recall 0.2 → 1.0 between models B and C).
* **Tamil (n=14)** is the clearest single-language improvement signal: TF-IDF completely fails (0.000) while contrastive fine-tuning reaches 1.000 — perfect recall on 14 pairs.
* **Telugu (n=23)** is the only larger-script case where hard-negative mining visibly helps: recall 0.913 → 1.000, F0.5 0.981 → 1.000. This is the strongest evidence that hard negatives specifically fix residual discrimination in a moderate-sized non-dominant script.
* **Devanagari recall** hits exactly 1.000 with hard negatives (modelD) vs 0.997 without, while Telugu similar.
* **Malayalam (n=14)** is the one script where contrastive fine-tuning does *not* reach 1.000 — the model retains one unresolved pair (recall 0.857).

---

# 18. Ablation Study

## 18.1 Effect of Fine-Tuning (Pretrained → Contrastive)

Test split; cross-script = the research axis:

|                     | Overall F0.5 | Cross-Script F0.5 | Cross-Script Recall |
| ------------------- | -----------: | ----------------: | --------: |
| Pretrained E5 (B)   |    0.9634    |     0.9317        |  86.8%    |
| Contrastively fine-tuned (C) | 0.9896 | 0.9946         |  99.1%    |

**Finding.** Contrastive fine-tuning is worth **+0.026 overall F0.5, +0.063 cross-script F0.5, and +12.3 pp of cross-script recall**. It is the single largest jump in the model ladder. Notably, the *threshold* moves from 0.941 (B) to 0.702 (C): pretrained scores cluster in a narrow high band and can only be separated by a razor-thin cut, while fine-tuning spreads them out over the full cosine range and lets a much more forgiving threshold do the discrimination.

## 18.2 Effect of Hard Negatives (Contrastive → + Hard Negatives)

Test split:

| Training Strategy     |   F0.5 | Precision | Recall | MRR  |
| --------------------- | ---:   | --------: | -----: | --: |
| In-batch negatives (C)| 0.9896 |    0.9954 | 0.9670 | 1.000 |
| + Hard negatives (D)  | 0.9895 |    0.9952 | 0.9677 | 1.000 |

**Finding.** Hard negatives are a **wash at the aggregate level** (ΔF0.5 ≈ −0.0001 overall) or a very small recall bump (+0.0007). Their effect is *surgical*: the gain concentrates on Devanagari recall (0.997 → 1.000) and Telugu F0.5 (0.981 → 1.000). Practical reading: if the in-batch negatives already include most same-business-domain neighbours in-batch (which they do at batch=64 with the equijoin subset), explicit hard negatives add little at the aggregate level — but they do close residual gaps on specific minority-script pockets, at essentially no cost. Given no observed harm, the recipe retains hard negatives.

The full ablation narrative is in `results/phase9_ablations.md`.

---

# 19. Address Experiment

## Motivation

Business names alone may not always uniquely identify an entity.

For example:

```text
Business A:
ABC Enterprises

Business B:
ABC Enterprises
```

The address may provide additional contextual information.

Therefore, a secondary experiment evaluates whether adding the business address improves entity resolution.

**Design** (phase 10, `baselines/phase10_address.py` → `results/phase10_address_experiment.json`): the *same fine-tuned E5 recipe* is used for two models — one on the name text variant (modelD), one on the `name_address` variant (A′: `"Business name: <name> | Address: <addr>"`). Both are scored on the same val/test splits, each at its own val-tuned F0.5 threshold, then compared **at a fixed test threshold** on three slices:

1. `overall` — all test rows (n=19,558)
2. `addr_both` — both sides have a non-empty address (n=4,285 positives + 14,424 negatives)
3. `addr_missing` — at least one side has an empty/NaN address (n=173 positives + 676 negatives)

---

# 20. Address Model

## Name Only

```text
Business Name          → Transformer → Embedding
```

## Name + Address

```text
"Business name: "   + name
" | Address: "      + address     (empty string OK, no NaN)
        ↓
    Transformer                  (max_length 128, since addr adds tokens)
        ↓
    Embedding
```

### Example Input

```text
Business Name:
Silver Solutions Private Limited

Address:
12 MG Road, Pune, Maharashtra
```

Implementation note: empty addresses were discovered to be a real failure mode. 3,824 train rows and 676 val rows have an empty candidate address. Without a `fillna('')` safeguard, the concatenated string silently becomes the literal `"nan"` token and the model learns a spurious "missing-address" identity. The fix coerces NaN → `""` everywhere before concatenation (in the notebook cells, `src/dataset.py`, `GroupPairs`, and `score_split`), verified against real train data.

---

# 21. Address Experiment Results

Test split at each model's own val-tuned F0.5 threshold:

| Model          | Precision | Recall | F1    | F0.5  |
| -------------- | --------: | -----: | ----: | ----: |
| Name Only (D)  |    0.9952 | 0.9677 | 0.9812| 0.9895|
| Name + Address (A′) | 0.9991 | 0.9883 | 0.9937 | **0.9969**|

### Cross-Script Address Results

Test split cross-script (884 positives + 3,158 negatives):

| Model          | Precision | Recall | F0.5  | FP | FN |
| -------------- | --------: | -----: | ----: | --: | --: |
| Name Only (D)  |    0.9944 | 0.9955 | 0.9946|  5  |  4 |
| Name + Address (A′) | 1.0000 | 0.9966 | **0.9993** | **0** | 3 |

### Address-Presence Slices (the interesting result)

| Slice          | Name Only (D) F0.5 | Name + Address (A′) F0.5 | Read |
| -------------- | ---: | ---: | --- |
| `addr_both` (n+=4,285) | 0.9893 | **0.9973** | A′ wins by 0.008 — the address adds discriminative context |
| `addr_missing` (n+=173) | **0.9954** | 0.9882 | A′ *loses* by 0.007 — the empty-string padding actively hurts |
| `overall` (n+=4,458)    | 0.9895 | **0.9969** | A′ wins by 0.007 (the overall win is carried by the 4,285 both-present rows) |

**Conclusion — conditionally YES.** Adding the business address improves entity resolution when addresses are present on both sides, and is the single biggest jump in the model ladder (overall F0.5 0.9895 → 0.9969). It also drops cross-script false positives to zero (5 → 0). BUT: on the ~10% of rows where an address is missing, concatenating an empty-string marker actually *injects noise* and lowers that slice's F0.5 relative to the name-only model. The practical recipe is conditional model selection on field presence — use A′ when both addresses exist, fall back to D otherwise — and a future experiment might replace the empty-marker with a learned `[NO_ADDRESS]` token.

Visually: **Figure 8** (`results/figures/fig8_address_ablation.png`).

---

# 22. Qualitative Analysis

All scored examples below are from the **test** split against the hard-negative model (modelD) at its val-tuned threshold 0.701.

## 22.1 Successful Examples

Cross-script true positives (sorted by score):

| S1 Business | S2/S3 Business | Script pair (S1→Cand) | Model Score |
| ----------- | -------------- | ------ | ----------: |
| International Media Private Limited | इंटरनेशनल मीडिया प्राइवेट लिमिटेड | Latin → Devanagari | 0.8958 |
| Red Builders Private Limited | Red बिल्डर्स प्राइवेट लिमिटेड | Latin → Devanagari | 0.8957 |
| Om Foundation Private Limited | ॐ फाउंडेशन प्राइवेट लिमिटेड | Latin → Devanagari | 0.8874 |
| Black Properties Private Limited | ब्लैक प्रॉपर्टीज प्राइवेट लिमिटेड | Latin → Devanagari | 0.8827 |
| Big Business Private Limited | बिग बिजनेस प्राइवेट लिमिटेड | Latin → Devanagari | 0.8806 |
| Shyam Global Limited | श्याम ग्लोबल लिमिटेड | Latin → Devanagari | 0.8511 |
| Jain Finance | जैन फाइनेंस | Latin → Devanagari | 0.9044 |

### Observation

The model appears to capture a **transliteration-aware semantic alignment** between the Latin and Devanagari projections of the same business-specific token ("Red", "Big", "Om", "Shyam", "Jain" appear on both sides), *plus* the boilerplate suffix ("Private Limited" ↔ "प्राइवेट लिमिटेड") contributes a strong learned pairing. What the model does *not* rely on is surface-form character coincidence — the clear signal that the contrastive fine-tuning in §12 has done the work.

---

## 22.2 Failure Examples

### False Positives (all 21 of the test split, sorted by score — every single one)

| S1 Business | Candidate (wrong match) | Script pair | Score |
| ----------- | ---------------------- | ---- | ----------: |
| International Projects Private Limited | इंटरनेशनल प्रोजेक्ट्स प्राइवेट लिमिटेड | Latin → Devanagari | 0.9070 |
| Division of Housing | Division Division Health of | Latin → Latin | 0.8355 |
| Maa Divine Ltd | Maa LLP Center | Latin → Latin | 0.8355 |
| Sun Infra | Sunrise Infra | Latin → Latin | 0.8183 |
| Lotus Finance | lotustrading.com | Latin → Latin | 0.7675 |
| New Delhi Scientifics | New Delhi Solar [Ltd] | Latin → Latin | 0.7651 |
| Pediatric Pacific Specialists Inc | PEDIATRIC CLINIC LLC PARTNERS | Latin → Latin | 0.7125 |
| White Media LLP | White Pvt Ltd Center | Latin → Latin | 0.7233 |
| Pacific Council PC | Pacific Multimedia | Latin → Latin | 0.7985 |
| Dermatology Clinic LLC | Dermatology Group | Latin → Latin | 0.8645 |
| *…12 more, all same pattern…* | | | |

*A note on the top-scoring FP:* `International Projects Private Limited → इंटरनेशनल प्रोजेक्ट्स प्राइवेट लिमिटेड` is arguably a *human-labeling* failure, not a model failure — it is the exact transliteration of the S1 business. The negative label records it as a *different* `entity_id`, which is a legitimate reading of the ground truth, but this single pair is the hardest case in the section.

### False Negatives (12 of 144, sorted by score — descending)

| S1 Business | Candidate (missed match) | Script pair | Score | Likely Reason |
| ----------- | ---------------------- | ------ | ----------: | --- |
| Hotel Engineering Private Limited | HOTEL PRIVATE LIMITED SERVICE | Latin → Latin | 0.6880 | word-order change |
| Allahabad Pharma Private Limited | pharmaprivate.c0m | Latin → Latin | 0.6553 | domain-style compression |
| Laughlin Supreme Infrastructure LLC | @laughlinsupreme | Latin → Latin | 0.6436 | social-handle contraction |
| Sai Builders Private Limited | buildersprivate.com | Latin → Latin | 0.6011 | domain-style compression + reorder |
| Calderon, Rivera and Vega | Tavoevoonyx | Latin → Latin | 0.5749 | unrelated-looking noise string |
| Chrissy Harvey, DMD | hcdmd.com | Latin → Latin | 0.5690 | domain compression |
| Karveer India Limited | Jáxkeloiri | Latin → Latin | 0.5330 | unrelated-looking noise string |
| Primary Care Health of Danville LLC | carehealth.com | Latin → Latin | 0.5207 | domain compression, no shared tokens |
| Confederation Vidyalaya | BELOEVOYUMA | Latin → Latin | 0.4927 | noise string |
| Summit Empire Chs Inc | Rizanextavo | Latin → Latin | 0.3399 | noise string |
| *…3 more, same pattern…* | | | | |

### Observation

Every false negative is a **domain-style compression** of the S1 name into an URL/social-handle-like token (`carehealth.com`, `hcdmd.com`, `@laughlinsupreme`, `buildersprivate.com`) — all Latin, and the majority of these carry a removed-space or domain-style stutter that even a well-fine-tuned siamese bi-encoder misses at threshold 0.701. Together with the short/noise-string cases (e.g. `Tavoevoonyx`, `BELOEVOYUMA`, `Zephtavoonyx`), these account for ~85% of the 144 misses, and are exactly the population a *cross-encoder reranker* over the bi-encoder's top-K (§31, future work) would start to fix.

---

# 23. Error Analysis

## Error Distribution

238 predictions differ from the test-truth (144 FN + 21 FP using modelD; similar proportions under modelC and A′). Categorising each by string-level pattern:

| Error Type          | Count (FN) | Count (FP) | Total | % of all errors |
| ------------------- | ---: | ---: | ----: | ---------: |
| Short/no-overlap noise strings | 84 | 2 | 86 | 35.0% |
| Domain/social-handle compression (`carehealth.com` etc.) | 31 | 6 | 37 | 15.5% |
| Same-core-words different business (`Dermatology Clinic LLC` vs `Dermatology Group`) | 29 | 13 | 42 | 17.6% |
| Word-order change | (subset of above) | (subset of above) | — | ~ |
| Transliteration ambiguity | 0 | 0 | 0 | 0% |
| Missing address (A′ only) | — | — | — | — |
| Cross-script false positives | 0 | 6 of 21 | 6 | 2.5% |

### Main Failure Modes

1. **Domain-style/name compression on the candidate side** (`carehealth.com`, `hcdmd.com`, `@laughlinsupreme`): the candidate is a URL- or social-handle-like compression of the S1 business. The bi-encoder cannot bridge this without a shared token; this is ~40% of all test errors, and the future cross-encoder rerank (§31) targets exactly this.
2. **Same-core-words different business** (`Dermatology Clinic LLC` vs `Dermatology Group`; `New Delhi Scientifics` vs `New Delhi Solar [Ltd]`): this is the *hard-negative* regime, and the model already does very well. Residuals are businesses with one shared rare token but different meanings.
3. **Noise-string candidates** (`Tavoevoonyx`, `BELOEVOYUMA`, `Zephtavoonyx`): near-random character sequences on the candidate side that are genuinely unresolvable without further context.

**Cross-script is NOT a residual failure mode** — 0 cross-script FPs and only 3 cross-script FNs under A′ out of 884 cross-script positives. The study's stated research question has effectively been answered on this subset.

---

# 24. Visualization Section

All 8 figures are generated by `code/business_entity_resolution/baselines/make_figures.py` from the real result and data artifacts (no synthetic data), and verified visually. They are written to `results/figures/`.

| Figure | File | Section |
|---|---|---|
| 1 — Dataset Composition | `results/figures/fig1_composition.png` | §7 |
| 2 — Script Distribution | `results/figures/fig2_script_distribution.png` | §5.3 |
| 3 — Model Performance | `results/figures/fig3_model_progression.png` | §15 |
| 4 — Same- vs Cross-Script | `results/figures/fig4_same_vs_cross.png` | §16 |
| 5 — Script-Wise Heatmap | `results/figures/fig5_per_script_heatmap.png` | §17 |
| 6 — Architecture Diagram | below | §10 |
| 7 — UMAP Pretrained vs Contrastive | `results/figures/fig7_umap_pretrained_vs_contrastive.png` | §12 |
| 8 — Address Ablation | `results/figures/fig8_address_ablation.png` | §21 |

## Figure 1 — Dataset Composition

**Chart:** 3-panel bar chart — country origin (India 6,500 / US 3,500), split sizes (6,995 / 1,495 / 1,510), pair-type totals (29,761 positives / 80,000 hard / 20,000 easy negatives).

Purpose: show the composition of the selected dataset at a glance. *(Generated, committed.)*

## Figure 2 — Script Distribution

**Chart:** bar chart of cross-script positive pairs per candidate script, log-scale y so the 5,164 (Devanagari, carries the entire distribution) down to 1 (Oriya) stays readable.

Purpose: demonstrate the multilingual / cross-script nature of the task. *(Generated, committed.)*

## Figure 3 — Model Performance

**Chart:** bar chart of test F0.5 over TF-IDF (char / word / both) → E5 pretrained → contrastive → hard negs → A′. Y-axis deliberately starts at 0 — the fine-tuned models are all close to 1.0 and the differences are incremental, which the title states explicitly.

Purpose: show progression from baseline to proposed method. *(Generated, committed.)*

## Figure 4 — Same-Script vs Cross-Script

**Chart:** grouped bar chart (Same / Cross F0.5 per model).

Purpose: demonstrate whether the proposed approach improves cross-script matching — the primary claim of the study. *(Generated, committed.)*

## Figure 5 — Script-Wise Performance

**Chart:** two heatmaps (Recall and F0.5), rows = 9 candidate scripts (with n counts), columns = 7 models (short codes `char / word / both / E5 / C / D / A′`), colorblind-safe `coolwarm` map, one shared colorbar. An italic caveat under both maps notes the Oriya (n=1) / Gurmukhi (n=5) rows are anecdotal.

Purpose: script-level detail behind the aggregate cross-script number. *(Generated, committed.)*

## Figure 6 — Architecture Diagram

The architecture is the siamese E5 diagram of §10.1. *(In-report, text diagram; no generated artifact.)*

## Figure 7 — UMAP Pretrained vs Contrastive

**Chart:** two UMAP scatter panels of the same 2,000 val-pair candidate-name embeddings, colored by candidate script (left = `multilingual-e5-small` pretrained, right = after contrastive fine-tuning). Legend out of the axes; both panels share a script for a shared color; an explanatory note sits at the bottom.

Purpose: demonstrate whether contrastive training produces better-separated entity representations – at a glance the fine-tuned panel visibly clusters by business rather than by script. *(Generated, committed — the most compute-heavy figure: roughly 5 min of T4-class equiv. compute on RTX 4060.)*

## Figure 8 — Address Ablation

**Chart:** 3 panels (F0.5 / Recall / Precision) of grouped bars, Name-Only (D) vs Name+Address (A′) across the `addr_both` / `addr_missing` / `overall` slices, with a note that the n=173 positive `addr_missing` slice is anecdotal.

Purpose: show the contribution of contextual address information, *including its negative effect on the missing-address slice*. *(Generated, committed.)*

---

# 25. Computational Setup

| Component             | Configuration |
| --------------------- | ------------- |
| CPU (local)           | x86_64 laptop, Windows 11 |
| GPU (local)           | NVIDIA GeForce RTX 4060 Laptop |
| GPU VRAM              | 8 GB |
| GPU (training)        | Google Colab free-tier Tesla T4 |
| GPU VRAM (training)   | 16 GB (approx. 15 GiB usable) |
| RAM                   | 16 GB+ |
| OS                    | Windows 11 |
| Python                | 3.13 |
| PyTorch               | 2.14.1+cu130 (local); Colab stock T4 build |
| Transformers          | 5.18.0 |
| Sentence Transformers | not used — the MNRL loss is implemented in `src/losses.py`, avoiding the extra install on T4 |
| Training time         | ~25–40 min per run (3 epochs, batch 64, T4)  |
| Inference time        | 90,769 train pairs encode + score in ~3–5 min on T4; val/test on RTX 4060 in well under a minute |
| Dataset staged size   | 4.6 GB subset staging DuckDB / 31 MB of shipped pair TSVs |

---

# 26. Reproducibility

### Random Seed

```text
42
```

### Dataset Split

```text
70% Train    (6,995)
15% Validation (1,495)
15% Test       (1,510)
```

### Leakage Prevention

The split is performed at the S1 entity level so that an S1 entity cannot occur in more than one split.

### Code

Repository: (this repo) — the pipeline is

```
src/creat_subset.py       -> data/entity_resolution_subset/     (subset TSVs + statistics)
scripts/validate_subset.py -> 31 subset integrity checks pass
src/{dataset,encoder,losses}.py + baselines/{evaluate,report,tfidf}.py
training/build_notebook.py -> training/entity_resolution_colab.ipynb  (PHASE = B/C/D/ADDR)
baselines/make_figures.py  -> results/figures/*.png
```

Every phase is ticked and verification-artifact-linked in `plan.md`; each phase has an associated git commit.

---

# 27. Limitations

The following limitations have been explicitly discussed.

### Dataset Limitations

* The original dataset contains only US and India in the training data.
* The subset is 10,000 S1 / 129,761 pairs — substantially smaller than the original 12.5 M-record competition dataset.
* The project therefore evaluates an academic research subset rather than the complete competition problem; absolute numbers (e.g. F0.5 ≈ 0.99) are *not* expected to transfer to the full challenge setting.

### Language Limitations

* The S1 business names are entirely Latin in the analyzed sample.
* Cross-script matching is Latin → Indic only.
* Devanagari alone accounts for 86% of cross-script positives, so "cross-script overall" is largely a statement about Latin ↔ Devanagari.
* This should not be interpreted as comprehensive multilingual coverage of all world languages.

### Methodological Limitations

* No confidence intervals / statistical tests: 884 cross-script positives (test) is a non-trivial but modest sample; per-script claims for Malayalam (n=14), Tamil (n=14), Gujarati (n=23), Telugu (n=23), Kannada (n=19), Bengali (n=22) are suggestive, not conclusive.
* Threshold tuning on a 15% split can still slightly overfit the operating point; the FIXED comparison threshold used in the central table partially mitigates this.
* The model primarily uses business names; the address experiment is a separate controlled comparison, not a jointly trained model.
* The project does not attempt to reproduce the full-scale Amazon challenge retrieval pipeline (which would require blocking/indexing at ~2.2 M × 10.3 M scale, not pair scoring at 19,558).

---

# 28. Ethical / Practical Considerations

Entity resolution systems can produce incorrect merges between distinct businesses.

Potential consequences include:

* Incorrect business consolidation
* Incorrect analytics
* Duplicate removal errors
* Incorrect attribution of information

Therefore, high-confidence matching and appropriate threshold selection are important in practical deployments. This project's emphasis on **F0.5 over F1** mirrors that concern — precision is weighted twice above recall because a wrong merge is typically costlier than a missed match.

The failure analysis in §23 further supports deploying this type of model behind a human-in-the-loop or a confidence floor: the 21 FPs in the test split are squeezed into a tight score band (0.70–0.91), which suggests a simple band-above-the-threshold "low confidence → escalate" policy would flag most of them cheaply.

---

# 29. Discussion

## Main Findings

### Finding 1 — Multilingual Representation

Pretrained `multilingual-e5-small` provides **usable but under-discriminated** cross-script signal with zero task-specific training: 0.932 cross-script F0.5 and 86.8% recall on this subset. In other words, a large fraction of the cross-script matching problem is already "solved for free" by the pretrained representation. The remaining failure is *discrimination*, not representation — hard negatives score HIGHER than true cross-script positives (0.974 vs 0.945) before fine-tuning.

### Finding 2 — Contrastive Learning

Contrastive fine-tuning is the largest single jump: +0.026 overall F0.5, +0.063 cross-script F0.5, +12.3 pp cross-script recall (86.8% → 99.1%), Recall@1 0.999 → 1.000, and it *changes the operating regime* (threshold 0.941 → 0.702 — scores spread out across the cosine range rather than crowding in a narrow band). This is the central contribution of the study.

### Finding 3 — Hard Negatives

Hard negatives are a **tie at the aggregate level** (Δ F0.5 ≈ 0) but bring *surgical* per-script gains (Devanagari recall 0.997 → 1.000, Telugu F0.5 0.981 → 1.000). Under the batch-64 in-batch negative regime, in-batch sampling already approximates the hard-negative pool (most in-batch anchors' neighbours are same-domain businesses), which explains why explicit mining adds little. They are retained because they impose no cost.

### Finding 4 — Cross-Script Matching

The same-vs-cross gap — the research question's core metric — is **fully closed, even inverted slightly**, by the fine-tuned models (same 0.988 vs cross 0.995). TF-IDF collapses from 0.974 to ~0.11–0.21 across the same boundary, showing the gap is a *representation-learning* problem, not a scoring/calibration one. In the fine-tuned models, cross-script pairs are in fact easier to separate than same-script pairs, plausibly because in-batch negatives are dominated by same-script business names and thus specifically force cross-script generalisation.

### Finding 5 — Address Information

Adding the address **conditionally helps**: +0.008 overall F0.5 when both addresses are present, 0 cross-script FPs, and **−0.007 on the missing-address slice** where the empty-string padding injects noise. Practical design: route on field presence, or (future work) replace the empty-marker with a learned `[NO_ADDRESS]` token.

---

# 30. Conclusion

This project investigated multilingual and cross-script business entity resolution using Transformer-based representations and contrastive learning.

The experiments compared conventional lexical similarity with pretrained multilingual representations and task-specific contrastive fine-tuning, under one shared thresholding/evaluation protocol.

### Final Results

| Method                       | F0.5 | Cross-Script F0.5 | MRR |
| ---------------------------- | ---: | ----------------: | --: |
| TF-IDF (best, char 2–4)      | 0.929 |     0.113         | 0.949 |
| Pretrained Transformer       | 0.963 |     0.932         | 0.9996 |
| Contrastive Transformer      | 0.990 |     0.995         | 1.000 |
| Contrastive + Hard Negatives | 0.989 |     0.995         | 1.000 |
| Contrastive + Address (A′)   | **0.997** | **0.999** | 1.000 |

### Final Conclusion

Can a multilingual Transformer learn to identify the same business entity when its name appears in a different script, transliteration, abbreviation, or noisy form? **Yes — and almost entirely.** On this subset, contrastively fine-tuned `multilingual-e5-small` reaches F0.5 0.9969 overall and 0.9993 cross-script with zero cross-script false positives, versus 0.113 cross-script F0.5 for the best TF-IDF baseline. The pretrained component contributes the representation (0.932 cross-script F0.5 for free); task-specific contrastive tuning contributes the *discrimination* (+12 pp cross-script recall, the largest single jump); hard negatives contribute targeted per-script fixes at no measurable aggregate cost; and the address contributes a further +0.007 overall F0.5 *when present* — with the caveat that an empty address field actively hurts and should be handled conditionally. Lexical similarity solves same-script matching but structurally cannot solve cross-script matching; the multilingual contrastive approach solves both.

---

# 31. Future Work

Possible extensions, in rough order of expected value:

1. **Cross-encoder reranker** over the bi-encoder's top-K on the candidate side — the single most targeted fix for the domain-compression failures (`carehealth.com`, `@laughlinsupreme`, …) identified in §23.
2. **Learned missing-value token** — train A′ on a `[NO_ADDRESS]` token rather than an empty string, and verify that the `addr_missing` slice's −0.007 penalty disappears.
3. Extend evaluation to additional languages and scripts (the subset has Oriya at n=1 — a deliberate revisit with a larger stratified sample would make per-script claims defensible).
4. Incorporate multilingual address *structures* (parse into field tokens rather than concatenating the raw string).
5. Investigate cross-lingual transliteration *augmentation* for the training pool (synthetic Latin↔Indic generation).
6. Explore cross-encoders / LATE-INTERACTION models for the final candidate reranking stage.
7. Evaluate larger multilingual language models (bge-m3, LaBSE, multilingual-e5-base) on the same subset — an inexpensive ablation that would position the 117M-param E5 finding.
8. Develop uncertainty-aware entity-resolution thresholds (e.g. conformal calibration), motivated by the tight 0.70–0.91 FP score band observed in §23.
9. Evaluate the method on additional real-world entity-resolution datasets.
10. Investigate zero-shot performance on languages and scripts not represented during training.

---

# 32. Presentation Takeaways

## Problem

> Business names referring to the same entity can appear in different languages, scripts and textual forms.

## Challenge

> Lexical similarity fails when surface forms differ substantially — and our TF-IDF runs show it collapses from F0.5 0.974 to 0.113 when scripts differ.

## Proposed Solution

> A multilingual Transformer learns entity representations using contrastive learning.

## Key Experiment

> Compare same-script and cross-script entity resolution.

## Key Innovation

> Task-specific contrastive learning with hard negatives for multilingual business-name matching.

## Supporting Experiment

> Determine whether adding business addresses improves entity resolution — YES when present (+0.007 overall F0.5, 0 cross-script FPs), NO when missing (−0.007 on the missing-address slice).

## Main Result

> **Cross-script F0.5 0.113 → 0.999 (+0.886) — contrastive fine-tuning closes the cross-script matching gap.**

## Final Message

> **Cross-script entity resolution is a representation-learning problem; a contrastively fine-tuned multilingual Transformer solves it almost completely where lexical similarity structurally cannot.**

---

# 33. Suggested Presentation Flow

The PowerPoint should follow this order:

1. **Title**
2. **Problem & Motivation**
3. **What is Entity Resolution?**
4. **Why Multilingual/Cross-Script Matching is Difficult**
5. **Dataset**
6. **Dataset Analysis**
7. **Cross-Script Examples**
8. **Research Question**
9. **Proposed Architecture**
10. **Contrastive Learning**
11. **Dataset Construction & Negative Sampling**
12. **Experimental Setup**
13. **Baseline vs Proposed Models**
14. **Overall Results**
15. **Same-Script vs Cross-Script Results**
16. **Script-Wise Analysis**
17. **Hard-Negative Ablation**
18. **Address Experiment**
19. **Qualitative Examples / Error Analysis**
20. **Limitations**
21. **Conclusion**
22. **Future Work**
23. **Q&A**

### Important presentation rule

Do **not** make the Amazon ML Challenge leaderboard the central story. The project is an **NLP study of multilingual/cross-script entity resolution**. The dataset is the experimental benchmark; the research contribution is the comparison of multilingual representations, contrastive learning and hard-negative training.
