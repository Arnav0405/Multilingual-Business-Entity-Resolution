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

| Model | Overall F0.5 | Cross-Script F0.5 | Recall@10 | MRR |
|---|---:|---:|---:|---:|
| TF-IDF | [ ] | [ ] | [ ] | [ ] |
| Pretrained Transformer | [ ] | [ ] | [ ] | [ ] |
| Contrastive Transformer | [ ] | [ ] | [ ] | [ ] |
| Contrastive + Hard Negatives | [ ] | [ ] | [ ] | [ ] |
| Contrastive + Address | [ ] | [ ] | [ ] | [ ] |

### Main Finding

[2–4 sentences describing the most important experimental result.]

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
| India | 883,188 | [ ] |
| US | 1,323,633 | [ ] |
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

Therefore, a stratified subset was constructed.

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

> **Replace the following with the actual output from `subset_statistics.json`.**

| Statistic | Actual Value |
|---|---:|
| Selected S1 entities | [ ] |
| Training S1 entities | [ ] |
| Validation S1 entities | [ ] |
| Test S1 entities | [ ] |
| Positive pairs | [ ] |
| Hard negatives | [ ] |
| Easy negatives | [ ] |
| Total pairs | [ ] |
| Cross-script positive pairs | [ ] |

---

# 8. Data Splitting

Splitting was performed at the **S1 entity level**.

```text
Selected S1 entities
        │
        ├── 70% Training
        ├── 15% Validation
        └── 15% Testing
````

This prevents the same S1 business entity from appearing in multiple splits.

## Split Statistics

| Split      | S1 Entities | Positive Pairs | Negative Pairs | Cross-Script Pairs |
| ---------- | ----------: | -------------: | -------------: | -----------------: |
| Train      |         [ ] |            [ ] |            [ ] |                [ ] |
| Validation |         [ ] |            [ ] |            [ ] |                [ ] |
| Test       |         [ ] |            [ ] |            [ ] |                [ ] |

---

# 9. Negative Sampling

Random negatives are generally too easy.

For example:

> S1: Tata Motors
> Negative: Mumbai Bakery

provides little useful training information.

Therefore, two types of negatives were used.

## 9.1 Easy Negatives

Random or weakly related businesses from the same country.

## 9.2 Hard Negatives

Businesses that are more difficult to distinguish because they have similar characteristics.

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
आల్ఫా ఇంటర్నేషనల్ ప్రైవేట్ లిమిటెడ్

Hard Negative:
Alpha International Trading Private Limited
```

### Negative Distribution

| Negative Type | Number | Percentage |
| ------------- | -----: | ---------: |
| Easy          |    [ ] |        [ ] |
| Hard          |    [ ] |        [ ] |
| Total         |    [ ] |       100% |

---

# 10. Proposed Method

## 10.1 Overall Architecture

```text
             Business Name A
                    │
                    ▼
        Multilingual Transformer
                    │
                    ▼
                Embedding A
                    │
                    │
             Cosine Similarity
                    │
                    │
                Embedding B
                    ▲
                    │
        Multilingual Transformer
                    ▲
                    │
             Business Name B
```

The two branches share the same Transformer parameters.

---

# 11. Model Architecture

## Encoder

**Model:** `[actual model]`

Example:

> `intfloat/multilingual-e5-small`

### Configuration

| Parameter               | Value |
| ----------------------- | ----- |
| Base model              | [ ]   |
| Embedding dimension     | [ ]   |
| Maximum sequence length | [ ]   |
| Batch size              | [ ]   |
| Learning rate           | [ ]   |
| Epochs                  | [ ]   |
| Optimizer               | [ ]   |
| Loss                    | [ ]   |
| Temperature             | [ ]   |
| Hardware                | [ ]   |

---

# 12. Contrastive Learning

The model learns to:

* Pull matching business representations closer.
* Push non-matching business representations apart.

Conceptually:

```text
Positive pair:

Business A ───────── Business B
      \                  /
       \                /
        Similar embeddings


Negative pair:

Business A ───────── Business C
       ↕
   Dissimilar embeddings
```

### Training Objective

[Insert exact loss function used by the implementation.]

For example:

```text
Multiple Negatives Ranking Loss / InfoNCE
```

---

# 13. Experimental Setup

Four principal models are evaluated.

## Model 1 — TF-IDF Baseline

```text
Business Name
      ↓
TF-IDF
      ↓
Cosine Similarity
```

Purpose:

> Establish a conventional lexical-similarity baseline.

---

## Model 2 — Pretrained Multilingual Transformer

```text
Business Name
      ↓
Pretrained multilingual Transformer
      ↓
Embedding
      ↓
Cosine Similarity
```

No task-specific fine-tuning.

Purpose:

> Measure how much multilingual semantic knowledge already exists in the pretrained model.

---

## Model 3 — Contrastive Transformer

```text
Pretrained Transformer
        ↓
Siamese architecture
        ↓
Contrastive fine-tuning
        ↓
Business embeddings
```

Purpose:

> Adapt the multilingual representation specifically to business entity resolution.

---

## Model 4 — Contrastive Transformer + Hard Negatives

```text
Pretrained Transformer
        ↓
Contrastive learning
        +
Hard-negative mining
        ↓
Fine-tuned embeddings
```

Purpose:

> Determine whether difficult negative examples improve entity discrimination.

---

# 14. Evaluation Metrics

## Classification Metrics

### Precision

Measures the proportion of predicted matches that are actually correct.

### Recall

Measures the proportion of true matches that are successfully identified.

### F1

Balances precision and recall.

### F0.5

Weights precision more heavily than recall.

$$
F_{0.5}
=
1.25
\frac{Precision \times Recall}
{0.25 \times Precision + Recall}
$$

---

## Retrieval Metrics

Because entity resolution can also be formulated as retrieving the correct entity from a candidate set:

* Recall@1
* Recall@5
* Recall@10
* Mean Reciprocal Rank (MRR)

---

# 15. Overall Results

## Main Results

| Model                        | Precision | Recall |  F1 | F0.5 | Recall@1 | Recall@5 | Recall@10 | MRR |
| ---------------------------- | --------: | -----: | --: | ---: | -------: | -------: | --------: | --: |
| TF-IDF                       |       [ ] |    [ ] | [ ] |  [ ] |      [ ] |      [ ] |       [ ] | [ ] |
| Pretrained Transformer       |       [ ] |    [ ] | [ ] |  [ ] |      [ ] |      [ ] |       [ ] | [ ] |
| Contrastive Transformer      |       [ ] |    [ ] | [ ] |  [ ] |      [ ] |      [ ] |       [ ] | [ ] |
| Contrastive + Hard Negatives |       [ ] |    [ ] | [ ] |  [ ] |      [ ] |      [ ] |       [ ] | [ ] |

### Key Result

[State which model produced the strongest result and the numerical improvement over the baseline.]

---

# 16. Same-Script vs Cross-Script Performance

This is the **primary research analysis**.

| Model                        | Same-Script F0.5 | Cross-Script F0.5 | Difference |
| ---------------------------- | ---------------: | ----------------: | ---------: |
| TF-IDF                       |              [ ] |               [ ] |        [ ] |
| Pretrained Transformer       |              [ ] |               [ ] |        [ ] |
| Contrastive Transformer      |              [ ] |               [ ] |        [ ] |
| Contrastive + Hard Negatives |              [ ] |               [ ] |        [ ] |

### Interpretation

[Explain whether Transformer-based representations reduce the performance gap between same-script and cross-script matching.]

---

# 17. Script-Wise Evaluation

| Script     | Number of Pairs | TF-IDF F0.5 | Pretrained F0.5 | Contrastive F0.5 | Hard-Negative F0.5 |
| ---------- | --------------: | ----------: | --------------: | ---------------: | -----------------: |
| Devanagari |             [ ] |         [ ] |             [ ] |              [ ] |                [ ] |
| Bengali    |             [ ] |         [ ] |             [ ] |              [ ] |                [ ] |
| Gujarati   |             [ ] |         [ ] |             [ ] |              [ ] |                [ ] |
| Kannada    |             [ ] |         [ ] |             [ ] |              [ ] |                [ ] |
| Malayalam  |             [ ] |         [ ] |             [ ] |              [ ] |                [ ] |
| Oriya      |             [ ] |         [ ] |             [ ] |              [ ] |                [ ] |
| Tamil      |             [ ] |         [ ] |             [ ] |              [ ] |                [ ] |
| Telugu     |             [ ] |         [ ] |             [ ] |              [ ] |                [ ] |

### Interpretation

[Identify scripts for which the model performs particularly well/poorly and possible reasons supported by the experiments.]

---

# 18. Ablation Study

## 18.1 Effect of Fine-Tuning

| Model                  | Overall F0.5 | Cross-Script F0.5 | Recall@10 |
| ---------------------- | -----------: | ----------------: | --------: |
| Pretrained Transformer |          [ ] |               [ ] |       [ ] |
| Contrastive Fine-Tuned |          [ ] |               [ ] |       [ ] |

### Finding

[ ]

---

## 18.2 Effect of Hard Negatives

| Training Strategy     | F0.5 | Precision | Recall | MRR |
| --------------------- | ---: | --------: | -----: | --: |
| Random/Easy Negatives |  [ ] |       [ ] |    [ ] | [ ] |
| Hard Negatives        |  [ ] |       [ ] |    [ ] | [ ] |

### Finding

[ ]

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

---

# 20. Address Model

## Name Only

```text
Business Name
       ↓
Transformer
       ↓
Embedding
```

## Name + Address

```text
Business Name
      +
Business Address
      ↓
Transformer
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

---

# 21. Address Experiment Results

| Model          | Precision | Recall |  F1 | F0.5 | Recall@10 | MRR |
| -------------- | --------: | -----: | --: | ---: | --------: | --: |
| Name Only      |       [ ] |    [ ] | [ ] |  [ ] |       [ ] | [ ] |
| Name + Address |       [ ] |    [ ] | [ ] |  [ ] |       [ ] | [ ] |

### Cross-Script Address Results

| Model          | Same-Script F0.5 | Cross-Script F0.5 |
| -------------- | ---------------: | ----------------: |
| Name Only      |              [ ] |               [ ] |
| Name + Address |              [ ] |               [ ] |

### Interpretation

[Explain whether address information improves entity resolution and whether its effect is different for same-script and cross-script cases.]

---

# 22. Qualitative Analysis

## 22.1 Successful Examples

| S1 Business | S2/S3 Business | Script | Model Score |
| ----------- | -------------- | ------ | ----------: |
| [ ]         | [ ]            | [ ]    |         [ ] |
| [ ]         | [ ]            | [ ]    |         [ ] |
| [ ]         | [ ]            | [ ]    |         [ ] |

### Observation

[Explain what the model appears to capture.]

---

## 22.2 Failure Examples

| S1 Business | Candidate | True Label | Model Score | Error Type |
| ----------- | --------- | ---------: | ----------: | ---------- |
| [ ]         | [ ]       |        [ ] |         [ ] | [ ]        |
| [ ]         | [ ]       |        [ ] |         [ ] | [ ]        |
| [ ]         | [ ]       |        [ ] |         [ ] | [ ]        |

Possible error categories:

* Transliteration ambiguity
* Similar business names
* Missing address
* Abbreviation
* Very short names
* Multiple valid entities
* Incorrect retrieval
* Script mismatch
* False positive
* False negative

---

# 23. Error Analysis

## Error Distribution

| Error Type          | Count | Percentage |
| ------------------- | ----: | ---------: |
| Similar names       |   [ ] |        [ ] |
| Transliteration     |   [ ] |        [ ] |
| Abbreviation        |   [ ] |        [ ] |
| Address ambiguity   |   [ ] |        [ ] |
| Short names         |   [ ] |        [ ] |
| Missing information |   [ ] |        [ ] |
| Other               |   [ ] |        [ ] |

### Main Failure Modes

1. [ ]
2. [ ]
3. [ ]

---

# 24. Visualization Section

The following figures should be generated for the presentation.

## Figure 1 — Dataset Composition

**Chart:** Bar chart

```text
US vs India
```

Purpose:

> Show the composition of the selected dataset.

---

## Figure 2 — Script Distribution

**Chart:** Bar chart

```text
Devanagari
Telugu
Malayalam
Kannada
Bengali
Gujarati
Tamil
Oriya
```

Purpose:

> Demonstrate the multilingual/cross-script nature of the task.

---

## Figure 3 — Model Performance

**Chart:** Grouped bar chart

```text
                F0.5
                  │
TF-IDF            ███
Pretrained        █████
Contrastive       ███████
Hard Negative     ████████
```

Purpose:

> Show progression from baseline to proposed method.

---

## Figure 4 — Same-Script vs Cross-Script

**Chart:** Grouped bar chart

```text
                Same Script    Cross Script

TF-IDF
Pretrained
Contrastive
Hard Negative
```

Purpose:

> Demonstrate whether the proposed approach improves cross-script matching.

---

## Figure 5 — Script-Wise Performance

**Chart:** Heatmap

Rows:

```text
Devanagari
Bengali
Gujarati
Kannada
Malayalam
Oriya
Tamil
Telugu
```

Columns:

```text
TF-IDF
Pretrained
Contrastive
Hard Negative
```

Values:

> F0.5

---

## Figure 6 — Architecture Diagram

```text
              S1 Business Name
                      │
                      ▼
             ┌────────────────┐
             │  Multilingual  │
             │  Transformer   │
             └───────┬────────┘
                     │
                     ▼
                Embedding A
                     │
                     │
                Similarity
                     │
                     ▲
                     │
                Embedding B
                     ▲
                     │
             ┌───────┴────────┐
             │  Multilingual  │
             │  Transformer   │
             └────────────────┘
                     ▲
                     │
              S2/S3 Business
                   Name
```

---

## Figure 7 — Contrastive Learning Visualization

Use a 2D projection such as **t-SNE or UMAP**.

Show:

```text
Same entity:
   ● ●
   ● ●

Different entities:
               ▲ ▲
                         ■ ■
```

Color/marker by entity ID or script.

Purpose:

> Demonstrate whether contrastive training produces more separated entity representations.

---

## Figure 8 — Address Ablation

**Chart:** Grouped bar chart

```text
              Name Only    Name + Address

Precision
Recall
F1
F0.5
```

Purpose:

> Show the contribution of contextual address information.

---

# 25. Computational Setup

| Component             | Configuration |
| --------------------- | ------------- |
| CPU                   | [ ]           |
| GPU                   | [ ]           |
| GPU VRAM              | [ ]           |
| RAM                   | [ ]           |
| OS                    | [ ]           |
| Python                | [ ]           |
| PyTorch               | [ ]           |
| Transformers          | [ ]           |
| Sentence Transformers | [ ]           |
| Training time         | [ ]           |
| Inference time        | [ ]           |

---

# 26. Reproducibility

### Random Seed

```text
42
```

### Dataset Split

```text
70% Train
15% Validation
15% Test
```

### Leakage Prevention

The split is performed at the S1 entity level so that an S1 entity cannot occur in more than one split.

### Code

Repository:

> [GitHub repository link]

---

# 27. Limitations

The following limitations should be explicitly discussed.

### Dataset Limitations

* The original dataset contains only US and India in the training data.
* The subset is substantially smaller than the original challenge dataset.
* The project therefore evaluates an academic research subset rather than the complete competition problem.

### Language Limitations

* The S1 business names are predominantly/entirely represented in Latin script in the analyzed sample.
* Cross-script matching is primarily between Latin and Indic scripts.
* This should not be interpreted as comprehensive multilingual coverage of all world languages.

### Methodological Limitations

* The model primarily uses business names.
* Address information is evaluated separately.
* The project does not attempt to reproduce the full-scale Amazon challenge retrieval pipeline.

---

# 28. Ethical / Practical Considerations

Entity resolution systems can produce incorrect merges between distinct businesses.

Potential consequences include:

* Incorrect business consolidation
* Incorrect analytics
* Duplicate removal errors
* Incorrect attribution of information

Therefore, high-confidence matching and appropriate threshold selection are important in practical deployments.

---

# 29. Discussion

## Main Findings

### Finding 1 — Multilingual Representation

[ ]

### Finding 2 — Contrastive Learning

[ ]

### Finding 3 — Hard Negatives

[ ]

### Finding 4 — Cross-Script Matching

[ ]

### Finding 5 — Address Information

[ ]

---

# 30. Conclusion

This project investigated multilingual and cross-script business entity resolution using Transformer-based representations and contrastive learning.

The experiments compared conventional lexical similarity with pretrained multilingual representations and task-specific contrastive fine-tuning.

### Final Results

| Method                       | F0.5 | Cross-Script F0.5 | MRR |
| ---------------------------- | ---: | ----------------: | --: |
| TF-IDF                       |  [ ] |               [ ] | [ ] |
| Pretrained Transformer       |  [ ] |               [ ] | [ ] |
| Contrastive Transformer      |  [ ] |               [ ] | [ ] |
| Contrastive + Hard Negatives |  [ ] |               [ ] | [ ] |
| Contrastive + Address        |  [ ] |               [ ] | [ ] |

### Final Conclusion

[Write 3–5 concise sentences describing what the experiments actually demonstrate.]

---

# 31. Future Work

Possible extensions:

1. Extend evaluation to additional languages and scripts.
2. Incorporate multilingual address representations.
3. Investigate cross-lingual transliteration augmentation.
4. Explore cross-encoders for final candidate reranking.
5. Evaluate larger multilingual language models.
6. Develop uncertainty-aware entity-resolution thresholds.
7. Evaluate the method on additional real-world entity-resolution datasets.
8. Investigate zero-shot performance on languages not represented during training.

---

# 32. Presentation Takeaways

## Problem

> Business names referring to the same entity can appear in different languages, scripts and textual forms.

## Challenge

> Lexical similarity fails when surface forms differ substantially.

## Proposed Solution

> A multilingual Transformer learns entity representations using contrastive learning.

## Key Experiment

> Compare same-script and cross-script entity resolution.

## Key Innovation

> Task-specific contrastive learning with hard negatives for multilingual business-name matching.

## Supporting Experiment

> Determine whether adding business addresses improves entity resolution.

## Main Result

> **[Insert strongest numerical result and improvement over baseline.]**

## Final Message

> **[One sentence summarizing the project's contribution.]**

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
