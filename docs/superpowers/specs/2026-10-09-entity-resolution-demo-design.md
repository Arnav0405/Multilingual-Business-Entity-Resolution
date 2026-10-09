# Entity Resolution Live Demo — Design Spec

- **Date:** 2026-10-09
- **Status:** Approved for implementation planning
- **Repo:** `nlp_ca` (business entity resolution project)

## 1. Purpose and Outcome

Build an interactive web demo of the fine-tuned multilingual business entity
resolution model, for the professor to operate and understand.

Two goals, in priority order:

1. **Let her use the model like a real business.** She picks or types a
   business name and the model returns the correct matching business entities
   (the duplicates of that business that exist in a catalog), with similarity
   scores. This is the "use it like Amazon" moment.
2. **Make the task and the model understandable.** Explain why entity
   resolution exists, show the before/after (lexical vs neural), and honestly
   show where the model fails.

**Audience assumption:** a professor who knows some ML but was confused about
why this task exists and what the model actually does. She will interact with
the app live (either she drives, or Arnav drives during the viva). The app
must be self-explanatory enough to stand alone, but is not a self-guided
course.

**Success criteria:**

- The "Try the model" tab returns correct ground-truth matches for a known
  query, including at least one cross-script (Latin → Indic) match.
- The cross-script contrast (TF-IDF misses, model catches) is visible in the
  UI.
- The three tabs render and the app launches on the local RTX 4060 (8 GB) with
  no out-of-memory.
- The demo corpus contains no entity present in the existing 10k train/val/test
  subset (genuinely unseen data).

## 2. Context and Constraints

Existing project (all on disk):

- Raw source: `data/train_source1.tsv` (2.2M S1), `train_source2.tsv` (5.0M S2),
  `train_source3.tsv` (5.3M S3), `train_ground_truth.tsv` (S1 → S2/S3 matches).
- Fine-tuned checkpoints: `results/modelD_epoch3/` (contrastive + hard negatives,
  name-only), `results/modelADDR_epoch3/` (contrastive + hard negatives +
  address), each ~470 MB safetensors, with `config.json` + tokenizer.
- Encoder helper: `src/encoder.py` (`E5Encoder`, `cross_cosine_matrix`).
- Venv: `code/business_entity_resolution/.venv`, torch 2.14.1+cu130, CUDA
  available, transformers 5.18.0.

Constraints:

- Runs locally on Windows 11, RTX 4060 Laptop (8 GB VRAM), Python 3.13.
- Do not modify the existing training/eval pipeline.
- Demo must reuse the exact model input formatting or scores will be wrong
  (see §6).

## 3. Decisions (agreed with Arnav)

- **Form factor:** interactive web app (Gradio), three tabs.
- **Pages:** "Why this exists", "Try the model" (search box + dropdown), and
  "Where it fails" (error analysis).
- **Corpus approach:** Approach A — catalog with planted duplicates. The corpus
  is S2/S3 records drawn to contain real duplicate clusters; matching means the
  model pulls up the other records of the same real-world business.
- **Corpus size/source:** ~5,000 fresh rows from the untrained S2/S3 corpus,
  with ground-truth labels, excluding the existing 10k subset.
- **Model:** modelADDR (contrastive + hard negatives + address) as primary;
  fall back to modelD (name-only) for free-text queries without an address.
  This follows the report's own "conditional selection on field presence"
  conclusion (§21): modelADDR needs addresses on both sides; an empty address
  field actively hurts.
- **Inference:** precompute corpus embeddings once (instant search) AND load
  the fine-tuned checkpoint live for arbitrary free-text queries.

## 4. Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Gradio app (demo/app.py)                                    │
│                                                              │
│  Tab 1: Why this exists   (static explainer)                 │
│  Tab 2: Try the model     (search box + dropdown + results)  │
│  Tab 3: Where it fails    (failure gallery)                  │
└──────────────┬───────────────────────────────────────────────┘
               │  query (name, optional address)
               ▼
        ┌──────────────────────────────┐
        │  Query encoder                │
        │  modelADDR (has address)      │
        │  modelD   (name-only fallback)│
        └──────────────┬───────────────┘
                       │  query embedding (384-d, L2)
                       ▼
        ┌──────────────────────────────────────────────┐
        │  Precomputed corpus index (5k × 384, .npy)    │
        │  cosine(query, corpus) → top-K               │
        └──────────────────────────────────────────────┘
                       │
                       ▼
        ┌──────────────────────────────────────────────┐
        │  Results renderer                             │
        │  green/red badge via ground truth entity_id   │
        │  TF-IDF side panel for the cross-script contrast│
        └──────────────────────────────────────────────┘
```

The corpus embeddings are precomputed offline (modelADDR matrix + modelD
matrix) and loaded at startup. The query is encoded live. Two checkpoints are
held in memory (each ~470 MB fp16 → ~1 GB combined, well within 8 GB VRAM).

## 5. Components

### 5.1 Corpus builder — `demo/build_corpus.py` (offline, run once)

Inputs: `data/train_source2.tsv`, `train_source3.tsv`, `train_ground_truth.tsv`.

Steps:

1. Load ground truth; map S1 entity_id → list of S2/S3 entity_ids.
2. Load the existing 10k subset entity_ids (from
   `data/entity_resolution_subset/*_s1.tsv` and the pairs files) and exclude
   them, so the demo corpus is unseen.
3. Select ~1,500–2,000 distinct real-world entities, favoring Indian
   cross-script cases (S2/S3 records whose name is in an Indic script while
   their paired record is Latin, or vice versa).
4. For each selected entity, keep 1–3 records (S2 and/or S3 versions,
   transliterations, abbreviations, typos) → ~5,000 records total.
5. Tag each record's script (reuse the Unicode-range tagging approach from the
   existing pipeline).
6. Output `demo/demo_corpus.tsv` with columns: `entity_id`, `business_name`,
   `business_address`, `country`, `script`.

Then encode every corpus record twice — once with modelADDR (name+address
variant, `"Business name: <name> | Address: <addr>"`, max_length 128), once
with modelD (name variant, max_length 64) — and save `demo/corpus_addr.npy`
and `demo/corpus_name.npy` (each 5k × 384 float32).

### 5.2 "Why this exists" — static tab

- Short motivation paragraph (marketplace holds millions of seller records;
  same business appears as many near-duplicates across scripts/transliterations/
  typos).
- One real side-by-side pair (e.g. "Silver Solutions Private Limited" ↔
  "सिल्वर सॉल्यूशंस प्राइवेट लिमिटेड").
- 2–3 consequence bullets: duplicate listings, fraud/mis-attribution, wrong
  analytics from uncollapsed records.
- Reuses framing from report §2 and §32. No model calls.

### 5.3 "Try the model" — interactive tab (centerpiece)

- **Dropdown:** ~1,000 pickable real business names from the demo corpus
  (full record, including address).
- **Search box:** free text (business name; optional address field).
- **Query path:**
  - Dropdown pick (or free text with a non-empty address) → modelADDR.
  - Free text name-only → modelD fallback.
- **Retrieval:** cosine(query embedding, corpus matrix) → top-K (K=8).
- **Result card per match:**
  - business name (original script), script tag, address, country,
  - similarity score,
  - green badge "same business" / red badge "different business" from ground
    truth entity_id.
- **TF-IDF side panel:** for the top result, show the TF-IDF cosine for the
  same pair (char 2–4g variant — the headline baseline, cross-script F0.5
  0.113), so the cross-script contrast is explicit (TF-IDF ≈ 0.1–0.2 vs
  model ≈ 0.85+).

### 5.4 "Where it fails" — error analysis tab

- Browsable gallery of the real modelD test-split failure cases from report
  §22–23, categorized:
  - domain/social-handle compression (`carehealth.com`, `@laughlinsupreme`),
  - short/no-overlap noise strings (`Tavoevoonyx`, `BELOEVOYUMA`),
  - same-core-words different business (`Dermatology Clinic LLC` vs
    `Dermatology Group`).
- Each card: S1 name, candidate name, script pair, model score, and the
  failure reason.
- Optional "run this case live" button that re-runs the model on that pair.

### 5.5 Model serving

- Load `modelADDR_epoch3` and `modelD_epoch3` at app startup via
  `AutoModel.from_pretrained(<local dir>)` in fp16 on CUDA.
- Reuse `E5Encoder` logic but parameterize the checkpoint path, max_length,
  and text variant. Preserve: `query:` prefix for the anchor side, `passage:`
  prefix for the corpus side, CLS pooling, L2 normalization.

## 6. Correctness risks (must reproduce exactly)

The model was trained with specific input formatting. Getting any of these
wrong silently corrupts scores:

- E5 `query:` / `passage:` prefixes (query = `query: `, corpus = `passage: `).
- CLS pooling (`last_hidden_state[:, 0]`), not mean pooling.
- L2 normalization of embeddings.
- modelADDR: max_length 128 and the `"Business name: <name> | Address: <addr>"`
  concatenation, `fillna('')` on address (empty string, never the literal
  `"nan"`).
- modelD: max_length 64, raw name only.

Mitigation: smoke test against `results/env_smoke.json` reference cosines
before wiring the app.

## 7. Error Handling

- Address missing on a dropdown pick: always fall back to modelD (address
  present by construction for dropdown, but guard anyway).
- Model load failure at startup: print a clear message and exit (do not run a
  half-working demo).
- Unknown ground-truth entity_id (should not happen): render the result without
  a badge, not as a crash.
- CUDA OOM: model loading is sized to fit 8 GB (two fp16 checkpoints ≈ 1 GB);
  if it ever OOMs, load one model at a time on demand rather than both
  resident.

## 8. Testing

1. **Embedding smoke test:** load each checkpoint, embed the known cross-script
   pairs from `env_smoke.json`, assert cosines within tolerance of the recorded
   values.
2. **Corpus disjointness:** assert `demo_corpus.tsv` entity_ids are disjoint
   from the 10k subset entity_ids.
3. **Corpus composition:** assert ~5,000 rows and a non-trivial fraction of
   cross-script records.
4. **Retrieval correctness:** for a handful of known queries, assert the
   correct ground-truth cluster appears in top-K.
5. **App launch:** app starts and all three tabs render; a dropdown pick and a
   free-text query both return results without error.

## 9. Out of Scope

- No retraining or new fine-tuning.
- No changes to the existing training/eval pipeline.
- No full-scale blocking/indexing at challenge scale (2.2M × 10.3M).
- No confidence-interval / calibration work.
- No deployment beyond localhost.

## 10. File Layout

```
code/business_entity_resolution/demo/
├── build_corpus.py        # subset build + corpus embeddings
├── app.py                 # Gradio app (three tabs)
├── demo_corpus.tsv        # generated artifact
├── corpus_addr.npy        # generated artifact (modelADDR embeddings)
├── corpus_name.npy        # generated artifact (modelD embeddings)
└── failures.py            # failure-case data for the "Where it fails" tab
```
