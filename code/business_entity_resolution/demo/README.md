# Entity Resolution — Live Demo

Three-tab Gradio demo of the fine-tuned multilingual business entity resolution
model. Search a fresh ~5,000-record business catalog with the fine-tuned encoder
and see correct cross-script matches, the "why this exists" framing, and the
model's real failure cases.

- **Tab 1 — Why this exists:** static motivation (report §2 / §32).
- **Tab 2 — Try the model:** pick or type a business name → top-K cosine
  matches from the precomputed corpus index. Green/red badges are ground truth;
  the TF-IDF column shows what character n-gram lexical matching would score.
- **Tab 3 — Where it fails:** the real modelD test-split errors (report §22–23).

## Layout

```
demo/
├── serving.py          checkpoint loading, E5 embedding, cosine top-K, TF-IDF scorer
├── build_corpus.py     ground-truth inversion → demo_corpus.tsv + .npy matrices
├── failures.py         failure gallery data (report §22–23)
├── app.py              Gradio app (three tabs)
├── tests/
│   ├── test_serving.py
│   └── test_build_corpus.py
├── demo_corpus.tsv     generated (gitignored)
├── corpus_addr.npy     generated — modelADDR embeddings, (N, 384)  (gitignored)
└── corpus_name.npy     generated — modelD embeddings,   (N, 384)  (gitignored)
```

## Prerequisites

Everything runs with the project venv (Python 3.13) **and `-X utf8`** (Indic text
breaks on Windows without it). Run all commands **from
`code/business_entity_resolution/`** so the `src/` + `baselines/` sys.path
bootstrap resolves.

```bash
cd code/business_entity_resolution

# one-time: Gradio (only the demo needs it)
uv pip install --python ./.venv/Scripts/python.exe gradio
```

## Rebuild the corpus (offline, run once — ~3 min)

Reads the 1 GB S2/S3 files + ground truth, selects clusters disjoint from the
existing 10k subset, and writes `demo/demo_corpus.tsv` + both embedding matrices.

```bash
./.venv/Scripts/python.exe -X utf8 demo/build_corpus.py
```

Expected: `demo_corpus.tsv` (~5k rows) and `corpus_addr.npy` / `corpus_name.npy`
(each `(N, 384)`).

## Run the tests

```bash
# serving module (loads both checkpoints; ~30 s) — the correctness core
./.venv/Scripts/python.exe -X utf8 -m pytest demo/tests/test_serving.py -q

# corpus builder: pure functions only (no 1 GB read)
./.venv/Scripts/python.exe -X utf8 -m pytest demo/tests/test_build_corpus.py -q -k "not slow"

# everything, including the artifact tests (needs a built corpus)
./.venv/Scripts/python.exe -X utf8 -m pytest demo/tests/ -q
```

## Launch the app

```bash
./.venv/Scripts/python.exe -X utf8 demo/app.py
# prints a localhost URL; Ctrl-C to stop
```

Non-interactive smoke check (loads state, runs one cross-script query, exits):

```bash
./.venv/Scripts/python.exe -X utf8 demo/app.py --smoke
```

## Professor demo script (3 lines)

1. **Tab 1 — "Why this exists":** point at the Silver Solutions ↔ सिल्वर सॉल्यूशंस
   pair; state the headline: cross-script F0.5 jumps **0.113 → 0.999** because
   this is a representation-learning problem, not a string-matching one.
2. **Tab 2 — "Try the model":** pick a curated cross-script name (e.g. *Jain
   Finance*), click **Find matches** → the Devanagari duplicate *जैन फाइनेंस*
   appears in top-K with a green **same business** badge and a near-zero
   **TF-IDF** column. That is the whole argument in one screen.
3. **Tab 3 — "Where it fails":** scroll to a *domain-compression* case
   (`carehealth.com`) and a *same-core-words* case (`Dermatology Clinic LLC` vs
   `Dermatology Group`) — honest limits, and exactly what a cross-encoder
   reranker (report §31) would target.

## Correctness notes

The model scores silently corrupt if the training-time input formatting is not
reproduced exactly — `serving.py` pins all of it:

- E5 prefixes: `query: ` (query/anchor side), `passage: ` (corpus side).
- CLS pooling + L2 normalization (handled inside `E5Encoder`).
- `modelD`: raw name, `max_length 64`; `modelADDR`: `"Business name: <name> |
  Address: <addr>"`, `max_length 128`, address `fillna('')`.
- Routing: non-empty address → `modelADDR`; empty address → `modelD` (an empty
  address field actively hurts `modelADDR` — report §21 / spec §3).

Verified against the report's reference: modelD cosine **0.9044** for
*"Jain Finance" ↔ "जैन फाइनेंस"* (`test_smoke_cosine_matches_reference`).
