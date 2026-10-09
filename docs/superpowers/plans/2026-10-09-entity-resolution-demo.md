# Entity Resolution Live Demo — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a 3-tab Gradio demo that lets the professor search a fresh 5k-record business catalog with the fine-tuned entity-resolution model and see correct cross-script matches, the "why this exists" framing, and the model's real failure cases.

**Architecture:** A static corpus (5k S2/S3 records with ground-truth groups) is embedded once with the fine-tuned checkpoints and stored as `.npy` matrices. At runtime a Gradio app loads the modelADDR checkpoint (hard-negatives + address) and the modelD checkpoint (name-only fallback), embeds the query live, and does cosine top-K against the precomputed index. Green/red badges come from ground-truth entity groups; a char 2–4g TF-IDF scorer provides the "what string-matching would score" contrast.

**Tech Stack:** Python 3.13, PyTorch 2.14.1+cu130, transformers 5.18.0, Gradio, pandas/numpy, DuckDB (corpus build), scikit-learn (TF-IDF scorer), pytest.

**Spec:** `docs/superpowers/specs/2026-10-09-entity-resolution-demo-design.md`

## Global Constraints

- Run every command with the venv at `code/business_entity_resolution/.venv` (Python 3.13) and with `-X utf8`.
- Run all demo scripts and pytest from cwd `code/business_entity_resolution/` so the `src/` + `baselines/` sys.path bootstrap resolves.
- Do NOT modify `src/`, `baselines/`, `training/`, or `results/` (read-only). All new code lives in `demo/`.
- Model input formatting is fixed and must be reproduced exactly: `query:` prefix on the query side, `passage:` prefix on the corpus side, CLS pooling (`last_hidden_state[:, 0]`), L2 normalization; modelD = name variant, max_length 64; modelADDR = `"Business name: <name> | Address: <addr>"` variant with `fillna('')`, max_length 128.
- Demo corpus must be disjoint from the existing 10k subset (unseen data).
- Reference values to test against: modelD threshold 0.701, modelADDR threshold 0.614, and modelD cosine 0.9044 for "Jain Finance" ↔ "जैन फाइनेंस" (report §22.1).

## Review Focus

The five conditions most likely to bite a person using this demo, and where each is pinned:

1. **Name-only query (no/empty address)** — must route to modelD; feeding an empty-address string to modelADDR silently degrades scores (spec §3, §7). → pinned by Task 1 `test_name_only_routes_to_modeld`.
2. **Cross-script query** — the whole point of the demo; a Latin query must retrieve its Indic duplicate in top-K. → pinned by Task 5 `test_cross_script_retrieval`.
3. **An S2/S3 id appearing under two S1 groups** (ground-truth inversion) — must dedupe deterministically, no duplicate `record_id`. → pinned by Task 2 `test_no_duplicate_record_ids`.
4. **Indic names surviving UTF-8 read/write** — mojibake would destroy the demo; script column must be populated with real Indic tags. → pinned by Task 2 `test_scripts_populated`.
5. **Wrong variant/max_length/prefix** — silently corrupts every score. → pinned by Task 1 `test_smoke_cosine_matches_reference`.

---

## File Structure

```
code/business_entity_resolution/demo/
├── serving.py            # checkpoint loading, embedding, cosine search, TF-IDF scorer
├── build_corpus.py       # ground-truth inversion → 5k-record demo corpus + embeddings
├── failures.py           # hardcoded failure-case gallery (from report §22–23)
├── app.py                # Gradio Blocks: 3 tabs
├── tests/
│   ├── test_serving.py
│   └── test_build_corpus.py
├── demo_corpus.tsv       # generated
├── corpus_addr.npy       # generated (modelADDR embeddings, 5k×384)
└── corpus_name.npy       # generated (modelD embeddings, 5k×384)
```

Every demo module starts with the same bootstrap (mirrors `training/train_contrastive.py`):

```python
import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[3]
BER = REPO_ROOT / "code" / "business_entity_resolution"
sys.path.insert(0, str(BER / "src"))
sys.path.insert(0, str(BER / "baselines"))
from encoder import E5Encoder
from dataset import apply_text_variant
import evaluate            # for script_tags
```

---

## Task 1: Serving module (correctness core)

**Files:**
- Create: `code/business_entity_resolution/demo/serving.py`
- Test: `code/business_entity_resolution/demo/tests/test_serving.py`

**Interfaces:**
- Consumes: `E5Encoder` from `src/encoder.py` (has `encode(texts, prefix, batch_size) -> np.ndarray [N,384]`), `apply_text_variant` from `src/dataset.py`.
- Produces:
  - `CHECKPOINT_D = BER / "results" / "modelD_epoch3"`
  - `CHECKPOINT_ADDR = BER / "results" / "modelADDR_epoch3"`
  - `QUERY_PREFIX = "query: "`, `PASSAGE_PREFIX = "passage: "`
  - `load_encoder(checkpoint_dir: Path, max_length: int) -> E5Encoder`
  - `encode_texts(encoder: E5Encoder, texts: list[str], prefix: str) -> np.ndarray`  (wraps `encoder.encode`; returns [N,384] float32)
  - `search(query_emb: np.ndarray, corpus_emb: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]`  (top-k row indices, top-k cosine scores; both already L2-normalized)
  - `class TfidfCharScorer` with `fit(names: list[str]) -> None` and `score(a: str, b: str) -> float` (char_wb 2–4, `sublinear_tf=True`, `min_df=2`)

- [ ] **Step 1: Write the failing tests** (`tests/test_serving.py`)

```python
import sys
from pathlib import Path
import numpy as np
DEMO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEMO))
import serving

def test_embeddings_normalized_and_384():
    enc = serving.load_encoder(serving.CHECKPOINT_D, max_length=64)
    emb = serving.encode_texts(enc, ["Jain Finance"], serving.QUERY_PREFIX)
    assert emb.shape == (1, 384)
    assert abs(float(np.linalg.norm(emb[0])) - 1.0) < 1e-4

def test_smoke_cosine_matches_reference():
    enc = serving.load_encoder(serving.CHECKPOINT_D, max_length=64)
    q = serving.encode_texts(enc, ["Jain Finance"], serving.QUERY_PREFIX)
    p = serving.encode_texts(enc, ["जैन फाइनेंस"], serving.PASSAGE_PREFIX)
    assert abs(float(q[0] @ p[0]) - 0.9044) < 0.02

def test_modeladdr_cross_script_positive_above_threshold():
    # load a cross-script positive with both addresses from test_pairs.tsv
    df = pd.read_csv(serving.REPO_ROOT / "data/entity_resolution_subset/test_pairs.tsv", sep="\t")
    row = df[(df["label"] == 1) & (df["s1_name"] != df["candidate_name"])
             & df["s1_address"].notna() & df["candidate_address"].notna()
             & (df["s1_address"].str.len() > 0) & (df["candidate_address"].str.len() > 0)]
    row = row[row.apply(lambda r: serving.script_tags(r["s1_name"]) != serving.script_tags(r["candidate_name"]), axis=1)].iloc[0]
    enc = serving.load_encoder(serving.CHECKPOINT_ADDR, max_length=128)
    q_txt = "Business name: " + row["s1_name"] + " | Address: " + row["s1_address"]
    p_txt = "Business name: " + row["candidate_name"] + " | Address: " + row["candidate_address"]
    q = serving.encode_texts(enc, [q_txt], serving.QUERY_PREFIX)
    p = serving.encode_texts(enc, [p_txt], serving.PASSAGE_PREFIX)
    assert float(q[0] @ p[0]) > 0.614

def test_name_only_routes_to_modeld():
    # the routing decision, not a model call: an empty address → name variant
    from dataset import apply_text_variant
    df = pd.DataFrame([{"name": "X", "addr": None}])
    out = apply_text_variant(df, ("name", "addr"), "name_address").iloc[0]
    assert "Address: " in out and "nan" not in out   # fillna('') sanity

def test_search_topk():
    corpus = np.eye(5, 384, dtype=np.float32)         # one-hot rows, all L2-normalized
    query = corpus[2:3]                                # exact match at index 2
    idx, scores = serving.search(query, corpus, k=3)
    assert idx[0] == 2 and abs(float(scores[0]) - 1.0) < 1e-6
```

Note: `serving.py` must re-export `script_tags` (import from `evaluate` and `from evaluate import script_tags`), and `test_serving.py` imports `pandas as pd` at the top.

- [ ] **Step 2: Run tests to verify they fail**

Run: `../.venv/Scripts/python.exe -X utf8 -m pytest demo/tests/test_serving.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'serving'`

- [ ] **Step 3: Implement `serving.py`**

`load_encoder` returns `E5Encoder(model_name=str(checkpoint_dir), max_length=max_length)`. `encode_texts` calls `encoder.encode(texts, prefix)`. `search` computes `query_emb @ corpus_emb.T` and returns `np.argsort` descending top-k (reuse `cross_cosine_matrix` from `src/encoder.py` if convenient). `TfidfCharScorer.fit` fits a single `TfidfVectorizer(analyzer="char_wb", ngram_range=(2,4), sublinear_tf=True, min_df=2)` on `names`; `.score` returns the cosine of the two row vectors via the same elementwise-product method as `baselines/tfidf.py::cross_scores`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `../.venv/Scripts/python.exe -X utf8 -m pytest demo/tests/test_serving.py -q`
Expected: PASS (5 passed). First run downloads nothing — checkpoints are local.

- [ ] **Step 5: Commit**

```bash
git add code/business_entity_resolution/demo/serving.py code/business_entity_resolution/demo/tests/test_serving.py
git commit -m "feat(demo): serving module — checkpoint loading, search, TF-IDF scorer"
```

---

## Task 2: Corpus builder

**Files:**
- Create: `code/business_entity_resolution/demo/build_corpus.py`
- Test: `code/business_entity_resolution/demo/tests/test_build_corpus.py`

**Interfaces:**
- Consumes: `serving.load_encoder`, `serving.encode_texts`, `serving.QUERY_PREFIX`/`PASSAGE_PREFIX`; `evaluate.script_tags`; `apply_text_variant`.
- Produces (all module-level in `build_corpus.py`):
  - Constants `DEMO_DIR`, `TARGET_CLUSTERS = 1800`, `MIN_RECORDS = 2`, `MAX_RECORDS = 6`, `SEED = 42`
  - `load_ground_truth() -> pd.DataFrame`  (columns `source1_entity_id`, `matched_entity_ids`)
  - `explode_matches(gt: pd.DataFrame) -> pd.DataFrame`  (columns `record_id`, `entity_id`; `record_id` is the S2/S3 id, `entity_id` the S1 group id; `record_id` is deduped first-wins when an id appears under multiple S1 groups)
  - `collect_subset_ids() -> tuple[set[str], set[str]]`  (s1_ids, candidate_ids) from the 3 `*_s1.tsv` and 3 `*_pairs.tsv` files
  - `select_clusters(matches: pd.DataFrame, excluded_s1: set[str], excluded_rec: set[str]) -> pd.DataFrame`  (filtered long table for the chosen clusters)
  - `build_records(clusters: pd.DataFrame, s2_path: Path, s3_path: Path) -> pd.DataFrame`  (columns `entity_id`, `record_id`, `business_name`, `business_address`, `country`, `source`, `script`)
  - `main() -> None`  (builds `demo_corpus.tsv`, `corpus_addr.npy`, `corpus_name.npy`)

- [ ] **Step 1: Write the failing tests** (`tests/test_build_corpus.py`) — pure-function tests over a tiny synthetic ground-truth frame, so they run without the 1 GB files:

```python
import sys
from pathlib import Path
import pandas as pd
DEMO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEMO))
import build_corpus

def test_explode_matches_columns():
    gt = pd.DataFrame([{"source1_entity_id": "S1-a", "matched_entity_ids": "S2-x,S3-y"}])
    long = build_corpus.explode_matches(gt)
    assert list(long.columns) == ["record_id", "entity_id"]
    assert set(long["record_id"]) == {"S2-x", "S3-y"}
    assert (long["entity_id"] == "S1-a").all()

def test_no_duplicate_record_ids():
    gt = pd.DataFrame([
        {"source1_entity_id": "S1-a", "matched_entity_ids": "S2-x,S3-y"},
        {"source1_entity_id": "S1-b", "matched_entity_ids": "S2-x,S3-z"},
    ])
    long = build_corpus.explode_matches(gt)
    # an S2/S3 id mapped under two S1s must be dropped or first-wins — never twice
    assert long["record_id"].is_unique
```

Plus two tests that run only against the real corpus (marked `@pytest.mark.slow`, or skipped when `demo_corpus.tsv` is absent):
- `test_corpus_disjoint_from_subset` — `set(corpus["entity_id"]) & subset_s1_ids == set()` and `set(corpus["record_id"]) & subset_candidate_ids == set()`.
- `test_scripts_populated` — `corpus["script"]` has no nulls and contains at least one Indic tag among `{"Devanagari","Tamil","Telugu","Bengali","Gujarati","Kannada","Malayalam","Gurmukhi","Oriya"}`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `../.venv/Scripts/python.exe -X utf8 -m pytest demo/tests/test_build_corpus.py -q`
Expected: FAIL (`ModuleNotFoundError: No module named 'build_corpus'`)

- [ ] **Step 3: Implement `build_corpus.py`**

Algorithm for `select_clusters` (this part is not determined by signatures, so it's specified):

1. `explode_matches(gt)` splits `matched_entity_ids` on `,` into one row per S2/S3 id, deduping `record_id` first-wins (the review-focus #3 rule).
2. Drop rows whose `record_id` is in `excluded_rec` or whose `entity_id` is in `excluded_s1`.
3. Group by `entity_id`, count remaining `record_id`s, keep groups with count in `[MIN_RECORDS, MAX_RECORDS]`.
4. Favor cross-script: for each surviving group, look up its record names (via a filtered DuckDB read of `train_source2.tsv`/`train_source3.tsv`, or a pandas read filtered to the kept record_ids), tag scripts, and score `cross_script = any(group record script != another record script)`. Sample `TARGET_CLUSTERS` groups with cross-script groups oversampled 3:1, `seed=SEED`.

`build_records` materializes those groups' records from S2/S3 (the four raw columns are `entity_id, business_name, business_address, country`) and appends `source` (`S2`/`S3`) and `script` (primary Indic tag from `script_tags(name)`, else `"Latin"`).

`main()`: run the selection, write `demo_corpus.tsv` (UTF-8), then embed — corpus side with `passage:` prefix:
- modelADDR matrix: text = `apply_text_variant(records, ("business_name","business_address"), "name_address")`, encoder max_length 128 → `corpus_addr.npy`.
- modelD matrix: text = raw `business_name`, encoder max_length 64 → `corpus_name.npy`.

Use DuckDB for the S2/S3 name lookup (mirror `src/creat_subset.py` pragmas: threads 16, memory 8GB) — the two files are 1 GB combined.

- [ ] **Step 4: Run tests to verify they pass (pure functions), then run the full build**

Run pure tests: `../.venv/Scripts/python.exe -X utf8 -m pytest demo/tests/test_build_corpus.py -q -k "not slow"`
Expected: PASS

Run build: `../.venv/Scripts/python.exe -X utf8 demo/build_corpus.py`
Expected: writes `demo_corpus.tsv` (~5k rows, 4000–6500), `corpus_addr.npy` + `corpus_name.npy` (each shape `(N, 384)`).

Then run slow tests: `../.venv/Scripts/python.exe -X utf8 -m pytest demo/tests/test_build_corpus.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add code/business_entity_resolution/demo/build_corpus.py code/business_entity_resolution/demo/tests/test_build_corpus.py
git commit -m "feat(demo): build 5k-record unseen corpus + precomputed embeddings"
```

(The generated `.tsv`/`.npy` are gitignored like the checkpoints — add `demo/demo_corpus.tsv` and `demo/*.npy` to `results/.gitignore` pattern or a new `demo/.gitignore`.)

---

## Task 3: Failure gallery data

**Files:**
- Create: `code/business_entity_resolution/demo/failures.py`

**Interfaces:**
- Produces: `FAILURES: list[dict]` — each dict has keys `category` (one of `"domain-compression"`, `"noise-string"`, `"same-core-words"`), `s1_name`, `candidate_name`, `script_pair`, `score` (float), `reason` (one sentence). Also `FAILURE_INTRO: str` — a two-sentence explainer of why these matter (the demo shows the model is honest about its limits).

- [ ] **Step 1: Populate `failures.py` from report §22–23** — transcribe the false-positive table (report §22.2, all 21 if practical, or the 9 shown plus the category summary) and the false-negative table (§22.3, the 12 shown), plus the error-type distribution (§23) as an optional summary list. Values are copy-paste from the report; no model calls.

- [ ] **Step 2: Verify the data loads**

Run: `../.venv/Scripts/python.exe -X utf8 -c "import sys; sys.path.insert(0,'demo'); import failures; print(len(failures.FAILURES), failures.FAILURE_INTRO[:40])"`
Expected: prints a count ≥ 15 and a non-empty intro.

- [ ] **Step 3: Commit**

```bash
git add code/business_entity_resolution/demo/failures.py
git commit -m "feat(demo): failure-case gallery data from report error analysis"
```

---

## Task 4: Gradio app (three tabs)

**Files:**
- Create: `code/business_entity_resolution/demo/app.py`

**Interfaces:**
- Consumes: `serving.*`, `build_corpus` constants, `failures.FAILURES`/`FAILURE_INTRO`, and the generated `demo_corpus.tsv` + `.npy` artifacts.
- Produces: a `gr.Blocks` app launched with `app.launch()`. Handlers:
  - `load_state() -> dict` — loads both encoders (modelADDR first), both corpus matrices, the corpus DataFrame, and a fitted `TfidfCharScorer`; cached once.
  - `run_search(query_name: str, query_address: str, k: int = 8) -> list[list]` — routes modelADDR if `query_address.strip()` non-empty else modelD; embeds query (`query:` prefix, variant per route), calls `serving.search`, and returns one row per top-k hit: `[name, script, country, address, score, badge, tfidf_score]` where `badge` is `"✓ same business"` if the hit's `entity_id == ` the picked record's group (for dropdown picks) else `"different business"` (free text gets badges only against ground-truth via exact-name match, else no badge).
  - Dropdown choices: the ~1,000 `business_name` values of the corpus records with an Indic-script sibling (so the cross-script examples are easy to find), built at startup.

- [ ] **Step 1: Install Gradio** (setup folded into this task — it's the only consumer)

Run: `uv pip install --python code/business_entity_resolution/.venv/Scripts/python.exe gradio`
Expected: installs into the venv.

- [ ] **Step 2: Implement `app.py`**

Three tabs via `gr.Tab`:
1. **"Why this exists"** — `gr.Markdown` with the motivation + the Silver Solutions ↔ सिल्वर सॉल्यूशंस side-by-side pair + consequence bullets (static, from report §2/§32).
2. **"Try the model"** — `gr.Dropdown` (the ~1,000 curated names), a `gr.Textbox` for free-text name, an optional `gr.Textbox` for address, a `gr.Button("Find matches")`, and a `gr.Dataframe` for results (headers: Name, Script, Country, Address, Score, Match, TF-IDF). The TF-IDF column shows `tfidf_char.score(query_name, hit_name)` for the hit.
3. **"Where it fails"** — a `gr.Dataframe` (or per-category `gr.Markdown` blocks) rendering `FAILURES`, grouped by `category`.

Wire `run_search` to the button; on a dropdown pick, prefill the name+address textboxes from the corpus row.

- [ ] **Step 3: Verify the app launches**

Run: `../.venv/Scripts/python.exe -X utf8 demo/app.py`
Expected: Gradio prints a localhost URL and the three tabs render. Then Ctrl-C.

- [ ] **Step 4: Commit**

```bash
git add code/business_entity_resolution/demo/app.py
git commit -m "feat(demo): three-tab Gradio app (why / try / failures)"
```

---

## Task 5: End-to-end verification + handoff note

**Files:**
- Modify: `docs/superpowers/plans/2026-10-09-entity-resolution-demo.md` (this file, not needed) — instead add `demo/README.md` (run commands + demo script).

**Interfaces:**
- Consumes: everything above.

- [ ] **Step 1: Cross-script retrieval test (review-focus #2)**

Run: `../.venv/Scripts/python.exe -X utf8 -m pytest demo/tests/test_serving.py -q` plus a manual check — launch the app, pick "Jain Finance" (or any curated cross-script name) from the dropdown, confirm the Devanagari duplicate "जैन फाइनेंस" appears in top-K with a green badge and a near-zero TF-IDF column.

- [ ] **Step 2: Write `demo/README.md`** — the exact commands to rebuild the corpus, run tests, and launch the app, plus the 3-line demo script for the professor (pick a cross-script name → see the Indic match → switch to a failure case).

- [ ] **Step 3: Commit**

```bash
git add code/business_entity_resolution/demo/README.md
git commit -m "docs(demo): run commands + professor demo script"
```

- [ ] **Step 4: Full plan verification** — run every test file and the build end-to-end once, confirm 4 smoke tests + corpus tests green, and the app launches cleanly.
