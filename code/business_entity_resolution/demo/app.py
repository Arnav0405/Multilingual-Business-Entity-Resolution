"""
Entity Resolution demo — three-tab Gradio app.

Tab 1 "Why this exists"  — static motivation (report §2 / §32).
Tab 2 "Try the model"    — search a fresh 5k catalog; modelADDR when an address
                           is present, modelD (name-only) otherwise; green/red
                           ground-truth badge + TF-IDF contrast column.
Tab 3 "Where it fails"   — the real modelD test-split errors (report §22–23).

Run (from code/business_entity_resolution/):
    ../.venv/Scripts/python.exe -X utf8 demo/app.py
Requires demo_corpus.tsv + corpus_addr.npy + corpus_name.npy (run build_corpus.py).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import gradio as gr

# --- sys.path bootstrap ------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[3]
BER = REPO_ROOT / "code" / "business_entity_resolution"
sys.path.insert(0, str(BER / "src"))
sys.path.insert(0, str(BER / "baselines"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset import apply_text_variant  # noqa: E402

import serving  # noqa: E402
import failures as failures_data  # noqa: E402

DEMO_DIR = Path(__file__).resolve().parent
CORPUS_TSV = DEMO_DIR / "demo_corpus.tsv"
ADDR_NPY = DEMO_DIR / "corpus_addr.npy"
NAME_NPY = DEMO_DIR / "corpus_name.npy"

DEFAULT_K = 8
RESULT_HEADERS = ["Name", "Script", "Country", "Address", "Score", "Match", "TF-IDF"]

_state: dict = {}


# ============================================================
# State loading
# ============================================================

def load_state() -> dict:
    """Load both encoders, both corpus matrices, the corpus frame, and the
    TF-IDF scorer once. modelADDR is loaded first (primary path)."""
    if _state:
        return _state
    for p in (CORPUS_TSV, ADDR_NPY, NAME_NPY):
        if not p.exists():
            raise SystemExit(
                f"missing demo artifact: {p.name}\n"
                f"run:  ../.venv/Scripts/python.exe -X utf8 demo/build_corpus.py"
            )
    try:
        enc_addr = serving.load_encoder(serving.CHECKPOINT_ADDR, serving.MAX_LEN_ADDR)
        enc_name = serving.load_encoder(serving.CHECKPOINT_D, serving.MAX_LEN_NAME)
    except Exception as e:  # pragma: no cover - startup guard
        raise SystemExit(f"failed to load model checkpoints: {e}")

    corpus = pd.read_csv(CORPUS_TSV, sep="\t", dtype=str).fillna("")
    addr = np.load(ADDR_NPY).astype(np.float32)
    name = np.load(NAME_NPY).astype(np.float32)

    tfidf = serving.TfidfCharScorer().fit(corpus["business_name"].tolist())
    dropdown, name_to_addr = _build_dropdown(corpus)

    _state.update(
        corpus=corpus, addr=addr, name=name,
        enc_addr=enc_addr, enc_name=enc_name, tfidf=tfidf,
        dropdown=dropdown, name_to_addr=name_to_addr,
    )
    return _state


def _build_dropdown(corpus: pd.DataFrame) -> tuple[list[str], dict[str, str]]:
    """Latin-script names whose entity group also contains an Indic record, so
    the cross-script duplicate is one click away."""
    sizes = corpus.groupby("entity_id")["script"].nunique()
    cross_groups = set(sizes[sizes > 1].index)
    sub = corpus[corpus["entity_id"].isin(cross_groups)]
    latin = sub[sub["script"] == "Latin"].copy()
    latin["_nm"] = latin["business_name"].str.strip()
    latin = latin[latin["_nm"] != ""]
    names = sorted(set(latin["_nm"]))
    name_to_addr = (
        latin.drop_duplicates("_nm").set_index("_nm")["business_address"].to_dict()
    )
    return names[:1200], name_to_addr


# ============================================================
# Search
# ============================================================

def run_search(query_name: str, query_address: str, k: int = DEFAULT_K):
    st = load_state()
    corpus = st["corpus"]
    qname = (query_name or "").strip()
    qaddr = (query_address or "").strip()
    if not qname:
        return [], "Enter a business name (or pick one from the dropdown)."

    use_addr = bool(qaddr)
    if use_addr:
        q_text = apply_text_variant(
            pd.DataFrame([{"business_name": qname, "business_address": qaddr}]),
            ("business_name", "business_address"), "name_address",
        ).iloc[0]
        emb = serving.encode_texts(st["enc_addr"], [q_text], serving.QUERY_PREFIX)
        matrix = st["addr"]
        route = "modelADDR (name + address)"
    else:
        emb = serving.encode_texts(st["enc_name"], [qname], serving.QUERY_PREFIX)
        matrix = st["name"]
        route = "modelD (name only)"

    k = int(k) if k else DEFAULT_K
    idx, scores = serving.search(emb, matrix, k=k)

    # ground-truth group(s) of the query, for badges
    exact = corpus[corpus["business_name"].str.strip() == qname]
    qgroups = set(exact["entity_id"]) if len(exact) else set()

    rows = []
    for rank, (i, s) in enumerate(zip(idx, scores)):
        hit = corpus.iloc[int(i)]
        if qgroups:
            badge = "✓ same business" if hit["entity_id"] in qgroups else "different business"
        else:
            badge = "—"
        rows.append([
            hit["business_name"],
            hit["script"],
            hit["country"],
            _short(hit["business_address"]),
            round(float(s), 4),
            badge,
            round(st["tfidf"].score(qname, hit["business_name"]), 4),
        ])
    return rows, f"route: {route}   |   {len(rows)} results"


def _short(s: str, n: int = 70) -> str:
    s = str(s)
    return s if len(s) <= n else s[: n - 1] + "…"


# ============================================================
# Static content
# ============================================================

WHY_MD = """
## Why this exists

A marketplace's catalog holds millions of seller records — and the **same real
business shows up many times**: across data sources, naming conventions,
abbreviations, typos, transliterations, and writing systems. Traditional lexical
matching relies on surface character similarity, so it **structurally fails**
when two records mean the same thing but look nothing alike.

| Source | Business name |
|---|---|
| S1 | Silver Solutions Private Limited |
| S2 | सिल्वर सॉल्यूशंस प्राइवेट लिमिटेड |

Both rows are the *same business*. Character overlap between the two strings is
near zero, so TF-IDF scores this pair ~0.1 — the model scores it ~0.95.

**Why it matters**

- **Duplicate listings** — the same seller appears as two, splitting reputation and inventory.
- **Fraud / mis-attribution** — a near-duplicate can inherit another business's history.
- **Broken analytics** — uncollapsed records corrupt counts, rankings, and per-seller metrics.

Cross-script matching collapses from **F0.5 0.974 → 0.113** for lexical methods;
a contrastively fine-tuned multilingual Transformer closes that gap to **0.999**
(report §15–16, §32).
"""


# ============================================================
# App
# ============================================================

def build_app() -> gr.Blocks:
    st = load_state()

    with gr.Blocks(title="Multilingual Business Entity Resolution") as app:
        gr.Markdown("# Multilingual & Cross-Script Business Entity Resolution\n"
                    "Fine-tuned `multilingual-e5-small` siamese encoder — live demo.")

        with gr.Tab("Why this exists"):
            gr.Markdown(WHY_MD)

        with gr.Tab("Try the model"):
            gr.Markdown(
                "Search a **fresh 5,000-record catalog** (unseen by the model). Pick a "
                "curated name, or type your own. Add an address to use the address-aware "
                "model (`modelADDR`); leave it blank to use the name-only model (`modelD`).\n\n"
                "**Match** is ground truth (green = same business); **TF-IDF** is what "
                "character n-gram lexical matching would score for the same pair."
            )
            with gr.Row():
                with gr.Column(scale=2):
                    dd = gr.Dropdown(
                        choices=st["dropdown"], value=None, label="Curated cross-script names",
                        filterable=True,
                    )
                with gr.Column(scale=3):
                    name_box = gr.Textbox(label="Business name", placeholder="e.g. Jain Finance")
                    addr_box = gr.Textbox(label="Address (optional)", placeholder="leave blank for name-only model")
            with gr.Row():
                btn = gr.Button("Find matches", variant="primary")
                k_slider = gr.Slider(1, 20, value=DEFAULT_K, step=1, label="Top-K")
            status = gr.Markdown("")
            results = gr.Dataframe(
                headers=RESULT_HEADERS, value=[], interactive=False, wrap=True,
            )

            def _prefill(pick):
                nm = (pick or "").strip()
                return nm, st["name_to_addr"].get(nm, "")

            dd.change(_prefill, inputs=dd, outputs=[name_box, addr_box])
            btn.click(run_search, inputs=[name_box, addr_box, k_slider],
                      outputs=[results, status])
            name_box.submit(run_search, inputs=[name_box, addr_box, k_slider],
                            outputs=[results, status])

        with gr.Tab("Where it fails"):
            gr.Markdown(failures_data.FAILURE_INTRO)
            fail_rows = [
                [failures_data.CATEGORY_TITLES.get(f["category"], f["category"]),
                 f["s1_name"], f["candidate_name"], f["script_pair"], f["score"], f["reason"]]
                for f in sorted(
                    failures_data.FAILURES,
                    key=lambda f: (failures_data.CATEGORIES.index(f["category"]), -f["score"]),
                )
            ]
            gr.Dataframe(
                headers=["Category", "S1 (query)", "Candidate", "Script pair", "Score", "Why it fails"],
                value=fail_rows, interactive=False, wrap=True,
            )
            gr.Markdown("### Error-type distribution (report §23)")
            gr.Dataframe(
                headers=["Error type", "FN", "FP", "Total", "% of errors"],
                value=[[d["error_type"], d["fn"], d["fp"], d["total"], d["pct"]]
                       for d in failures_data.ERROR_DISTRIBUTION],
                interactive=False,
            )

    return app


def smoke_test() -> None:
    """Load state, print corpus stats and one cross-script search result."""
    st = load_state()
    print(f"corpus: {len(st['corpus'])} records, "
          f"{st['corpus']['entity_id'].nunique()} entities, "
          f"{len(st['dropdown'])} dropdown names")
    # find a curated name whose group has an Indic sibling
    pick = st["dropdown"][0]
    rows, status = run_search(pick, st["name_to_addr"].get(pick, ""), DEFAULT_K)
    print(f"query: {pick!r}  ({status})")
    for r in rows[:5]:
        print("  ", r)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--smoke":
        smoke_test()
    else:
        build_app().launch()
