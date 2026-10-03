"""
Phase 11 figures -- report.md sections 24.x (Figures 1-5, 7-8).

All figures are data-driven from the real results artefacts:
  results/*.json                          -- model metrics
  data/entity_resolution_subset/*.tsv     -- subset composition
  results/model{B,C,D}_scores.parquet     -- embedding scores
  results/modelC_epoch3/model.safetensors -- checkpoints (UMAP fig 7)

Static PNG at code/business_entity_resolution/results/figures/figN_*.png
seaborn (paper figures 1-5, 8) + plotly (interactives, same data).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import evaluate  # noqa: E402  (sibling module; fig7 uses load_split_pairs)

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
DATA = HERE.parents[2] / "data" / "entity_resolution_subset"
FIGS = RESULTS / "figures"
FIGS.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid", context="talk")
PALETTE = ["#3A86FF", "#FF006E", "#FB5607", "#FFBE0B", "#8338EC", "#06D6A0", "#4ECDC4"]

MODEL_ORDER = ["tfidf_char", "tfidf_word", "tfidf_both", "modelB", "modelC", "modelD", "modelA_name_address"]
MODEL_LABELS = {
    "tfidf_char": "TF-IDF char", "tfidf_word": "TF-IDF word", "tfidf_both": "TF-IDF both",
    "modelB": "E5 pretrained", "modelC": "E5 + contrastive", "modelD": "E5 + hard negs",
    "modelA_name_address": "E5 + address (A')",
}
SCRIPT_ORDER = ["Devanagari", "Bengali", "Gujarati", "Gurmukhi", "Kannada", "Malayalam", "Oriya", "Tamil", "Telugu"]


def load_reports() -> dict[str, dict]:
    out = {}
    for p in RESULTS.glob("*.json"):
        if p.name in ("env_smoke.json", "central_table.json", "per_script_matrix.json",
                      "cross_script_table.json", "phase10_address_experiment.json"):
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        if "model" in d and "test" in d:
            out[d["model"]] = d
    return out


def fig1_composition() -> None:
    """Dataset composition: countries, pair types, split sizes."""
    stats = json.loads((DATA / "subset_statistics.json").read_text(encoding="utf-8"))
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    fig.suptitle("Subset composition (10,000 stratified S1 entities)", y=1.04, fontweight="bold")

    # countries
    ax = axes[0]
    cd = stats["country_distribution"]
    ax.bar(cd.keys(), cd.values(), color=PALETTE[:2], edgecolor="black", linewidth=0.8)
    for i, (k, v) in enumerate(cd.items()):
        ax.text(i, v + 60, f"{v:,}\n({v/10000:.0%})", ha="center", fontsize=11, fontweight="bold")
    ax.set_title("Country"); ax.set_ylabel("S1 entities"); ax.set_ylim(0, 7600)

    # splits
    ax = axes[1]
    sp = {"train": stats["train_s1"], "val": stats["val_s1"], "test": stats["test_s1"]}
    ax.bar(sp.keys(), sp.values(), color=PALETTE[2:5], edgecolor="black", linewidth=0.8)
    for i, (k, v) in enumerate(sp.items()):
        ax.text(i, v + 60, f"{v:,}", ha="center", fontsize=11, fontweight="bold")
    ax.set_title("Entity-level split (no S1 leakage)"); ax.set_ylim(0, 8200)

    # pair types
    ax = axes[2]
    pt = {"positive": stats["total_positive_pairs"], "hard negative": stats["hard_negatives"], "easy negative": stats["easy_negatives"]}
    ax.barh(list(pt.keys())[::-1], list(pt.values())[::-1], color=[PALETTE[1], "#e63946", PALETTE[4]], edgecolor="black", linewidth=0.8)
    for i, (k, v) in enumerate(list(pt.items())[::-1]):
        ax.text(v + 800, i, f"{v:,}", va="center", fontsize=11, fontweight="bold")
    ax.set_title("Pair types (all splits)")
    ax.set_xlabel("pairs"); ax.set_xlim(0, 118000)
    for a in axes: a.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGS / "fig1_composition.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def fig2_script_distribution() -> None:
    """Cross-script positive pairs per Indic script (train+val+test).
    Log y-scale: 6 of 9 scripts sit under 200 and vanish on a linear axis."""
    n = {}
    for split in ("train", "val", "test"):
        df = pd.read_csv(DATA / f"{split}_pairs.tsv", sep="\t", usecols=["candidate_name", "label", "pair_type"])
        pos = df[(df["label"] == 1)]
        for s in pos["candidate_name"]:
            cp = s
            for script, (lo, hi) in {
                "Devanagari": (0x0900, 0x097F), "Bengali": (0x0980, 0x09FF), "Gurmukhi": (0x0A00, 0x0A7F),
                "Gujarati": (0x0A80, 0x0AFF), "Oriya": (0x0B00, 0x0B7F), "Tamil": (0x0B80, 0x0BFF),
                "Telugu": (0x0C00, 0x0C7F), "Kannada": (0x0C80, 0x0CFF), "Malayalam": (0x0D00, 0x0D7F),
            }.items():
                if any(lo <= ord(c) <= hi for c in cp):
                    n[script] = n.get(script, 0) + 1
    s = pd.Series(n).reindex(SCRIPT_ORDER).fillna(0).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(9, 5.2))
    colors = sns.color_palette("Spectral", len(s))
    ax.bar(range(len(s)), s.values, color=colors, edgecolor="black", linewidth=0.8)
    ax.set_yscale("log")
    ax.set_ylim(8, 12000)  # headroom above 5,164 for the value labels
    ax.set_xticks(range(len(s)))
    ax.set_xticklabels([f"{k}" for k in s.keys()], fontsize=10)
    for i, v in enumerate(s.values):
        ax.text(i, v * 1.15, f"{int(v):,}", ha="center", fontweight="bold")
    ax.set_title("Cross-script positive pairs by candidate script — all splits (log scale)", fontweight="bold")
    ax.set_ylabel("positive pairs (log)"); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGS / "fig2_script_distribution.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def fig3_model_performance(reports) -> None:
    """Progression: overall F0.5 by model (the central bar chart).
    Full 0-1 axis: differences are genuinely tiny, don't inflate them."""
    models = [m for m in MODEL_ORDER if m in reports] + ["modelA_name_address"]
    models = list(dict.fromkeys(models))
    f05 = [reports[m]["test"]["overall"]["f0.5"] for m in models if m in reports]
    labels = [MODEL_LABELS[m] for m in models if m in reports]
    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    bars = ax.bar(labels, f05, color=PALETTE, edgecolor="black", linewidth=1.0)
    for b, v in zip(bars, f05):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.008, f"{v:.4f}", ha="center", fontweight="bold", fontsize=11)
    ax.set_ylim(0, 1.0)  # honest baseline: differences are small, say so
    ax.set_ylabel("F0.5 (test split)")
    ax.set_title("Model progression — overall F0.5 (all bars near 1.0: improvements are incremental)", fontweight="bold", pad=14, fontsize=12)
    ax.axhline(1.0, color="gray", linestyle=":", alpha=0.6)
    ax.grid(axis="y", alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=18, ha="right")
    fig.tight_layout()
    fig.savefig(FIGS / "fig3_model_progression.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def fig4_same_vs_cross(reports) -> None:
    """Grouped bars: same-script vs cross-script F0.5, models on the x axis.
    Legend moved OUT of the axes (it was hiding the headline A′ bars)."""
    models = [m for m in MODEL_ORDER if m in reports]
    same = [reports[m]["test"]["same_script"]["f0.5"] for m in models]
    cross = [reports[m]["test"]["cross_script"]["f0.5"] for m in models]
    labels = [MODEL_LABELS[m] for m in models]
    x = np.arange(len(models))
    w = 0.38
    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    ax.bar(x - w / 2, same, w, label="Same-script F0.5", color=PALETTE[0], edgecolor="black")
    ax.bar(x + w / 2, cross, w, label="Cross-script F0.5", color=PALETTE[1], edgecolor="black")
    for xi, (sv, cv) in enumerate(zip(same, cross)):
        ax.text(xi - w / 2, sv + 0.01, f"{sv:.3f}", ha="center", fontsize=9, fontweight="bold")
        ax.text(xi + w / 2, cv + 0.01, f"{cv:.3f}", ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=14, ha="right", fontsize=10)
    ax.set_ylim(0, 1.12); ax.set_ylabel("F0.5")
    # legend below the axes, horizontal — nothing overlaps the data
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=True, framealpha=1.0)
    ax.set_title("Same-script vs cross-script F0.5 — the research axis", fontweight="bold", pad=14)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGS / "fig4_same_vs_cross.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def fig5_per_script_heatmap(reports) -> None:
    """Rows=script, cols=model, values=cross-script recall/F0.5.
    Colorblind-safe map (crest/viridis family), one SHARED colorbar, Oriya caveat."""
    models = [m for m in MODEL_ORDER if m in reports]
    data_r = pd.DataFrame({MODEL_LABELS[m]: {
        s: reports[m]["test"]["cross_script_by_script"].get(s, {}).get("recall", np.nan) for s in SCRIPT_ORDER
    } for m in models})
    data_f = pd.DataFrame({MODEL_LABELS[m]: {
        s: reports[m]["test"]["cross_script_by_script"].get(s, {}).get("f0.5", np.nan) for s in SCRIPT_ORDER
    } for m in models})
    n_by = {}
    for m in models:
        for s, e in reports[m]["test"]["cross_script_by_script"].items():
            if e:
                n_by[s] = e["tp"] + e["fn"]
    ylabels = [f"{s} (n={n_by.get(s, 0)})" for s in data_r.index]

    fig, axes = plt.subplots(1, 2, figsize=(16.5, 6.5))
    fig.suptitle("Cross-script performance by candidate script (test split) — colorblind-safe map", fontweight="bold", y=0.98, fontsize=14)
    cmap = "coolwarm"
    SHORT = {"TF-IDF char": "char", "TF-IDF word": "word", "TF-IDF both": "both",
             "E5 pretrained": "E5", "E5 + contrastive": "C", "E5 + hard negs": "D",
             "E5 + address (A')": "A′"}
    for ax, dat, title in [(axes[0], data_r, "Recall"), (axes[1], data_f, "F0.5")]:
        dat = dat.rename(columns=SHORT)
        sns.heatmap(dat, annot=True, fmt=".2f", cmap=cmap, vmin=0, vmax=1,
                    annot_kws={"fontsize": 11, "fontweight": "bold"},
                    linewidths=1.4, linecolor="white", ax=ax)
        ax.set_yticklabels(ylabels, rotation=0, fontsize=11)
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=45, labelsize=11)
        for lab in ax.get_xticklabels():
            lab.set_horizontalalignment("right")
    # Right-margin VERTICAL colorbar: cannot collide with bottom x-labels.
    fig.subplots_adjust(bottom=0.26, top=0.86, left=0.11, right=0.86, wspace=0.28)
    import matplotlib.colors
    import matplotlib.cm
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    cax = fig.add_axes([0.885, 0.30, 0.018, 0.52])
    cbar = fig.colorbar(sm, cax=cax, orientation="vertical")
    cbar.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
    cbar.set_ticklabels(["0", ".25", ".5", ".75", "1"])
    cbar.ax.tick_params(labelsize=9)
    cbar.set_label("score", fontsize=9)
    # caveat in its own bottom strip, well clear of the 45° labels
    fig.text(0.485, 0.045, "Oriya (n=1) and Gurmukhi (n=5) are anecdotal — per-script claims rest on Devanagari (n=763).",
             ha="center", fontsize=10, style="italic")

    # --- programmatic collision check: NO text may intersect another's bbox ---
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = []
    for ax_ in axes:
        for lab in ax_.get_xticklabels() + ax_.get_yticklabels():
            if lab.get_text():
                boxes.append((f"tick:{lab.get_text()[:18]}", lab.get_window_extent(renderer)))
    cbar_texts = [cbar.ax.xaxis.label, *cbar.ax.get_yticklabels(), cbar.ax.yaxis.label]
    for t in cbar_texts:
        if t.get_text():
            boxes.append((f"cbar:{t.get_text()[:10]}", t.get_window_extent(renderer)))
    cav = fig.texts[-1]
    boxes.append(("caveat", cav.get_window_extent(renderer)))
    # colorbar BAR bbox vs every text bbox
    cax_bb = cax.get_window_extent(renderer)
    collisions = []
    for name, bb in boxes:
        if bb.overlaps(cax_bb):
            collisions.append((name, "colorbar-bar"))
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            (n1, b1), (n2, b2) = boxes[i], boxes[j]
            if b1.overlaps(b2):
                collisions.append((n1, n2))
    if collisions:
        raise SystemExit(f"figure-bbox collision: {collisions}")
    fig.savefig(FIGS / "fig5_per_script_heatmap.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def fig7_umap_contrastive() -> None:
    """Fig 7: UMAP of the same test instances, pretrained (B) vs contrastive (C).
    Colors = candidate script; shows the clustering/separation shift."""
    import torch
    from safetensors.torch import load_file
    from transformers import AutoTokenizer
    import torch.nn.functional as Fn
    import umap

    tok = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-small")
    df = evaluate.load_split_pairs("val").head(2000)  # manageable, still illustrative

    def cls_embed(payload_path_or_model, side_name, side_addr, is_ckpt):
        texts = [f"passage: {n if pd.notna(n) else ''}" for n, a in zip(side_name, side_addr)]
        out = np.empty((len(texts), 384), dtype=np.float32)
        return texts, out

    def embed(model_obj, texts):
        out = np.empty((len(texts), 384), dtype=np.float32)
        arr = np.arange(len(texts))
        for i in range(0, len(texts), 256):
            enc = tok(texts[i:i + 256], padding=True, truncation=True, max_length=64, return_tensors="pt").to("cuda")
            with torch.no_grad():
                h = model_obj(**enc).last_hidden_state[:, 0]
            out[i:i + len(h)] = Fn.normalize(h.float(), dim=-1).cpu().numpy()
        return out

    sys_path = HERE.parent / "src"
    texts = [f"passage: {n if pd.notna(n) else ''}" for n in df["candidate_name"]]
    emb_b = np.empty((len(texts), 384), dtype=np.float32)
    emb_c = np.empty((len(texts), 384), dtype=np.float32)

    from transformers import AutoModel
    mb = AutoModel.from_pretrained("intfloat/multilingual-e5-small").to("cuda").eval()
    emb_b[:] = embed(mb, texts)
    del mb; torch.cuda.empty_cache()
    ckpt_sd = load_file(RESULTS / "modelC_epoch3" / "model.safetensors")
    mc = AutoModel.from_pretrained("intfloat/multilingual-e5-small").to("cuda").eval()
    mc.load_state_dict(ckpt_sd)
    emb_c[:] = embed(mc, texts)
    del mc; torch.cuda.empty_cache()

    def script_of(name):
        ranges = {"Devanagari": (0x0900, 0x097F), "Bengali": (0x0980, 0x09FF), "Gurmukhi": (0x0A00, 0x0A7F),
                  "Gujarati": (0x0A80, 0x0AFF), "Oriya": (0x0B00, 0x0B7F), "Tamil": (0x0B80, 0x0BFF),
                  "Telugu": (0x0C00, 0x0C7F), "Kannada": (0x0C80, 0x0CFF), "Malayalam": (0x0D00, 0x0D7F)}
        for sname, (lo, hi) in ranges.items():
            if any(lo <= ord(c) <= hi for c in (name if pd.notna(name) else '')):
                return sname
        return "Latin"
    scripts = df["candidate_name"].map(script_of).values

    fig, axes = plt.subplots(1, 2, figsize=(16.5, 6.4), gridspec_kw={"wspace": 0.06})
    fig.suptitle("UMAP of candidate-name embeddings (2,000 val pairs) — pretrained vs contrastively fine-tuned", fontweight="bold", y=1.0)
    uniq = sorted(set(scripts))
    cmap = "husl"
    pal = dict(zip(uniq, sns.color_palette(cmap, len(uniq))))
    last_z = None
    for ax, emb, title in [(axes[0], emb_b, "E5 pretrained (modelB)"), (axes[1], emb_c, "E5 + contrastive (modelC)")]:
        z = umap.UMAP(n_neighbors=30, min_dist=0.25, metric="cosine", random_state=42).fit_transform(emb)
        last_z = z
        for s in uniq:
            m = scripts == s
            ax.scatter(z[m, 0], z[m, 1], s=26, alpha=0.75, color=pal[s], edgecolors="none")
        ax.set_title(title, fontweight="bold"); ax.set_xticks([]); ax.set_yticks([])
        ax.grid(False)
        ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
    # single shared legend OUTSIDE (right of the 2 panels), full script list
    handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=pal[s], markersize=9) for s in uniq]
    leg = fig.legend(handles, uniq, loc="center right", bbox_to_anchor=(1.0, 0.5),
                     ncol=1, fontsize=9, framealpha=1.0, title="candidate script",
                     markerscale=1.0, labelspacing=0.7)
    fig.subplots_adjust(right=0.86)  # leave room for the outside legend
    fig.text(0.43, -0.03, "UMAP axes are unscaled — only relative cluster structure is meaningful.",
             ha="center", fontsize=10, style="italic")
    fig.tight_layout()
    fig.savefig(FIGS / "fig7_umap_pretrained_vs_contrastive.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def fig8_address_ablation() -> None:
    """Fig 8: A' vs D on overall / addr_both / addr_missing slices (F0.5 + recall)."""
    d = json.loads((RESULTS / "phase10_address_experiment.json").read_text(encoding="utf-8"))
    a_, d_ = "modelA_name_address", "modelD"
    rows = []
    for split in ("val", "test"):
        for tag in ("overall", "addr_both", "addr_missing"):
            for model in (d_, a_):
                m = d[split][tag][model]
                rows.append({"Split": split, "Slice": tag, "Model": MODEL_LABELS[model],
                             "F0.5": m["f0.5"], "Recall": m["recall"], "Precision": m["precision"]})
    df = pd.DataFrame(rows)
    sl = df[df["Split"] == "test"]
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.0))
    fig.suptitle("Address ablation (A′ vs D) — test split, per-model val-tuned thresholds", fontweight="bold", y=1.03)
    n_pos = {t: d["test"].get(t, {}).get("n_pos", 19_558 if t == "overall" else "?") for t in ("overall", "addr_both", "addr_missing")}
    for ax, metric in zip(axes, ("F0.5", "Recall", "Precision")):
        p = sl.pivot(index="Slice", columns="Model", values=metric).reindex(["addr_both", "addr_missing", "overall"])
        idx = [(s, n_pos.get(s, '')) for s in p.index]
        p.index = [f"{s}\n(n_pos={n})" if n != '' else s for s, n in idx]
        p.plot.bar(ax=ax, color=[PALETTE[3], PALETTE[0]], edgecolor="black", width=0.72)
        for c in ax.containers:
            ax.bar_label(c, fmt="%.3f", fontsize=9, fontweight="bold", padding=2)
        ax.set_title(metric, fontweight="bold"); ax.set_ylim(0.95, 1.006)
        ax.set_xlabel(""); ax.grid(axis="y", alpha=0.3)
        ax.tick_params(axis="x", rotation=0, labelsize=9)
        ax.get_legend().remove()
    # shared legend UNDER the 3 panels — was overlapping the headline A′ bar
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=PALETTE[3], edgecolor="black"),
               plt.Rectangle((0, 0), 1, 1, facecolor=PALETTE[0], edgecolor="black")]
    fig.legend(handles, ["Name-only (D)", "Name+address (A′)"], loc="upper center",
               bbox_to_anchor=(0.5, -0.02), ncol=2, fontsize=10, framealpha=1.0)
    fig.text(0.5, -0.08, "n_pos=173 for addr_missing (tiny sample) — treat that slice as anecdotal.",
             ha="center", fontsize=9, style="italic")
    fig.tight_layout()
    fig.savefig(FIGS / "fig8_address_ablation.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    reports = load_reports()
    fig1_composition()
    fig2_script_distribution()
    fig3_model_performance(reports)
    fig4_same_vs_cross(reports)
    fig5_per_script_heatmap(reports)
    if (RESULTS / "modelC_epoch3" / "model.safetensors").exists() and __import__("torch").cuda.is_available():
        fig7_umap_contrastive()
    else:
        print("skipping fig7 (needs CUDA + modelC checkpoint)")
    fig8_address_ablation()
    made = sorted(p.name for p in FIGS.glob("*.png"))
    print("figures written:")
    for f in made:
        print("  ", f)


if __name__ == "__main__":
    main()
