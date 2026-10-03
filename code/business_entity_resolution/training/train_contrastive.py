"""
Contrastive fine-tuning of multilingual-e5-small -- handoff sections 5-9.

Models:
    C: contrastive (in-batch MNRL), positives only
    D: contrastive + explicit hard negatives appended per-sample

Pure-transformers backend (zero extra deps on Colab T4). Runs locally
too (RTX 4060, fp16). Checkpoint per epoch under results/<--out>/epochN;
the notebook's scoring phase then writes the score book and the actual
evaluation happens locally via baselines/report.py.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "code" / "business_entity_resolution" / "src"))

from dataset import apply_text_variant, load_pairs  # noqa: E402
from losses import mnrl  # noqa: E402

SUBSET = REPO_ROOT / "data" / "entity_resolution_subset"
RESULTS = REPO_ROOT / "code" / "business_entity_resolution" / "results"

MODEL_NAME = "intfloat/multilingual-e5-small"


# ============================================================
# Dataset: one item = one S1 group with pre-extracted raw strings
# ============================================================

class GroupPairs(Dataset):
    """One item per S1 group (anchor + its positives + up to `max_hard`
    hard-negative candidate strings). Zero-match S1s are skipped: they
    contribute nothing to an alignment loss."""

    def __init__(self, df: pd.DataFrame, max_hard: int = 4):
        groups = []
        for s1_id, g in df.groupby("s1_id", sort=False):
            pos = g[g["label"] == 1]
            if not len(pos):
                continue
            neg = g[g["label"] == 0]
            hard = neg[neg["pair_type"] == "hard_negative"].head(max_hard)
            groups.append({
                "anchor": str(pos["s1_name"].iloc[0]),
                "positives": [str(x) for x in pos["candidate_name"].tolist()],
                "hards": [str(x) for x in hard["candidate_name"].tolist()] if max_hard else [],
            })
        self.groups = groups

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, i: int) -> dict:
        return self.groups[i]


def collate(batch, tokenizer, max_length=64, max_pos=2, max_extra_neg=4):
    """
    Returns tensors:
        anchor  [B * max_pos] query strings (anchor repeated per positive)
        pos     [B * max_pos] passage strings
        hard    [A * K]      passage strings (K uniform per anchor row)
    MNRL treats in-batch positives as easy negatives; `hard` entries
    append per-sample extra negatives via losses.mnrl.

    Anchors are flattened PER POSITIVE ROW (a group with 2 positives
    contributes 2 anchor rows), so hard negatives must be aligned per
    anchor row too: exactly `max_extra_neg` per row (uniform K), the
    group's own stored hards first, then the batch pool fills the rest.
    A pool entry is "some other business" — a valid extra negative for
    any anchor, so cross-assignment adds no label noise. This keeps
    h.shape[0] == a.shape[0] * K, which the train loop's view() needs.
    """
    anchors, positives, group_hards = [], [], []
    for item in batch:
        a = item["anchor"]
        hs = list(item["hards"][:max_extra_neg]) if max_extra_neg else []
        for p in item["positives"][:max_pos]:
            anchors.append(a)
            positives.append(p)
            group_hards.append(hs)

    hards: list[str] = []
    if max_extra_neg and any(group_hards):
        pool = []
        for hs in group_hards:
            for h in hs:
                if h not in pool:
                    pool.append(h)
        # drop a hard that is exactly equal to its OWN row's positive
        # (never train "same string = negative"); other rows' positives are
        # fine — they are in-batch negatives anyway.
        for hs in group_hards:
            own = [h for h in hs if h not in pool or True]
            own = hs[:]
            fill_count = max_extra_neg - len(own)
            if fill_count > 0:
                distinct = [h for h in pool if h not in own]
                if len(distinct) >= fill_count:
                    extra = distinct[:fill_count]
                else:
                    # with-replacement fill to reach exactly K: a repeated
                    # negative string is still a negative; a ragged hard
                    # tensor crashes the train loop
                    src = distinct or pool
                    extra = [src[i % len(src)] for i in range(fill_count)] if src else []
                own = own + extra
            hards.extend(own[:max_extra_neg])
    K = max_extra_neg if hards else 0

    enc = lambda texts, prefix: tokenizer(
        [prefix + t for t in texts], padding=True, truncation=True,
        max_length=max_length, return_tensors="pt",
    )
    return {
        "anchor": enc(anchors, "query: "),
        "positive": enc(positives, "passage: "),
        "hard": enc(hards, "passage: ") if hards else None,
    }


# ============================================================
# Train loop
# ============================================================

def train_one_epoch(model, loader, optimizer, scheduler, device) -> float:
    model.train()
    total, n, skipped = 0.0, 0, 0
    optimizer.zero_grad()
    for step, batch in enumerate(loader):
        batch = {k: (v.to(device) if v is not None else None) for k, v in batch.items()}
        # fp32 master weights; fp16 autocast only on the encoder forwards
        with torch.autocast(device_type=device if device != "cpu" else "cpu",
                            dtype=torch.float16, enabled=(device != "cpu")):
            a_raw = model(**batch["anchor"]).last_hidden_state[:, 0]
            p_raw = model(**batch["positive"]).last_hidden_state[:, 0]
        a_emb = torch.nn.functional.normalize(a_raw.float(), dim=-1)
        p_emb = torch.nn.functional.normalize(p_raw.float(), dim=-1)
        extra = None
        if batch["hard"] is not None:
            with torch.autocast(device_type=device if device != "cpu" else "cpu",
                                dtype=torch.float16, enabled=(device != "cpu")):
                h_raw = model(**batch["hard"]).last_hidden_state[:, 0]
            h_emb = torch.nn.functional.normalize(h_raw.float(), dim=-1)
            K = h_emb.shape[0] // a_emb.shape[0]
            extra = h_emb.view(a_emb.shape[0], K, -1)
        loss = mnrl(a_emb, p_emb, extra_neg_emb=extra)
        # NaN guard: one poisoned batch costs one step, not the whole run
        if not torch.isfinite(loss):
            skipped += 1
            optimizer.zero_grad()
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        total += loss.item()
        n += 1
        if step == 49:
            assert torch.isfinite(loss), "loss non-finite after 50 steps -- divergence, abort early"
            print(f"    [health] step-50 loss={loss.item():.4f} (finite)")
    if skipped:
        print(f"    [warn] skipped {skipped} non-finite-loss batches")
    return total / max(n, 1)


def save_checkpoint(model, tokenizer, out: Path, epoch: int) -> None:
    ckpt = out / f"epoch{epoch}"
    ckpt.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ckpt)
    tokenizer.save_pretrained(ckpt)
    (ckpt / "trainer_state.json").write_text(
        json.dumps({"epoch": epoch, "saved_at": time.time()}), encoding="utf-8"
    )
    print(f"[ckpt] {ckpt}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="C", choices=["C", "D"])
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--max-pos", type=int, default=2)
    ap.add_argument("--max-hard", type=int, default=4)
    ap.add_argument("--out", default="modelC")
    ap.add_argument("--limit-groups", type=int, default=0, help="cap S1 groups for smoke tests")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)
    random.seed(42)

    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    # fp32 master weights (training); inference scoring may use half separately

    df = load_pairs(SUBSET / "train_pairs.tsv")
    if args.limit_groups:
        keep = set(df["s1_id"].drop_duplicates().head(args.limit_groups))
        df = df[df["s1_id"].isin(keep)].reset_index(drop=True)
        print(f"[smoke] {args.limit_groups} groups -> {len(df)} pairs")

    max_hard = args.max_hard if args.model == "D" else 0
    ds = GroupPairs(df, max_hard=max_hard)
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=lambda b: collate(b, tok, max_pos=args.max_pos, max_extra_neg=max_hard),
    )

    steps_per_epoch = max(math.ceil(len(loader)), 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=steps_per_epoch * args.epochs
    )

    out = RESULTS / args.out
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        loss = train_one_epoch(model, loader, optimizer, scheduler, device)
        print(f"epoch {epoch}/{args.epochs}: loss={loss:.4f} ({time.time() - t0:.0f}s)")
        save_checkpoint(model, tok, out, epoch)

    print("[done] final checkpoint dir:", out)


if __name__ == "__main__":
    main()
