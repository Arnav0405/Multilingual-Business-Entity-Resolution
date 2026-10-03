"""
Build the Colab notebook (phase 5-7): entity_resolution_colab.ipynb.

Also re-runnable any time the embedded code blocks change -- run this,
then re-upload the notebook to Colab.
"""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
NB_PATH = HERE / "entity_resolution_colab.ipynb"


def code(source: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": source}


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source}


CELLS = [
    md("""# Multilingual Cross-Script Business Entity Resolution — Colab (T4)

Phase-flagged notebook. Set `PHASE` and run all cells top-to-bottom.

- **B** — pretrained `intfloat/multilingual-e5-small`, no fine-tuning; score all three splits
- **C** — contrastive fine-tune (in-batch MNRL), then score with the fine-tuned encoder
- **D** — contrastive + explicit hard negatives, then score

**Local artifacts this notebook uploads from your machine** (left panel → Files, or
`google.colab.files.upload()`):

1. `data/entity_resolution_subset/{train,val,test}_pairs.tsv` (~31 MB total)
2. *(optional, phase C/D)* nothing else — the model downloads from HF hub directly

**Local artifacts the notebook produces for download back** (~15 MB each):

- `modelB_scores.parquet`, `modelC_scores.parquet`, `modelD_scores.parquet` — the score book
- *(phase C/D)* `modelC_checkpoint.zip` / `modelD_checkpoint.zip` — fine-tuned weights, used
  by local scoring/eval, no need to keep them on Colab
"""),

    md("""## Setup
"""),

    code("""PHASE = "B"   # "B" | "C" | "D"
TEXT_VARIANT = "name"   # "name" | "name_address" (phase 10 experiment)
assert PHASE in ("B", "C", "D")
assert TEXT_VARIANT in ("name", "name_address")
"""),

    code("""import torch, platform
print(torch.__version__, torch.cuda.is_available(), platform.machine())
assert torch.cuda.is_available(), 'Runtime → Change runtime type → T4 GPU'
print(torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).total_memory / 1e9)
"""),

    code("""!pip -q install transformers sentencepiece
"""),

    md("""## Data upload

Run once per session. In Colab: left Files panel → upload the 3 TSVs into `/content/data`.
"""),
    code("""import pathlib
DATA = pathlib.Path('/content/data')
for split in ('train', 'val', 'test'):
    p = DATA / f'{split}_pairs.tsv'
    assert p.exists(), f'missing {p} — upload the subset TSVs first'
    print(p, p.stat().st_size // 1024, 'KB')
"""),

    md("""## Shared helpers (mirrors src/encoder.py + baselines/evaluate.py)
"""),
    code("""import numpy as np, pandas as pd, re, torch, torch.nn.functional as F
from pathlib import Path
from transformers import AutoModel, AutoTokenizer

MODEL_NAME = 'intfloat/multilingual-e5-small'
QUERY, PASSAGE = 'query: ', 'passage: '
SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED)

DEVICE = 'cuda'
MAX_LENGTH = 64

def load_pairs(split):
    df = pd.read_csv(DATA / f'{split}_pairs.tsv', sep='\\t')
    df['row_idx'] = df.index
    return df

INDIC_RANGES = {
    'Devanagari': (0x0900, 0x097F), 'Bengali': (0x0980, 0x09FF),
    'Gurmukhi': (0x0A00, 0x0A7F), 'Gujarati': (0x0A80, 0x0AFF),
    'Oriya': (0x0B00, 0x0B7F), 'Tamil': (0x0B80, 0x0BFF),
    'Telugu': (0x0C00, 0x0C7F), 'Kannada': (0x0C80, 0x0CFF),
    'Malayalam': (0x0D00, 0x0D7F),
}
_script_res = {n: re.compile(f'[{chr(lo)}-{chr(hi)}]') for n, (lo, hi) in INDIC_RANGES.items()}
_cache = {}
def script_tags(s):
    if not isinstance(s, str) or not s:
        return frozenset({'<empty>'})
    hit = _cache.get(s)
    if hit is not None:
        return hit
    tags = {n for n, rx in _script_res.items() if rx.search(s)}
    if not tags:
        tags = {'Latin'}
    out = frozenset(tags)
    _cache[s] = out
    return out

def is_cross_script(a, b):
    return script_tags(a) != script_tags(b)

def apply_variant(df, side, variant):
    if variant == 'name':
        return df[side + '_name']
    return 'Business name: ' + df[side + '_name'] + ' | Address: ' + df[side + '_address']

@torch.no_grad()
def encode(texts, prefix='passage: ', batch_size=256):
    out = np.empty((len(texts), 384), dtype=np.float32)
    for i in range(0, len(texts), batch_size):
        batch = [prefix + t for t in texts[i:i+batch_size]]
        enc = tok(batch, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors='pt').to(DEVICE)
        emb = model(**enc).last_hidden_state[:, 0]
        out[i:i+len(emb)] = F.normalize(emb.float(), dim=-1).cpu().numpy().astype(np.float32)
        if (i // batch_size) % 20 == 0:
            print(f'  encode {i}/{len(texts)}', flush=True)
    return out

def rowwise_cos(a, b):
    return (torch.from_numpy(a) * torch.from_numpy(b)).sum(-1).numpy()

def score_split(df, variant='name'):
    left = encode(apply_variant(df, 's1', variant).tolist(), QUERY)
    right = encode(apply_variant(df, 'candidate', variant).tolist(), PASSAGE)
    return rowwise_cos(left, right)
"""),

    md("""## Model / tokenizer load
"""),
    code("""from transformers import AutoModel, AutoTokenizer
tok = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE).eval()
model = model.half()
print(model.config.num_hidden_layers, 'layers,', sum(p.numel() for p in model.parameters())/1e6, 'M params')
"""),

    md("""## Phase B — pretrained, no fine-tuning
"""),
    code("""if PHASE == 'B':
    scores = {}
    for split in ('train', 'val', 'test'):
        df = load_pairs(split)
        scores[split] = score_split(df, TEXT_VARIANT)
        print(split, len(df), 'pairs, mean cos', float(np.mean(scores[split])))
    pd.concat(
        [pd.DataFrame({'split': s, 'row_idx': load_pairs(s)['row_idx'], 'score': scores[s]}) for s in scores]
    ).to_parquet('/content/modelB_scores.parquet', index=False)
    print('wrote /content/modelB_scores.parquet')
"""),

    md("""## Phase C / D — contrastive training
C: in-batch MNRL only. D: also appends stored hard negatives per sample.
"""),
    code("""import math, random
from torch.utils.data import DataLoader, Dataset

class GroupPairs(Dataset):
    '''one item = one S1 group: anchor string, its positives, up to 4 stored hard negatives'''
    def __init__(self, df, max_hard=4):
        self.groups = []
        for s1_id, g in df.groupby('s1_id', sort=False):
            pos = g[g['label'] == 1]
            if not len(pos):
                continue
            neg = g[g['label'] == 0]
            hard = neg[neg['pair_type'] == 'hard_negative'].head(max_hard)
            self.groups.append({
                'anchor': str(pos['s1_name'].iloc[0]),
                'positives': [str(x) for x in pos['candidate_name'].tolist()],
                'hards': [str(x) for x in hard['candidate_name'].tolist()],
            })
    def __len__(self):
        return len(self.groups)
    def __getitem__(self, i):
        return self.groups[i]

def collate(batch, tokenizer, max_length=MAX_LENGTH, max_pos=2, max_extra_neg=4):
    '''Flattens each group into its top `max_pos` positives (anchor repeated);
    `max_extra_neg` hard negatives per group. `max_extra_neg=0` for phase C.'''
    anchors, positives, hards = [], [], []
    for item in batch:
        a = item['anchor']
        for p in item['positives'][:max_pos]:
            anchors.append(a); positives.append(p)
        for h in (item['hards'][:max_extra_neg] if max_extra_neg else []):
            hards.append(h)
    enc = lambda texts, prefix: tokenizer(
        [prefix + t for t in texts], padding=True, truncation=True,
        max_length=max_length, return_tensors='pt')
    return {
        'anchor': enc(anchors, QUERY),
        'positive': enc(positives, PASSAGE),
        'hard': enc(hards, PASSAGE) if hards else None,
    }

def mnrl(a, p, extra_neg=None, scale=20.0):
    '''Cross-entropy over cosine logits; in-batch + optional per-sample extra negatives.'''
    sim = a @ p.T * scale
    if extra_neg is not None:
        neg = torch.einsum('bd,bkd->bk', a, extra_neg) * scale
        sim = torch.cat([sim, neg], dim=1)
    labels = torch.arange(sim.shape[0], device=sim.device)
    return F.cross_entropy(sim, labels)

@torch.no_grad()
def dummy():
    pass
"""),

    code("""if PHASE in ('C', 'D'):
    train_df = load_pairs('train')
    max_hard = 4 if PHASE == 'D' else 0
    ds = GroupPairs(train_df, max_hard=max_hard)
    print(f'{len(ds):,} S1 groups with >= 1 positive')
    loader = DataLoader(ds, batch_size=64, shuffle=True,
        collate_fn=lambda b: collate(b, tok, max_pos=2, max_extra_neg=max_hard), num_workers=2)
    steps = max(len(loader), 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)
    SCHED_TOTAL = steps * 3   # default epochs=3
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=3e-5, total_steps=SCHED_TOTAL)
    EPOCHS = 3
"""),

    code("""if PHASE in ('C', 'D'):
    import time
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time(); model.train(); total, n = 0.0, 0
        for batch in loader:
            batch = {k: (v.to(DEVICE) if v is not None else None) for k, v in batch.items()}
            a = model(**batch['anchor']).last_hidden_state[:, 0]
            p = model(**batch['positive']).last_hidden_state[:, 0]
            a = F.normalize(a.float(), dim=-1); p = F.normalize(p.float(), dim=-1)
            extra = None
            if batch['hard'] is not None:
                h = model(**batch['hard']).last_hidden_state[:, 0]
                h = F.normalize(h.float(), dim=-1)
                extra = h.view(a.shape[0], h.shape[0] // a.shape[0], -1)
            loss = mnrl(a, p, extra_neg=extra)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); scheduler.step(); optimizer.zero_grad()
            total += loss.item(); n += 1
        print(f'epoch {epoch}/{EPOCHS}: loss={total/max(n,1):.4f} ({time.time()-t0:.0f}s)')
        ckpt = f'/content/model{PHASE}_epoch{epoch}'
        model.save_pretrained(ckpt, safe_serialization=True)
        tok.save_pretrained(ckpt)
"""),

    md("""## Phase C / D — score with the fine-tuned encoder
"""),
    code("""if PHASE in ('C', 'D'):
    LATEST = f'/content/model{PHASE}_epoch{EPOCHS}'
    model = AutoModel.from_pretrained(LATEST).to(DEVICE).eval().half()
    scores = {}
    for split in ('train', 'val', 'test'):
        df = load_pairs(split)
        scores[split] = score_split(df, TEXT_VARIANT)
        print(split, len(df), 'pairs, mean cos', float(np.mean(scores[split])))
    pd.concat(
        [pd.DataFrame({'split': s, 'row_idx': load_pairs(s)['row_idx'], 'score': scores[s]}) for s in scores]
    ).to_parquet(f'/content/model{PHASE}_scores.parquet', index=False)
    print(f'wrote /content/model{PHASE}_scores.parquet')
"""),

    md("""## Download artifacts
"""),
    code("""from google.colab import files
files.download('/content/model_scores.parquet')   # adjust filename per PHASE
# phase C/D also: files.download('/content/modelC_epoch3.zip') after archiving:
# !cd /content && zip -rq modelC_epoch3.zip modelC_epoch3/
"""),

    md("""## Sanity checks (run these in Colab before the long cells)
"""),
    code("""assert len(load_pairs('val')) == len(load_pairs('test')) == len(pd.read_csv(DATA / 'test_pairs.tsv', sep='\\t'))
assert set(load_pairs('train')['label'].unique()) == {0, 1}
print('data shape ok; PHASE =', PHASE, '; TEXT_VARIANT =', TEXT_VARIANT)
"""),

    md("""## End
"""),
]


def build() -> Path:
    nb = {
        "cells": CELLS,
        "metadata": {
            "colab": {"provenance": [], "gpuType": "T4"},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }
    NB_PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    return NB_PATH


if __name__ == "__main__":
    print("wrote", build())
