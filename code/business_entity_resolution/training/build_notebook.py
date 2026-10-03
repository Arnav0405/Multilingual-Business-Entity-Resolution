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
- **ADDR** — phase 10: contrastive + hard negatives trained on `name_address` variant
  (`Business name: ... | Address: ...` raw concatenation, MAX_LENGTH 128); answers
  "does contextual address information improve multilingual entity resolution?"

**Local artifacts this notebook uploads from your machine** (left panel → Files, or
`google.colab.files.upload()`):

1. `data/entity_resolution_subset/{train,val,test}_pairs.tsv` (~31 MB total)
2. *(optional, phase C/D/ADDR)* nothing else — the model downloads from HF hub directly

**Local artifacts the notebook produces for download back** (~15 MB each):

- `modelB_scores.parquet`, `modelC_scores.parquet`, `modelD_scores.parquet`, `modelA_name_address_scores.parquet` — the score book
- *(phase C/D/ADDR)* `modelC_checkpoint.zip` / `modelD_checkpoint.zip` / `modelA_name_address_checkpoint.zip` — fine-tuned weights, used
  by local scoring/eval, no need to keep them on Colab
"""),

    md("""## Setup
"""),

    code("""PHASE = "B"   # "B" | "C" | "D" | "ADDR"
TEXT_VARIANT = "name"   # "name" | "name_address" (phase 10 experiment: set TEXT_VARIANT="name_address" with PHASE="ADDR")
assert PHASE in ("B", "C", "D", "ADDR")
if PHASE == "ADDR":
    TEXT_VARIANT = "name_address"   # forced: the address experiment IS the variant
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
    # NaN (empty optional address fields) must never survive: coerce to ''
    def clean(col):
        return df[col].fillna('').astype(str)
    if variant == 'name':
        return clean(side + '_name')
    return 'Business name: ' + clean(side + '_name') + ' | Address: ' + clean(side + '_address')

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

    md("""## Phase C / D / ADDR — contrastive training
C: in-batch MNRL only. D: also appends stored hard negatives per sample.
ADDR: D's trainer but on `name_address` strings with MAX_LENGTH=128
(raw `Business name: ... | Address: ...` concatenation, no normalization).
"""),
    code("""import math, random
from torch.utils.data import DataLoader, Dataset

MAX_LENGTH = 128 if PHASE == 'ADDR' else 64   # address variant needs longer inputs

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
                'anchor': str(pos['s1_name'].iloc[0]) if pd.notna(pos['s1_name'].iloc[0]) else '',
                'anchor_address': str(pos['s1_address'].iloc[0]) if ('s1_address' in pos.columns and pd.notna(pos['s1_address'].iloc[0])) else '',
                'positives': [str(x) if pd.notna(x) else '' for x in pos['candidate_name'].tolist()],
                'positives_address': [str(x) if pd.notna(x) else '' for x in (pos['candidate_address'].tolist() if 'candidate_address' in pos.columns else [])],
                'hards': [str(x) if pd.notna(x) else '' for x in hard['candidate_name'].tolist()],
                'hards_address': [str(x) if pd.notna(x) else '' for x in (hard['candidate_address'].tolist() if 'candidate_address' in hard.columns else [])],
            })
    def __len__(self):
        return len(self.groups)
    def __getitem__(self, i):
        return self.groups[i]

def _variant(item, kind, j, variant):
    '''kind: anchor|pos|hard; j: which candidate within the kind.
    name variant: plain candidate_name; name_address: raw concat string.
    Empty (NaN) addresses are coerced to '' before concatenation.'''
    if variant == 'name':
        return str(item['anchor'] if kind == 'anchor' else (
            item['positives'][j] if kind == 'pos' else item['hards'][j]))
    addr = ''
    if kind == 'anchor':
        addr = str(item.get('anchor_address', '') or '')
        return 'Business name: ' + str(item['anchor']) + ' | Address: ' + addr
    if kind == 'pos':
        addr = str(item['positives_address'][j] or '') if j < len(item['positives_address']) else ''
        return 'Business name: ' + str(item['positives'][j]) + ' | Address: ' + addr
    addr = str(item['hards_address'][j] or '') if j < len(item['hards_address']) else ''
    return 'Business name: ' + str(item['hards'][j]) + ' | Address: ' + addr

def collate(batch, tokenizer, max_length=MAX_LENGTH, max_pos=2, max_extra_neg=4, variant=TEXT_VARIANT):
    '''Flattens each group into its top `max_pos` positives (anchor repeated
    per positive row); hard negatives are aligned PER ANCHOR ROW — exactly
    `max_extra_neg` per row (uniform K), the group's own stored hards first
    and the batch pool filling the rest (a pool entry is a valid negative
    for any anchor). Keeps h.shape[0] == a.shape[0] * K, which the train
    loop's view() reshape needs. `max_extra_neg=0` for phase C.
    variant selects raw-name vs name+address concatenation.'''
    anchors, positives, group_hards = [], [], []
    for item in batch:
        hs = [_variant(item, 'hard', j, variant)
              for j in range(min(len(item['hards']), max_extra_neg))] if max_extra_neg else []
        for p_idx, p in enumerate(item['positives'][:max_pos]):
            anchors.append(_variant(item, 'anchor', p_idx, variant))
            positives.append(_variant(item, 'pos', p_idx, variant))
            group_hards.append(hs)
    hards = []
    if max_extra_neg and any(group_hards):
        pool = []
        for hs in group_hards:
            for h in hs:
                if h not in pool:
                    pool.append(h)
        for hs in group_hards:
            own = hs[:]
            fill_count = max_extra_neg - len(own)
            if fill_count > 0:
                distinct = [h for h in pool if h not in own]
                if len(distinct) >= fill_count:
                    extra = distinct[:fill_count]
                else:
                    src = distinct or pool
                    extra = [src[i % len(src)] for i in range(fill_count)] if src else []
                own = own + extra
            hards.extend(own[:max_extra_neg])
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

    code("""if PHASE in ('C', 'D', 'ADDR'):
    # PHASE 'ADDR' = Model A' (name+address): same trainer as D (hard negatives),
    # different text variant and longer inputs.
    train_df = load_pairs('train')
    max_hard = 0 if PHASE == 'C' else 4
    ds = GroupPairs(train_df, max_hard=max_hard)
    print(f'{len(ds):,} S1 groups with >= 1 positive')
    loader = DataLoader(ds, batch_size=32 if PHASE == 'ADDR' else 64, shuffle=True,
        collate_fn=lambda b: collate(b, tok, max_pos=2, max_extra_neg=max_hard),
        num_workers=2)
    steps = max(len(loader), 1)
    # NOTE: model stays fp32; autocast is applied per-forward in the train loop
    # (fresh fp32 model, NOT the half=model above which is for scoring only)
    model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE).eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)
    EPOCHS = 3
    SCHED_TOTAL = steps * EPOCHS
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=3e-5, total_steps=SCHED_TOTAL)
"""),

    code("""if PHASE in ('C', 'D', 'ADDR'):
    import time
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time(); model.train(); skipped = 0
        total, n = 0.0, 0
        for step, batch in enumerate(loader):
            batch = {k: (v.to(DEVICE) if v is not None else None) for k, v in batch.items()}
            # fp32 params + fp16 autocast on encoder forwards only (fp16 master
            # weights diverged pre-fix: attempt 1 came back all-NaN)
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                a = model(**batch['anchor']).last_hidden_state[:, 0]
                p = model(**batch['positive']).last_hidden_state[:, 0]
            a = F.normalize(a.float(), dim=-1); p = F.normalize(p.float(), dim=-1)
            extra = None
            if batch['hard'] is not None:
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    h = model(**batch['hard']).last_hidden_state[:, 0]
                h = F.normalize(h.float(), dim=-1)
                extra = h.view(a.shape[0], h.shape[0] // a.shape[0], -1)
            loss = mnrl(a, p, extra_neg=extra)
            if not torch.isfinite(loss):
                skipped += 1; optimizer.zero_grad(); continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); scheduler.step(); optimizer.zero_grad()
            total += loss.item(); n += 1
            if step == 49:
                assert torch.isfinite(loss), 'loss non-finite after 50 steps -- divergence'
                print(f'    [health] step-50 loss={loss.item():.4f}')
        if skipped:
            print(f'    [warn] skipped {skipped} non-finite batches')
        print(f'epoch {epoch}/{EPOCHS}: loss={total/max(n,1):.4f} ({time.time()-t0:.0f}s)')
        ckpt = f'/content/model{PHASE}_epoch{epoch}'
        model.save_pretrained(ckpt, safe_serialization=True)
        tok.save_pretrained(ckpt)
"""),

    md("""## Phase C / D / ADDR — score with the fine-tuned encoder
"""),
    code("""if PHASE in ('C', 'D', 'ADDR'):
    # load FRESH from the fine-tuned checkpoint (fp32->half for scoring only)
    LATEST = f'/content/model{PHASE}_epoch{EPOCHS}'
    model = AutoModel.from_pretrained(LATEST).to(DEVICE).eval()
    model = model.half()
    # NaN check BEFORE wasting 30 minutes scoring
    import torch as _t
    bad = [k for k, v in model.state_dict().items() if _t.isnan(v.float()).any()]
    assert not bad, f'checkpoint has NaN tensors: {bad[:5]} -- retrain, do not score'
    scores = {}
    for split in ('train', 'val', 'test'):
        df = load_pairs(split)
        scores[split] = score_split(df, TEXT_VARIANT)
        assert np.isfinite(scores[split]).all(), f'{split}: non-finite scores'
        print(split, len(df), 'pairs, mean cos', float(np.mean(scores[split])))
    out_name = 'modelA_name_address_scores.parquet' if PHASE == 'ADDR' else f'model{PHASE}_scores.parquet'
    pd.concat(
        [pd.DataFrame({'split': s, 'row_idx': load_pairs(s)['row_idx'], 'score': scores[s]}) for s in scores]
    ).to_parquet(f'/content/{out_name}', index=False)
    print(f'wrote /content/{out_name}')
"""),

    md("""## Download artifacts
"""),
    code("""from google.colab import files
NAME = 'modelA_name_address' if PHASE == 'ADDR' else f'model{PHASE}'
files.download(f'/content/{NAME}_scores.parquet')
# phase C/D/ADDR checkpoint archive:
# !cd /content && zip -rq {NAME}_epoch3.zip {NAME}_epoch3/
# files.download(f'/content/{NAME}_epoch3.zip')
"""),

    md("""## Sanity checks (run these in Colab before the long cells)
"""),
    code("""rows = {s: len(load_pairs(s)) for s in ('train', 'val', 'test')}
raw  = {s: len(pd.read_csv(DATA / f'{s}_pairs.tsv', sep='\\t')) for s in ('train', 'val', 'test')}
assert rows == raw, f'score-book row counts must equal the TSVs on disk: {rows} vs {raw}'
assert set(load_pairs('train')['label'].unique()) == {0, 1}
assert len(load_pairs('train')['s1_id'].unique()) == 6995, 'train should have 6995 S1 entities'
assert len(load_pairs('val')['s1_id'].unique()) == 1495, 'val should have 1495 S1 entities'
assert len(load_pairs('test')['s1_id'].unique()) == 1510, 'test should have 1510 S1 entities'
print('data shape ok:', rows, '; PHASE =', PHASE, '; TEXT_VARIANT =', TEXT_VARIANT)
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
