import json
import pathlib
import time

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

t0 = time.time()
tok = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-small")
model = AutoModel.from_pretrained("intfloat/multilingual-e5-small")
print("download+load %.1fs" % (time.time() - t0))


@torch.no_grad()
def emb(s, prefix):
    h = model(**tok(prefix + " " + s, return_tensors="pt")).last_hidden_state[:, 0]
    return F.normalize(h, dim=1).squeeze(0)


def cos(a, b):
    return float(a @ b)


pairs = [
    ("latin_latin", emb("Silver Solutions Private Limited", "query"),
     emb("Alpha International Private Limited", "query")),
    ("cross_silver_devanagari", emb("Silver Solutions Private Limited", "query"),
     emb("सिल्वर सॉल्यूशंस प्राइवेट लिमिटेड", "passage")),
    ("cross_alpha_telugu", emb("Alpha International Private Limited", "query"),
     emb("ఆల్ఫా ఇంటర్నేషనల్ ప్రైవేట్ లిమిటెడ్", "passage")),
    ("hardneg_silver_trading", emb("Silver Solutions Private Limited", "query"),
     emb("Silver Solutions Trading Private Limited", "passage")),
]

out = {}
for n, a, b in pairs:
    out["cosine_" + n] = round(cos(a, b), 4)
out.update({
    "dim": 384,
    "model_params_M": round(sum(p.numel() for p in model.parameters()) / 1e6, 1),
    "torch": torch.__version__,
    "cuda": torch.cuda.is_available(),
})
print(json.dumps(out, indent=2, ensure_ascii=False))

p = pathlib.Path("results")
p.mkdir(exist_ok=True)
(p / "env_smoke.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
