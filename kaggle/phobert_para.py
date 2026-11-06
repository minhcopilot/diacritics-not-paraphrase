"""Kaggle GPU: PhoBERT trên SMS thật và tin do LLM viết lại, cùng cách chia với scripts/eval_para.py.

Đầu vào: sms_input.csv và para_all_checked.csv (đã đánh dấu bản viết lại hỏng, sinh bởi eval_para.py).
Mô hình: sach, tang_cuong (thêm bản bỏ dấu), tang_cuong_llm (thêm bản bỏ dấu và bản viết lại của các LLM khác LLM bị test).
Ghi kết quả sau từng lần chạy.
"""
import subprocess
import sys

subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'pyvi', 'sentence-transformers'], check=True)

import glob
import os
import random
import time
import unicodedata

import numpy as np
import pandas as pd
import torch
from pyvi import ViTokenizer
from sklearn.metrics import f1_score, roc_auc_score
from transformers import AutoModelForSequenceClassification as M
from transformers import AutoTokenizer as Tk
from transformers import get_linear_schedule_with_warmup as sched

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
S = pd.read_csv(glob.glob('/kaggle/input/**/sms_input.csv', recursive=True)[0])
P = pd.read_csv(glob.glob('/kaggle/input/**/para_all_checked.csv', recursive=True)[0])
# Độ giữ nghĩa giữa tin gốc và bản viết lại (LaBSE, cosine), tính cho mọi dòng kể cả dòng bị loại
from sentence_transformers import SentenceTransformer
lab = SentenceTransformer('sentence-transformers/LaBSE', device='cuda')
eg = lab.encode(list(P.goc.astype(str)), batch_size=128, normalize_embeddings=True, convert_to_numpy=True)
ep = lab.encode(list(P.text.fillna('').astype(str)), batch_size=128, normalize_embeddings=True, convert_to_numpy=True)
P['labse'] = (eg * ep).sum(1)
P[['id', 'model', 'prompt', 'hong', 'labse']].to_csv('/kaggle/working/labse_sim.csv', index=False)
print(P.groupby(['model', 'prompt', 'hong']).labse.describe().round(3).to_string(), flush=True)
del lab
torch.cuda.empty_cache()
P = P[~P.hong].copy()
S['text'] = S.text.map(lambda t: unicodedata.normalize('NFC', t))
P['text'] = P.text.fillna('').astype(str).map(lambda t: unicodedata.normalize('NFC', t))
NAME, MAXLEN, BS, EP, LR, SEEDS = 'vinai/phobert-base-v2', 160, 16, 3, 2e-5, 3
tok = Tk.from_pretrained(NAME)
dev = 'cuda'
OUT = '/kaggle/working/phobert_para.csv'
LLMS = sorted(P.model.unique())
segc = {}


def strip_dia(t):
    t = t.replace('đ', 'd').replace('Đ', 'D')
    return unicodedata.normalize('NFC', ''.join(c for c in unicodedata.normalize('NFD', t) if unicodedata.category(c) != 'Mn'))


def enc(X):
    out = []
    for x in X:
        if x not in segc:
            segc[x] = tok(ViTokenizer.tokenize(x), add_special_tokens=False, truncation=False)['input_ids'][:MAXLEN - 2]
        out.append([tok.cls_token_id] + segc[x] + [tok.sep_token_id])
    return out


def batches(ids, y, shuffle, seed=0):
    o = np.random.default_rng(seed).permutation(len(ids)) if shuffle else np.argsort([len(i) for i in ids])
    for s in range(0, len(ids), BS):
        b = o[s:s + BS]
        a = torch.full((len(b), max(len(ids[j]) for j in b)), tok.pad_token_id)
        for n, j in enumerate(b):
            a[n, :len(ids[j])] = torch.tensor(ids[j])
        yield a.to(dev), (a != tok.pad_token_id).long().to(dev), torch.tensor(y[b]).to(dev), b


@torch.no_grad()
def score(m, X):
    ids = enc(X)
    m.eval()
    p = np.zeros(len(ids))
    for a, mk, _, b in batches(ids, np.zeros(len(ids), dtype=int), False):
        with torch.amp.autocast('cuda'):
            l = m(input_ids=a, attention_mask=mk).logits
        p[b] = (l[:, 1] - l[:, 0]).float().cpu().numpy()
    return p


def train(X, y, Xv, yv, seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    Xt = enc(X)
    m = M.from_pretrained(NAME, num_labels=2).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=.01)
    nb = int(np.ceil(len(Xt) / BS))
    sc = sched(opt, int(.1 * nb * EP), nb * EP)
    scaler = torch.amp.GradScaler('cuda')
    w = torch.tensor([len(y) / (2 * (y == 0).sum()), len(y) / (2 * (y == 1).sum())], dtype=torch.float, device=dev)
    best, bs = -1, None
    for e in range(EP):
        m.train()
        for a, mk, yy, _ in batches(Xt, y, True, seed * 100 + e):
            with torch.amp.autocast('cuda'):
                l = torch.nn.functional.cross_entropy(m(input_ids=a, attention_mask=mk).logits, yy, weight=w)
            opt.zero_grad()
            scaler.scale(l).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sc.step()
        f = roc_auc_score(yv, score(m, Xv))
        if f > best:
            best, bs = f, {k: v.cpu().clone() for k, v in m.state_dict().items()}
    m.load_state_dict(bs)
    pv = score(m, Xv)
    th = max(np.unique(np.quantile(pv, np.linspace(0, 1, 201))), key=lambda t: f1_score(yv, pv >= t, zero_division=0))
    return m, th


X, Y, IDS = list(S.text), S.label.values, S.id.values
CD = S.set_index('id').text.map(lambda t: strip_dia(t) != t)
rows = []
for s in range(SEEDS):
    o = np.random.default_rng(s).permutation(len(X))
    a, b, c = o[:int(.8 * len(o))], o[int(.8 * len(o)):int(.9 * len(o))], o[int(.9 * len(o)):]
    tr_ids, te_ids = set(IDS[a]), set(IDS[c])
    Tr, Va = [X[i] for i in a], [X[i] for i in b]
    te = S[S.id.isin(te_ids)]
    plan = [('sach', None, Tr, Y[a]), ('tang_cuong', None, Tr + [strip_dia(t) for t in Tr], np.r_[Y[a], Y[a]])]
    for held in LLMS:
        A = P[P.id.isin(tr_ids) & (P.model != held)]
        T2 = Tr + list(A.text)
        plan.append(('tang_cuong_llm', held, T2 + [strip_dia(t) for t in T2], np.r_[Y[a], A.label.values, Y[a], A.label.values]))
    for name, held, T, y in plan:
        t0 = time.time()
        m, th = train(T, y, Va, Y[b], s)
        sets = {} if held else {('goc', '-', 'nguyen_van'): te, ('goc', '-', 'bo_dau'): te.assign(text=te.text.map(strip_dia))}
        for (llm, pr), g in P[P.id.isin(te_ids)].groupby(['model', 'prompt']):
            if held and llm != held:
                continue
            sets[(llm, pr, 'nguyen_van')] = g
            sets[(llm, pr, 'bo_dau')] = g.assign(text=g.text.map(strip_dia))
        for (llm, pr, bien), g in sets.items():
            yy, cd = g.label.values, CD.loc[g.id].values
            sc = score(m, list(g.text))
            p = sc >= th
            rows.append(dict(seed=s, model='phobert_' + name, llm=llm, prompt=pr, bien=bien, n_pos=int(yy.sum()), n_neg=int((1 - yy).sum()),
                             recall=p[yy == 1].mean(), recall_co_dau=p[(yy == 1) & cd].mean(), recall_khong_dau=p[(yy == 1) & ~cd].mean(),
                             fpr=p[yy == 0].mean() if (yy == 0).any() else np.nan,
                             f1=f1_score(yy, p) if (yy == 0).any() else np.nan,
                             auroc=roc_auc_score(yy, sc) if (yy == 0).any() else np.nan))
        pd.DataFrame(rows).to_csv(OUT, index=False)
        print(s, name, held, len(T), 'mẫu train', round(time.time() - t0), 'giây', flush=True)
        del m
        torch.cuda.empty_cache()
print('xong')
