"""Bổ sung cho phobert_para.py: chỉ chạy tang_cuong_llm, đo cả trên tập gốc, rồi đo kích thước và độ trễ CPU (fp32, INT8 động).

Cùng cách chia với scripts/eval_para.py.

Đầu vào: sms_input.csv và para_all_checked.csv (đã đánh dấu bản viết lại hỏng, sinh bởi eval_para.py).
Mô hình: sach, tang_cuong (thêm bản bỏ dấu), tang_cuong_llm (thêm bản bỏ dấu và bản viết lại của các LLM khác LLM bị test).
Ghi kết quả sau từng lần chạy.
"""
import subprocess
import sys

subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'pyvi'], check=True)

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
P = P[~P.hong].copy()
S['text'] = S.text.map(lambda t: unicodedata.normalize('NFC', t))
P['text'] = P.text.fillna('').astype(str).map(lambda t: unicodedata.normalize('NFC', t))
NAME, MAXLEN, BS, EP, LR, SEEDS = 'vinai/phobert-base-v2', 160, 16, 3, 2e-5, 3
tok = Tk.from_pretrained(NAME)
dev = 'cuda'
OUT = '/kaggle/working/phobert_extra.csv'
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


def bench(m, th, te):
    """Kích thước và độ trễ trên CPU, từng tin một (batch 1), fp32 và INT8 động; kèm recall/FPR của bản INT8."""
    import copy
    import io
    cpu = copy.deepcopy(m).float().cpu().eval()
    q = torch.ao.quantization.quantize_dynamic(copy.deepcopy(cpu), {torch.nn.Linear}, dtype=torch.qint8)
    msgs = list(te.text)[:200]
    ids = enc(msgs)
    out = []
    for tag, mod in [('fp32', cpu), ('int8', q)]:
        buf = io.BytesIO()
        torch.save(mod.state_dict(), buf)
        for nt in (1, 4):
            torch.set_num_threads(nt)
            with torch.no_grad():
                for i in ids[:10]:
                    mod(input_ids=torch.tensor([i]))
                t = []
                for i in ids:
                    t0 = time.perf_counter()
                    mod(input_ids=torch.tensor([i]))
                    t.append(time.perf_counter() - t0)
            out.append(dict(dang=tag, threads=nt, mb=round(buf.getbuffer().nbytes / 2**20, 1), ms_tb=round(1000 * np.mean(t), 1), ms_p95=round(1000 * np.quantile(t, .95), 1)))
        torch.set_num_threads(4)
        for bien, T in [('nguyen_van', list(te.text)), ('bo_dau', [strip_dia(x) for x in te.text])]:
            with torch.no_grad():
                sc = []
                for i in enc(T):
                    l = mod(input_ids=torch.tensor([i])).logits[0]
                    sc.append(float(l[1] - l[0]))
            p = np.array(sc) >= th
            yy = te.label.values
            out.append(dict(dang=tag, bien=bien, recall=p[yy == 1].mean(), fpr=p[yy == 0].mean(), f1=f1_score(yy, p)))
    pd.DataFrame(out).to_csv('/kaggle/working/phobert_bench.csv', index=False)
    print(pd.DataFrame(out).to_string(), flush=True)


X, Y, IDS = list(S.text), S.label.values, S.id.values
CD = S.set_index('id').text.map(lambda t: strip_dia(t) != t)
rows = []
for s in range(SEEDS):
    o = np.random.default_rng(s).permutation(len(X))
    a, b, c = o[:int(.8 * len(o))], o[int(.8 * len(o)):int(.9 * len(o))], o[int(.9 * len(o)):]
    tr_ids, te_ids = set(IDS[a]), set(IDS[c])
    Tr, Va = [X[i] for i in a], [X[i] for i in b]
    te = S[S.id.isin(te_ids)]
    plan = []
    for held in LLMS:
        A = P[P.id.isin(tr_ids) & (P.model != held)]
        T2 = Tr + list(A.text)
        plan.append(('tang_cuong_llm', held, T2 + [strip_dia(t) for t in T2], np.r_[Y[a], A.label.values, Y[a], A.label.values]))
    for name, held, T, y in plan:
        t0 = time.time()
        m, th = train(T, y, Va, Y[b], s)
        sets = {('goc', '-', 'nguyen_van'): te, ('goc', '-', 'bo_dau'): te.assign(text=te.text.map(strip_dia))}
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
        if s == 0 and name == 'tang_cuong_llm' and held == LLMS[0]:
            bench(m, th, te)
        del m
        torch.cuda.empty_cache()
print('xong')
