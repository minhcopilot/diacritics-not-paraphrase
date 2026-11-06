"""Đánh giá bộ lọc SMS lừa đảo trên tin do LLM viết lại (sinh bởi kaggle/para_kernel).

Cách chia giống exp_p3_transfer.py (cùng thứ tự tin sau khi bỏ trùng, 5 seed, 80/10/10), nên số liệu so được với nhau.
- Kiểm tra chất lượng bản viết lại: rỗng, từ chối, lệch độ dài, mất <URL>, đổi kiểu có dấu/không dấu.
- Tập test: tin gốc, bản viết lại trung tính (cả hai lớp, đo được FPR), bản viết lại né lọc (chỉ tin dương),
  và bản bỏ dấu của từng loại.
- Mô hình: sach, tang_cuong (thêm bản bỏ dấu), chuan_hoa (bỏ dấu khi học và khi đoán),
  tang_cuong_llm (chuan_hoa cộng bản viết lại của tin train, sinh bởi các LLM khác LLM đang bị test).
"""
import glob
import pathlib
import re
import sys
import unicodedata
import warnings

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score

warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')
pd.set_option('display.width', 230, 'display.max_rows', 300, 'display.max_columns', 40)
SRC, PARA_DIR, OUT = sys.argv[1], pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3])  # tùy chọn: argv[4] labse_sim.csv, argv[5] ngưỡng
OUT.mkdir(parents=True, exist_ok=True)
REFUSE = re.compile(r'(xin lỗi|tôi không thể|không thể (giúp|hỗ trợ|thực hiện)|i cannot|i can\'t|i\'m sorry|as an ai)', re.I)


def strip_dia(t):
    t = t.replace('đ', 'd').replace('Đ', 'D')
    return unicodedata.normalize('NFC', ''.join(c for c in unicodedata.normalize('NFD', t) if unicodedata.category(c) != 'Mn'))


EN = set('the you your to and of for a is are we our this that with have has be will please click account money from it'.split())


def tieng_anh(t):
    w = re.findall(r'[a-zA-Z]+', t.lower())
    return len(w) >= 5 and sum(x in EN for x in w) / len(w) > .15


def bo_cau_dan(t):
    """Bỏ dòng dẫn kiểu 'Bản sao tin nhắn sau:' mà LLM tự thêm ở đầu."""
    L = t.strip().split('\n')
    if len(L) > 1 and L[0].strip().endswith(':') and len(L[0]) < 80:
        t = '\n'.join(L[1:])
    return re.sub(r'^(dạ,?\s*)?(xin gửi thông tin sau|bản sao tin nhắn sau|đây là tin nhắn( đã)? viết lại)\s*:\s*', '', t.strip(), flags=re.I).strip()


def roi(t):
    """Tin gốc bị làm rối (kiểu chèn ký tự, bỏ khoảng trắng), dùng lại heuristic của pilot_sms.py."""
    s = re.sub(r'[^a-z]', '', strip_dia(t).lower())
    jam = sum(a == b for a, b in zip(s, s[1:])) / (len(s) - 1) if len(s) >= 8 else 0.
    nospace = len(re.findall(r'\s', t)) / max(1, len(t)) < .03 and len(t) > 40
    sym = len(re.findall(r'[^\w\s<>\[\]/.,:()-]', t)) / max(1, len(t)) > .15
    return jam > .2 or nospace or sym


def has_dia(t):
    return strip_dia(t) != t


def grams(t, n=3):
    t = re.sub(r'\s+', ' ', t.lower())
    return {t[i:i + n] for i in range(max(1, len(t) - n + 1))}


def jaccard(a, b):
    A, B = grams(a), grams(b)
    return len(A & B) / max(1, len(A | B))


S = pd.read_csv(SRC)
S['text'] = S.text.map(lambda t: unicodedata.normalize('NFC', t))
S['co_dau'] = S.text.map(has_dia)
P = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(PARA_DIR / 'para_*.csv')))], ignore_index=True)
P['text'] = P.text.fillna('').astype(str).map(lambda t: bo_cau_dan(unicodedata.normalize('NFC', t)))
P = P.merge(S[['id', 'label', 'text', 'co_dau']].rename(columns={'text': 'goc'}), on='id')

# --- Chất lượng bản viết lại ---
P['rong'] = P.text.str.strip().str.len() < 10
P['tu_choi'] = P.text.map(lambda t: bool(REFUSE.search(t[:200])))
P['ti_le_dai'] = P.text.str.len() / P.goc.str.len()
P['lech_dai'] = ~P.ti_le_dai.between(.3, 3)
P['mat_url'] = P.goc.str.contains('<URL>', regex=False) & ~P.text.str.contains('<URL>', regex=False)
P['doi_dau'] = P.text.map(has_dia) != P.co_dau
P['jaccard'] = [jaccard(a, b) for a, b in zip(P.goc, P.text)]
P['chep'] = P.jaccard >= .9
P['tieng_anh'] = P.text.map(tieng_anh)
P['goc_roi'] = P.goc.map(roi)
P['hong'] = P.rong | P.tu_choi | P.lech_dai | P.chep | P.tieng_anh | P.goc_roi
if len(sys.argv) > 4:
    # Phân tích độ nhạy: loại thêm bản viết lại có độ giống nghĩa LaBSE dưới ngưỡng (labse_sim.csv từ kernel PhoBERT)
    L = pd.read_csv(sys.argv[4])
    P = P.merge(L[['id', 'model', 'prompt', 'labse']], on=['id', 'model', 'prompt'], how='left')
    P['lech_nghia'] = P.labse < float(sys.argv[5])
    P['hong'] = P.hong | P.lech_nghia
q = P.groupby(['model', 'prompt']).agg(n=('id', 'size'), rong=('rong', 'mean'), tu_choi=('tu_choi', 'mean'), lech_dai=('lech_dai', 'mean'),
                                       mat_url=('mat_url', 'mean'), doi_dau=('doi_dau', 'mean'), chep=('chep', 'mean'), tieng_anh=('tieng_anh', 'mean'),
                                       goc_roi=('goc_roi', 'mean'), jaccard_tv=('jaccard', 'median'), hong=('hong', 'mean')).round(3)
print('== Chất lượng bản viết lại (tỷ lệ)\n', q.to_string())
P.to_csv(OUT / 'para_all_checked.csv', index=False)
P = P[~P.hong].copy()
MODELS_LLM = sorted(P.model.unique())


def fit(T, y):
    w = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=200000)
    c = TfidfVectorizer(analyzer='char', ngram_range=(2, 5), min_df=2, sublinear_tf=True, max_features=300000)
    M = hstack([w.fit_transform(T), c.fit_transform(T)]).tocsr()
    return w, c, LogisticRegression(max_iter=3000, class_weight='balanced', solver='liblinear').fit(M, y)


def score(m, T):
    return m[2].decision_function(hstack([m[0].transform(T), m[1].transform(T)]).tocsr())


def thr(p, y):
    return max(np.unique(np.quantile(p, np.linspace(0, 1, 201))), key=lambda t: f1_score(y, p >= t, zero_division=0))


def ident(T):
    return list(T)


def strip_all(T):
    return [strip_dia(t) for t in T]


rows, preds = [], []
X, Y, IDS = list(S.text), S.label.values, S.id.values
for s in range(5):
    o = np.random.default_rng(s).permutation(len(X))
    a, b, c = o[:int(.8 * len(o))], o[int(.8 * len(o)):int(.9 * len(o))], o[int(.9 * len(o)):]
    tr_ids, te_ids = set(IDS[a]), set(IDS[c])
    Tr, Va = [X[i] for i in a], [X[i] for i in b]
    base = {
        'sach': ((Tr, Y[a]), ident),
        'tang_cuong': ((Tr + strip_all(Tr), np.r_[Y[a], Y[a]]), ident),
        'chuan_hoa': ((strip_all(Tr), Y[a]), strip_all),
    }
    fitted = {k: (fit(*d), f) for k, (d, f) in base.items()}
    # tang_cuong_llm: một mô hình riêng cho mỗi LLM bị test, chỉ học bản viết lại của LLM khác (tránh rò rỉ phong cách)
    for held in MODELS_LLM:
        A = P[P.id.isin(tr_ids) & (P.model != held)]
        fitted[f'tang_cuong_llm|{held}'] = (fit(strip_all(Tr + list(A.text)), np.r_[Y[a], A.label.values]), strip_all)
    for key, (m, f) in fitted.items():
        name, held = (key.split('|') + [None])[:2]
        th = thr(score(m, f(Va)), Y[b])
        te = S[S.id.isin(te_ids)]
        sets = {('goc', '-', 'goc'): (te.text, te.label.values, te.co_dau.values, te.id.values)}
        for (llm, pr), g in P[P.id.isin(te_ids)].groupby(['model', 'prompt']):
            if held is not None and llm != held:
                continue
            sets[(llm, pr, 'nguyen_van')] = (g.text, g.label.values, g.co_dau.values, g.id.values)
            sets[(llm, pr, 'bo_dau')] = (strip_all(g.text), g.label.values, g.co_dau.values, g.id.values)
        sets[('goc', '-', 'bo_dau')] = (strip_all(te.text), te.label.values, te.co_dau.values, te.id.values)
        for (llm, pr, bien), (T, y, cd, ids) in sets.items():
            sc = score(m, f(T))
            p = sc >= th
            r = dict(seed=s, model=name, llm=llm, prompt=pr, bien=bien, n_pos=int(y.sum()), n_neg=int((1 - y).sum()),
                     recall=p[y == 1].mean(), recall_co_dau=p[(y == 1) & cd].mean(), recall_khong_dau=p[(y == 1) & ~cd].mean(),
                     fpr=p[y == 0].mean() if (y == 0).any() else np.nan,
                     f1=f1_score(y, p) if (y == 0).any() else np.nan,
                     auroc=roc_auc_score(y, sc) if (y == 0).any() else np.nan)
            rows.append(r)
            preds.append(pd.DataFrame(dict(seed=s, model=name, llm=llm, prompt=pr, bien=bien, id=ids, label=y, pred=p.astype(int))))
R = pd.DataFrame(rows)
R.to_csv(OUT / 'eval_para.csv', index=False)
pd.concat(preds).to_csv(OUT / 'preds.csv.gz', index=False)  # dự đoán từng tin, dùng cho khoảng tin cậy (ci_para.py)
g = R.groupby(['llm', 'prompt', 'bien', 'model'])
T = g.agg(recall=('recall', 'mean'), sd=('recall', 'std'), fpr=('fpr', 'mean'), f1=('f1', 'mean'), auroc=('auroc', 'mean'),
          rec_co_dau=('recall_co_dau', 'mean'), rec_khong_dau=('recall_khong_dau', 'mean'), n_pos=('n_pos', 'mean')).round(3)
print('\n== Kết quả (trung bình 5 seed)\n', T.to_string())
print('\n== Recall gọn: hàng = tập test, cột = mô hình')
print(R.pivot_table(index=['llm', 'prompt', 'bien'], columns='model', values='recall').round(3).to_string())
print('\n== FPR gọn (chỉ tập có tin âm)')
print(R.dropna(subset=['fpr']).pivot_table(index=['llm', 'prompt', 'bien'], columns='model', values='fpr').round(3).to_string())
