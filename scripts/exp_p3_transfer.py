"""Thử nhanh: mô hình học trên SMS thật có bắt được tin lừa đảo do LLM viết lại (PhishVN p3) không.

p3 đã bị bỏ dấu sẵn, nên so recall trên p3 với recall trên tin dương thật cũng bị bỏ dấu
để tách tác động của việc viết lại khỏi tác động của việc mất dấu.
Mọi URL được thay bằng <URL> để mô hình không học link giả sim.example.vn.
"""
import csv
import re
import sys
import unicodedata
import warnings

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')
SMS, P3, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
URL = re.compile(r'(https?://\S+|www\.\S+|\b[\w-]+(\.[\w-]+)*\.(vn|com|net|top|vip|me|link|xyz|info|cc)\b\S*)', re.I)


def norm_key(t):
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFC', t).lower()).strip()


def strip_dia(t):
    t = t.replace('đ', 'd').replace('Đ', 'D')
    return unicodedata.normalize('NFC', ''.join(c for c in unicodedata.normalize('NFD', t) if unicodedata.category(c) != 'Mn'))


def mask(t):
    return URL.sub(' <URL> ', t)


D = {}
with open(SMS, encoding='utf-8-sig', newline='') as fh:
    for x in csv.DictReader(fh):
        if x['message'].strip():
            D.setdefault(norm_key(x['message']), (mask(unicodedata.normalize('NFC', x['message'])), int(x['label'])))
X = [v[0] for v in D.values()]
Y = np.array([v[1] for v in D.values()])
p3 = pd.read_csv(P3)
p3 = p3[p3.channel == 'sms'].copy()
p3['text'] = p3.text.map(mask)
p3['band'] = pd.qcut(p3.jaccard, 3, labels=['viet_lai_manh', 'vua', 'nhe']) if 'jaccard' in p3 else 'all'
print('SMS thật không trùng', len(X), 'dương', int(Y.sum()), '| p3 kênh SMS', len(p3))


def fit(T, y):
    w = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=200000)
    c = TfidfVectorizer(analyzer='char', ngram_range=(2, 5), min_df=2, sublinear_tf=True, max_features=300000)
    M = hstack([w.fit_transform(T), c.fit_transform(T)]).tocsr()
    return w, c, LogisticRegression(max_iter=3000, class_weight='balanced', solver='liblinear').fit(M, y)


def score(m, T):
    return m[2].decision_function(hstack([m[0].transform(T), m[1].transform(T)]).tocsr())


def thr(p, y):
    return max(np.unique(np.quantile(p, np.linspace(0, 1, 201))), key=lambda t: f1_score(y, p >= t, zero_division=0))


# sach: học trên tin gốc | tang_cuong: thêm bản bỏ dấu của tin train | chuan_hoa: bỏ dấu cả lúc học lẫn lúc đoán
MODELS = {
    'sach': (lambda T, y: (T, y), lambda T: T),
    'tang_cuong': (lambda T, y: (T + [strip_dia(t) for t in T], np.r_[y, y]), lambda T: T),
    'chuan_hoa': (lambda T, y: ([strip_dia(t) for t in T], y), lambda T: [strip_dia(t) for t in T]),
}
rows = []
for s in range(5):
    o = np.random.default_rng(s).permutation(len(X))
    a, b, c = o[:int(.8 * len(o))], o[int(.8 * len(o)):int(.9 * len(o))], o[int(.9 * len(o)):]
    Tr, Va, Te = [X[i] for i in a], [X[i] for i in b], [X[i] for i in c]
    for name, (prep_train, prep_in) in MODELS.items():
        m = fit(*prep_train(Tr, Y[a]))
        th = thr(score(m, prep_in(Va)), Y[b])
        sets = {
            'that_goc': (Te, Y[c]),
            'that_bo_dau': ([strip_dia(t) for t in Te], Y[c]),
        }
        for k, (T, y) in sets.items():
            p = score(m, prep_in(T)) >= th
            rows.append(dict(seed=s, model=name, test=k, recall=p[y == 1].mean(), fpr=p[y == 0].mean(), n=int((y == 1).sum())))
        p = score(m, prep_in(list(p3.text))) >= th
        for (var, band), g in pd.DataFrame(dict(var=p3.variant.values, band=p3.band.values, hit=p)).groupby(['var', 'band'], observed=True):
            rows.append(dict(seed=s, model=name, test=f'p3_{var}_{band}', recall=g.hit.mean(), fpr=np.nan, n=len(g)))
        rows.append(dict(seed=s, model=name, test='p3_tat_ca', recall=p.mean(), fpr=np.nan, n=len(p)))
R = pd.DataFrame(rows)
R.to_csv(OUT, index=False)
print('\nRecall (trung bình ± độ lệch chuẩn, 5 seed); FPR chỉ có ở tập SMS thật')
agg = R.groupby(['test', 'model']).agg(recall=('recall', 'mean'), sd=('recall', 'std'), fpr=('fpr', 'mean'), n=('n', 'mean')).round(3)
print(agg.unstack('model').to_string())
