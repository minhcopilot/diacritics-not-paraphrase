"""Kích thước và độ trễ CPU của bộ lọc TF-IDF (mô hình đề xuất: chuẩn hóa bỏ dấu, học thêm tin viết lại của mọi LLM).

Huấn luyện một lần trên toàn bộ tin gốc cộng mọi bản viết lại hợp lệ, rồi đo đoán từng tin một (batch 1) trên 500 tin.
"""
import pickle
import platform
import sys
import time
import unicodedata

import numpy as np
import pandas as pd
import sklearn
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

sys.stdout.reconfigure(encoding='utf-8')
SRC, CHECKED = sys.argv[1], sys.argv[2]


def strip_dia(t):
    t = t.replace('đ', 'd').replace('Đ', 'D')
    return unicodedata.normalize('NFC', ''.join(c for c in unicodedata.normalize('NFD', t) if unicodedata.category(c) != 'Mn'))


S = pd.read_csv(SRC)
P = pd.read_csv(CHECKED)
P = P[~P.hong]
T = [strip_dia(t) for t in list(S.text) + list(P.text.astype(str))]
y = np.r_[S.label.values, P.label.values]
w = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=200000)
c = TfidfVectorizer(analyzer='char', ngram_range=(2, 5), min_df=2, sublinear_tf=True, max_features=300000)
t0 = time.time()
M = hstack([w.fit_transform(T), c.fit_transform(T)]).tocsr()
lr = LogisticRegression(max_iter=3000, class_weight='balanced', solver='liblinear').fit(M, y)
fit_s = time.time() - t0
blob = pickle.dumps((w, c, lr), protocol=5)
# Bỏ stop_words_ (tập n-gram bị loại, chỉ cần khi gỡ lỗi) để có kích thước triển khai thực tế
for v in (w, c):
    v.stop_words_ = None
blob_slim = pickle.dumps((w, c, lr), protocol=5)
msgs = list(S.text.sample(500, random_state=0))
for m in msgs[:20]:
    lr.decision_function(hstack([w.transform([strip_dia(m)]), c.transform([strip_dia(m)])]))
lat = []
for m in msgs:
    t1 = time.perf_counter()
    lr.decision_function(hstack([w.transform([strip_dia(m)]), c.transform([strip_dia(m)])]))
    lat.append(time.perf_counter() - t1)
print(dict(n_train=len(T), n_feat=M.shape[1], fit_s=round(fit_s, 1), mb_pickle=round(len(blob) / 2**20, 1), mb_trien_khai=round(len(blob_slim) / 2**20, 1),
           ms_tb=round(1000 * np.mean(lat), 2), ms_p95=round(1000 * np.quantile(lat, .95), 2), cpu=platform.processor(), python=platform.python_version(), sklearn=sklearn.__version__))
