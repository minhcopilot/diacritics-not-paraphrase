"""Khoảng tin cậy 95% cho recall/FPR từ dự đoán từng tin (preds.csv.gz của eval_para.py).

Gộp dự đoán của mọi seed, bootstrap theo tin gốc (cùng một id ở nhiều seed và nhiều bản viết lại được lấy cùng nhau), B=2000.
Hiệu số giữa hai mô hình là bootstrap ghép cặp trên cùng các id.
"""
import sys

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')
pd.set_option('display.width', 230, 'display.max_rows', 300)
D = pd.read_csv(sys.argv[1])
OUT = sys.argv[2]
B, rng = 2000, np.random.default_rng(0)
BASE, PROP = 'sach', 'tang_cuong_llm'


def boot(g, metric_label):
    """Trả về ước lượng điểm và CI của tỷ lệ dự đoán dương trên các dòng có nhãn metric_label, bootstrap theo id."""
    g = g[g.label == metric_label]
    if g.empty:
        return np.nan, np.nan, np.nan
    per = g.groupby('id').pred.agg(['sum', 'count'])
    s, c = per['sum'].values, per['count'].values
    idx = rng.integers(0, len(per), size=(B, len(per)))
    bs = s[idx].sum(1) / c[idx].sum(1)
    return s.sum() / c.sum(), *np.quantile(bs, [.025, .975])


rows = []
for (llm, pr, bien, model), g in D.groupby(['llm', 'prompt', 'bien', 'model']):
    r, rl, rh = boot(g, 1)
    f, fl, fh = boot(g, 0)
    rows.append(dict(llm=llm, prompt=pr, bien=bien, model=model, recall=r, recall_ci=f'[{rl:.3f}, {rh:.3f}]',
                     fpr=f, fpr_ci=f'[{fl:.3f}, {fh:.3f}]' if not np.isnan(f) else ''))
R = pd.DataFrame(rows)
# Hiệu recall ghép cặp: mô hình đề xuất trừ mô hình thường, trên cùng tập id dương
diff = []
for (llm, pr, bien), g in D[D.label == 1].groupby(['llm', 'prompt', 'bien']):
    a = g[g.model == PROP].groupby('id').pred.mean()
    b = g[g.model == BASE].groupby('id').pred.mean()
    ids = a.index.intersection(b.index)
    if not len(ids):
        continue
    d = (a[ids] - b[ids]).values
    bs = d[rng.integers(0, len(d), size=(B, len(d)))].mean(1)
    lo, hi = np.quantile(bs, [.025, .975])
    diff.append(dict(llm=llm, prompt=pr, bien=bien, n_id=len(ids), delta_recall=d.mean(), ci=f'[{lo:.3f}, {hi:.3f}]', ci_khong_chua_0=(lo > 0) or (hi < 0)))
R.to_csv(OUT, index=False)
print(R.round(3).to_string(index=False))
print(f'\n== Hiệu recall ghép cặp {PROP} - {BASE}')
print(pd.DataFrame(diff).round(3).to_string(index=False))
