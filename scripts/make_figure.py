"""Hình 1 của bài báo: recall theo từng phép né và FPR, cho 4 chế độ huấn luyện TF-IDF, kèm CI 95%.

Gộp ba LLM cho mỗi phép né. CI bootstrap theo tin gốc (B=2000), giống ci_para.py.
Bảng màu: 4 ô đầu của bảng categorical trong skill dataviz (đã chạy validate_palette.js, đạt).
Thêm hoa văn sọc để in đen trắng vẫn phân biệt được.
"""
import sys

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

D = pd.read_csv(sys.argv[1])
OUT = sys.argv[2]
rng = np.random.default_rng(0)
MODELS = [('sach', 'Base', '#2a78d6', ''), ('tang_cuong', '+Strip', '#eb6834', '////'),
          ('chuan_hoa', 'Normalize', '#1baf7a', '\\\\\\\\'), ('tang_cuong_llm', '+LLM (proposed)', '#eda100', 'xxx')]
REC = [('Original', lambda d: (d.llm == 'goc') & (d.bien == 'goc')),
       ('Original,\nstripped', lambda d: (d.llm == 'goc') & (d.bien == 'bo_dau')),
       ('Neutral\nrewrite', lambda d: (d.prompt == 'trung_tinh') & (d.bien == 'nguyen_van')),
       ('Evasive\nrewrite', lambda d: (d.prompt == 'ne_loc') & (d.bien == 'nguyen_van')),
       ('Evasive rewrite\n+ stripped', lambda d: (d.prompt == 'ne_loc') & (d.bien == 'bo_dau'))]
FPR = [('Original', lambda d: (d.llm == 'goc') & (d.bien == 'goc')),
       ('Neutral\nrewrite', lambda d: (d.prompt == 'trung_tinh') & (d.bien == 'nguyen_van'))]


def est(g):
    per = g.groupby('id').pred.agg(['sum', 'count'])
    s, c = per['sum'].values, per['count'].values
    idx = rng.integers(0, len(per), size=(2000, len(per)))
    bs = s[idx].sum(1) / c[idx].sum(1)
    p = s.sum() / c.sum()
    lo, hi = np.quantile(bs, [.025, .975])
    return p, p - lo, hi - p


plt.rcParams.update({'font.size': 7, 'font.family': 'serif', 'axes.edgecolor': '#8a8a85', 'axes.linewidth': .6,
                     'xtick.color': '#3d3d3a', 'ytick.color': '#3d3d3a', 'hatch.linewidth': .5, 'hatch.color': 'white'})
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 2.1), gridspec_kw=dict(width_ratios=[5, 2], wspace=.18))
W = .19
for ax, conds, label, ycol in [(a1, REC, 1, 'Recall on scam/spam'), (a2, FPR, 0, 'FPR on legitimate')]:
    for j, (m, name, col, hat) in enumerate(MODELS):
        vals = [est(D[f(D) & (D.model == m) & (D.label == label)]) for _, f in conds]
        x = np.arange(len(conds)) + (j - 1.5) * (W + .02)
        ax.bar(x, [v[0] for v in vals], W, color=col, hatch=hat, edgecolor='white', linewidth=.6, label=name, zorder=2)
        ax.errorbar(x, [v[0] for v in vals], yerr=np.array([[v[1] for v in vals], [v[2] for v in vals]]),
                    fmt='none', ecolor='#3d3d3a', elinewidth=.6, capsize=1.2, zorder=3)
    ax.set_xticks(np.arange(len(conds)))
    ax.set_xticklabels([c for c, _ in conds])
    ax.set_ylabel(ycol)
    ax.yaxis.grid(True, color='#e4e3df', linewidth=.5, zorder=0)
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(length=0)
a1.set_ylim(0, 1.0)
a2.set_ylim(0, .14)
a1.legend(ncol=4, loc='lower center', bbox_to_anchor=(.7, 1.0), frameon=False, handlelength=1.6, columnspacing=1.2)
fig.savefig(OUT, bbox_inches='tight')
fig.savefig(OUT.replace('.pdf', '.png'), dpi=200, bbox_inches='tight')
print('ok', OUT)
