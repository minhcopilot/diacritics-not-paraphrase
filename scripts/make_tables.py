"""Sinh các bảng LaTeX của bài báo từ kết quả đã lưu, tránh chép tay số liệu.

Đầu vào: results/eval_para (cấu hình chính), results/eval_para_labse70 (độ nhạy), results/phobert_para, results/phobert_extra,
results/bench_tfidf.txt. Đầu ra: paper/tables/*.tex.
"""
import ast
import pathlib
import sys

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')
R = pathlib.Path(sys.argv[1])
OUT = pathlib.Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
LLM = {'qwen25_7b': 'Qwen2.5-7B', 'seallm3_7b': 'SeaLLMs-v3-7B', 'nemo_12b': 'Mistral-NeMo-12B'}
PR = {'trung_tinh': 'neutral', 'ne_loc': 'evasive'}


def f3(x):
    return '--' if pd.isna(x) else f'{x:.3f}'.lstrip('0') if x < 1 else '1.00'


def ci(s):
    lo, hi = ast.literal_eval(s)
    return f'\\tiny[{f3(lo)}, {f3(hi)}]'


# --- Bảng 1: chất lượng bản viết lại ---
P = pd.read_csv(R / 'eval_para/para_all_checked.csv')
L = pd.read_csv(R / 'phobert_para/labse_sim.csv')
P = P.merge(L[['id', 'model', 'prompt', 'labse']], on=['id', 'model', 'prompt'], how='left')
lines = []
for m in LLM:
    for pr in PR:
        g = P[(P.model == m) & (P.prompt == pr)]
        k = g[~g.hong]
        lines.append(f"{LLM[m]} & {PR[pr]} & {len(g)} & {f3(g.tu_choi.mean())} & {f3(g.chep.mean())} & {f3(g.hong.mean())} & "
                     f"{f3(g.doi_dau.mean())} & {f3(k.labse.median())} & {f3((k.labse < .7).mean())} \\\\")
(OUT / 'quality.tex').write_text('\n'.join(lines) + '\n', encoding='utf-8')

# --- Bảng 2: TF-IDF, recall và FPR kèm CI (cấu hình chính) ---
C = pd.read_csv(R / 'eval_para/ci.csv')
MOD = ['sach', 'tang_cuong', 'chuan_hoa', 'tang_cuong_llm']


def row(llm, pr, bien, label, metric='recall'):
    cells = []
    for m in MOD:
        x = C[(C.llm == llm) & (C.prompt == pr) & (C.bien == bien) & (C.model == m)]
        if x.empty or pd.isna(x[metric].iloc[0]):
            cells.append('--')
        else:
            cells.append(f"{f3(x[metric].iloc[0])} {ci(x[metric + '_ci'].iloc[0])}")
    return f'{label} & ' + ' & '.join(cells) + ' \\\\'


lines = ['\\multicolumn{5}{l}{\\textit{Recall on scam/spam messages}} \\\\',
         row('goc', '-', 'goc', 'Original'), row('goc', '-', 'bo_dau', 'Original, diacritics stripped')]
for m in LLM:
    lines += [row(m, 'trung_tinh', 'nguyen_van', f'{LLM[m]}, neutral'), row(m, 'ne_loc', 'nguyen_van', f'{LLM[m]}, evasive'),
              row(m, 'ne_loc', 'bo_dau', f'{LLM[m]}, evasive + stripped')]
lines += ['\\midrule', '\\multicolumn{5}{l}{\\textit{False-positive rate on legitimate messages}} \\\\',
          row('goc', '-', 'goc', 'Original', 'fpr')]
for m in LLM:
    lines.append(row(m, 'trung_tinh', 'nguyen_van', f'{LLM[m]}, neutral rewrite', 'fpr'))
(OUT / 'tfidf_main.tex').write_text('\n'.join(lines) + '\n', encoding='utf-8')

# --- Bảng 3: PhoBERT (3 seed, trung bình ± độ lệch chuẩn) so với TF-IDF cùng 3 seed đầu ---
Pb = pd.read_csv(R / 'phobert_para/phobert_para.csv')
ex = R / 'phobert_extra/phobert_extra.csv'
if ex.exists():
    # phobert_extra chạy lại tang_cuong_llm và có thêm tập gốc: dùng bản này thay cho bản cũ.
    # Tập gốc của tang_cuong_llm là trung bình trên 3 mô hình (mỗi LLM bị giữ lại một mô hình), giống TF-IDF.
    Pb = pd.concat([Pb[Pb.model != 'phobert_tang_cuong_llm'], pd.read_csv(ex)])
T = pd.read_csv(R / 'eval_para/eval_para.csv')
T = T[T.seed < 3]
Pb.loc[(Pb.llm == 'goc') & (Pb.bien == 'nguyen_van'), 'bien'] = 'goc'


def ms(df, model, llm, pr, bien, metric):
    x = df[(df.model == model) & (df.llm == llm) & (df.prompt == pr) & (df.bien == bien)][metric]
    return '--' if x.empty or x.isna().all() else f'{f3(x.mean())}\\tiny$\\pm${f3(x.std()) if len(x) > 1 else "0"}'


COLS = [(T, 'sach'), (T, 'tang_cuong_llm'), (Pb, 'phobert_sach'), (Pb, 'phobert_tang_cuong'), (Pb, 'phobert_tang_cuong_llm')]
lines = []
spec = [('goc', '-', 'goc', 'Original', 'recall'), ('goc', '-', 'bo_dau', 'Original, stripped', 'recall')]
for m in LLM:
    spec += [(m, 'trung_tinh', 'nguyen_van', f'{LLM[m]}, neutral', 'recall'), (m, 'ne_loc', 'bo_dau', f'{LLM[m]}, evasive + stripped', 'recall')]
spec += [('goc', '-', 'goc', 'FPR, original', 'fpr')] + [(m, 'trung_tinh', 'nguyen_van', f'FPR, {LLM[m]} rewrite', 'fpr') for m in LLM]
for llm, pr, bien, lab, met in spec:
    lines.append(f'{lab} & ' + ' & '.join(ms(df, mo, llm, pr, bien, met) for df, mo in COLS) + ' \\\\')
(OUT / 'phobert.tex').write_text('\n'.join(lines) + '\n', encoding='utf-8')

# --- Bảng phụ (bản JCC): recall theo kiểu dấu của tin gốc, TF-IDF 5 seed ---
E5 = pd.read_csv(R / 'eval_para/eval_para.csv')
lines = []
spec = [('goc', '-', 'goc', 'Original'), ('goc', '-', 'bo_dau', 'Original, stripped')]
spec += [(m, 'ne_loc', 'bo_dau', f'{LLM[m]}, evasive + stripped') for m in LLM]
for llm, pr, bien, lab in spec:
    cells = []
    for m in ['sach', 'tang_cuong_llm']:
        x = E5[(E5.llm == llm) & (E5.prompt == pr) & (E5.bien == bien) & (E5.model == m)]
        cells += [f3(x.recall_co_dau.mean()), f3(x.recall_khong_dau.mean())]
    lines.append(f'{lab} & ' + ' & '.join(cells) + ' \\\\')
(OUT / 'by_source_style.tex').write_text('\n'.join(lines) + '\n', encoding='utf-8')

# --- Bảng 4: hiệu năng triển khai ---
bt = (R / 'bench_tfidf.txt').read_text(encoding='utf-8').strip()
bt = ast.literal_eval(bt.replace('np.float64(', '(')) if bt else {}
lines = [f"TF-IDF + LR (proposed) & {bt.get('mb_trien_khai', '--')} & {bt.get('ms_tb', '--')} & {bt.get('ms_p95', '--')} \\\\"]
bp = R / 'phobert_extra/phobert_bench.csv'
if bp.exists():
    B = pd.read_csv(bp)
    for d in ['fp32', 'int8']:
        acc = B[(B.dang == d) & (B.bien == 'nguyen_van')].recall
        if acc.empty or acc.iloc[0] < .5:
            print('bỏ', d, '- bản này hỏng độ chính xác (recall', acc.tolist(), ')')
            continue
        x = B[(B.dang == d) & (B.threads == 1)].iloc[0]
        lines.append(f"PhoBERT-base {d.upper()} & {x.mb} & {x.ms_tb} & {x.ms_p95} \\\\")
(OUT / 'efficiency.tex').write_text('\n'.join(lines) + '\n', encoding='utf-8')
for f in sorted(OUT.glob('*.tex')):
    print('==', f.name)
    print(f.read_text(encoding='utf-8'))
