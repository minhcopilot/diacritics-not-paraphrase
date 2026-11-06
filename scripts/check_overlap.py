"""Tự kiểm tra trùng lặp: tìm các đoạn từ 7 từ liên tiếp trở lên trùng giữa phần chữ của bài và các nguồn được trích.

Nguồn: tóm tắt arXiv của các bài trích dẫn (file XML từ arXiv API), toàn văn PhishVN (pdftotext), README bộ dữ liệu.
Đây chỉ là kiểm tra sơ bộ, không thay được Turnitin hay iThenticate.
"""
import re
import sys
import xml.etree.ElementTree as E

sys.stdout.reconfigure(encoding='utf-8')
TEX, AX, *OTHERS = sys.argv[1:]
N = 7
tex = open(TEX, encoding='utf-8').read()
tex = tex[tex.index(r'\begin{abstract}'):tex.index(r'\bibliographystyle')]
tex = re.sub(r'\\begin\{table\*?\}.*?\\end\{table\*?\}', ' ', tex, flags=re.S)
tex = re.sub(r'%.*', ' ', tex)
tex = re.sub(r'\\[a-zA-Z]+\*?(\[[^]]*\])?', ' ', tex)


def words(t):
    return re.findall(r"[a-z0-9]+(?:[-'][a-z0-9]+)*", t.lower())


W = words(tex)
grams = {tuple(W[i:i + N]) for i in range(len(W) - N + 1)}
src = {}
ns = {'a': 'http://www.w3.org/2005/Atom'}
for e in E.parse(AX).getroot().findall('a:entry', ns):
    src['arXiv ' + e.find('a:id', ns).text.split('/')[-1]] = e.find('a:summary', ns).text + ' ' + e.find('a:title', ns).text
for f in OTHERS:
    src[f] = open(f, encoding='utf-8', errors='ignore').read()
print('số từ trong bài', len(W), '| số nguồn', len(src))
tot = 0
for k, t in src.items():
    w = words(t)
    hit = sorted({' '.join(g) for g in (tuple(w[i:i + N]) for i in range(len(w) - N + 1)) if g in grams})
    tot += len(hit)
    if hit:
        print(f'{k}: {len(hit)} đoạn trùng ->', hit[:6])
print('tổng số đoạn trùng', tot)
