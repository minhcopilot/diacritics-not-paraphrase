"""Chuẩn bị đầu vào cho bước sinh tin viết lại: bỏ trùng như các thí nghiệm trước, che URL bằng <URL>, gắn id ổn định."""
import csv
import hashlib
import re
import sys
import unicodedata

sys.stdout.reconfigure(encoding='utf-8')

SRC, OUT = sys.argv[1], sys.argv[2]
URL = re.compile(r'(https?://\S+|www\.\S+|\b[\w-]+(\.[\w-]+)*\.(vn|com|net|top|vip|me|link|xyz|info|cc)\b\S*)', re.I)


def norm_key(t):
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFC', t).lower()).strip()


seen, rows = set(), []
with open(SRC, encoding='utf-8-sig', newline='') as fh:
    for x in csv.DictReader(fh):
        if not x['message'].strip() or norm_key(x['message']) in seen:
            continue
        seen.add(norm_key(x['message']))
        t = unicodedata.normalize('NFC', x['message'])
        rows.append(dict(id=hashlib.sha1(norm_key(t).encode()).hexdigest()[:12], message_id=x['message_id'],
                         label=int(x['label']), text=URL.sub('<URL>', t).strip()))
with open(OUT, 'w', encoding='utf-8', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
L = sorted(len(r['text']) for r in rows)
print(len(rows), 'tin | dương', sum(r['label'] for r in rows), '| độ dài ký tự trung vị', L[len(L) // 2], 'p95', L[int(.95 * len(L))], 'max', L[-1])
