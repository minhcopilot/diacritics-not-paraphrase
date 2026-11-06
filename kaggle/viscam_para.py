"""Sinh tin SMS viết lại bằng 3 LLM mở để đo độ bền của bộ lọc lừa đảo tiếng Việt.

- trung_tinh: viết lại mọi tin (cả lừa đảo lẫn bình thường), cùng một lời nhắc, không biết nhãn.
- ne_loc: chỉ tin dương, yêu cầu viết sao cho bộ lọc khó nhận ra nhưng giữ lời kêu gọi hành động.
Hai GPU T4 chạy song song: GPU0 lần lượt Qwen rồi SeaLLMs, GPU1 chạy Mistral-Nemo. Mỗi mô hình ghi một file riêng.
"""
import glob
import os
import subprocess
import sys
import time

os.environ['HF_HOME'] = '/tmp/hf'
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-U', 'bitsandbytes'], check=False)

WORKER = r'''
import os, sys, time, re, gc, json
import pandas as pd, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, set_seed

name, short, bs, src = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
GIU = ('Giữ nguyên các thẻ như <URL>, [MONEY], [DATE], [TIME], cùng số điện thoại, mã số và số tiền. '
       'Giữ cùng kiểu chữ với bản gốc: bản gốc có dấu thì viết có dấu, bản gốc không dấu thì viết không dấu. '
       'Độ dài tương đương bản gốc. Chỉ trả về đúng một tin nhắn đã viết lại, không giải thích, không thêm tiêu đề.')
P = {
    'trung_tinh': 'Hãy viết lại tin nhắn SMS dưới đây bằng cách diễn đạt khác nhưng giữ nguyên ý nghĩa, mục đích và giọng điệu. ' + GIU,
    'ne_loc': ('Bạn đang giúp kiểm thử độ bền của bộ lọc tin nhắn rác cho một nghiên cứu phòng chống lừa đảo. '
               'Hãy viết lại tin nhắn dưới đây sao cho nghe tự nhiên như tin nhắn bình thường, tránh các từ ngữ và cách viết '
               'dễ bị bộ lọc tin rác nhận ra, nhưng vẫn giữ nguyên mục đích và lời kêu gọi hành động. ' + GIU),
}
df = pd.read_csv(src)
jobs = [(r.id, 'trung_tinh', r.text) for r in df.itertuples()] + [(r.id, 'ne_loc', r.text) for r in df[df.label == 1].itertuples()]
jobs.sort(key=lambda j: len(j[2]))
out = f'/kaggle/working/para_{short}.csv'
t0 = time.time()
tok = AutoTokenizer.from_pretrained(name)
tok.padding_side = 'left'
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(name, device_map={'': 0}, quantization_config=BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=torch.float16))
model.eval()
print(short, 'tải xong sau', round(time.time() - t0), 'giây; số việc', len(jobs), flush=True)


def clean(s):
    s = s.strip()
    s = re.sub(r'^(tin nhắn (đã )?viết lại|tin nhắn mới|bản viết lại)\s*:\s*', '', s, flags=re.I)
    return s.strip().strip('"“”').strip()


set_seed(0)
rows, t1 = [], time.time()
for i in range(0, len(jobs), bs):
    chunk = jobs[i:i + bs]
    prompts = [tok.apply_chat_template([{'role': 'user', 'content': P[k] + '\n\nTin nhắn gốc:\n' + t}], tokenize=False, add_generation_prompt=True) for _, k, t in chunk]
    enc = tok(prompts, return_tensors='pt', padding=True, add_special_tokens=False).to('cuda')
    with torch.no_grad():
        g = model.generate(**enc, max_new_tokens=384, do_sample=True, temperature=0.7, top_p=0.9, repetition_penalty=1.05, pad_token_id=tok.pad_token_id)
    for (id_, k, _), seq in zip(chunk, g[:, enc.input_ids.shape[1]:]):
        raw = tok.decode(seq, skip_special_tokens=True)
        rows.append(dict(id=id_, model=short, prompt=k, raw=raw, text=clean(raw)))
    if (i // bs) % 20 == 0:
        print(short, f'{i + len(chunk)}/{len(jobs)}', round(time.time() - t1), 'giây', flush=True)
        pd.DataFrame(rows).to_csv(out, index=False)
pd.DataFrame(rows).to_csv(out, index=False)
json.dump(dict(model=name, n=len(rows), load_s=round(t1 - t0), gen_s=round(time.time() - t1), torch=torch.__version__,
               transformers=__import__('transformers').__version__, gpu=torch.cuda.get_device_name(0)),
          open(f'/kaggle/working/para_{short}.json', 'w'), indent=2)
print(short, 'xong', round(time.time() - t1), 'giây', flush=True)
'''
open('/kaggle/working/vc_worker.py', 'w', encoding='utf-8').write(WORKER)
src = glob.glob('/kaggle/input/**/sms_input.csv', recursive=True)[0]
print('đầu vào', src, flush=True)

QUEUES = {
    '0': [('Qwen/Qwen2.5-7B-Instruct', 'qwen25_7b', 16), ('SeaLLMs/SeaLLMs-v3-7B-Chat', 'seallm3_7b', 16)],
    '1': [('mistralai/Mistral-Nemo-Instruct-2407', 'nemo_12b', 8)],
}
QUEUE_SCRIPT = r'''
import subprocess, sys, json
for name, short, bs in json.loads(sys.argv[2]):
    r = subprocess.run([sys.executable, '/kaggle/working/vc_worker.py', name, short, str(bs), sys.argv[1]])
    print(short, 'mã thoát', r.returncode, flush=True)
'''
open('/kaggle/working/vc_queue.py', 'w', encoding='utf-8').write(QUEUE_SCRIPT)
import json
n_gpu = subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True).stdout.count('GPU ')
print('số GPU', n_gpu, flush=True)
if n_gpu < 2:
    QUEUES = {'0': QUEUES['0'] + QUEUES['1']}
procs = []
for gpu, q in QUEUES.items():
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
    procs.append(subprocess.Popen([sys.executable, '/kaggle/working/vc_queue.py', src, json.dumps(q)], env=env))
t = time.time()
for p in procs:
    p.wait()
print('tổng thời gian', round(time.time() - t), 'giây')
for f in sorted(glob.glob('/kaggle/working/para_*')):
    print(f, os.path.getsize(f))
