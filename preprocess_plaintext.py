"""预处理openwebtext数据集，因为数据不干净已经弃用"""
import math, os
from multiprocessing import Queue, Process
import numpy as np
from tqdm import tqdm
from config import *
from encoder import Encoder

def _preprocess(qi: Queue, encoder: Encoder, outfname: str): # 把所有文字用<eos>隔开然后连接在一起
    with open(outfname, "ab") as f_out:
        while True:
            d = qi.get()
            if d is None:
                break
            codes = encoder.encode(d) + [SPECIAL_TOKENS_IDS["<eos>"]]
            f_out.write(np.array(codes, dtype=np.uint16).tobytes())

def preprocess(fname: str, encoder: Encoder):
    qi: Queue = Queue(maxsize=64)
    procs = []
    for i in range(32):
        p = Process(target=_preprocess, args=(qi, encoder, f"{fname}.bin.part{i}"))
        p.start()
        procs.append(p)

    with open(fname, "r", encoding="utf-8") as f_in:
        content = ""
        blankline_count = 0
        for line in tqdm(f_in):
            line = line.strip()
            if line:
                blankline_count = 0
                content += line + " "
            else:
                blankline_count += 1
                if blankline_count > 2 and content:
                    qi.put(content)
                    content = ""

    for i in range(32):
        qi.put(None)
    for p in procs:
        p.join()
    os.system(f"cat {fname}.bin.part* > {fname}.bin")
    os.system(f"rm {fname}.bin.part*")

if __name__ == "__main__":
    import sys
    if len(sys.argv) <= 1:
        print("usage: python preprocess_plaintext.py <path> [<path>, ...]")
        exit(1)
    fnames = sys.argv[1:]
    for fname in fnames:
        print("Preprocessing", fname)
        preprocess(fname, Encoder.from_path("encoder.json"))

