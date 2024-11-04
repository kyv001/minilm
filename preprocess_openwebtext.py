"""预处理openwebtext数据集，因为数据不干净已经弃用"""
import math
import numpy as np
from tqdm import tqdm
from config import *
from encoder import Encoder

def preprocess(fname: str, encoder: Encoder): # 把所有文字用<eos>隔开然后连接在一起
    with open(fname) as f_in, open(fname + ".bin", "ba") as f_out:
        codes = []
        for l in tqdm(f_in):
            if len(l) > 30:
                codes += encoder.encode(l.strip())
            if len(codes) > MAX_LENGTH:
                f_out.write(np.array(
                    codes[:MAX_LENGTH] + [SPECIAL_TOKENS_IDS["<eos>"]],
                    dtype=np.uint16
                ).tobytes())
                codes = codes[MAX_LENGTH:]

if __name__ == "__main__":
    import sys
    if len(sys.argv) <= 1:
        print("usage: python preprocess_wanjuan.py <path> [<path>, ...]")
        exit(1)
    fnames = sys.argv[1:]
    for fname in fnames:
        print("Preprocessing", fname)
        preprocess(fname, Encoder.from_path("encoder.json"))

