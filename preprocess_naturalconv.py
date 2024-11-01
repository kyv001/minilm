"""预处理多轮对话数据集"""
import math
import json
import numpy as np
from tqdm import tqdm
from config import *
from encoder import Encoder

def preprocess(fname: str, encoder: Encoder):
    with open(fname) as f_in, open(fname + ".bin", "ba") as f_out:
        lines = json.load(f_in)
        i = 0
        data = []
        for j in tqdm(lines):
            is_a = True
            start = 2 if i else 0
            for turn in j["content"][start:-4]:
                data += (
                    encoder.encode(f"{'甲' if is_a else '乙'}：" + turn + "\n\n")
                )
                is_a = not is_a
            i += 1
            if i % 1000 == 0:
                i = 0
                arr = np.array(
                    data + [SPECIAL_TOKENS_IDS["<pad>"]] * (MAX_LENGTH + 1 - len(data) % (MAX_LENGTH + 1)),
                    dtype=np.int16
                )
                f_out.write(arr.tobytes())
                data = []

if __name__ == "__main__":
    import sys
    if len(sys.argv) <= 1:
        print("usage: python preprocess_neuralconv.py <path> [<path>, ...]")
        exit(1)
    fnames = sys.argv[1:]
    for fname in fnames:
        print("Preprocessing", fname)
        preprocess(fname, Encoder.from_path("encoder.json"))