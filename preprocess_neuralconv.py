"""预处理多轮对话数据集"""
import math
import json
import numpy as np
from tqdm import tqdm
from config import *
from encoder import Encoder

def preprocess(fname: str, encoder: Encoder): # <ins>A/B</ins>content<eos>
    with open(fname) as f_in, open(fname + ".bin", "ba") as f_out:
        lines = json.load(f_in)
        for j in tqdm(lines):
            data = [SPECIAL_TOKENS_IDS["<ins>"]]
            is_a = True
            for turn in j["content"]:
                data += (
                    [SPECIAL_TOKENS_IDS["<ins>"]] +
                    encoder.encode("A" if is_a else "B") +
                    [SPECIAL_TOKENS_IDS["</ins>"]] +
                    encoder.encode(turn) + 
                    [SPECIAL_TOKENS_IDS["<eos>"]]
                )
                is_a = not is_a
            arr = np.array(
                data + [SPECIAL_TOKENS_IDS["<pad>"]] * (MAX_LENGTH + 1 - len(data) % (MAX_LENGTH + 1)),
                dtype=np.int16
            )
            f_out.write(arr.tobytes())

if __name__ == "__main__":
    import sys
    if len(sys.argv) <= 1:
        print("usage: python preprocess_neuralconv.py <path> [<path>, ...]")
        exit(1)
    fnames = sys.argv[1:]
    for fname in fnames:
        print("Preprocessing", fname)
        preprocess(fname, Encoder.from_path("encoder.json"))