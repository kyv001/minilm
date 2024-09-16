"""预处理指令微调数据集，已用多轮对话数据集替代，已被弃用"""
import math
import json
import numpy as np
from tqdm import tqdm
from config import *
from encoder import Encoder

def preprocess(fname: str, encoder: Encoder): # <ins>{指令}\n{输入}</ins>{输出}<eos>
    with open(fname) as f_in, open(fname + ".bin", "ba") as f_out:
        lines = json.load(f_in)
        for j in tqdm(lines):
            instr = j["instruction"]
            inp = j["input"]
            out = j["output"]
            data = [SPECIAL_TOKENS_IDS["<ins>"], *encoder.encode(instr + "\n" + inp), SPECIAL_TOKENS_IDS["</ins>"],
                    *encoder.encode(out), SPECIAL_TOKENS_IDS["<eos>"]][:MAX_LENGTH + 1]
            arr = np.array(
                data + [SPECIAL_TOKENS_IDS["<pad>"]] * (MAX_LENGTH - len(data) + 1),
                dtype=np.int16
            )
            f_out.write(arr.tobytes())

if __name__ == "__main__":
    import sys
    if len(sys.argv) <= 1:
        print("usage: python preprocess_instructions.py <path> [<path>, ...]")
        exit(1)
    fnames = sys.argv[1:]
    for fname in fnames:
        print("Preprocessing", fname)
        preprocess(fname, Encoder.from_path("encoder.json"))