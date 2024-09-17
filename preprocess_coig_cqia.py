"""预处理COIG-CQIA指令微调数据集"""
import math
import json
import numpy as np
from tqdm import tqdm
from config import *
from encoder import Encoder

def preprocess(fname: str, encoder: Encoder): 
    """
    <ins>{指令}
    {输入}</ins>{输出}<eos>
    """
    with open(fname) as f_in, open(fname + ".bin", "ba") as f_out:
        for l in tqdm(f_in):
            j = json.loads(l)
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
        print("usage: python preprocess_coig_cqia.py <path> [<path>, ...]")
        exit(1)
    fnames = sys.argv[1:]
    for fname in fnames:
        print("Preprocessing", fname)
        preprocess(fname, Encoder.from_path("encoder.json"))
