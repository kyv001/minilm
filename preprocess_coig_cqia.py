"""预处理COIG-CQIA指令微调数据集"""
import math
import json
import numpy as np
from tqdm import tqdm
from config import *
from encoder import Encoder

def preprocess(fname: str, encoder: Encoder): 
    """
用户：{instruction}
{input}


MiniLM：{output}


    """
    with open(fname) as f_in, open(fname + ".bin", "ba") as f_out:
        data: list[int] = []
        for l in tqdm(f_in):
            j = json.loads(l)
            instr = j["instruction"].replace("\n", "")
            inp = j["input"].replace("\n", "")
            out = j["output"].replace("\n", "")
            data += [*encoder.encode("甲：" + instr + "\n" + inp + "\n"),
                    *encoder.encode("乙：" + out + "\n\n")]
            if len(data) > MAX_LENGTH + 1:
                data = data[:MAX_LENGTH + 1]
                arr = np.array(
                    data,
                    dtype=np.int16
                )
                f_out.write(arr.tobytes())
                data = []

if __name__ == "__main__":
    import sys
    if len(sys.argv) <= 1:
        print("usage: python preprocess_coig_cqia.py <path> [<path>, ...]")
        exit(1)
    fnames = sys.argv[1:]
    for fname in fnames:
        print("Preprocessing", fname)
        preprocess(fname, Encoder.from_path("encoder.json"))
