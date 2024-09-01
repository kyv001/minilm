"""预处理COIG-CQIA指令微调数据集"""
import math
import json
import numpy as np
from tqdm import tqdm
from config import *
from encoder import Encoder

def preprocess(fname: str, encoder: Encoder): # <ins>{指令}：{输入}</ins>{输出}<eos>
    with open(fname) as f_in, open(fname + ".bin", "ba") as f_out:
        for l in tqdm(f_in):
            j = json.loads(l)
            instr = j["instruction"].replace("\n", "\\n") # 训练模型输出“\n”而不是直接换行
            inp = j["input"].replace("\n", "\\n")
            if inp:
                if instr.endswith("。"):
                    instr = instr[:-1] + "：" # 把最后一个句号换成冒号来将指令和输入连接在一起
                elif not instr.endswith("："):
                    instr += "：" # 如果指令本来就没有句号，直接加上冒号
            out = j["output"].replace("\n", "\\n")
            data = [SPECIAL_TOKENS_IDS["<ins>"], *encoder.encode(instr + inp), SPECIAL_TOKENS_IDS["</ins>"],
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