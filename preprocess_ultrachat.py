"""预处理Pile数据集"""
import math, os
import ijson.backends.yajl2_c as ijson # type: ignore
from multiprocessing import Queue, Process
import numpy as np
from tqdm import tqdm
from config import *
from encoder import Encoder

def _preprocess(qi: Queue, encoder: Encoder, outfname: str):
    prefixes = [encoder.encode(prefix) for prefix in ROLE_PREFIXES]
    with open(outfname, "ab") as f_out, open(outfname + ".mask", "ab") as f_mask:
        while True:
            j = qi.get()
            if j is None:
                break
            d = next(ijson.items(j, "data"))
            codes = []
            mask = []
            role = True
            for turn in d:
                context = encoder.encode(turn + "\n\n")
                role_prefix = prefixes[role]
                codes += role_prefix + context
                if role:
                    mask += [0] * len(role_prefix) + [0] * len(context)
                else:
                    mask += [0] * len(role_prefix) + [1] * len(context)
                role = not role
            assert len(codes) == len(mask)
            codes += [SPECIAL_TOKENS_IDS["<pad>"]] * (MAX_LENGTH + 1 - len(codes) % (MAX_LENGTH + 1))
            mask += [0] * (MAX_LENGTH + 1 - len(mask) % (MAX_LENGTH + 1))
            f_out.write(np.array(codes, dtype=np.uint16).tobytes())
            f_mask.write(np.array(mask, dtype=np.uint8).tobytes())

def preprocess(fname: str, encoder: Encoder):
    qi: Queue = Queue(maxsize=64)
    procs = []
    for i in range(32):
        p = Process(target=_preprocess, args=(qi, encoder, f"{fname}.bin.part{i}"))
        p.start()
        procs.append(p)

    with open(fname, "r", encoding="utf-8") as f_in:
        for line in tqdm(f_in):
            if len(line) > 2:
                qi.put(line.strip())

    for i in range(32):
        qi.put(None)
    for p in procs:
        p.join()
    
    with open(f"{fname}.bin", "ab") as f_out, open(f"{fname}.mask.bin", "ab") as f_mask:
        for i in tqdm(range(32)):
            if os.path.exists(f"{fname}.bin.part{i}"):
                with open(f"{fname}.bin.part{i}", "rb") as f_in:
                    f_out.write(f_in.read())
                os.remove(f"{fname}.bin.part{i}")
            if os.path.exists(f"{fname}.bin.part{i}.mask"):
                with open(f"{fname}.bin.part{i}.mask", "rb") as f_in:
                    f_mask.write(f_in.read())
                os.remove(f"{fname}.bin.part{i}.mask")

if __name__ == "__main__":
    import sys
    if len(sys.argv) <= 1:
        print("usage: python preprocess_ultrachat.py <path> [<path>, ...]")
        exit(1)
    fnames = sys.argv[1:]
    for fname in fnames:
        print("Preprocessing", fname)
        preprocess(fname, Encoder.from_path("encoder.json"))

