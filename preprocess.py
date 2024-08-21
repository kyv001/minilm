import math
import numpy as np
from tqdm import tqdm
from config import *
from encoder import Encoder

def _get_lines(fname: str, max_length: int) -> str:
    print("Splitting lines.")
    out_fname = fname + ".lines.txt"
    with open(fname) as f_in, open(out_fname, "a") as f_out:
        for l in tqdm(f_in):
            l = l[:-1]
            length = len(l)
            if length > 1:
                for i in range(math.ceil(length / max_length)):
                    batch = l[i * max_length: i * max_length + max_length]
                    f_out.write(batch + "\n")
    return out_fname

def _encode_lines(fname: str, encoder: Encoder, line_sep: str) -> str:
    print("Encoding lines.")
    out_fname = fname + ".encoded.bin"
    with open(fname) as f_in, open(out_fname, "a") as f_out:
        for l in tqdm(f_in):
            l = l[:-1]
            c = encoder.encode(l)
            s = "".join(map(chr, c)) + line_sep
            f_out.write(s)
    return out_fname

def preprocess(fname: str, encoder: Encoder):
    with open(fname) as f_in, open(fname + ".bin", "ba") as f_out:
        for l in tqdm(f_in):
            if len(l) > 40:
                arr = np.array(
                    encoder.encode(l[40:-3]) + [SPECIAL_TOKENS_IDS["<eos>"]],
                    dtype=np.int16
                )
                f_out.write(arr.tobytes())

if __name__ == "__main__":
    import sys
    if len(sys.argv) <= 1:
        print("usage: python preprocess.py <path> [<path>, ...]")
        exit(1)
    fnames = sys.argv[1:]
    for fname in fnames:
        print("Preprocessing", fname)
        preprocess(fname, Encoder.from_path("encoder.json"))