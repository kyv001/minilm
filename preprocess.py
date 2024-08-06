import math
from tqdm import tqdm
from config import *
from encoder import Encoder

def _get_contents(fname: str) -> str:
    print("Extracting contents.")
    out_fname = fname + ".contents.txt"
    with open(fname) as f_in, open(out_fname, "a") as f_out:
        for l in tqdm(f_in):
            if len(l) > 40:
                f_out.write(l[40:-3] + "\n")
    return out_fname

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

def preprocess(fname: str, max_length: int, encoder: Encoder, line_sep: str):
    return _encode_lines(
        _get_lines(
            _get_contents(fname),
            max_length
        ),
        encoder,
        line_sep
    )

if __name__ == "__main__":
    preprocess("tiny-example-news.jsonl", MAX_LENGTH + 1, Encoder.from_path("encoder.json"), LINE_SEP)