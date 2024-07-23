import json
import os
from config import *
from encoder import Encoder
import multiprocessing

def preprocess(fname: str, length: int, encoder: Encoder, target: str):
    with open(fname) as f:
        lines = f.readlines()
    n_lines = len(lines)
    ns_lines = [n_lines // 8] * 7
    ns_lines.append(n_lines - sum(ns_lines))

    q = multiprocessing.Queue()
    def _preprocess(lines: list, length: int, encoder: Encoder, target: str, rank: int):
        preprocessed_lines = []
        for line in lines:
            content = json.loads(line)["content"]
            n_lines = len(content) // length
            left = len(content) % length
            for i in range(n_lines):
                x = encoder.encode(content[i * length: (i + 1) * length])
                if i == n_lines - 1 and not left:
                    y = x[1: ] + [SPECIAL_TOKENS_IDS["<eos>"]]
                else:
                    y = x[1: ] + encoder.encode([content[(i + 1) * length]])
                # print(encoder.decode(x))
                # print(encoder.decode(y))
                preprocessed_lines.append(json.dumps({"x": x, "y": y}))
            if left:
                x = encoder.encode(content[-left: ]) + [SPECIAL_TOKENS_IDS["<eos>"]] + [SPECIAL_TOKENS_IDS["<pad>"]] * (length - left - 1)
                y = x[1: ] + [SPECIAL_TOKENS_IDS["<pad>"]]
                # print(encoder.decode(x))
                # print(encoder.decode(y))
                preprocessed_lines.append(json.dumps({"x": x, "y": y}))
        with open(target + str(rank), "w") as f:
            f.write("\n".join(preprocessed_lines))
        q.put(rank)

    for rank in range(8):
        i = sum(ns_lines[:rank])
        j = i + ns_lines[rank]
        p = multiprocessing.Process(target=_preprocess, args=(lines[i: j], length, encoder, target, rank))
        p.start()
    for i in range(8):
        q.get()
    all_lines = []
    for rank in range(8):
        all_lines += open(target + str(rank)).readlines()
        all_lines[-1] += "\n"
        os.remove(target + str(rank))
    with open(target, "w") as f:
        f.write("".join(all_lines))

if __name__ == "__main__":
    encoder = Encoder.from_path("encoder.json")
    preprocess("WanJuan-News/news1.jsonl", 1600, encoder, "WanJuan-News/news1.jsonl.preprocessed.jsonl")
