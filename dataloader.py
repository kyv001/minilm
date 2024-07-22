import json
import os
import torch
from config import *
from preprocess import preprocess

class WanJuanLoader:
    def __init__(self, fname, encoder, batch_size, length):
        self.fname = fname
        self.batch_size = batch_size
        self.length = length
        self.encoder = encoder

        if not os.path.exists(fname + ".parts"):
            os.mkdir(fname + ".parts")
        parts = []
        lines = []
        for line in open(fname):
            lines.append(line)
            if len(lines) >= 100:
                parts.append(lines)
                with open(fname + f".parts/{len(parts)}.jsonl", "w") as f:
                    f.write("".join(lines))
                lines = []
        if lines:
            parts.append(lines)
            with open(fname + f".parts/{len(parts)}.jsonl", "w") as f:
                f.write("".join(lines))
            lines = []
        self.n_parts = len(parts)
        self.total_lines = 100 * (self.n_parts - 1) + len(parts[-1])
        self.line = 0
        self.part_lines = {}
        self.ended = False
        del parts, lines

    def get_data(self):
        x_l = []
        y_l = []
        for i in range(self.batch_size):
            x, y = self.fetch_line()
            x_l.append(torch.tensor(x))
            y_l.append(torch.tensor(y))
        x = torch.stack(x_l)
        y = torch.stack(y_l)
        n_tokens = sum(sum(x != SPECIAL_TOKENS_IDS["<pad>"])).item()
        return x, y, n_tokens

    def fetch_line(self):
        part = (self.line // 100) + 1
        part_fname = self.fname + f".parts/{part}.jsonl"
        preprocessed_part_fname = part_fname + ".preprocessed"
        if not os.path.exists(preprocessed_part_fname):
            preprocess(part_fname, self.length, self.encoder, preprocessed_part_fname)
        if self.part_lines.get(part) is None:
            with open(preprocessed_part_fname) as f:
                self.part_lines[part] = f.readlines()
        l = self.part_lines[part][self.line % 100]
        j = json.loads(l)
        x = j["x"]
        y = j["y"]
        self.line += 1
        if self.line >= self.total_lines:
            self.ended = True
            self.line = 0
        return x, y

if __name__ == "__main__":
    from encoder import Encoder
    encoder = Encoder.from_path("encoder.json")
    loader = WanJuanLoader("tiny-example-news.jsonl", encoder, 5, 5)
    while not loader.ended:
        x, y, n = loader.get_data()
        print(n)
