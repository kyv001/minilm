import json
import os
from tqdm import tqdm
import torch
from config import *
from preprocess import preprocess

class WanJuanLoader:
    def __init__(self, fname, encoder, batch_size, length):
        self.fname = fname
        self.batch_size = batch_size
        self.length = length

        preprocessed_fname = fname + ".preprocessed"
        self.preprocessed = os.path.exists(preprocessed_fname)
        if self.preprocessed:
            l = next(open(preprocessed_fname))
            if len(json.loads(l)["x"]) != length:
                self.preprocessed = False
        if not self.preprocessed:
            print(f"Preprocessing: {fname}")
            preprocess(fname, length, encoder, preprocessed_fname)

        self.batches, x_batch, y_batch = [], [], []
        i = 0
        n_tokens = 0
        with open(preprocessed_fname) as ppf:
            lines = ppf.readlines()
        for l in tqdm(lines):
            i += 1
            xy = json.loads(l)
            x = torch.tensor(xy["x"])
            y = torch.tensor(xy["y"])
            n_tokens += sum(x != SPECIAL_TOKENS_IDS["<pad>"])
            x_batch.append(x)
            y_batch.append(y)
            if i % batch_size == 0:
                x = torch.stack(x_batch)
                y = torch.stack(y_batch)
                self.batches.append((x, y, n_tokens))
                x_batch, y_batch = [], []
                n_tokens = 0
        self.total_lines = i
        self.line = 0
        self.ended = False

    def get_data(self):
        x, y, n_tokens = self.batches[self.line]
        self.line += 1
        if self.line == self.total_lines:
            self.ended = True
        return x, y, n_tokens

if __name__ == "__main__":
    from encoder import Encoder
    encoder = Encoder.from_path("encoder.json")
    loader = WanJuanLoader("tiny-example-news.jsonl", encoder, 1, 5)
    x, y, n = loader.get_data()
    print(x, y, n)
