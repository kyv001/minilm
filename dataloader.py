import json
import torch
from config import *

class WanJuanLoader:
    def __init__(self, fname, encoder):
        with open(fname) as f:
            self.lines = f.readlines()
        self.line = 0
        self.total_lines = len(self.lines)
        self.encoder = encoder
        self.ended = False

    def get_data(self, batch_size, length):
        full_length = length + 1
        total_amount = batch_size * full_length
        data_l = []
        n_batches_left = batch_size
        while n_batches_left > 0:
            line = self._fetch_line()
            if line is not None:
                content = json.loads(line)["content"]
            else:
                content = ""
                self.ended = True
                data_l.append("")
                n_batches_left -= 1
                continue
            for i in range(len(content) // full_length):
                data_l.append(content[i * full_length:(i + 1) * full_length])
                n_batches_left -= 1
            rest = len(content) % full_length
            if rest:
                data_l.append(content[-rest:])
                n_batches_left -= 1
        data_l = data_l[:batch_size]
        x_l = []
        y_l = []
        for i in range(len(data_l)):
            data_length = len(data_l[i])
            n_paddings = full_length - data_length
            codes = self.encoder.encode(data_l[i])
            if n_paddings:
                codes += [SPECIAL_TOKENS_IDS["<eos>"]]
                n_paddings -= 1
            padded_tensor = torch.nn.functional.pad(
                torch.tensor(codes),
                (0, n_paddings),
                value=SPECIAL_TOKENS_IDS["<pad>"]
            )
            x_l.append(padded_tensor[:-1])
            y_l.append(padded_tensor[1:])
        x = torch.stack(x_l)
        y = torch.stack(y_l)
        n_tokens = (x != SPECIAL_TOKENS_IDS["<pad>"]).sum()
        return x, y, n_tokens
    
    def _fetch_line(self):
        self.line += 1
        if self.line <= self.total_lines:
            return self.lines[self.line - 1]
        else:
            return None

if __name__ == "__main__":
    from encoder import Encoder
    encoder = Encoder.from_path("encoder.json")
    loader = WanJuanLoader("tiny-example-news.jsonl", encoder)
    x, y, n = loader.get_data(1, 751)
    print(x, y, n)
