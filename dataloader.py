import torch
from tqdm import tqdm
from torch.utils.data import Dataset
from torch.nn.functional import pad
from config import *

class BinaryDataset(Dataset):
    def __init__(self, path: str, line_sep: str):
        byte_sep = line_sep.encode("utf-8")
        self.lines = []
        left = b""
        with open(path, "rb") as f:
            while True:
                d = left + f.read(1024 * 1024)
                if len(d) == 0:
                    break
                l = d.split(byte_sep)
                self.lines.extend(l[:-1])
                left = l[-1]
    
    def __getitem__(self, index: int) -> torch.Tensor:
        l = self.lines[index].decode("utf-8")
        c = list(map(ord, l))
        return torch.tensor(c)

    def __len__(self) -> int:
        return len(self.lines)

def collate_fn(batch: list[torch.Tensor]) -> tuple:
    l_x = []
    l_y = []
    for line in batch:
        if len(line) < MAX_LENGTH + 1:
            line = torch.cat((line, SPECIAL_TOKENS_TENSORS["<eos>"].unsqueeze(0)))
            line = pad(line, (0, MAX_LENGTH + 1 - len(line)), value=SPECIAL_TOKENS_IDS["<pad>"])
        l_x.append(line[:-1])
        l_y.append(line[1:])
    x, y = torch.stack(l_x), torch.stack(l_x)
    n_tokens = (x != SPECIAL_TOKENS_IDS["<pad>"]).sum()
    return x, y, n_tokens


if __name__ == "__main__":
    from encoder import Encoder
    from torch.utils.data import DataLoader
    dts = BinaryDataset("WanJuan-News/part-006853-a894b46e.jsonl.contents.txt.lines.txt.encoded.bin", LINE_SEP)
    # dts = BinaryDataset("tiny-example-news.jsonl.contents.txt.lines.txt.encoded.bin", LINE_SEP)
    print(len(dts))
    d = dts[0]
    e = Encoder.from_path("encoder.json")
    print(d)
    print(e.decode(list(d)))
    print(len(e.decode(list(d))))
    loader = DataLoader(dts, 5, True, collate_fn=collate_fn, num_workers=2)
    
    for x, y, n_tokens in loader:
        print(x.shape, y.shape, n_tokens)
        break
