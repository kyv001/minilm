from typing import Optional
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
from torch.nn.functional import pad
from config import *

class BinaryDataset(Dataset):
    def __init__(self, path: str, max_length: int):
        with open(path, "rb") as f:
            array = np.fromfile(path, dtype=np.uint16)
        full_length = array.size
        self.n_lines = full_length // (max_length + 1)
        length = self.n_lines * (max_length + 1)
        array = array[:length] # 裁剪到填满每一行
        self.data = torch.from_numpy(array).view(-1, max_length + 1)
    
    def __getitem__(self, index: int) -> torch.Tensor:
        return self.data[index]

    def __len__(self) -> int:
        return self.n_lines

def collate_fn(batch: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    l_x = []
    l_y = []
    # l: 这是一个测试。<eos>
    # x: 这是一个测试。
    # y: 是一个测试。<eos>
    for line in batch:
        if len(line) < MAX_LENGTH + 1:
            line = torch.cat((line, SPECIAL_TOKENS_TENSORS["<eos>"].unsqueeze(0)))
            line = pad(line, (0, MAX_LENGTH + 1 - len(line)), value=SPECIAL_TOKENS_IDS["<pad>"])
        l_x.append(line[:-1])
        l_y.append(line[1:])
    x = torch.stack(l_x).type_as(SPECIAL_TOKENS_TENSORS["<eos>"]) # int16 -> int防止类型不一致
    y = torch.stack(l_y).type_as(SPECIAL_TOKENS_TENSORS["<eos>"])
    n_tokens = (x != SPECIAL_TOKENS_IDS["<pad>"]).sum()
    return x, y, n_tokens


if __name__ == "__main__":
    from encoder import Encoder
    from torch.utils.data import DataLoader
    dts = BinaryDataset("ultrachat/ultrachat_release_230407.json.bin", MAX_LENGTH)
    print(len(dts))
    d = dts[0]
    e = Encoder.from_path("encoder.json")
    print(d)
    print(e.decode(list(d)))
    print(len(e.decode(list(d))))
    loader = DataLoader(dts, 5, True, collate_fn=collate_fn, num_workers=2)
    
    i = 0
    for x, y, n_tokens in loader:
        print(x.shape, y.shape, n_tokens)
        i += 1
        if i > 5:
            print("Y = ")
            print(e.decode(list(y[0][:60])))
            print("X = ")
            print(e.decode(list(x[0][:60])))
            break
