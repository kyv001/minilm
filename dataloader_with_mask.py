from typing import Optional
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
from torch.nn.functional import pad
from config import *

class BinaryDatasetWithMask(Dataset):
    def __init__(self, path: str, path_mask: str, max_length: int):
        with open(path, "rb") as f:
            data = np.fromfile(path, dtype=np.uint16)
        with open(path_mask, "rb") as f:
            mask = np.fromfile(path_mask, dtype=np.uint8)
        full_length = data.size
        assert data.shape == mask.shape, "掩码和数据不匹配！"
        self.n_lines = full_length // (max_length + 1)
        length = self.n_lines * (max_length + 1)
        data = data[:length] # 裁剪到填满每一行
        mask = mask[:length]
        self.data = torch.from_numpy(data).view(-1, max_length + 1)
        self.mask = torch.from_numpy(mask).view(-1, max_length + 1)
    
    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.data[index], self.mask[index]

    def __len__(self) -> int:
        return self.n_lines

def collate_fn_with_mask(batch: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    l_x = []
    l_y = []
    l_my = []
    # l: 这是一个测试。<eos>
    # x: 这是一个测试。
    # y: 是一个测试。<eos>
    # m: 1111111
    for i in range(len(batch)):
        line, mask = batch[i]
        if len(line) < MAX_LENGTH + 1:
            line = torch.cat((line, SPECIAL_TOKENS_TENSORS["<eos>"].unsqueeze(0)))
            line = pad(line, (0, MAX_LENGTH + 1 - len(line)), value=SPECIAL_TOKENS_IDS["<pad>"])
            mask = torch.cat((mask, torch.tensor([True])))
            mask = pad(mask, (0, MAX_LENGTH + 1 - len(mask)), value=0)
        l_x.append(line[:-1])
        l_y.append(line[1:])
        l_my.append(mask[1:])
    x = torch.stack(l_x).type_as(SPECIAL_TOKENS_TENSORS["<eos>"]) # int16 -> int防止类型不一致
    y = torch.stack(l_y).type_as(SPECIAL_TOKENS_TENSORS["<eos>"])
    my = torch.stack(l_my).type_as(SPECIAL_TOKENS_TENSORS["<eos>"])
    n_tokens = my.sum()
    return x, y, my, n_tokens


if __name__ == "__main__":
    from encoder import Encoder
    from torch.utils.data import DataLoader
    dts = BinaryDatasetWithMask("ultrachat/ultrachat_release_230407.json.bin", "ultrachat/ultrachat_release_230407.json.mask.bin", MAX_LENGTH)
    e = Encoder.from_path("encoder.json")
    loader = DataLoader(dts, 1, True, collate_fn=collate_fn_with_mask, num_workers=2)
    
    i = 0
    for x, y, my, n_tokens in loader:
        print(x.shape, y.shape, n_tokens)
        i += 1
        if i > 5:
            for i in range(len(y[0])):
                if my[0][i]:
                    print(f"{e.decode([y[0][i].item()])}", end="")
                else:
                    print(f"\033[31m{e.decode([y[0][i].item()])}\033[0m", end="")
            break
