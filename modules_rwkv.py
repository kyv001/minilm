"""RWKV模型：https://arxiv.org/abs/2305.13048"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import *
class WKV(nn.Module):
    """RWKV版的注意力机制"""
    def __init__(self, dim: int):
        super().__init__()
        raise NotImplementedError

    def forward(self, r: torch.Tensor, w: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        raise NotImplementedError

class MultiHeadWKV(nn.Module):
    """多头WKV"""
    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        raise NotImplementedError

    def forward(self, r: torch.Tensor, w: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor):
        raise NotImplementedError

class TimeMixing(nn.Module):
    def __init__(self, dim: int, lora_dim: int, n_heads: int):
        super().__init__()
        raise NotImplementedError
        
    def forward(self, x: torch.Tensor):
        raise NotImplementedError

class ChannelMixing(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        raise NotImplementedError

    def forward(self, x: torch.Tensor):
        raise NotImplementedError

class Block(nn.Module):
    def __init__(self, dim: int, lora_dim: int, n_heads: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.time_mixing = TimeMixing(dim, lora_dim, n_heads)
        self.channel_mixing = ChannelMixing(dim)

    def forward(self, x: torch.Tensor):
        x = x + self.time_mixing(self.ln1(x))
        x = x + self.channel_mixing(self.ln2(x))
        return x

class LLM(nn.Module):
    """RWKV模型"""
    def __init__(self, vocab_size: int, dim: int, lora_dim: int,
                n_blocks: int, n_heads: int):
        super().__init__()
        assert dim % n_heads == 0
        self.wte = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([
            Block(dim, lora_dim, n_heads) for _ in range(n_blocks)
        ])
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.lmhead = nn.Linear(dim, vocab_size, bias=False)
        self.lmhead.weight = self.wte.weight # 共享权重

    def forward(self, x: torch.Tensor):
        x = self.wte(x)
        x = self.ln1(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln2(x)
        x = self.lmhead(x)
        return x

    def save(self, path: str):
        torch.save(self.state_dict(), path)
