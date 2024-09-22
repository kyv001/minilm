"""RWKV模型：https://arxiv.org/abs/2305.13048"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import *

class WKV(nn.Module):
    """RWKV版的注意力机制"""
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.w = nn.Parameter(torch.empty(dim))
        self.u = nn.Parameter(torch.empty(dim))

    def forward(self, k: torch.Tensor, v: torch.Tensor):
        wkv = torch.zeros_like(k)
        for t in range(k.size(-2)):
            vt_score = torch.exp(u + k[..., t, :])
            for i in range(1, t):
                vi_score = torch.exp(-(t - 1 - i) * self.w + k[..., i, :])
                wkv[..., t, :] += (
                    vi_score * v[..., i, :] + vt_score * v[..., t, :] /
                    vi_score + vt_score
                )
        return wkv

class TimeMixing(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.r_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)
        self.r_mu = nn.Parameter(torch.empty(dim))
        self.k_mu = nn.Parameter(torch.empty(dim))
        self.v_mu = nn.Parameter(torch.empty(dim))
        self.pad = nn.ZeroPad2d((0, 0, 1, -1))
        self.wkv = WKV(dim, dropout)
        
    def forward(self, x: torch.Tensor):
        x_1 = self.pad(x)
        r = self.r_proj(x * self.r_mu + x_1 * (1 - self.r_mu))
        k = self.k_proj(x * self.k_mu + x_1 * (1 - self.k_mu))
        v = self.v_proj(x * self.v_mu + x_1 * (1 - self.v_mu))
        return self.o_proj(F.sigmoid(r) * self.wkv(k, v))

class ChannelMixing(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.r_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.r_mu = nn.Parameter(torch.empty(dim))
        self.k_mu = nn.Parameter(torch.empty(dim))
        self.pad = nn.ZeroPad2d((0, 0, 1, -1))

    def forward(self, x: torch.Tensor):
        x_1 = self.pad(x)
        r = self.r_proj(x * self.r_mu + x_1 * (1 - self.r_mu))
        k = self.k_proj(x * self.k_mu + x_1 * (1 - self.k_mu))
        v = self.v_proj(x)
        return F.sigmoid(r) * (v * (F.relu(k)) ** 2)

class Block(nn.Module):
    def __init__(self, dim: int, n_heads: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.time_mixing = TimeMixing(dim, dropout)
        self.channel_mixing = ChannelMixing(dim, dropout)

    def forward(self, x: torch.Tensor):
        x = x + self.time_mixing(self.ln1(x))
        x = x + self.channel_mixing(self.ln2(x))
        return x

class RWKV(nn.Module):
    """RWKV4单头模型"""
    def __init__(self, vocab_size: int, dim: int,
                n_blocks: int, dropout: float):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([
            Block(dim, n_heads, dropout) for _ in range(n_blocks)
        ])
        self.ln = nn.LayerNorm(dim)
        self.lmhead = nn.Linear(dim, vocab_size, bias=False)
        self.lmhead.weight = self.wte.weight # 共享权重

    def forward(self, x: torch.Tensor):
        x = self.wte(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln(x)
        x = self.lmhead(x)
        return x

    def save(self, path: str):
        torch.save(self.state_dict(), path)
