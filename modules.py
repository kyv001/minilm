import torch
from torch import nn
from config import *

class MLP(nn.Module):
    def __init__(self, dim, device):
        super().__init__()
        self.device = device
        self.linear = nn.Linear(dim, dim * 4)
        self.gelu = nn.GELU()
        self.proj = nn.Linear(dim * 4, dim)

    def forward(self, x):
        x = self.linear(x)
        x = self.gelu(x)
        return self.proj(x)

class CausalSelfAttention(nn.Module):
    def __init__(self, dim, max_length, n_heads, dropout, device):
        super().__init__()
        self.device = device
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.mha = nn.MultiheadAttention(dim, n_heads, dropout, batch_first=True)
        self.proj = nn.Linear(dim, dim)
        causal_mask = torch.tril(torch.ones((max_length, max_length))).to(self.device) == 0
        self.register_buffer("causal_mask", causal_mask)

    def forward(self, x, key_padding_mask):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        x = self.mha(
            q, k, v,
            key_padding_mask=key_padding_mask,
            attn_mask=self.causal_mask[:x.size(1), :x.size(1)],
            is_causal=True,
            need_weights=False
        )[0]
        return self.proj(x)

class Block(nn.Module):
    def __init__(self, dim, max_length, n_heads, dropout, device):
        super().__init__()
        self.device = device
        self.ln1 = nn.LayerNorm(dim)
        self.attn = CausalSelfAttention(dim, max_length, n_heads, dropout, device)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, device)

    def forward(self, x, key_padding_mask):
        x = x + self.attn(self.ln1(x), key_padding_mask)
        x = x + self.mlp(self.ln2(x))
        return x

class LLM(nn.Module):
    def __init__(self, vocab_size, dim, max_length, n_heads, n_blocks, dropout, device):
        super().__init__()
        self.device = device
        self.wte = nn.Embedding(vocab_size, dim)
        self.pe = nn.Embedding(max_length, dim)
        self.blocks = nn.ModuleList([
            Block(dim, max_length, n_heads, dropout, device) for _ in range(n_blocks)
        ])
        self.ln = nn.LayerNorm(dim)
        self.lmhead = nn.Linear(dim, vocab_size)
        self.lmhead.weight = self.wte.weight # 共享权重减少参数数量
        positions_buffer = torch.arange(max_length, 0, -1) - 1
        self.register_buffer("positions_buffer", positions_buffer)

    def forward(self, x):
        key_padding_mask = (x == SPECIAL_TOKENS_IDS["<pad>"])
        pos = self.positions_buffer[-x.size(-1):]
        x = self.wte(x) + self.pe(pos)
        for block in self.blocks:
            x = block(x, key_padding_mask)
        x = self.ln(x)
        x = self.lmhead(x)
        return x