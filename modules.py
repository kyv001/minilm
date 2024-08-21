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
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.device = device
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, T, _ = x.shape
        # (B, T, V) -proj-> (B, T, V)
        # -view-> (B, T, n_heads, head_dim)
        # -T(1, 2)-> (B, n_heads, T, head_dim)
        q = self.q_proj(x).view(B, T, self.n_heads, -1)
        k = self.k_proj(x).view(B, T, self.n_heads, -1)
        v = self.v_proj(x).view(B, T, self.n_heads, -1)
        # (B, n_heads, T, head_dim) -T(1, 2) -> (B, T, n_heads, head_dim)
        # -view-> (B, T, V)
        x = nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)\
            .transpose(1, 2).contiguous().view(B, T, -1)
        return self.proj(x)

class Block(nn.Module):
    def __init__(self, dim, max_length, n_heads, dropout, device):
        super().__init__()
        self.device = device
        self.ln1 = nn.LayerNorm(dim)
        self.attn = CausalSelfAttention(dim, max_length, n_heads, dropout, device)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, device)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class LLM(nn.Module):
    def __init__(self, vocab_size, dim, max_length, n_heads, n_blocks, dropout, device):
        super().__init__()
        self.device = device
        self.wte = nn.Embedding(vocab_size, dim)
        self.pe = nn.Embedding(max_length, dim) # TODO：将位置嵌入换成Llama风格的旋转位置编码
        self.blocks = nn.ModuleList([
            Block(dim, max_length, n_heads, dropout, device) for _ in range(n_blocks)
        ])
        self.ln = nn.LayerNorm(dim)
        self.lmhead = nn.Linear(dim, vocab_size)
        self.lmhead.weight = self.wte.weight # 共享权重减少参数数量
        positions_buffer = torch.arange(max_length, 0, -1) - 1
        self.register_buffer("positions_buffer", positions_buffer)

    def forward(self, x):
        pos = self.positions_buffer[-x.size(-1):]
        x = self.wte(x) + self.pe(pos)
        for block in self.blocks:
            x = block(x)
        x = self.ln(x)
        x = self.lmhead(x)
        return x