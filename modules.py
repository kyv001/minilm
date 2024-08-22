import torch
from torch import nn
from config import *

class RotatoryPositionalEncoding(nn.Module):
    """旋转位置编码"""
    def __init__(self, dim: int, max_length: int):
        super().__init__()
        assert dim % 2 == 0
        positions = torch.arange(0, max_length, 1)
        theta = 1 / 10000 ** (torch.arange(0, dim, 2) / dim) # thetai = 1/10000^(2i/dim)
        """
            theta0  theta1  theta2  theta3 ... theta(dim/2-1)
        m=0 0theta0 0theta1 0theta2 0theta3
        m=1 1theta0 1theta1 1theta2 1theta3
        m=2 2theta0 2theta1 2theta2 2theta3
        m=3 3theta0 3theta1 3theta2 3theta3
        ...
        m=max_length-1                         ...
        """
        positions_theta = positions.unsqueeze(1) * theta.unsqueeze(0) # (max_length, dim//2)
        positions_sin = torch.sin(positions_theta)
        positions_cos = torch.cos(positions_theta)
        self.register_buffer('positions_sin', positions_sin)
        self.register_buffer('positions_cos', positions_cos)
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_real = x[..., :self.dim // 2] # (x.size(-2), dim//2)
        x_imag = x[...,  self.dim // 2:]
        pos_cos = self.positions_cos[:x.size(-2)] # (x.size(-2), dim//2)
        pos_sin = self.positions_sin[:x.size(-2)]
        y_real = x_real * pos_cos - x_imag * pos_sin
        y_imag = x_real * pos_sin + x_imag * pos_cos
        return torch.cat([y_real, y_imag], dim=-1)

class MLP(nn.Module):
    """Feed forward层，一个普通的多层感知机"""
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim * 4)
        self.linear2 = nn.Linear(dim, dim * 4)
        self.proj = nn.Linear(dim * 4, dim)
        self.silu = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        x1 = self.linear1(x)
        x2 = self.linear2(x)
        y = self.proj(self.silu(x1) * x2)
        return self.dropout(y)

class CausalSelfAttention(nn.Module):
    """带因果关系的多头自注意力，使用Flash Attention和RoPE"""
    def __init__(self, dim: int, max_length: int, n_heads: int, dropout: float):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.pe = RotatoryPositionalEncoding(self.head_dim, max_length)
        self.dropout = dropout
        self.final_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        B, T, _ = x.shape
        # (B, T, V) -proj-> (B, T, V)
        # -view-> (B, T, n_heads, head_dim)
        # -T(1, 2)-> (B, n_heads, T, head_dim)
        q = self.pe(self.q_proj(x).view(B, T, self.n_heads, -1))
        k = self.pe(self.k_proj(x).view(B, T, self.n_heads, -1))
        v = self.pe(self.v_proj(x).view(B, T, self.n_heads, -1))
        # (B, n_heads, T, head_dim) -T(1, 2) -> (B, T, n_heads, head_dim)
        # -view-> (B, T, V)
        x = nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.dropout)\
            .transpose(1, 2).contiguous().view(B, T, -1)
        return self.final_dropout(self.proj(x))

class Block(nn.Module):
    """一个Encoder块"""
    def __init__(self, dim: int, max_length: int, n_heads: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = CausalSelfAttention(dim, max_length, n_heads, dropout)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, dropout)

    def forward(self, x: torch.Tensor):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class LLM(nn.Module):
    """大模型本体"""
    def __init__(self, vocab_size: int, dim: int, max_length: int, n_heads: int,
                n_blocks: int, dropout: float):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([
            Block(dim, max_length, n_heads, dropout) for _ in range(n_blocks)
        ])
        self.ln = nn.LayerNorm(dim)
        self.lmhead = nn.Linear(dim, vocab_size)
        self.lmhead.weight = self.wte.weight # 共享权重减少参数数量

    def forward(self, x: torch.Tensor):
        x = self.wte(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln(x)
        x = self.lmhead(x)
        return x