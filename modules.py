# Nvidia的新模型：nGPT

import torch
from torch import nn
import torch.nn.functional as F
from config import *

normalize = lambda x: F.normalize(x, dim=-1)

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
    def __init__(self, dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.dim = dim
        self.u_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.o_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

        # uv的缩放因子
        suinit = torch.tensor(1.0)
        suscale = torch.tensor(1.0)
        self.register_buffer('suinit', suinit)
        self.register_buffer('suscale', suscale)
        self.su = nn.Parameter(suscale.detach()) # su可以是一个标量
        svinit = torch.tensor(1.0)
        svscale = torch.tensor(1.0)
        self.register_buffer('svinit', svinit)
        self.register_buffer('svscale', svscale)
        self.sv = nn.Parameter(torch.ones(hidden_dim) * svscale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        actual_su = self.su * self.suinit / self.suscale
        actual_sv = self.sv * self.svinit / self.svscale
        u = self.u_proj(x) * actual_su
        v = self.v_proj(x) * actual_sv * self.dim ** 0.5
        return self.o_proj(self.dropout(u * nn.functional.silu(v)))

    def normalize(self) -> None:
        self.u_proj.weight.data = normalize(self.u_proj.weight.data)
        self.v_proj.weight.data = normalize(self.v_proj.weight.data)
        self.o_proj.weight.data = normalize(self.o_proj.weight.data)

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
        self.o_proj = nn.Linear(dim, dim, bias=False)
        self.pe = RotatoryPositionalEncoding(self.head_dim, max_length)
        self.dropout = dropout

        # QK的缩放因子
        sqkinit = torch.tensor(1.0)
        sqkscale = torch.tensor(1 / dim ** 0.5)
        self.register_buffer('sqkinit', sqkinit)
        self.register_buffer('sqkscale', sqkscale)
        self.sqk = nn.Parameter(torch.ones(dim) * sqkscale)

    def forward(self, x: torch.Tensor):
        B, T, _ = x.shape
        # (V) -view-> (n_heads, head_dim) -> (n_heads, 1, head_dim)
        actual_sqk = (self.sqk * self.sqkinit / self.sqkscale).view(self.n_heads, 1, -1)
        # (B, T, V) -proj-> (B, T, V)
        # -view-> (B, T, n_heads, head_dim)
        # -T(1, 2)-> (B, n_heads, T, head_dim)
        q = self.pe(self.q_proj(x).view(B, T, self.n_heads, -1).transpose(1, 2)) * actual_sqk
        k = self.pe(self.k_proj(x).view(B, T, self.n_heads, -1).transpose(1, 2)) * actual_sqk
        v = self.v_proj(x).view(B, T, self.n_heads, -1).transpose(1, 2)
        # (B, n_heads, T, head_dim) -T(1, 2)-> (B, T, n_heads, head_dim)
        # -view-> (B, T, V)
        x = (
            nn.functional.scaled_dot_product_attention(q, k, v,
                    is_causal=True, dropout_p=self.dropout,
                    scale=self.head_dim ** 0.5)
            .transpose(1, 2)
            .view(B, T, -1)
        )
        return self.o_proj(x)

    def normalize(self) -> None:
        self.q_proj.weight.data = normalize(self.q_proj.weight.data)
        self.k_proj.weight.data = normalize(self.k_proj.weight.data)
        self.v_proj.weight.data = normalize(self.v_proj.weight.data)
        self.o_proj.weight.data = normalize(self.o_proj.weight.data)

class Block(nn.Module):
    """一个Decoder块"""
    def __init__(self, dim: int, max_length: int, n_heads: int, dropout: float):
        super().__init__()
        self.attn = CausalSelfAttention(dim, max_length, n_heads, dropout)
        self.mlp = MLP(dim, dim * 4, dropout)

        # 自带的学习率
        self.lrinit_a = torch.tensor(0.05)
        self.lrscale_a = torch.tensor(1 / dim ** 0.5)
        self.lr_a = nn.Parameter(torch.ones(dim) * self.lrscale_a)
        self.lrinit_m = torch.tensor(0.05)
        self.lrscale_m = torch.tensor(1 / dim ** 0.5)
        self.lr_m = nn.Parameter(torch.ones(dim) * self.lrscale_m)

    def forward(self, x: torch.Tensor):
        actual_lr_a = self.lr_a * self.lrinit_a / self.lrscale_a
        actual_lr_m = self.lr_m * self.lrinit_m / self.lrscale_m
        x = normalize(x + (self.attn(x) - x) * actual_lr_a)
        x = normalize(x + (self.mlp(x) - x) * actual_lr_m)
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
        self.lmhead = nn.Linear(dim, vocab_size)

        # Logit缩放因数
        szinit = torch.tensor(1.0)
        szscale = torch.tensor(1 / dim ** 0.5)
        self.register_buffer('szinit', szinit)
        self.register_buffer('szscale', szscale)
        self.sz = nn.Parameter(torch.ones(vocab_size) * szscale)

        self.normalize()

    def forward(self, x: torch.Tensor):
        x = self.wte(x)
        for block in self.blocks:
            x = block(x)
        x = self.lmhead(x)
        actual_sz = self.sz * self.szinit / self.szscale
        return x * actual_sz

    def save(self, path: str):
        torch.save(self.state_dict(), path) # 保存模型参数防止带上不必要的前缀

    def normalize(self) -> None:
        self.wte.weight.data = normalize(self.wte.weight.data)
        self.lmhead.weight.data = normalize(self.lmhead.weight.data)
        for block in self.blocks:
            block.attn.normalize()
            block.mlp.normalize()
        