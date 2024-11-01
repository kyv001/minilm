import torch
from torch import nn
import torch.nn.functional as F
from config import *

class LoRA(nn.Module):
    def __init__(self, dim: int, lora_dim: int):
        super().__init__()
        self.a = nn.Linear(dim, lora_dim, bias=False)
        nn.init.normal_(self.a.weight, std=10e-4)
        self.b = nn.Linear(lora_dim, dim, bias=False)
        nn.init.normal_(self.b.weight, std=10e-4)
        self.lambd = nn.Parameter(torch.empty(dim))
        nn.init.normal_(self.lambd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lambd + self.b(F.tanh(self.a(x)))

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

class ChannelMixing(nn.Module):
    """来源于RWKV的通道混合"""
    def __init__(self, dim: int, block_id: int, n_blocks: int):
        super().__init__()
        r0 = block_id / (n_blocks - 1)
        r1 = 1 - block_id / n_blocks
        i = torch.arange(dim)
        self.r_proj = nn.Linear(dim, dim, bias=False)
        nn.init.zeros_(self.r_proj.weight)
        self.k_proj = nn.Linear(dim, dim * 4, bias=False)
        nn.init.orthogonal_(self.k_proj.weight, gain=4)
        self.v_proj = nn.Linear(dim * 4, dim, bias=False)
        nn.init.zeros_(self.v_proj.weight)
        self.r_weight = nn.Parameter(1 - (i / dim) ** r1)
        self.k_weight = nn.Parameter(1 - (i / dim) ** r1)
        self.pad = nn.ZeroPad2d((0, 0, 1, -1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xx = self.pad(x)
        xxx = xx - x
        r = self.r_proj(x + xxx * self.r_weight)
        k = self.k_proj(x + xxx * self.k_weight)
        v = self.v_proj(F.relu(k) ** 2)
        return F.sigmoid(r) * v

class CausalSelfAttention(nn.Module):
    """带因果关系的多头自注意力，使用Flash Attention和RoPE"""
    def __init__(self, dim: int, max_length: int, n_heads: int, dropout: float, lora_dim: int):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.q_lora = LoRA(dim, lora_dim)
        self.k_lora = LoRA(dim, lora_dim)
        self.v_lora = LoRA(dim, lora_dim)
        self.pad = nn.ZeroPad2d((0, 0, 1, -1))
        self.proj = nn.Linear(dim, dim)
        self.pe = RotatoryPositionalEncoding(self.head_dim, max_length)
        self.dropout = dropout

    def forward(self, x: torch.Tensor):
        B, T, _ = x.shape
        xx = self.pad(x) # 来自RWKV的Token Shifting
        xxx = xx - x
        q = self.q_proj(x + xxx * self.q_lora(x))
        k = self.k_proj(x + xxx * self.k_lora(x))
        v = self.v_proj(x + xxx * self.v_lora(x))
        # (B, T, V) -proj-> (B, T, V)
        # -view-> (B, T, n_heads, head_dim)
        # -T(1, 2)-> (B, n_heads, T, head_dim)
        q = self.pe(q.view(B, T, self.n_heads, -1).transpose(1, 2))
        k = self.pe(k.view(B, T, self.n_heads, -1).transpose(1, 2))
        v = v.view(B, T, self.n_heads, -1).transpose(1, 2)
        # (B, n_heads, T, head_dim) -T(1, 2) -> (B, T, n_heads, head_dim)
        # -view-> (B, T, V)
        x = (
            nn.functional.scaled_dot_product_attention(q, k, v,
                    is_causal=True, dropout_p=self.dropout)
            .transpose(1, 2)
            .contiguous()
            .view(B, T, -1)
        )
        return self.proj(x)

class Block(nn.Module):
    """一个Decoder块"""
    def __init__(self, dim: int, max_length: int, n_heads: int, dropout: float, block_id: int, n_blocks: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = CausalSelfAttention(dim, max_length, n_heads, dropout, 64)
        self.ln2 = nn.LayerNorm(dim)
        self.cmix = ChannelMixing(dim, block_id, n_blocks)

    def forward(self, x: torch.Tensor):
        x = x + self.attn(self.ln1(x))
        x = x + self.cmix(self.ln2(x))
        return x

class LLM(nn.Module):
    """大模型本体"""
    def __init__(self, vocab_size: int, dim: int, max_length: int, n_heads: int,
                n_blocks: int, dropout: float):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([
            Block(dim, max_length, n_heads, dropout, block_id=_, n_blocks=n_blocks) for _ in range(n_blocks)
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

    def save(self, path: str):
        torch.save(self.state_dict(), path) # 保存模型参数防止带上不必要的前缀
        