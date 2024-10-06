"""RWKV模型：https://arxiv.org/abs/2305.13048"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import *

class Lerp(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.w = nn.Parameter(torch.empty(dim))

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        return x + (y - x) * self.w

class LoRA(nn.Module):
    def __init__(self, dim: int, lora_dim: int):
        super().__init__()
        self.proj_down = nn.Parameter(torch.empty(dim, lora_dim))
        self.proj_up = nn.Parameter(torch.empty(lora_dim, dim))
        self.bias = nn.Parameter(torch.empty(dim))

    def forward(self, x: torch.Tensor):
        return self.bias + self.proj_up(F.tanh(self.proj_down(x)))

class DDLerp(nn.Module):
    def __init__(self, dim: int, lora_dim: int):
        super().__init__()
        self.lora = LoRA(dim, lora_dim)
        self.w = nn.Parameter(torch.empty(dim))

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        return x + (y - x) * self.lora(x + (y - x) * self.w)

class LoRADDLerp(nn.Module):
    def __init__(self, dim: int, lora_dim: int):
        super().__init__()
        self.lora = LoRA(dim, lora_dim)
        self.w = nn.Parameter(torch.empty(dim))

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        return self.lora(x + (y - x) * self.lora(x + (y - x) * self.w))

class WKV(nn.Module):
    """RWKV版的注意力机制"""
    def __init__(self, dim: int):
        super().__init__()
        self.s = nn.Parameter(torch.empty(dim))
        self.u = nn.Parameter(torch.empty(dim))

    def forward(self, r: torch.Tensor, w: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        """
        S_t = u A_t + A_{t-1} + w_{t-1} A_{t-2} + w_{t-1} w_{t-2} A_{t-3} + ...

        [   1   x   1   ]xA_{t-1}
        [   1   xW_{t-1}]xA_{t-2}

        [0, 0] [W_{t-2}, W_{t-1}] A_{t-1}
        [0, 1] [W_{t-2}, W_{t-1}] A_{t-2}
        RWKV注意力，并行计算
        """
        a = torch.einsum('...j,...k->...jk', k, v).to("cuda")
        wkv = torch.sum(torch.prod(
            torch.tril(
                torch.ones((k.size(-2) - 1, k.size(-2) - 1)).to("cuda"),
                diagonal=-1
            ).flip(-1).unsqueeze(-1).unsqueeze(-4) *
            (w[..., :-1, :] - 1).unsqueeze(-3).repeat(1, k.size(-2) - 1, 1, 1) + 1,
            dim=-2
        ).unsqueeze(-2) * a[..., :-1, :, :].flip(-3), dim=-3) + self.u * a[..., -1, :, :]
        return (r @ wkv).view(w.shape)

class MultiHeadWKV(nn.Module):
    """多头WKV"""
    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        self.heads = nn.ModuleList([
            WKV(dim // n_heads) for _ in range(n_heads)
        ])
        self.ln = nn.LayerNorm(dim // n_heads)

    def forward(self, r: torch.Tensor, w: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor):
        r_heads = r.chunk(self.heads, dim=-1)
        w_heads = w.chunk(self.heads, dim=-1)
        k_heads = k.chunk(self.heads, dim=-1)
        v_heads = v.chunk(self.heads, dim=-1)
        g_heads = g.chunk(self.heads, dim=-1)
        wkv_heads = [
            head(r_head, w_head, k_head, v_head)
            for head, r_head, w_head, k_head, v_head
            in zip(self.heads, r_heads, w_heads, k_heads, v_heads)
        ]
        return torch.cat((
            self.ln(wkv_head) * F.silu(g_head)
            for wkv_head, g_head
            in zip(wkv_heads, g_heads)
        ), dim=-1)

class TimeMixing(nn.Module):
    def __init__(self, dim: int, lora_dim: int, n_heads: int):
        super().__init__()
        self.r_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.g_proj = nn.Linear(dim, dim, bias=False)
        self.r_ddlerp = DDLerp(dim, lora_dim)
        self.k_ddlerp = DDLerp(dim, lora_dim)
        self.v_ddlerp = DDLerp(dim, lora_dim)
        self.g_ddlerp = DDLerp(dim, lora_dim)
        self.loraddlerp = LoRADDLerp(dim, lora_dim)
        self.pad = nn.ZeroPad2d((0, 0, 1, -1))
        self.mhwkv = MultiHeadWKV(dim, n_heads)
        
    def forward(self, x: torch.Tensor):
        x_1 = self.pad(x)
        r = self.r_proj(self.r_ddlerp(x, x_1))
        k = self.k_proj(self.k_ddlerp(x, x_1))
        v = self.v_proj(self.v_ddlerp(x, x_1))
        g = self.g_proj(self.g_ddlerp(x, x_1))
        d = self.loraddlerp(x, x_1)
        w = F.exp(-F.exp(d))
        return self.mhwkv(w, k, v, g)

class ChannelMixing(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.r_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim * 4, bias=False)
        self.v_proj = nn.Linear(dim * 4, dim, bias=False)
        self.r_lerp = Lerp(dim)
        self.k_lerp = Lerp(dim)
        self.pad = nn.ZeroPad2d((0, 0, 1, -1))

    def forward(self, x: torch.Tensor):
        x_1 = self.pad(x)
        r = self.r_proj(self.r_lerp(x, x_1))
        k = self.k_proj(self.k_lerp(x, x_1))
        v = self.v_proj(F.relu(k) ** 2)
        return F.sigmoid(r) * v

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

class RWKV(nn.Module):
    """RWKV模型"""
    def __init__(self, vocab_size: int, dim: int, lora_dim: int,
                n_blocks: int, n_heads: int):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([
            Block(dim, lora_dim, n_heads) for _ in range(n_blocks)
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
