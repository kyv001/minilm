"""RWKV6"""
from typing import Optional, Sequence
import torch
import torch.nn as nn
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

class WKV(nn.Module):
    """RWKV版的注意力机制"""
    def __init__(self, dim: int, block_id: int, n_blocks: int):
        super().__init__()
        r0 = block_id / (n_blocks - 1)
        r1 = 1 - block_id / n_blocks
        i = torch.arange(dim)
        self.init_state = nn.Parameter(torch.empty(dim))
        nn.init.normal_(self.init_state)
        self.u = nn.Parameter(r0 * (1 - i / (dim - 1)) + 0.1 * (i + 1) % 3)

    def forward(self, r: torch.Tensor, w: torch.Tensor, k: torch.Tensor, v: torch.Tensor, 
                state: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:
        wkv = torch.zeros(*k.size()[:-1], k.size(-1), v.size(-1)).to(k.device)
        state1: torch.Tensor
        if state is None:
            state1 = self.init_state
        else:
            state1 = state
        for i in range(k.size(-2)):
            wkv[..., i, :, :] = state1 + torch.diag(self.u) @ k[..., i, :].mT @ v[..., i, :]
            state1 = self.diag(w[..., i, :]) @ state1 + k[..., i, :].mT @ v[..., i, :]

        return (r.unsqueeze(-2) @ wkv).squeeze(-2), state1
    
    @staticmethod
    def diag(wi):
        if len(wi.shape) == 2:
            diagw = torch.stack([torch.diag(t) for t in wi.unbind(dim=-2)])
        else:
            diagw = torch.diag(wi)
        return diagw

class MultiHeadWKV(nn.Module):
    """多头WKV"""
    def __init__(self, dim: int, n_heads: int, block_id: int, n_blocks: int):
        super().__init__()
        assert dim % n_heads == 0, "模型维度必须是n_heads的整数倍"
        self.n_heads = n_heads
        self.heads = nn.ModuleList([
            WKV(dim // n_heads, block_id, n_blocks) for _ in range(n_heads)
        ])
        self.ln = nn.LayerNorm(dim // n_heads)
        nn.init.constant_(self.ln.weight, ((1 + block_id) / n_blocks) ** 0.7)
        self.o_proj = nn.Linear(dim, dim, bias=False)
        nn.init.zeros_(self.o_proj.weight)

    def forward(self, r: torch.Tensor, w: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor,
                state: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:
        r_heads = r.chunk(self.n_heads, dim=-1)
        w_heads = w.chunk(self.n_heads, dim=-1)
        k_heads = k.chunk(self.n_heads, dim=-1)
        v_heads = v.chunk(self.n_heads, dim=-1)
        g_heads = g.chunk(self.n_heads, dim=-1)
        state_heads: Sequence[Optional[torch.Tensor]]
        if state is None:
            state_heads = [None] * self.n_heads
        else:
            state_heads = state.chunk(self.n_heads, dim=-1)
        o_s_heads = [self.heads[i](
            r_heads[i],
            w_heads[i],
            k_heads[i],
            v_heads[i],
            state_heads[i]
        ) for i in range(self.n_heads)]
        o = torch.cat([self.ln(o_s_heads[i][0]) * F.silu(g_heads[i]) for i in range(self.n_heads)], dim=-1)
        state = torch.cat([o_s_head[1] for o_s_head in o_s_heads], dim=-1)
        del o_s_heads, w_heads, k_heads, v_heads, g_heads, state_heads
        return self.o_proj(o), state

class TimeMixing(nn.Module):
    def __init__(self, dim: int, lora_dim: int, n_heads: int, block_id: int, n_blocks: int):
        super().__init__()
        r0 = block_id / (n_blocks - 1)
        r1 = 1 - block_id / n_blocks
        self.r_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.g_proj = nn.Linear(dim, dim, bias=False)
        self.r_lora = LoRA(dim, lora_dim)
        self.k_lora = LoRA(dim, lora_dim)
        self.v_lora = LoRA(dim, lora_dim)
        self.g_lora = LoRA(dim, lora_dim)
        self.d_lora = LoRA(dim, lora_dim)
        i = torch.arange(dim)
        self.x_weight = nn.Parameter(1 - (i / dim) ** r1)
        self.pad = nn.ZeroPad2d((0, 0, 1, -1))
        self.wkv = MultiHeadWKV(dim, n_heads, block_id, n_blocks)
        
    def forward(self, x: torch.Tensor, state: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:
        xx = self.pad(x)
        xxx = xx - x
        lerpx = x + xxx * self.x_weight
        r = self.r_proj(x + xxx * self.r_lora(lerpx))
        k = self.k_proj(x + xxx * self.k_lora(lerpx))
        v = self.v_proj(x + xxx * self.v_lora(lerpx))
        g = self.g_proj(x + xxx * self.g_lora(lerpx))
        w = torch.exp(-torch.exp(self.d_lora(x + xxx * self.d_lora(lerpx))))
        return self.wkv(r, w, k, v, g, state)

class ChannelMixing(nn.Module):
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

class Block(nn.Module):
    def __init__(self, dim: int, lora_dim: int, n_heads: int, block_id: int, n_blocks: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.time_mixing = TimeMixing(dim, lora_dim, n_heads, block_id, n_blocks)
        self.channel_mixing = ChannelMixing(dim, block_id, n_blocks)

    def forward(self, x: torch.Tensor, state: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:
        x1, state1 = self.time_mixing(self.ln1(x), state)
        x = x1 + x + self.channel_mixing(self.ln2(x))
        return x, state1

class LLM(nn.Module):
    """RWKV模型"""
    def __init__(self, vocab_size: int, dim: int, lora_dim: int,
                n_blocks: int, n_heads: int, max_lr: float = 3e-4):
        super().__init__()
        assert dim % n_heads == 0
        self.wte = nn.Embedding(vocab_size, dim)
        nn.init.normal_(self.wte.weight, std=max_lr)
        self.blocks = nn.ModuleList([
            Block(dim, lora_dim, n_heads, block_id, n_blocks) for block_id in range(n_blocks)
        ])
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.lmhead = nn.Linear(dim, vocab_size, bias=False)
        nn.init.orthogonal_(self.lmhead.weight, gain=0.5)

    def forward(self, x: torch.Tensor, states: Optional[list[torch.Tensor]] = None) -> tuple[torch.Tensor, list[torch.Tensor]]:
        x = self.wte(x)
        x = self.ln1(x)
        states1: list[torch.Tensor] = []
        states_i: list[Optional[torch.Tensor]]
        if states is None:
            states_i = [None] * len(self.blocks)
        else:
            states_i = states # type: ignore # 此处的states不可能是None，而是list[torch.Tensor]
        for i in range(len(self.blocks)):
            x, s = self.blocks[i](x, states_i[i])
            states1.append(s)
        x = self.ln2(x)
        x = self.lmhead(x)
        return x, states1

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)
