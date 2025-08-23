#!/usr/bin/env python3
"""
Multi-modal components (fMRI-focused) with axial RoPE and true S = T×V flow.

Key points:
- No fixed 512 anywhere. The sequence length is the true S = T×V.
- fMRIInputAdapterConv1d: (L,B,Ttok,Dh) -> (B,S,D) with optional retarget to S.
- AxialRotaryMHA: applies Rotary Positional Embeddings along time and voxel axes.
- TransformerLayer / HierarchicalEncoder require (T, V) to compute axial RoPE.
- fMRIDecodingAdapter2D: predicts (B,T,V) directly (time-only resize), removing the
  rank-1-over-voxels issue in the old Lite decoder.

Kept:
- EEGDecodingAdapterLite (unchanged, included for completeness).
- fMRIDecodingAdapterLite retained for backward-compat but DO NOT use for fMRI;
  it’s rank-1 across voxels by design. Use fMRIDecodingAdapter2D instead.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------------
# EEG (unchanged)
# -------------------------------
class EEGDecodingAdapterLite(nn.Module):
    """
    Very small EEG decoder.
    Input:  (B, Tfused, D)
    Output: (B, C, P, S)  with P=patch_num (grouped seconds), S=patch_size (e.g., 200)
    """
    def __init__(self, channels, patch_num, patch_size=200, d_model=128, rank=32):
        super().__init__()
        self.channels   = int(channels)
        self.patch_num  = int(patch_num)
        self.patch_size = int(patch_size)
        self.d_model    = int(d_model)
        self.rank       = int(rank)
        self.target_tokens = self.channels * self.patch_num

        self.seq_adjust = nn.Sequential(
            nn.Linear(self.d_model, self.d_model, bias=True),
            nn.GELU(),
            nn.LayerNorm(self.d_model),
        )
        self.to_rank   = nn.Linear(self.d_model, self.rank, bias=False)
        self.from_rank = nn.Linear(self.rank, self.patch_size, bias=True)

    def forward(self, fused: torch.Tensor) -> torch.Tensor:
        B, T, D = fused.shape
        x = self.seq_adjust(fused)
        xt = x.transpose(1, 2)  # (B,D,T)
        if T != self.target_tokens:
            if T < self.target_tokens:
                xt = F.interpolate(xt, size=self.target_tokens, mode="linear", align_corners=False)
            else:
                xt = F.adaptive_avg_pool1d(xt, self.target_tokens)
        x = xt.transpose(1, 2)  # (B,target_tokens,D)
        z = self.from_rank(self.to_rank(x))  # (B,target_tokens,S)
        z = z.view(B, self.channels, self.patch_num, self.patch_size)
        return z

# -------------------------------
# (Deprecated) fMRI Lite decoder
# -------------------------------
class fMRIDecodingAdapterLite(nn.Module):
    """
    DEPRECATED for fMRI: rank-1 across voxels (kept for backwards-compat).
    """
    def __init__(self, target_T: int, target_V: int, d_model: int = 128, rank: int = 16):
        super().__init__()
        self.target_tokens = int(target_T) * int(target_V)
        self.d_model = int(d_model)
        self.rank = int(rank)

        self.seq_adjust = nn.Sequential(
            nn.Linear(self.d_model, self.d_model, bias=True),
            nn.GELU(),
            nn.LayerNorm(self.d_model),
        )
        self.to_rank   = nn.Linear(self.d_model, self.rank, bias=False)
        self.from_rank = nn.Linear(self.rank, 1, bias=True)

    def forward(self, fused: torch.Tensor) -> torch.Tensor:
        B, T, D = fused.shape
        x = self.seq_adjust(fused)
        z = self.from_rank(self.to_rank(x))  # (B,T,1)
        z = z.transpose(1, 2)                # (B,1,T)
        if T != self.target_tokens:
            if T < self.target_tokens:
                z = F.interpolate(z, size=self.target_tokens, mode="linear", align_corners=False)
            else:
                z = F.adaptive_avg_pool1d(z, self.target_tokens)
        z = z.squeeze(1)                     # (B, target_tokens)
        return z

# -------------------------------
# fMRI 2-D decoder (FIXED)
# -------------------------------
class fMRIDecodingAdapter2D(nn.Module):
    """
    Predicts V voxel channels first, then resizes along time only.
    Input:  (B, Tfused, D)
    Output: (B, T, V)
    """
    def __init__(self, target_T: int, target_V: int, d_model: int = 128, rank: int = 32):
        super().__init__()
        self.target_T = int(target_T)
        self.target_V = int(target_V)
        self.d_model  = int(d_model)
        self.rank     = int(rank)

        self.seq_adjust = nn.Sequential(
            nn.Linear(self.d_model, self.d_model, bias=True),
            nn.GELU(),
            nn.LayerNorm(self.d_model),
        )
        # low-rank factorization D -> rank -> V
        self.to_rank   = nn.Linear(self.d_model, self.rank, bias=False)
        self.to_voxels = nn.Linear(self.rank, self.target_V, bias=True)

    def forward(self, fused: torch.Tensor) -> torch.Tensor:
        """
        fused: (B, Tf, D) with Tf ~ S or any intermediate fused length
        returns: (B, T, V) where T=self.target_T, V=self.target_V
        """
        B, Tf, D = fused.shape
        x = self.seq_adjust(fused)           # (B,Tf,D)
        z = self.to_voxels(self.to_rank(x))  # (B,Tf,V)
        z = z.transpose(1, 2)                # (B,V,Tf)
        if Tf != self.target_T:
            # time-only resize
            if Tf < self.target_T:
                z = F.interpolate(z, size=self.target_T, mode="linear", align_corners=False)
            else:
                z = F.adaptive_avg_pool1d(z, self.target_T)
        z = z.transpose(1, 2)                # (B,T,V)
        return z

# -------------------------------
# Input adapter (no fixed 512)
# -------------------------------
class fMRIInputAdapterConv1d(nn.Module):
    """
    x: (n_layers, batch, seq_len_tokens, input_dim)
    Projects (n_layers * input_dim) -> output_dim, optional retarget to target_seq_len (= S=T×V).
    """
    def __init__(self, seq_len: int, n_layers: int, input_dim: int,
                 output_dim: int = 256, target_seq_len: Optional[int] = None):
        super().__init__()
        self.n_layers = int(n_layers)
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.target_seq_len = int(target_seq_len) if target_seq_len is not None else None
        self.proj = nn.Linear(self.n_layers * self.input_dim, self.output_dim)
        self.norm = nn.LayerNorm(self.output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (L,B,Ttok,Din)
        returns: (B, S', D) where S' = target_seq_len (if set) else Ttok
        """
        L, B, T, Din = x.shape
        assert L == self.n_layers and Din == self.input_dim
        x = x.permute(1, 2, 0, 3).contiguous()         # (B,T,L,D)
        x = x.view(B, T, L * Din)                      # (B,T,L*D)
        xt = x.transpose(1, 2)                         # (B,L*D,T)
        if self.target_seq_len is not None and T != self.target_seq_len:
            if T < self.target_seq_len:
                xt = F.interpolate(xt, size=self.target_seq_len, mode="linear", align_corners=False)
            else:
                xt = F.adaptive_avg_pool1d(xt, self.target_seq_len)
        x = xt.transpose(1, 2)                         # (B,S',L*D)
        x = self.proj(x)                               # (B,S',out)
        return self.norm(x)

# -------------------------------
# Axial Rotary MHA
# -------------------------------
def _build_rope_cos_sin(idx: torch.Tensor, rot_dim: int, base: float = 10000.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    idx: (S,) integer positions
    rot_dim must be even. returns cos,sin: (S, rot_dim)
    """
    if rot_dim % 2 != 0:
        rot_dim -= 1  # ensure even
    half = rot_dim // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, half, device=idx.device, dtype=torch.float32) / half))
    angles = idx.float().unsqueeze(1) * inv_freq.unsqueeze(0)  # (S, half)
    cos = torch.cos(angles).repeat_interleave(2, dim=-1)       # (S, rot_dim)
    sin = torch.sin(angles).repeat_interleave(2, dim=-1)       # (S, rot_dim)
    return cos, sin

def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    x:   (B, H, S, R)   R even
    cos: (S, R)
    sin: (S, R)
    """
    x_even = x[..., ::2]
    x_odd  = x[..., 1::2]
    cos_e  = cos[..., ::2].unsqueeze(0).unsqueeze(0)  # (1,1,S,R/2)
    sin_e  = sin[..., ::2].unsqueeze(0).unsqueeze(0)
    cos_o  = cos[..., 1::2].unsqueeze(0).unsqueeze(0)
    sin_o  = sin[..., 1::2].unsqueeze(0).unsqueeze(0)
    # Interleave apply
    out_even = x_even * cos_e - x_odd * sin_e
    out_odd  = x_even * sin_o + x_odd * cos_o
    out = torch.empty_like(x)
    out[..., ::2] = out_even
    out[..., 1::2] = out_odd
    return out

class AxialRotaryMHA(nn.Module):
    """
    Multi-head self-attention with axial Rotary Positional Embeddings.
    Splits head dim into: [time-rotated | voxel-rotated | (optional) rest]
    """
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0,
                 rope_fraction: float = 1.0, rope_base: float = 10000.0):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim  = embed_dim // num_heads
        self.scale     = self.head_dim ** -0.5
        self.rope_base = float(rope_base)
        # rotate at most rope_fraction of head_dim (rounded to nearest even)
        rot_total = int(self.head_dim * max(0.0, min(1.0, rope_fraction)))
        rot_total = (rot_total // 2) * 2
        self.rot_t = rot_total // 2
        self.rot_v = rot_total - self.rot_t
        self.rest  = self.head_dim - (self.rot_t + self.rot_v)

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.o_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        x = x.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)  # (B,H,S,hd)
        return x

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, H, S, hd = x.shape
        x = x.transpose(1, 2).contiguous().view(B, S, H * hd)  # (B,S,D)
        return x

    def forward(self, x: torch.Tensor, *, T: int, V: int,
                attn_mask: Optional[torch.Tensor] = None,
                key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: (B, S, D) where S = T×V
        """
        B, S, D = x.shape
        assert S == T * V, f"Expected S=T×V, got S={S}, T×V={T*V}"

        q = self._split_heads(self.q_proj(x))  # (B,H,S,hd)
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))

        # Axial indices
        p = torch.arange(S, device=x.device)
        t_idx = torch.div(p, V, rounding_mode='floor')  # (S,)
        v_idx = p % V                                   # (S,)

        # Build cos/sin for each axis on the split dims
        if self.rot_t > 0:
            cos_t, sin_t = _build_rope_cos_sin(t_idx, self.rot_t, base=self.rope_base)
        if self.rot_v > 0:
            cos_v, sin_v = _build_rope_cos_sin(v_idx, self.rot_v, base=self.rope_base)

        # Slice head dim: [t | v | rest]
        if self.rot_t > 0:
            q_t, k_t = q[..., :self.rot_t], k[..., :self.rot_t]
            q[..., :self.rot_t] = _apply_rope(q_t, cos_t, sin_t)
            k[..., :self.rot_t] = _apply_rope(k_t, cos_t, sin_t)
        if self.rot_v > 0:
            t_end = self.rot_t
            v_end = self.rot_t + self.rot_v
            q_v, k_v = q[..., t_end:v_end], k[..., t_end:v_end]
            q[..., t_end:v_end] = _apply_rope(q_v, cos_v, sin_v)
            k[..., t_end:v_end] = _apply_rope(k_v, cos_v, sin_v)
        # rest (if any) is left unrotated

        # Scaled dot-product attention
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B,H,S,S)

        if attn_mask is not None:
            attn = attn + attn_mask  # broadcast as needed
        if key_padding_mask is not None:
            # key_padding_mask: (B,S) -> (B,1,1,S)
            attn = attn.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        y = torch.matmul(attn, v)  # (B,H,S,hd)
        y = self._merge_heads(y)   # (B,S,D)
        y = self.o_proj(y)
        y = self.proj_drop(y)
        return y

# -------------------------------
# Transformer blocks (with axial RoPE)
# -------------------------------
class TransformerLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float, rope_fraction: float = 1.0):
        super().__init__()
        self.self_attn = AxialRotaryMHA(d_model, n_heads, dropout=dropout, rope_fraction=rope_fraction)
        self.n1 = nn.LayerNorm(d_model)
        self.n2 = nn.LayerNorm(d_model)
        self.do = nn.Dropout(dropout)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, *, T: int, V: int,
                attn_mask: Optional[torch.Tensor] = None,
                key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_out = self.self_attn(x, T=T, V=V, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        x = self.n1(x + self.do(attn_out))
        x = self.n2(x + self.ff(x))
        return x

class HierarchicalEncoder(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float, n_layers_per_stack: int, rope_fraction: float = 1.0):
        super().__init__()
        self.lower_stack = nn.ModuleList([
            TransformerLayer(d_model, n_heads, d_ff, dropout, rope_fraction=rope_fraction)
            for _ in range(n_layers_per_stack)
        ])
        self.higher_stack = nn.ModuleList([
            TransformerLayer(d_model, n_heads, d_ff, dropout, rope_fraction=rope_fraction)
            for _ in range(n_layers_per_stack)
        ])

    def forward(self, x_lower: torch.Tensor, x_higher: torch.Tensor, *, T: int, V: int) -> Tuple[torch.Tensor, torch.Tensor]:
        for layer in self.lower_stack:
            x_lower = layer(x_lower, T=T, V=V)
        for layer in self.higher_stack:
            x_higher = layer(x_higher, T=T, V=V)
        return x_lower, x_higher
