#!/usr/bin/env python3
# module.py

"""
Multi-modal neural translation model with hierarchical Transformer encoders and cross-attention
Handles missing modalities and leverages multi-layer latent representations with BidirectionalAdaptiveCompressor
Uses geometric mean strategy for optimal sequence alignment
Includes decoders for reconstructing EEG (CBraMod latent format) and fMRI (BrainLM latent format)

Stage-2 updates:
- Default dropout lowered to 0.05
- Compressor supports target_scale/min_target_len (Translator uses 1.25/128)
- Lite decoder ranks: EEG=48, fMRI=32
- Translator.forward can return pre-tanh fMRI for variance-floor loss
"""

import math
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------------
# Path setup for local modules
# -------------------------------
sys.path.append(str(Path(__file__).parent.parent / "CBraMod"))
sys.path.append(str(Path(__file__).parent.parent / "BrainLM"))

class EEGDecodingAdapterLite(nn.Module):
    """
    Very small EEG decoder.
    Input:  (B, Tfused, D)
    Output: (B, C, P, S)  with P=patch_num (grouped seconds), S=patch_size (e.g., 200)
    """
    def __init__(self, channels, patch_num, patch_size=200, d_model=128, rank=48):
        super().__init__()
        self.channels = int(channels)
        self.patch_num = int(patch_num)
        self.patch_size = int(patch_size)
        self.d_model = int(d_model)
        self.rank = int(rank)
        self.target_tokens = self.channels * self.patch_num

        self.seq_adjust = nn.Sequential(
            nn.Linear(self.d_model, self.d_model, bias=True),
            nn.GELU(),
            nn.LayerNorm(self.d_model),
        )
        # low-rank factorization D -> r -> S
        self.to_rank   = nn.Linear(self.d_model, self.rank, bias=False)
        self.from_rank = nn.Linear(self.rank, self.patch_size, bias=True)

    def forward(self, fused: torch.Tensor) -> torch.Tensor:
        B, T, D = fused.shape
        x = self.seq_adjust(fused)

        # resample sequence len to channels*patch_num
        xt = x.transpose(1, 2)                       # (B, D, T)
        if T != self.target_tokens:
            if T < self.target_tokens:
                xt = F.interpolate(xt, size=self.target_tokens, mode="linear", align_corners=False)
            else:
                xt = F.adaptive_avg_pool1d(xt, self.target_tokens)
        x = xt.transpose(1, 2)                        # (B, target_tokens, D)

        # low-rank projection per token -> patch_size
        z = self.to_rank(x)                           # (B, target_tokens, r)
        z = self.from_rank(z)                         # (B, target_tokens, S)

        # reshape tokens back to (C, P)
        z = z.view(B, self.channels, self.patch_num, self.patch_size)  # (B,C,P,S)
        return z


class fMRIDecodingAdapterLite(nn.Module):
    """
    Very small fMRI decoder that outputs scalar tokens directly.
    Input:  (B, Tfused, D)
    Output: (B, T*V) tokens (scalar), caller reshapes to (B,T,V)
    """
    def __init__(self, target_T: int, target_V: int, d_model: int = 128, rank: int = 32):
        super().__init__()
        self.target_tokens = int(target_T) * int(target_V)
        self.d_model = int(d_model)
        self.rank = int(rank)

        self.seq_adjust = nn.Sequential(
            nn.Linear(self.d_model, self.d_model, bias=True),
            nn.GELU(),
            nn.LayerNorm(self.d_model),
        )
        # low-rank projection D -> r -> 1
        self.to_rank   = nn.Linear(self.d_model, self.rank, bias=False)
        self.from_rank = nn.Linear(self.rank, 1, bias=True)

    def forward(self, fused: torch.Tensor) -> torch.Tensor:
        B, T, D = fused.shape
        x = self.seq_adjust(fused)                    # (B,T,D)

        # produce scalar token per fused step
        z = self.from_rank(self.to_rank(x))           # (B,T,1)
        z = z.transpose(1, 2)                         # (B,1,T)

        # resize to exactly T*V scalar tokens
        if T != self.target_tokens:
            if T < self.target_tokens:
                z = F.interpolate(z, size=self.target_tokens, mode="linear", align_corners=False)
            else:
                z = F.adaptive_avg_pool1d(z, self.target_tokens)

        z = z.squeeze(1)                              # (B, target_tokens)
        return z

# -------------------------------
# Bidirectional Adaptive Compressor (scaled)
# -------------------------------
class BidirectionalAdaptiveCompressor(nn.Module):
    def __init__(self, target_scale: float = 1.0, min_target_len: int = 1):
        super().__init__()
        self.target_scale = float(target_scale)
        self.min_target_len = int(min_target_len)

    def forward(self, eeg_seq, fmri_seq):
        """
        Args:
            eeg_seq: (batch, eeg_len, d_model)
            fmri_seq: (batch, fmri_len, d_model)
        Returns:
            compressed_eeg: (batch, target_len, d_model)
            compressed_fmri: (batch, target_len, d_model)
            target_length: int
        """
        batch_size, eeg_len, d_model = eeg_seq.shape
        _, fmri_len, _ = fmri_seq.shape
        t = math.ceil((eeg_len * fmri_len) ** 0.5 * self.target_scale)
        target_length = max(self.min_target_len, t)
        return (
            self._resize(eeg_seq, target_length),
            self._resize(fmri_seq, target_length),
            target_length,
        )

    def _resize(self, x, target_length):
        B, T, D = x.shape
        if T == target_length:
            return x
        xt = x.transpose(1, 2)
        if T < target_length:
            xt = F.interpolate(xt, size=target_length, mode="linear", align_corners=False)
        else:
            xt = F.adaptive_avg_pool1d(xt, target_length)
        return xt.transpose(1, 2)

# -------------------------------
# Input Adapters
# -------------------------------
class ConvEEGInputAdapter(nn.Module):
    """
    Accepts either:
    - x: (n_layers, batch, seq_len, channels, input_dim)
    - x: (n_layers, batch, T, D) where T == channels * patch_num and D == input_dim
    Projects (n_layers * input_dim) -> output_dim per token.
    """
    def __init__(self, seq_len, n_layers, channels, input_dim, output_dim=256):
        super().__init__()
        self.seq_len = seq_len
        self.n_layers = n_layers
        self.channels = channels
        self.input_dim = input_dim
        self.proj = nn.Linear(n_layers * input_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x):
        if x.dim() == 4:
            L, B, T, D = x.shape
            assert L == self.n_layers and D == self.input_dim
            assert T == self.seq_len * self.channels
            x = x.view(L, B, self.seq_len, self.channels, self.input_dim)
        L, B, S, C, D = x.shape  # (L,B,seq_len,channels,input_dim)
        x = x.permute(1, 2, 3, 0, 4).contiguous()          # (B, seq_len, channels, L, D)
        x = x.view(B, S * C, L * D)                        # (B, S*C, L*D)
        x = self.proj(x)                                   # (B, S*C, out_dim)
        return self.norm(x)

class fMRIInputAdapterConv1d(nn.Module):
    """
    x: (n_layers, batch, seq_len, input_dim)
    Projects (n_layers * input_dim) -> output_dim, resamples to target_seq_len.
    """
    def __init__(self, seq_len, n_layers, input_dim, output_dim=256, target_seq_len=512):
        super().__init__()
        self.n_layers = n_layers
        self.input_dim = input_dim
        self.target_seq_len = target_seq_len
        self.proj = nn.Linear(n_layers * input_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x):
        L, B, T, D = x.shape
        assert L == self.n_layers and D == self.input_dim
        x = x.permute(1, 2, 0, 3).contiguous()   # (B, T, L, D)
        x = x.view(B, T, L * D)                  # (B, T, L*D)
        xt = x.transpose(1, 2)                   # (B, L*D, T)
        if T != self.target_seq_len:
            if T < self.target_seq_len:
                xt = F.interpolate(xt, size=self.target_seq_len, mode="linear", align_corners=False)
            else:
                xt = F.adaptive_avg_pool1d(xt, self.target_seq_len)
        x = xt.transpose(1, 2)                   # (B, target, L*D)
        x = self.proj(x)
        return self.norm(x)

# -------------------------------
# Transformer Blocks
# -------------------------------
class TransformerLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout),
        )
        self.n1 = nn.LayerNorm(d_model)
        self.n2 = nn.LayerNorm(d_model)
        self.do = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_out, _ = self.self_attn(x, x, x, attn_mask=mask)
        x = self.n1(x + self.do(attn_out))
        x = self.n2(x + self.ff(x))
        return x

class CrossAttentionLayer(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        self.cross = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.n = nn.LayerNorm(d_model)
        self.do = nn.Dropout(dropout)

    def forward(self, query, key_value, mask=None, key_padding_mask=None):
        attn_out, _ = self.cross(query, key_value, key_value, attn_mask=mask, key_padding_mask=key_padding_mask)
        return self.n(query + self.do(attn_out))

class HierarchicalEncoder(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout, n_layers_per_stack):
        super().__init__()
        self.lower_stack = nn.ModuleList([TransformerLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers_per_stack)])
        self.higher_stack = nn.ModuleList([TransformerLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers_per_stack)])

    def forward(self, x_lower, x_higher, mask=None):
        for layer in self.lower_stack:
            x_lower = layer(x_lower, mask)
        for layer in self.higher_stack:
            x_higher = layer(x_higher, mask)
        return x_lower, x_higher

# -------------------------------
# Model Loading (CBraMod / BrainLM)
# -------------------------------
def _safe_load_state_dict(model, weights_path):
    try:
        checkpoint = torch.load(weights_path, map_location='cpu', weights_only=True)  # PyTorch 2+
    except TypeError:
        checkpoint = torch.load(weights_path, map_location='cpu')                     # PyTorch <2
    missing, unexpected = model.load_state_dict(checkpoint, strict=False)
    print(f"   Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")

def load_cbramod(channels, patch_num, patch_size=200, n_layer=12, nhead=8):
    from models.cbramod import CBraMod
    model = CBraMod(
        in_dim=patch_size, out_dim=patch_size, d_model=patch_size,
        seq_len=patch_num, n_layer=n_layer, nhead=nhead
    )
    weights_path = Path(__file__).parent.parent / "CBraMod" / "pretrained_weights" / "pretrained_weights.pth"
    try:
        _safe_load_state_dict(model, weights_path)
        print("✅ CBraMod loaded")
    except Exception as e:
        print(f"❌ Failed to load CBraMod weights: {e}")
        return None
    batch_size = 1
    x = torch.randn(batch_size, channels, patch_num, patch_size)
    model.eval()
    with torch.no_grad():
        patch_emb = model.patch_embedding(x)
        model.encoder.eval()
        cur = patch_emb
        outs = []
        for layer in model.encoder.layers:
            cur = layer(cur)
            outs.append(cur)
        combined_latent = torch.stack(outs, dim=0)  # (L,B,T,D) or (L,B,seq_len,channels,input_dim) depending on model
        print(f"CBraMod combined latent shape: {tuple(combined_latent.shape)}")
        return model, x, combined_latent

def load_brainlm(num_voxels=424, timepoints_per_voxel=200, n_layer=5):
    from brainlm_mae.modeling_brainlm import BrainLMForPretraining
    from brainlm_mae.configuration_brainlm import BrainLMConfig
    config_path = Path(__file__).parent.parent / "BrainLM" / "pretrained_models" / "2023-06-06-22_15_00-checkpoint-1400" / "config.json"
    with open(config_path, 'r') as f:
        config_data = json.load(f)
    config = BrainLMConfig(**config_data)
    model = BrainLMForPretraining(config)
    weights_path = Path(__file__).parent.parent / "BrainLM" / "pretrained_models" / "2023-06-06-22_15_00-checkpoint-1400" / "pytorch_model.bin"
    try:
        _safe_load_state_dict(model, weights_path)
        print("✅ BrainLM loaded")
    except Exception as e:
        print(f"❌ Failed to load BrainLM weights: {e}")
        return None
    batch_size = 1
    signal_vectors = torch.randn(batch_size, num_voxels, timepoints_per_voxel)
    xyz_vectors = torch.randn(batch_size, num_voxels, 3)
    noise = torch.rand(batch_size, num_voxels * (timepoints_per_voxel // 20))
    model.eval()
    with torch.no_grad():
        embeddings, mask, ids_restore = model.vit.embeddings(
            signal_vectors=signal_vectors, xyz_vectors=xyz_vectors, noise=noise
        )
        encoder_outputs = model.vit.encoder(
            hidden_states=embeddings, output_hidden_states=True, return_dict=True
        )
        all_hidden_states = encoder_outputs.hidden_states
        combined_latent = torch.stack(all_hidden_states, dim=0)  # (L,B,T,D)
        print(f"BrainLM combined latent shape: {tuple(combined_latent.shape)}")
        return model, signal_vectors, xyz_vectors, noise, combined_latent

# -------------------------------
# Decoding Adapters (EEG / fMRI)
# -------------------------------
class EEGDecodingAdapter(nn.Module):
    """
    Input:  (B, fused_seq_len, d_model)
    Output: (n_layers, B, patch_num, channels, patch_size)
    """
    def __init__(self, channels, patch_num, n_layers=12, patch_size=200, d_model=256):
        super().__init__()
        self.channels = channels
        self.patch_num = patch_num
        self.n_layers = n_layers
        self.patch_size = patch_size
        self.d_model = d_model
        self.original_seq_len = channels * patch_num

        self.seq_adjuster = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(), nn.Dropout(0.05), nn.LayerNorm(d_model)
        )
        self.layer_projector = nn.Linear(d_model, n_layers * patch_size)
        self.layer_norm = nn.LayerNorm(n_layers * patch_size)
        self.refinement = nn.Sequential(
            nn.Linear(n_layers * patch_size, n_layers * patch_size * 2),
            nn.ReLU(), nn.Dropout(0.05),
            nn.Linear(n_layers * patch_size * 2, n_layers * patch_size),
            nn.LayerNorm(n_layers * patch_size)
        )
        self.final_activation = nn.Tanh()

    def forward(self, fused_representation):
        B, T, D = fused_representation.shape
        if T != self.original_seq_len:
            xt = fused_representation.transpose(1, 2)
            xt = F.adaptive_avg_pool1d(xt, self.original_seq_len)
            x = xt.transpose(1, 2)
        else:
            x = fused_representation
        x = self.seq_adjuster(x)
        x = self.layer_projector(x)
        x = self.layer_norm(x)
        x = self.refinement(x)
        x = self.final_activation(x)
        x = x.view(B, self.channels, self.patch_num, self.n_layers * self.patch_size)
        x = x.view(B, self.channels, self.patch_num, self.n_layers, self.patch_size)
        x = x.permute(3, 0, 2, 1, 4)  # (L,B,patch_num,channels,patch_size)
        return x

class fMRIDecodingAdapter(nn.Module):
    """
    Fast + memory-safe fMRI decoder.
    - Single-shot linear interpolation
    - Optional cap on target tokens with automatic downsampling
    Returns:
      reconstructed: (n_layers, B, effective_target_len, hidden_size)
      effective_target_len: int
      downsample_factor: int
    """
    def __init__(self, num_voxels=424, timepoints_per_voxel=200, n_layers=5,
                 hidden_size=256, d_model=256,
                 max_target_tokens=100_000, downsample_to_cap=True):
        super().__init__()
        self.num_voxels = num_voxels
        self.timepoints_per_voxel = timepoints_per_voxel
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.d_model = d_model

        self.nominal_target_len = num_voxels * timepoints_per_voxel
        self.max_target_tokens = max_target_tokens
        self.downsample_to_cap = downsample_to_cap

        if self.nominal_target_len > self.max_target_tokens and self.downsample_to_cap:
            factor = math.ceil(self.nominal_target_len / self.max_target_tokens)
            self.effective_target_len = self.nominal_target_len // factor
            self.downsample_factor = factor
        else:
            self.effective_target_len = self.nominal_target_len
            self.downsample_factor = 1

        self.seq_adjuster = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.ReLU(), nn.Dropout(0.05),
            nn.Linear(d_model * 2, d_model), nn.LayerNorm(d_model)
        )
        self.layer_projector = nn.Linear(d_model, n_layers * hidden_size)
        self.layer_norm = nn.LayerNorm(n_layers * hidden_size)
        self.smooth_conv = nn.Conv1d(
            in_channels=n_layers * hidden_size, out_channels=n_layers * hidden_size,
            kernel_size=3, padding=1, groups=n_layers * hidden_size
        )
        self.refinement = nn.Sequential(
            nn.Linear(n_layers * hidden_size, n_layers * hidden_size * 2),
            nn.ReLU(), nn.Dropout(0.05),
            nn.Linear(n_layers * hidden_size * 2, n_layers * hidden_size),
            nn.LayerNorm(n_layers * hidden_size)
        )
        self.final_activation = nn.Identity() 

    def forward(self, fused_representation):
        B, T, D = fused_representation.shape
        x = self.seq_adjuster(fused_representation)     # (B,T,D)
        x = self.layer_projector(x)                     # (B,T,L*H)
        x = self.layer_norm(x)

        tgt = self.effective_target_len
        xt = x.transpose(1, 2)                          # (B,L*H,T)
        if T != tgt:
            xt = F.interpolate(xt, size=tgt, mode="linear", align_corners=False)
            xt = self.smooth_conv(xt)
        x = xt.transpose(1, 2)                          # (B,tgt,L*H)

        x = self.refinement(x)
        x = self.final_activation(x)

        x = x.view(B, tgt, self.n_layers, self.hidden_size).permute(2, 0, 1, 3)
        return x, tgt, self.downsample_factor

# -------------------------------
# Main Translator
# -------------------------------
class TranslatorModel(nn.Module):
    def __init__(
        self,
        eeg_channels: int,
        eeg_patch_num: int,
        eeg_n_layers: int,
        eeg_input_dim: int,
        fmri_n_layers: int,
        fmri_hidden_size: int,
        fmri_tokens_target: int,
        fmri_target_T: int,
        fmri_target_V: int,
        d_model: int = 128,
        n_heads: int = 4,
        d_ff: int = 512,
        dropout: float = 0.05,
        voxel_count: int = 424,
        debug: bool = False,
    ) -> None:
        super().__init__()
        self.debug = debug
        self._voxel_count = int(voxel_count)
        self._fmri_n_layers = int(fmri_n_layers)
        self._fmri_hidden_size = int(fmri_hidden_size)
        self._d_model = int(d_model)

        # Adapters (trainable)
        self.adapter_eeg = ConvEEGInputAdapter(
            seq_len=eeg_patch_num,
            n_layers=eeg_n_layers,
            channels=eeg_channels,
            input_dim=eeg_input_dim,
            output_dim=d_model,
        )
        self.adapter_fmri = fMRIInputAdapterConv1d(
            seq_len=fmri_tokens_target,
            n_layers=fmri_n_layers,
            input_dim=fmri_hidden_size,
            output_dim=d_model,
            target_seq_len=512,
        )

        # Positional encodings (buffers)
        self.pos_eeg = PositionalEncoding(d_model)
        self.pos_fmri = PositionalEncoding(d_model)

        # Encoders & fusion
        self.eeg_encoder = HierarchicalEncoder(d_model, n_heads, d_ff, dropout, n_layers_per_stack=1)
        self.fmri_encoder = HierarchicalEncoder(d_model, n_heads, d_ff, dropout, n_layers_per_stack=1)
        self.cross_attn = CrossAttentionLayer(d_model, n_heads, dropout)
        self.compressor = BidirectionalAdaptiveCompressor(target_scale=1.25, min_target_len=128)

        # Lite decoders (trainable)
        self.eeg_decoder = EEGDecodingAdapterLite(
            channels=eeg_channels,
            patch_num=eeg_patch_num,
            patch_size=eeg_input_dim,
            d_model=d_model,
            rank=32
        )
        self.fmri_decoder = fMRIDecodingAdapterLite(
            target_T=fmri_target_T,
            target_V=fmri_target_V,
            d_model=d_model,
            rank=32,
        )

        # Output affine for fMRI (trainable)
        self.fmri_out_scale = nn.Parameter(torch.tensor(1.0))
        self.fmri_out_bias  = nn.Parameter(torch.tensor(0.0))

    def _dbg(self, msg: str) -> None:
        if self.debug:
            print(msg, flush=True)

    def forward(
        self,
        eeg_latents: torch.Tensor,        # (L,B,S,C,D)
        fmri_latents: torch.Tensor,       # (L,B,T,H)
        fmri_target_T: int,
        fmri_target_V: int,
        return_pretanh: bool = False,
    ):
        self._dbg("\n[Forward] start")
        if fmri_target_V != self._voxel_count:
            raise ValueError(f"TranslatorModel expects V={self._voxel_count}, got V={fmri_target_V}")

        eeg_adapted = self.adapter_eeg(eeg_latents)    # (B, N_eeg_tokens, D)
        fmri_adapted = self.adapter_fmri(fmri_latents) # (B, N_fmri_tokens, D)

        eeg_adapted = self.pos_eeg(eeg_adapted)
        fmri_adapted = self.pos_fmri(fmri_adapted)

        _, eeg_higher = self.eeg_encoder(eeg_adapted, eeg_adapted)
        _, fmri_higher = self.fmri_encoder(fmri_adapted, fmri_adapted)

        eeg_c, fmri_c, _ = self.compressor(eeg_higher, fmri_higher)
        fused = self.cross_attn(eeg_c, fmri_c)         # (B, Tfused, D)

        eeg_signal = self.eeg_decoder(fused)                    # (B,C,Pe,S)
        fmri_flat  = self.fmri_decoder(fused)                   # (B, T*V)
        pre_tanh   = fmri_flat.view(-1, int(fmri_target_T), int(fmri_target_V))  # (B,T,V)
        fmri_signal = torch.tanh(pre_tanh)
        fmri_signal = self.fmri_out_scale * fmri_signal + self.fmri_out_bias
        self._dbg("[Forward] end\n")
        if return_pretanh:
            return eeg_signal, fmri_signal, pre_tanh
        return eeg_signal, fmri_signal

# -------------------------------
# Positional Encoding (sin/cos)
# -------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 100_000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[: x.size(1), :].unsqueeze(0)
