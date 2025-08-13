#!/usr/bin/env python3
"""
Multi-modal neural translation model with hierarchical Transformer encoders and cross-attention
Handles missing modalities and leverages multi-layer latent representations with BidirectionalAdaptiveCompressor
Uses geometric mean strategy for optimal sequence alignment
Includes decoders for reconstructing EEG (CBraMod latent format) and fMRI (BrainLM latent format)
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

# -------------------------------
# Simplified Bidirectional Adaptive Compressor
# -------------------------------
class BidirectionalAdaptiveCompressor(nn.Module):
    def __init__(self):
        super().__init__()

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
        target_length = math.ceil((eeg_len * fmri_len) ** 0.5)
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
            nn.Linear(d_model, d_model), nn.ReLU(), nn.Dropout(0.1), nn.LayerNorm(d_model)
        )
        self.layer_projector = nn.Linear(d_model, n_layers * patch_size)
        self.layer_norm = nn.LayerNorm(n_layers * patch_size)
        self.refinement = nn.Sequential(
            nn.Linear(n_layers * patch_size, n_layers * patch_size * 2),
            nn.ReLU(), nn.Dropout(0.1),
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
            nn.Linear(d_model, d_model * 2), nn.ReLU(), nn.Dropout(0.1),
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
            nn.ReLU(), nn.Dropout(0.1),
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
# Main experiment
# -------------------------------
def main():
    print("=" * 60)
    print("🧠 Multi-Modal Neural Translation Model with Geometric Mean Compression")
    print("=" * 60)

    # Only TWO test cases (Test 3 removed)
    eeg_configs = [
        {"channels": 34, "patch_num": 4},   # 136 tokens
        {"channels": 64, "patch_num": 8},   # 512 tokens
    ]
    fmri_configs = [
        {"timepoints_per_voxel": 200},
        {"timepoints_per_voxel": 400},
    ]

    compressor = BidirectionalAdaptiveCompressor()

    for i, (eeg_config, fmri_config) in enumerate(zip(eeg_configs, fmri_configs)):
        print(f"\n{'='*60}")
        print(f"Test Case {i+1}: EEG(ch={eeg_config['channels']}, patches={eeg_config['patch_num']}), "
              f"fMRI(voxels=424, t/voxel={fmri_config['timepoints_per_voxel']})")
        print(f"{'='*60}")

        modality_mask = torch.tensor([[1, 1]], dtype=torch.bool)

        # ---- Load CBraMod and adapt EEG
        print("\n🔄 Loading CBraMod...")
        cbramod_result = load_cbramod(**eeg_config)
        if cbramod_result is None:
            print("❌ CBraMod loading failed, skipping test case...")
            continue
        _, _, combined_latent_eeg = cbramod_result

        print("\n🔧 Applying EEG adapter...")
        adapter_eeg = ConvEEGInputAdapter(
            seq_len=eeg_config["patch_num"],
            n_layers=12,
            channels=eeg_config["channels"],
            input_dim=200,
            output_dim=256
        )
        adapted_eeg = adapter_eeg(combined_latent_eeg)  # (B, channels*patch_num, 256)
        eeg_lower = adapted_eeg
        eeg_higher = adapted_eeg

        # ---- Load BrainLM and adapt fMRI
        print("\n🔄 Loading BrainLM...")
        brainlm_result = load_brainlm(**fmri_config)
        if brainlm_result is None:
            print("❌ BrainLM loading failed, skipping test case...")
            continue
        _, _, _, _, combined_latent_fmri = brainlm_result

        print("\n🔧 Applying fMRI Conv1d adapter...")
        adapter_fmri = fMRIInputAdapterConv1d(
            seq_len=424 * fmri_config["timepoints_per_voxel"],
            n_layers=combined_latent_fmri.shape[0],
            input_dim=combined_latent_fmri.shape[-1],
            output_dim=256,
            target_seq_len=512
        )
        adapted_fmri = adapter_fmri(combined_latent_fmri)  # (B, 512, 256)
        fmri_lower = adapted_fmri
        fmri_higher = adapted_fmri

        if not modality_mask[0, 0]:
            eeg_lower = torch.zeros_like(eeg_lower)
            eeg_higher = torch.zeros_like(eeg_higher)
        if not modality_mask[0, 1]:
            fmri_lower = torch.zeros_like(fmri_lower)
            fmri_higher = torch.zeros_like(fmri_higher)

        # ---- Transformer params
        d_model = 256
        n_heads = 8
        d_ff = 1024
        dropout = 0.1
        n_layers_per_stack = 2

        eeg_encoder = HierarchicalEncoder(d_model, n_heads, d_ff, dropout, n_layers_per_stack)
        fmri_encoder = HierarchicalEncoder(d_model, n_heads, d_ff, dropout, n_layers_per_stack)
        cross_attn = CrossAttentionLayer(d_model, n_heads, dropout)

        eeg_lower_enc, eeg_higher_enc = eeg_encoder(eeg_lower, eeg_higher)
        fmri_lower_enc, fmri_higher_enc = fmri_encoder(fmri_lower, fmri_higher)

        print(f"Before compression - EEG Higher: {tuple(eeg_higher_enc.shape)}, fMRI Higher: {tuple(fmri_higher_enc.shape)}")

        print("\n🔧 Applying Geometric Mean Compression...")
        eeg_compressed, fmri_compressed, target_length = compressor(eeg_higher_enc, fmri_higher_enc)
        eeg_lower_compressed, fmri_lower_compressed, _ = compressor(eeg_lower_enc, fmri_lower_enc)

        print(f"After compression - EEG: {tuple(eeg_compressed.shape)}, fMRI: {tuple(fmri_compressed.shape)}")
        print(f"📏 Geometric mean target length: {target_length}")

        fused_output = cross_attn(eeg_compressed, fmri_compressed)  # (B, target_len, d_model)

        print("\n" + "="*60)
        print("📊 TRANSLATOR RESULTS SUMMARY")
        print("="*60)
        print(f"EEG Lower Input:        {tuple(eeg_lower.shape)}")
        print(f"EEG Lower Compressed:   {tuple(eeg_lower_compressed.shape)}")
        print(f"EEG Higher Input:       {tuple(eeg_higher.shape)}")
        print(f"EEG Higher Compressed:  {tuple(eeg_compressed.shape)}")
        print(f"fMRI Lower Input:       {tuple(fmri_lower.shape)}")
        print(f"fMRI Lower Compressed:  {tuple(fmri_lower_compressed.shape)}")
        print(f"fMRI Higher Input:      {tuple(fmri_higher.shape)}")
        print(f"fMRI Higher Compressed: {tuple(fmri_compressed.shape)}")
        print(f"Fused Output:           {tuple(fused_output.shape)}")

        eeg_ratio = eeg_higher_enc.shape[1] / eeg_compressed.shape[1]
        fmr_ratio = fmri_higher_enc.shape[1] / fmri_compressed.shape[1]
        print(f"\n📈 COMPRESSION ANALYSIS (Geometric Mean Strategy)")
        print(f"Target length (√{eeg_higher_enc.shape[1]}×{fmri_higher_enc.shape[1]}): {target_length}")
        print(f"EEG ratio: {eeg_ratio:.2f}:1 {'(compressed' if eeg_ratio>1 else '(upsampled' if eeg_ratio<1 else '(unchanged)'} "
              f" {eeg_higher_enc.shape[1]} → {eeg_compressed.shape[1]})")
        print(f"fMRI ratio: {fmr_ratio:.2f}:1 {'(compressed' if fmr_ratio>1 else '(upsampled' if fmr_ratio<1 else '(unchanged)'} "
              f" {fmri_higher_enc.shape[1]} → {fmri_compressed.shape[1]})")

        # ---- Decode
        print("\n🔁 Decoding fused representation back to modality latents...")

        eeg_decoder = EEGDecodingAdapter(
            channels=eeg_config["channels"],
            patch_num=eeg_config["patch_num"],
            n_layers=12,
            patch_size=200,
            d_model=d_model
        )
        fmri_decoder = fMRIDecodingAdapter(
            num_voxels=424,
            timepoints_per_voxel=fmri_config["timepoints_per_voxel"],
            n_layers=5,
            hidden_size=256,
            d_model=d_model,
            max_target_tokens=100_000,
            downsample_to_cap=True
        )

        reconstructed_eeg = eeg_decoder(fused_output)
        reconstructed_fmri, eff_len, ds_factor = fmri_decoder(fused_output)

        expected_eeg_shape = (12, fused_output.shape[0], eeg_config["patch_num"], eeg_config["channels"], 200)
        print("\n" + "="*60)
        print("🧩 DECODE RESULTS")
        print("="*60)
        print(f"EEG reconstructed:   {tuple(reconstructed_eeg.shape)}")
        print(f"EEG expected:        {expected_eeg_shape}")
        print(f"EEG shape OK?        {'✅' if reconstructed_eeg.shape == expected_eeg_shape else '❌'}")

        nominal_len = 424 * fmri_config["timepoints_per_voxel"]
        print(f"\nfMRI reconstructed:  {tuple(reconstructed_fmri.shape)}")
        print(f"[fMRI decoder] nominal_len={nominal_len}, effective_len={eff_len}, downsample_factor={ds_factor}")

        print("\n✅ Structural end-to-end pass complete for this test case.")

if __name__ == "__main__":
    main()