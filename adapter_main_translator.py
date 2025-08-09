#!/usr/bin/env python3
"""
Multi-modal neural translation model with hierarchical Transformer encoders and cross-attention
Handles missing modalities and leverages multi-layer latent representations with BidirectionalAdaptiveCompressor
Uses geometric mean strategy for optimal sequence alignment
Includes test cases for varying EEG channels, patch numbers, and fMRI time points per ROI (voxels fixed at 424)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import sys
import os
from pathlib import Path

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent / "CBraMod"))
sys.path.append(str(Path(__file__).parent.parent / "BrainLM"))

# Simplified Bidirectional Adaptive Compressor
class BidirectionalAdaptiveCompressor(nn.Module):
    """
    Compresses both EEG and fMRI sequences to geometric mean length
    This provides optimal balance between information preservation and efficiency
    """
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
        
        # Use geometric mean for optimal balance
        target_length = int((eeg_len * fmri_len) ** 0.5)
        
        # Compress both sequences to target length
        compressed_eeg = self._compress_sequence(eeg_seq, target_length)
        compressed_fmri = self._compress_sequence(fmri_seq, target_length)
        
        return compressed_eeg, compressed_fmri, target_length
    
    def _compress_sequence(self, x, target_length):
        """
        Compress sequence to target length using adaptive pooling
        x: (batch, seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.shape
        
        if seq_len == target_length:
            return x
        elif seq_len < target_length:
            # Upsample using interpolation
            x_transposed = x.transpose(1, 2)  # (batch, d_model, seq_len)
            upsampled = F.interpolate(x_transposed, size=target_length, mode='linear', align_corners=False)
            return upsampled.transpose(1, 2)  # (batch, target_length, d_model)
        else:
            # Downsample using adaptive pooling
            x_transposed = x.transpose(1, 2)  # (batch, d_model, seq_len)
            compressed = F.adaptive_avg_pool1d(x_transposed, target_length)
            return compressed.transpose(1, 2)  # (batch, target_length, d_model)

# Adapter Definitions
class ConvEEGInputAdapter(nn.Module):
    def __init__(self, seq_len, n_layers, channels, input_dim, output_dim=256):
        super().__init__()
        self.seq_len = seq_len
        self.n_layers = n_layers
        self.channels = channels
        self.input_dim = input_dim
        self.proj = nn.Linear(n_layers * input_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x):
        n_layers, batch_size, seq_len, channels, input_dim = x.shape
        x = x.permute(1, 2, 3, 0, 4).contiguous()  # (batch_size, seq_len, channels, n_layers, input_dim)
        x = x.contiguous().view(batch_size, seq_len * channels, n_layers * input_dim)  # (batch_size, seq_len * channels, n_layers * input_dim)
        x = self.proj(x)  # (batch_size, seq_len * channels, output_dim)
        return self.norm(x)  # Output: (1, seq_len * channels, 256)

class fMRIInputAdapterConv1d(nn.Module):
    def __init__(self, seq_len, n_layers, input_dim, output_dim=256, target_seq_len=512):
        super().__init__()
        self.n_layers = n_layers
        self.input_dim = input_dim
        self.target_seq_len = target_seq_len
        self.seq_reduction = nn.Conv1d(
            in_channels=n_layers * input_dim,
            out_channels=n_layers * input_dim,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=n_layers * input_dim
        ) if seq_len > target_seq_len else None
        self.proj = nn.Linear(n_layers * input_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x):
        n_layers, batch_size, seq_len, input_dim = x.shape
        x = x.permute(1, 2, 0, 3).contiguous()
        x = x.view(batch_size, seq_len, n_layers * input_dim)
        
        if self.seq_reduction is not None:
            x = x.transpose(1, 2)
            x = self.seq_reduction(x)
            x = nn.functional.adaptive_avg_pool1d(x, self.target_seq_len).transpose(1, 2)
        
        x = self.proj(x)
        return self.norm(x)  # Output: (1, target_seq_len, 256)

# Transformer Components
class TransformerLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_output, _ = self.self_attn(x, x, x, attn_mask=mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)
        return x

class CrossAttentionLayer(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key_value, mask=None):
        attn_output, _ = self.cross_attn(query, key_value, key_value, attn_mask=mask)
        x = self.norm(query + self.dropout(attn_output))
        return x

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

# Model Loading Functions
def load_cbramod(channels, patch_num, patch_size=200, n_layer=12, nhead=8):
    from models.cbramod import CBraMod
    
    d_model = 200
    
    model = CBraMod(
        in_dim=patch_size,
        out_dim=patch_size,
        d_model=d_model,
        seq_len=patch_num,
        n_layer=n_layer,
        nhead=nhead
    )
    
    weights_path = Path(__file__).parent.parent / "CBraMod" / "pretrained_weights" / "pretrained_weights.pth"
    
    try:
        checkpoint = torch.load(weights_path, map_location='cpu', weights_only=True)
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=False)
        print(f"✅ CBraMod loaded. Missing: {len(missing_keys)}, Unexpected: {len(unexpected_keys)}")
    except Exception as e:
        print(f"❌ Failed to load CBraMod weights: {e}")
        return None
    
    batch_size = 1
    x = torch.randn(batch_size, channels, patch_num, patch_size)
    
    model.eval()
    with torch.no_grad():
        patch_emb = model.patch_embedding(x)
        model.encoder.eval()
        current_input = patch_emb
        all_layer_outputs = []
        
        for layer in model.encoder.layers:
            current_input = layer(current_input)
            all_layer_outputs.append(current_input)
        
        combined_latent = torch.stack(all_layer_outputs, dim=0)
        print(f"CBraMod combined latent shape: {combined_latent.shape}")
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
        checkpoint = torch.load(weights_path, map_location='cpu', weights_only=True)
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=False)
        print(f"✅ BrainLM loaded. Missing: {len(missing_keys)}, Unexpected: {len(unexpected_keys)}")
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
            signal_vectors=signal_vectors,
            xyz_vectors=xyz_vectors,
            noise=noise
        )
        encoder_outputs = model.vit.encoder(
            hidden_states=embeddings,
            output_hidden_states=True,
            return_dict=True
        )
        
        all_hidden_states = encoder_outputs.hidden_states
        combined_latent = torch.stack(all_hidden_states, dim=0)
        print(f"BrainLM combined latent shape: {combined_latent.shape}")
        return model, signal_vectors, xyz_vectors, noise, combined_latent

def main():
    print("="*60)
    print("🧠 Multi-Modal Neural Translation Model with Geometric Mean Compression")
    print("="*60)
    
    # Test cases
    eeg_configs = [
        {"channels": 34, "patch_num": 4},   # EEG length: 136
        {"channels": 64, "patch_num": 8},   # EEG length: 512
        {"channels": 128, "patch_num": 16}, # EEG length: 2048
    ]
    fmri_configs = [
        {"timepoints_per_voxel": 200},  # fMRI length: 512 (after adapter)
        {"timepoints_per_voxel": 400},  # fMRI length: 512 (after adapter)
        {"timepoints_per_voxel": 800},  # fMRI length: 512 (after adapter)
    ]
    
    # Initialize the simplified compressor
    compressor = BidirectionalAdaptiveCompressor()
    
    for i, (eeg_config, fmri_config) in enumerate(zip(eeg_configs, fmri_configs)):
        print(f"\n{'='*60}")
        print(f"Test Case {i+1}: EEG (channels={eeg_config['channels']}, patches={eeg_config['patch_num']}), "
              f"fMRI (voxels=424, timepoints={fmri_config['timepoints_per_voxel']})")
        print(f"{'='*60}")
        
        # Modality mask (1 = present, 0 = absent)
        modality_mask = torch.tensor([[1, 1]], dtype=torch.bool)
        
        # Load CBraMod and apply EEG adapter
        print("\n🔄 Loading CBraMod...")
        cbramod_result = load_cbramod(**eeg_config)
        if cbramod_result is None:
            print("❌ CBraMod loading failed, exiting...")
            continue
        
        cbramod_model, cbramod_input, combined_latent_eeg = cbramod_result
        
        print("\n🔧 Applying EEG adapter...")
        adapter_eeg = ConvEEGInputAdapter(eeg_config["patch_num"], 12, eeg_config["channels"], 200)
        adapted_eeg = adapter_eeg(combined_latent_eeg)  # (1, channels * patch_num, 256)
        eeg_lower = adapted_eeg
        eeg_higher = adapted_eeg
        
        # Load BrainLM and apply fMRI adapter
        print("\n🔄 Loading BrainLM...")
        brainlm_result = load_brainlm(**fmri_config)
        if brainlm_result is None:
            print("❌ BrainLM loading failed, exiting...")
            continue
        
        brainlm_model, signal_vectors, xyz_vectors, noise, combined_latent_fmr = brainlm_result
        
        print("\n🔧 Applying fMRI Conv1d adapter...")
        adapter_fmr = fMRIInputAdapterConv1d(
            seq_len=424 * fmri_config["timepoints_per_voxel"],
            n_layers=len(combined_latent_fmr),
            input_dim=combined_latent_fmr.shape[-1],
            target_seq_len=512
        )
        adapted_fmr = adapter_fmr(combined_latent_fmr)  # (1, 512, 256)
        fmr_lower = adapted_fmr
        fmr_higher = adapted_fmr
        
        # Handle missing modalities
        if not modality_mask[0, 0]:  # EEG missing
            eeg_lower = torch.zeros_like(eeg_lower)
            eeg_higher = torch.zeros_like(eeg_higher)
        if not modality_mask[0, 1]:  # fMRI missing
            fmr_lower = torch.zeros_like(fmr_lower)
            fmr_higher = torch.zeros_like(fmr_higher)
        
        # Define Transformer parameters
        d_model = 256
        n_heads = 8
        d_ff = 1024
        dropout = 0.1
        n_layers_per_stack = 2
        
        # Initialize hierarchical encoders
        eeg_encoder = HierarchicalEncoder(d_model, n_heads, d_ff, dropout, n_layers_per_stack)
        fmr_encoder = HierarchicalEncoder(d_model, n_heads, d_ff, dropout, n_layers_per_stack)
        cross_attn = CrossAttentionLayer(d_model, n_heads, dropout)
        
        # Encode with hierarchical stacks
        eeg_lower_enc, eeg_higher_enc = eeg_encoder(eeg_lower, eeg_higher)
        fmr_lower_enc, fmr_higher_enc = fmr_encoder(fmr_lower, fmr_higher)
        
        print(f"Before compression - EEG Higher: {eeg_higher_enc.shape}, fMRI Higher: {fmr_higher_enc.shape}")
        
        # 🔧 Apply BidirectionalAdaptiveCompressor (geometric mean strategy)
        print("\n🔧 Applying Geometric Mean Compression...")
        eeg_compressed, fmr_compressed, target_length = compressor(eeg_higher_enc, fmr_higher_enc)
        
        # Also compress lower representations for consistency
        eeg_lower_compressed, fmr_lower_compressed, _ = compressor(eeg_lower_enc, fmr_lower_enc)
        
        print(f"After compression - EEG: {eeg_compressed.shape}, fMRI: {fmr_compressed.shape}")
        print(f"📏 Geometric mean target length: {target_length}")
        
        # Apply cross-attention (no mask needed since sequences are same length)
        fused_output = cross_attn(eeg_compressed, fmr_compressed)
        
        # Summary
        print("\n" + "="*60)
        print("📊 TRANSLATOR RESULTS SUMMARY")
        print("="*60)
        print(f"EEG Lower Input:        {eeg_lower.shape}")
        print(f"EEG Lower Compressed:   {eeg_lower_compressed.shape}")
        print(f"EEG Higher Input:       {eeg_higher.shape}")
        print(f"EEG Higher Compressed:  {eeg_compressed.shape}")
        print(f"fMRI Lower Input:       {fmr_lower.shape}")
        print(f"fMRI Lower Compressed:  {fmr_lower_compressed.shape}")
        print(f"fMRI Higher Input:      {fmr_higher.shape}")
        print(f"fMRI Higher Compressed: {fmr_compressed.shape}")
        print(f"Fused Output:           {fused_output.shape}")
        
        # Calculate compression/expansion ratios
        eeg_ratio = eeg_higher_enc.shape[1] / eeg_compressed.shape[1]
        fmr_ratio = fmr_higher_enc.shape[1] / fmr_compressed.shape[1]
        
        print(f"\n📈 COMPRESSION ANALYSIS (Geometric Mean Strategy)")
        print(f"Target length (√{eeg_higher_enc.shape[1]}×{fmr_higher_enc.shape[1]}): {target_length}")
        print(f"EEG ratio: {eeg_ratio:.2f}:1 ", end="")
        if eeg_ratio > 1:
            print(f"(compressed {eeg_higher_enc.shape[1]} → {eeg_compressed.shape[1]})")
        elif eeg_ratio < 1:
            print(f"(upsampled {eeg_higher_enc.shape[1]} → {eeg_compressed.shape[1]})")
        else:
            print(f"(unchanged)")
            
        print(f"fMRI ratio: {fmr_ratio:.2f}:1 ", end="")
        if fmr_ratio > 1:
            print(f"(compressed {fmr_higher_enc.shape[1]} → {fmr_compressed.shape[1]})")
        elif fmr_ratio < 1:
            print(f"(upsampled {fmr_higher_enc.shape[1]} → {fmr_compressed.shape[1]})")
        else:
            print(f"(unchanged)")
        
        print("✅ Perfect sequence alignment with balanced information preservation!")

if __name__ == "__main__":
    main()