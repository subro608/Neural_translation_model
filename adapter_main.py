#!/usr/bin/env python3
"""
Simple test script for CBraMod and BrainLM models with corrected Convolutional EEG Adapter
Ensures 256 hidden dimension for Transformer usage
"""

import torch
import torch.nn as nn
import json
import sys
import os
from pathlib import Path

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent / "CBraMod"))
sys.path.append(str(Path(__file__).parent.parent / "BrainLM"))

# Adapter Definitions
class ConvEEGInputAdapter(torch.nn.Module):
    def __init__(self, seq_len, n_layers, channels, input_dim, output_dim=256):
        super().__init__()
        self.seq_len = seq_len
        self.n_layers = n_layers
        self.channels = channels
        self.input_dim = input_dim
        # No channel reduction - preserve all channels as sequence positions
        # Project from (n_layers * input_dim) to output_dim
        self.proj = torch.nn.Linear(n_layers * input_dim, output_dim)
        self.norm = torch.nn.LayerNorm(output_dim)

    def forward(self, x):
        # Input: (n_layers, batch_size, seq_len, channels, input_dim) = (12, 1, 4, 37, 200)
        n_layers, batch_size, seq_len, channels, input_dim = x.shape
        # Permute to (batch_size, seq_len, channels, n_layers, input_dim)
        x = x.permute(1, 2, 3, 0, 4)  # (1, 4, 37, 12, 200)
        # Reshape to (batch_size, seq_len * channels, n_layers * input_dim)
        # ADD .contiguous() before .view()
        x = x.contiguous().view(batch_size, seq_len * channels, n_layers * input_dim)  # (1, 148, 2400)
        # Project to output dimension
        x = self.proj(x)  # (1, 148, 256)
        x = self.norm(x)
        return x  # Output: (1, patch_nos * channel_nos, 256) = (1, 148, 256)

class fMRIInputAdapterConv1d(nn.Module):
    """
    Apply conv1d directly to the flattened sequence to reduce length
    No assumptions about ROI/patch structure needed
    """
    def __init__(self, seq_len, n_layers, input_dim, output_dim=256, target_seq_len=512):
        super().__init__()
        self.seq_len = seq_len
        self.n_layers = n_layers
        self.input_dim = input_dim
        self.target_seq_len = target_seq_len
        
        # Calculate conv1d parameters to reduce sequence length
        if seq_len > target_seq_len:
            # Use stride to reduce sequence length
            stride = max(1, seq_len // target_seq_len)
            kernel_size = min(3, stride + 1)
            
            self.seq_reduction = nn.Conv1d(
                in_channels=n_layers * input_dim,
                out_channels=n_layers * input_dim,
                kernel_size=kernel_size,
                stride=stride,
                padding=kernel_size//2,
                groups=n_layers * input_dim  # Depthwise conv to preserve features
            )
        else:
            self.seq_reduction = None
            
        # Project to output dimension
        self.proj = nn.Linear(n_layers * input_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x):
        # Input: (n_layers, batch_size, seq_len, input_dim)
        n_layers, batch_size, seq_len, input_dim = x.shape
        
        # Permute to (batch_size, seq_len, n_layers * input_dim)
        x = x.permute(1, 2, 0, 3).contiguous()
        x = x.view(batch_size, seq_len, n_layers * input_dim)
        
        # Apply sequence reduction if needed
        if self.seq_reduction is not None:
            # Conv1d expects (batch, channels, sequence)
            x = x.transpose(1, 2)  # (batch_size, n_layers * input_dim, seq_len)
            x = self.seq_reduction(x)  # Reduce sequence length
            x = x.transpose(1, 2)  # Back to (batch_size, reduced_seq_len, n_layers * input_dim)
        
        # Project to output dimension
        x = self.proj(x)
        x = self.norm(x)
        
        return x

# Model Loading Functions
def load_cbramod():
    from models.cbramod import CBraMod
    
    patch_size = 200
    patch_num = 4
    d_model = 200
    n_layer = 12
    nhead = 8
    channels = 37
    
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

def load_brainlm():
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
    num_voxels = 424
    num_timepoints_per_voxel = 200
    
    signal_vectors = torch.randn(batch_size, num_voxels, num_timepoints_per_voxel)
    xyz_vectors = torch.randn(batch_size, num_voxels, 3)
    noise = torch.rand(batch_size, num_voxels * (num_timepoints_per_voxel // 20))
    
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
    print("🧠 Testing Neural Translation Model Adapters")
    print("="*60)
    
    # Load CBraMod and apply EEG adapter
    print("\n🔄 Loading CBraMod...")
    cbramod_result = load_cbramod()
    if cbramod_result is None:
        print("❌ CBraMod loading failed, exiting...")
        return
    
    cbramod_model, cbramod_input, combined_latent_eeg = cbramod_result
    
    print("\n🔧 Applying EEG adapter...")
    adapter_eeg = ConvEEGInputAdapter(4, 12, 37, 200)
    adapted_eeg = adapter_eeg(combined_latent_eeg)
    print(f"✅ Adapted EEG shape: {adapted_eeg.shape}")
    
    # Load BrainLM and apply fMRI adapter
    print("\n🔄 Loading BrainLM...")
    brainlm_result = load_brainlm()
    if brainlm_result is None:
        print("❌ BrainLM loading failed, exiting...")
        return
        
    brainlm_model, signal_vectors, xyz_vectors, noise, combined_latent_fmr = brainlm_result
    
    print("\n🔧 Applying fMRI Conv1d adapter...")
    adapter_fmr = fMRIInputAdapterConv1d(
        seq_len=combined_latent_fmr.shape[2], 
        n_layers=len(combined_latent_fmr), 
        input_dim=combined_latent_fmr.shape[-1],
        target_seq_len=512  # Reduce to manageable size
    )
    adapted_fmr = adapter_fmr(combined_latent_fmr)
    print(f"✅ Adapted fMRI shape: {adapted_fmr.shape}")
    
    # Summary
    print("\n" + "="*60)
    print("📊 ADAPTER RESULTS SUMMARY")
    print("="*60)
    print(f"EEG Input:  {combined_latent_eeg.shape}")
    print(f"EEG Output: {adapted_eeg.shape}")
    print(f"fMRI Input: {combined_latent_fmr.shape}")  
    print(f"fMRI Output: {adapted_fmr.shape}")
    
    # Check sequence length ratio
    eeg_seq_len = adapted_eeg.shape[1]
    fmri_seq_len = adapted_fmr.shape[1]
    ratio = fmri_seq_len / eeg_seq_len
    print(f"\nSequence length ratio (fMRI:EEG): {ratio:.1f}:1")
    
    if ratio < 10:
        print("✅ Good sequence length balance for cross-attention!")
    else:
        print("⚠️  Consider further reducing fMRI sequence length")

if __name__ == "__main__":
    main()