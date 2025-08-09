#!/usr/bin/env python3
"""
Simple test script for CBraMod and BrainLM models
Tests both models with random input data to verify they load and run correctly
"""

import torch
import numpy as np
import sys
import os
from pathlib import Path

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent / "CBraMod"))
sys.path.append(str(Path(__file__).parent.parent / "BrainLM"))

def test_cbramod():
    from models.cbramod import CBraMod
    
    # Model parameters
    patch_size = 200
    patch_num = 4
    d_model = 200
    n_layer = 12
    nhead = 8
    
    # Create model
    model = CBraMod(
        in_dim=patch_size,
        out_dim=patch_size,
        d_model=d_model,
        seq_len=patch_num,
        n_layer=n_layer,
        nhead=nhead
    )
    
    # Load pretrained weights
    weights_path = Path(__file__).parent.parent / "CBraMod" / "pretrained_weights" / "pretrained_weights.pth"
    checkpoint = torch.load(weights_path, map_location='cpu')
    missing_keys, unexpected_keys = model.load_state_dict(checkpoint)
    print(f"LLLLLLLLLLsLoaded pretrained weights from {weights_path}, missing keys: {missing_keys}, unexpected keys: {unexpected_keys}")

    
    # Create random EEG data - CBraMod expects (batch, channels, patch_num, patch_size)
    batch_size = 1
    channels = 37
    patch_num = 30
    patch_size = 200  # Must match in_dim parameter
    
    x = torch.randn(batch_size, channels, patch_num, patch_size)
    print(f"   Input shape: {x.shape}")
    
    # Forward pass to get latent representation after transformer encoder
    model.eval()
    with torch.no_grad():
        # Get patch embeddings first
        patch_emb = model.patch_embedding(x)
        print(f"   Patch embeddings shape: {patch_emb.shape}")
        
        # Get latent representation from transformer encoder
        # For CBraMod, we need to manually extract from each layer
        model.encoder.eval()
        current_input = patch_emb
        all_layer_outputs = []
        
        # Extract from each layer
        for i, layer in enumerate(model.encoder.layers):
            current_input = layer(current_input)
            all_layer_outputs.append(current_input.clone())
            print(f"   Layer {i} output shape: {current_input.shape}")
        
        # Stack all layer outputs
        combined_latent = torch.stack(all_layer_outputs, dim=0)  # Shape: (num_layers, batch, channels, patches, features)
        print(f"   Combined latent shape: {combined_latent.shape}")
        
        # Also get the final layer output for comparison
        final_latent = current_input
        print(f"   Final layer latent shape: {final_latent.shape}")
        
        # Alternative: Concatenate all layers along the feature dimension
        concatenated_latent = torch.cat(all_layer_outputs, dim=-1)  # Shape: (batch, channels, patches, features * num_layers)
        print(f"   Concatenated latent shape: {concatenated_latent.shape}")
        
        # Also get final output for comparison
        output = model(x)
        print(f"   Final output shape: {output.shape}")
    
    print(f"   ✅ CBraMod test passed!")
    
    return True

def test_brainlm():
    from brainlm_mae.modeling_brainlm import BrainLMForPretraining
    from brainlm_mae.configuration_brainlm import BrainLMConfig
    
    # Load config from pretrained model
    config_path = Path(__file__).parent.parent / "BrainLM" / "pretrained_models" / "2023-06-06-22_15_00-checkpoint-1400" / "config.json"
    
    if config_path.exists():
        import json
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        
        config = BrainLMConfig(**config_data)
        print(f"   ✅ Loaded config from {config_path}")
    else:
        print(f"   ⚠️  Config not found, using default")
        config = BrainLMConfig()
    
    # Create model
    model = BrainLMForPretraining(config)
    
    # Load pretrained weights
    weights_path = Path(__file__).parent.parent / "BrainLM" / "pretrained_models" / "2023-06-06-22_15_00-checkpoint-1400" / "pytorch_model.bin"
    checkpoint = torch.load(weights_path, map_location='cpu')
    missing_keys, unexpected_keys = model.load_state_dict(checkpoint)
    print(f"LLLLLLLLLLsLoaded pretrained weights from {weights_path}, missing keys: {missing_keys}, unexpected keys: {unexpected_keys}")

    
    # Create random fMRI data
    batch_size = 1
    num_voxels = 424
    num_timepoints_per_voxel = 200  # Must be divisible by timepoint_patching_size (20)
    
    signal_vectors = torch.randn(batch_size, num_voxels, num_timepoints_per_voxel)
    xyz_vectors = torch.randn(batch_size, num_voxels, 3)  # 3D coordinates
    
    # Calculate expected sequence length for noise
    num_patch_tokens = num_timepoints_per_voxel // 20  # timepoint_patching_size = 20
    seq_length = num_voxels * num_patch_tokens
    noise = torch.rand(batch_size, seq_length)  # Noise for masking
    
    print(f"   Signal vectors shape: {signal_vectors.shape}")
    print(f"   XYZ vectors shape: {xyz_vectors.shape}")
    print(f"   Noise shape: {noise.shape}")
    
    # Forward pass to get latent representation after transformer encoder
    model.eval()
    with torch.no_grad():
        # Get embeddings from the model's embedding layer
        embeddings, mask, ids_restore = model.vit.embeddings(
            signal_vectors=signal_vectors,
            xyz_vectors=xyz_vectors,
            noise=noise
        )
        print(f"   Embeddings shape: {embeddings.shape}")
        
        # Get latent representation from transformer encoder
        encoder_outputs = model.vit.encoder(
            hidden_states=embeddings,
            output_hidden_states=True,
            return_dict=True
        )
        
        # Extract all hidden states from all layers
        all_hidden_states = encoder_outputs.hidden_states
        print(f"   Number of encoder layers: {len(all_hidden_states)}")
        
        # Combine all latent representations from all layers
        # Stack all hidden states along a new dimension
        combined_latent = torch.stack(all_hidden_states, dim=0)  # Shape: (num_layers, batch, seq_len, hidden_dim)
        print(f"   Combined latent shape: {combined_latent.shape}")
        
        # Also get the final layer output for comparison
        final_latent = encoder_outputs.last_hidden_state
        print(f"   Final layer latent shape: {final_latent.shape}")
        
        # Alternative: Concatenate all layers along the feature dimension
        # This would give us all layer features concatenated
        concatenated_latent = torch.cat(all_hidden_states, dim=-1)  # Shape: (batch, seq_len, hidden_dim * num_layers)
        print(f"   Concatenated latent shape: {concatenated_latent.shape}")
        
        # Also get final output for comparison
        outputs = model(
            signal_vectors=signal_vectors,
            xyz_vectors=xyz_vectors,
            noise=noise,
            return_dict=True
        )
        print(f"   Loss: {outputs.loss.item():.4f}")
    
    print(f"   ✅ BrainLM test passed!")
    
    return True


def main():
    """Run tests for both models"""
    print("🚀 Testing CBraMod and BrainLM Models")
    print("=" * 50)
    
    # Test CBraMod
    cbramod_success = test_cbramod()
    print()
    
    # Test BrainLM
    brainlm_success = test_brainlm()
    print()
    
    # Summary
    print("=" * 50)
    print("📊 Test Results:")
    print(f"   CBraMod: {'✅ PASSED' if cbramod_success else '❌ FAILED'}")
    print(f"   BrainLM: {'✅ PASSED' if brainlm_success else '❌ FAILED'}")
    
    if cbramod_success and brainlm_success:
        print("🎉 All tests passed! Both models are working correctly.")
    else:
        print("⚠️  Some tests failed. Check the error messages above.")

if __name__ == "__main__":
    main() 