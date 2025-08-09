import torch
import torch.nn as nn
import torch.nn.functional as F

class EEGDecodingAdapter(nn.Module):
    """
    Decodes fused representation back to EEG format
    Input: (batch, fused_seq_len, d_model=256)
    Output: (n_layers, batch, patch_num, channels, patch_size=200)
    
    This adapter reconstructs the original CBraMod latent representation format
    """
    def __init__(self, channels, patch_num, n_layers=12, patch_size=200, d_model=256):
        super().__init__()
        self.channels = channels
        self.patch_num = patch_num
        self.n_layers = n_layers
        self.patch_size = patch_size
        self.d_model = d_model
        self.original_seq_len = channels * patch_num
        
        # Sequence length adjustment layer
        self.seq_adjuster = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.LayerNorm(d_model)
        )
        
        # Project back to multi-layer EEG dimensions
        self.layer_projector = nn.Linear(d_model, n_layers * patch_size)
        self.layer_norm = nn.LayerNorm(n_layers * patch_size)
        
        # Refinement layers for better reconstruction quality
        self.refinement = nn.Sequential(
            nn.Linear(n_layers * patch_size, n_layers * patch_size * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(n_layers * patch_size * 2, n_layers * patch_size),
            nn.LayerNorm(n_layers * patch_size)
        )
        
        # Final activation
        self.final_activation = nn.Tanh()
        
    def forward(self, fused_representation):
        """
        Args:
            fused_representation: (batch, fused_seq_len, d_model)
        Returns:
            reconstructed_eeg: (n_layers, batch, patch_num, channels, patch_size)
        """
        batch_size, fused_seq_len, d_model = fused_representation.shape
        
        # Step 1: Adjust sequence length to match original EEG sequence length
        if fused_seq_len != self.original_seq_len:
            # Use adaptive pooling to get back to original EEG sequence length
            x_transposed = fused_representation.transpose(1, 2)  # (batch, d_model, fused_seq_len)
            x_adjusted = F.adaptive_avg_pool1d(x_transposed, self.original_seq_len)
            x = x_adjusted.transpose(1, 2)  # (batch, original_seq_len, d_model)
        else:
            x = fused_representation
        
        # Step 2: Apply sequence adjustment
        x = self.seq_adjuster(x)  # (batch, original_seq_len, d_model)
        
        # Step 3: Project to multi-layer EEG dimensions
        x = self.layer_projector(x)  # (batch, original_seq_len, n_layers * patch_size)
        x = self.layer_norm(x)
        
        # Step 4: Apply refinement
        x = self.refinement(x)  # (batch, original_seq_len, n_layers * patch_size)
        
        # Step 5: Apply final activation
        x = self.final_activation(x)
        
        # Step 6: Reshape to EEG format
        # First reshape to (batch, channels, patch_num, n_layers * patch_size)
        x = x.view(batch_size, self.channels, self.patch_num, self.n_layers * self.patch_size)
        
        # Then to (batch, channels, patch_num, n_layers, patch_size)
        x = x.view(batch_size, self.channels, self.patch_num, self.n_layers, self.patch_size)
        
        # Final format: (n_layers, batch, patch_num, channels, patch_size)
        x = x.permute(3, 0, 1, 2, 4)
        
        return x

class fMRIDecodingAdapter(nn.Module):
    """
    Decodes fused representation back to fMRI format
    Input: (batch, fused_seq_len, d_model=256)
    Output: (n_layers, batch, num_voxels * timepoints_per_voxel, hidden_size=256)
    
    This adapter reconstructs the original BrainLM latent representation format
    """
    def __init__(self, num_voxels=424, timepoints_per_voxel=200, n_layers=5, 
                 hidden_size=256, d_model=256):
        super().__init__()
        self.num_voxels = num_voxels
        self.timepoints_per_voxel = timepoints_per_voxel
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.d_model = d_model
        self.target_seq_len = num_voxels * timepoints_per_voxel
        
        # Sequence length adjustment with learnable features
        self.seq_adjuster = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model)
        )
        
        # Project back to multi-layer fMRI dimensions
        self.layer_projector = nn.Linear(d_model, n_layers * hidden_size)
        self.layer_norm = nn.LayerNorm(n_layers * hidden_size)
        
        # Learnable upsampling for sequence length expansion
        self.upsampler = nn.ModuleList([
            nn.ConvTranspose1d(
                in_channels=n_layers * hidden_size,
                out_channels=n_layers * hidden_size,
                kernel_size=4,
                stride=2,
                padding=1,
                groups=n_layers * hidden_size
            ),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=n_layers * hidden_size,
                out_channels=n_layers * hidden_size,
                kernel_size=3,
                padding=1,
                groups=n_layers * hidden_size
            ),
            nn.LayerNorm(n_layers * hidden_size)
        ])
        
        # Refinement layers for better reconstruction
        self.refinement = nn.Sequential(
            nn.Linear(n_layers * hidden_size, n_layers * hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(n_layers * hidden_size * 2, n_layers * hidden_size),
            nn.LayerNorm(n_layers * hidden_size)
        )
        
        # Final activation
        self.final_activation = nn.Tanh()
        
    def forward(self, fused_representation):
        """
        Args:
            fused_representation: (batch, fused_seq_len, d_model)
        Returns:
            reconstructed_fmri: (n_layers, batch, num_voxels * timepoints_per_voxel, hidden_size)
        """
        batch_size, fused_seq_len, d_model = fused_representation.shape
        
        # Step 1: Apply sequence adjustment
        x = self.seq_adjuster(fused_representation)  # (batch, fused_seq_len, d_model)
        
        # Step 2: Project to multi-layer fMRI dimensions
        x = self.layer_projector(x)  # (batch, fused_seq_len, n_layers * hidden_size)
        x = self.layer_norm(x)
        
        # Step 3: Expand sequence length to target fMRI length
        if fused_seq_len < self.target_seq_len:
            # Use learnable upsampling
            x_transposed = x.transpose(1, 2)  # (batch, n_layers * hidden_size, fused_seq_len)
            
            # Apply upsampling layers iteratively
            current_len = fused_seq_len
            upsampling_iterations = 0
            max_iterations = 10  # Prevent infinite loops
            
            while current_len < self.target_seq_len and upsampling_iterations < max_iterations:
                # Apply ConvTranspose1d
                x_transposed = self.upsampler[0](x_transposed)
                # Apply ReLU
                x_transposed = self.upsampler[1](x_transposed)
                # Apply Conv1d
                x_transposed = self.upsampler[2](x_transposed)
                
                current_len = x_transposed.shape[2]
                upsampling_iterations += 1
                
                # If we overshoot, use adaptive pooling to get exact length
                if current_len >= self.target_seq_len:
                    x_transposed = F.adaptive_avg_pool1d(x_transposed, self.target_seq_len)
                    break
            
            x = x_transposed.transpose(1, 2)  # (batch, target_seq_len, n_layers * hidden_size)
            
            # Apply layer norm after upsampling
            batch_size_new, seq_len_new, feature_dim = x.shape
            x = x.view(-1, feature_dim)  # Flatten for LayerNorm
            x = self.upsampler[3](x)  # Apply LayerNorm
            x = x.view(batch_size_new, seq_len_new, feature_dim)  # Reshape back
            
        elif fused_seq_len > self.target_seq_len:
            # Downsample if needed
            x_transposed = x.transpose(1, 2)  # (batch, n_layers * hidden_size, fused_seq_len)
            x_transposed = F.adaptive_avg_pool1d(x_transposed, self.target_seq_len)
            x = x_transposed.transpose(1, 2)  # (batch, target_seq_len, n_layers * hidden_size)
        
        # Step 4: Apply refinement
        x = self.refinement(x)  # (batch, target_seq_len, n_layers * hidden_size)
        
        # Step 5: Apply final activation
        x = self.final_activation(x)
        
        # Step 6: Reshape to fMRI format
        # (batch, target_seq_len, n_layers, hidden_size)
        x = x.view(batch_size, self.target_seq_len, self.n_layers, self.hidden_size)
        
        # Final format: (n_layers, batch, target_seq_len, hidden_size)
        x = x.permute(2, 0, 1, 3)
        
        return x

# Test function for the decoding adapters
def test_decoding_adapters():
    """Test both decoding adapters with sample data"""
    
    print("🧪 Testing Decoding Adapters")
    print("="*50)
    
    # Test configurations
    eeg_config = {"channels": 64, "patch_num": 8}
    fmri_config = {"timepoints_per_voxel": 400}
    
    # Sample fused representation (output from cross-attention)
    batch_size = 1
    fused_seq_len = 512  # From geometric mean compression
    d_model = 256
    fused_repr = torch.randn(batch_size, fused_seq_len, d_model)
    
    print(f"Input fused representation: {fused_repr.shape}")
    
    # Test EEG Decoding Adapter
    print("\n🔄 Testing EEG Decoding Adapter...")
    eeg_decoder = EEGDecodingAdapter(
        channels=eeg_config["channels"],
        patch_num=eeg_config["patch_num"],
        n_layers=12,
        patch_size=200,
        d_model=d_model
    )
    
    eeg_reconstructed = eeg_decoder(fused_repr)
    expected_eeg_shape = (12, batch_size, eeg_config["patch_num"], eeg_config["channels"], 200)
    
    print(f"✅ EEG reconstructed shape: {eeg_reconstructed.shape}")
    print(f"📋 Expected EEG shape: {expected_eeg_shape}")
    print(f"🎯 Shape match: {'✅' if eeg_reconstructed.shape == expected_eeg_shape else '❌'}")
    
    # Test fMRI Decoding Adapter
    print("\n🔄 Testing fMRI Decoding Adapter...")
    fmri_decoder = fMRIDecodingAdapter(
        num_voxels=424,
        timepoints_per_voxel=fmri_config["timepoints_per_voxel"],
        n_layers=5,
        hidden_size=256,
        d_model=d_model
    )
    
    fmri_reconstructed = fmri_decoder(fused_repr)
    expected_fmri_shape = (5, batch_size, 424 * fmri_config["timepoints_per_voxel"], 256)
    
    print(f"✅ fMRI reconstructed shape: {fmri_reconstructed.shape}")
    print(f"📋 Expected fMRI shape: {expected_fmri_shape}")
    print(f"🎯 Shape match: {'✅' if fmri_reconstructed.shape == expected_fmri_shape else '❌'}")
    
    # Test different fused sequence lengths
    print("\n🔄 Testing with different fused sequence lengths...")
    
    # Test with length 263 (from Case 1)
    fused_repr_263 = torch.randn(batch_size, 263, d_model)
    eeg_recon_263 = eeg_decoder(fused_repr_263)
    fmri_recon_263 = fmri_decoder(fused_repr_263)
    
    print(f"With fused length 263:")
    print(f"  EEG: {fused_repr_263.shape} → {eeg_recon_263.shape}")
    print(f"  fMRI: {fused_repr_263.shape} → {fmri_recon_263.shape}")
    
    # Test with length 1024 (from Case 3)
    fused_repr_1024 = torch.randn(batch_size, 1024, d_model)
    eeg_recon_1024 = eeg_decoder(fused_repr_1024)
    fmri_recon_1024 = fmri_decoder(fused_repr_1024)
    
    print(f"With fused length 1024:")
    print(f"  EEG: {fused_repr_1024.shape} → {eeg_recon_1024.shape}")
    print(f"  fMRI: {fused_repr_1024.shape} → {fmri_recon_1024.shape}")
    
    print("\n✅ All decoding adapter tests completed!")
    
    return eeg_decoder, fmri_decoder

if __name__ == "__main__":
    test_decoding_adapters()