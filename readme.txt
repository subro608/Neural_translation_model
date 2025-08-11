Train (YAML config):
  python train_translator_paired.py --config configs/translator.yaml

Test (YAML config):
  python test_translator_paired.py --config configs/translator.yaml

Notes:
- CLI flags override YAML values if provided (e.g., add `--device cuda`).
- To enable Weights & Biases logging, set `wandb_off: false` and fill `wandb_project` and `wandb_run_name` in `configs/translator.yaml`.



Summary of Conditional Multimodal Masking Training Additions
1. Condition Embeddings - Core Addition
A. Add to TranslatorModel
pythonclass TranslatorModel(nn.Module):
    def __init__(self, ...):
        # ... existing code ...
        
        # NEW: Condition embeddings for different masking scenarios
        self.condition_embed = nn.Embedding(5, d_model)
        # 0: both_available, 1: eeg_missing, 2: fmri_missing, 3: eeg_partial, 4: fmri_partial
        
    def forward(self, eeg_latents, fmri_latents, fmri_target_T, fmri_target_V, 
                condition_ids=None):  # NEW parameter
        
        eeg_adapted = self.adapter_eeg(eeg_latents)
        fmri_adapted = self.adapter_fmri(fmri_latents)
        
        # NEW: Add condition information to both modalities
        if condition_ids is not None:
            cond_embeds = self.condition_embed(condition_ids)  # (B, d_model)
            eeg_adapted = eeg_adapted + cond_embeds.unsqueeze(1)  # Broadcast to all tokens
            fmri_adapted = fmri_adapted + cond_embeds.unsqueeze(1)
        
        # Rest of forward pass unchanged...
2. Training Loop Modifications
A. Create Condition IDs
pythondef run_epoch(mode: str, epoch: int):
    # ... existing masking logic ...
    
    for i, batch in enumerate(iterator):
        # ... existing code up to condition assignment ...
        
        # NEW: Create condition IDs for each sample
        condition_ids = torch.zeros(B, dtype=torch.long, device=device)
        condition_ids[cond == 0] = 0  # both_available
        condition_ids[miss_eeg] = 1   # eeg_missing  
        condition_ids[miss_fmri] = 2  # fmri_missing
        condition_ids[partial_eeg] = 3  # eeg_partial
        condition_ids[partial_fmr] = 4  # fmri_partial
        
        # ... existing frozen latent extraction ...
        
        # NEW: Pass condition information to translator
        with torch.amp.autocast('cuda', enabled=(cfg.amp and device.type == 'cuda')):
            recon_eeg, recon_fmri = translator(
                eeg_latents_t, fmri_latents,
                fmri_target_T=int(fmri_t.shape[1]),
                fmri_target_V=int(fmri_t.shape[2]),
                condition_ids=condition_ids,  # NEW
            )
3. Enhanced Cross-Modal Loss Weighting
A. Emphasize Cross-Modal Learning
python# In loss calculation section:
# CHANGE: Increase weight for cross-modal scenarios
alpha_miss, beta_pres = 1.5, 1.0  # Instead of 1.0, 1.0

w_eeg = torch.where(miss_eeg, torch.as_tensor(alpha_miss, device=device),
                              torch.as_tensor(beta_pres,  device=device))
w_fmr = torch.where(miss_fmri, torch.as_tensor(alpha_miss, device=device),
                              torch.as_tensor(beta_pres,  device=device))

# This gives higher weight to EEG->fMRI and fMRI->EEG reconstruction
4. Optional: Skip Frozen Encoder for Missing Data
A. Efficient Processing
pythondef run_epoch(mode: str, epoch: int):
    # ... existing setup ...
    
    with torch.no_grad():
        # Always process EEG
        eeg_latents = frozen_cbramod.extract_latents(x_eeg_obs)
        
        # NEW: Smart fMRI processing - skip BrainLM for missing samples
        if miss_fmri.any():
            # Create placeholder latents for missing samples
            L_fmri = 5  # BrainLM layers
            T_tokens = 424 * int(round(cfg.window_sec / 2.0)) // 20
            H = 256  # BrainLM hidden size
            fmri_latents = torch.zeros(L_fmri, B, T_tokens, H, device=device)
            
            # Only process non-missing samples through BrainLM
            available_mask = ~miss_fmri
            if available_mask.any():
                fmri_subset = fmri_obs[available_mask]
                fmri_padded = pad_timepoints_for_brainlm_torch(fmri_subset, patch_size=20)
                signal_vectors = fmri_padded.permute(0, 2, 1).contiguous()
                
                # Process coordinates for subset only
                if V == 424:
                    xyz_subset = xyz[available_mask]
                else:
                    xyz_subset = torch.zeros(available_mask.sum(), V, 3, device=device)
                
                # Get latents only for available samples
                fmri_latents_subset = frozen_brainlm.extract_latents(signal_vectors, xyz_subset, noise=None)
                fmri_latents[:, available_mask, :, :] = fmri_latents_subset
        else:
            # All available - process normally
            fmri_latents = frozen_brainlm.extract_latents(...)
5. Optional: Missing Modality Tokens
A. Alternative to Zero Inputs
pythonclass TranslatorModel(nn.Module):
    def __init__(self, ...):
        # ... existing code ...
        
        # NEW: Learnable tokens for completely missing modalities
        self.eeg_missing_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.fmri_missing_token = nn.Parameter(torch.randn(1, 1, d_model))
        
    def forward(self, ..., condition_ids=None):
        eeg_adapted = self.adapter_eeg(eeg_latents)
        fmri_adapted = self.adapter_fmri(fmri_latents)
        
        # NEW: Replace completely missing modalities with learned tokens
        if condition_ids is not None:
            eeg_missing_mask = (condition_ids == 1)  # EEG missing
            fmri_missing_mask = (condition_ids == 2)  # fMRI missing
            
            if eeg_missing_mask.any():
                B, seq_len, d_model = eeg_adapted.shape
                missing_tokens = self.eeg_missing_token.expand(eeg_missing_mask.sum(), seq_len, d_model)
                eeg_adapted[eeg_missing_mask] = missing_tokens
                
            if fmri_missing_mask.any():
                B, seq_len, d_model = fmri_adapted.shape
                missing_tokens = self.fmri_missing_token.expand(fmri_missing_mask.sum(), seq_len, d_model)
                fmri_adapted[fmri_missing_mask] = missing_tokens
Minimal Implementation (Essential Only):
python# 1. Add to TranslatorModel.__init__:
self.condition_embed = nn.Embedding(5, d_model)

# 2. Modify TranslatorModel.forward() signature:
def forward(self, eeg_latents, fmri_latents, fmri_target_T, fmri_target_V, condition_ids=None):
    eeg_adapted = self.adapter_eeg(eeg_latents)
    fmri_adapted = self.adapter_fmri(fmri_latents)
    
    # Add condition embeddings
    if condition_ids is not None:
        cond_embeds = self.condition_embed(condition_ids)
        eeg_adapted = eeg_adapted + cond_embeds.unsqueeze(1)
        fmri_adapted = fmri_adapted + cond_embeds.unsqueeze(1)
    
    # Rest unchanged...

# 3. In training loop, before translator call:
condition_ids = torch.zeros(B, dtype=torch.long, device=device)
condition_ids[miss_eeg] = 1
condition_ids[miss_fmri] = 2
condition_ids[partial_eeg] = 3
condition_ids[partial_fmr] = 4

# 4. Update translator call:
recon_eeg, recon_fmri = translator(
    eeg_latents_t, fmri_latents,
    fmri_target_T=int(fmri_t.shape[1]),
    fmri_target_V=int(fmri_t.shape[2]),
    condition_ids=condition_ids
)

# 5. Optional - increase cross-modal emphasis:
alpha_miss, beta_pres = 1.5, 1.0  # Instead of 1.0, 1.0
This gives your translator explicit awareness of:

What type of masking is applied to each sample
Which modality needs cross-modal reconstruction
How to adapt its processing strategy accordingly

The core insight: Instead of the model having to figure out "why is this input zero?", you're explicitly telling it "this is zero because the modality is missing, please reconstruct it from the other modality."