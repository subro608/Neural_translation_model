#!/usr/bin/env python3
"""
Train the adapter_main_translatordecoder using Oddball EEG and fMRI for reconstruction.

Overview
 - EEG: load .mat, downsample to 200 Hz, select 34 EEG channels, patchify into 1s patches
 - fMRI: load BOLD NIfTI, extract A424 parcel time series in native space, prepare BrainLM inputs
 - Feature extractors are frozen (CBraMod, BrainLM). We train:
    - ConvEEGInputAdapter, fMRIInputAdapterConv1d
    - Two HierarchicalEncoders (EEG/fMRI), CrossAttentionLayer, BidirectionalAdaptiveCompressor
    - EEGDecodingAdapter + simple fMRI signal head
 - Loss: MSE on reconstructed raw signals vs. preprocessed inputs
 - Data split: 70% train, 10% val, 20% test at the file level

Notes
 - We can train on single-modality windows by zeroing the missing pathway.
 - Debug logs are enabled with --debug to trace shapes and info flow.
"""

from __future__ import annotations

import os
import sys
import json
import glob
import time
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# wandb
import wandb

# Local paths to import CBraMod and BrainLM modules
THIS_DIR = Path(__file__).parent
REPO_ROOT = THIS_DIR.parent
CBRAMOD_DIR = REPO_ROOT / "CBraMod"
BRAINLM_DIR = REPO_ROOT / "BrainLM"

sys.path.append(str(CBRAMOD_DIR))
sys.path.append(str(BRAINLM_DIR))

# Adapter and core blocks
from adapter_main_translatordecoder import (  # type: ignore
    BidirectionalAdaptiveCompressor,
    ConvEEGInputAdapter,
    fMRIInputAdapterConv1d,
    HierarchicalEncoder,
    CrossAttentionLayer,
    EEGDecodingAdapter,
)

# Models
from models.cbramod import CBraMod  # type: ignore
from brainlm_mae.modeling_brainlm import BrainLMForPretraining  # type: ignore
from brainlm_mae.configuration_brainlm import BrainLMConfig  # type: ignore

# EEG processing utils
import scipy.io as sio
import scipy.signal

# fMRI processing utils
import nibabel as nib
from nilearn.image import resample_to_img


# -----------------------------
# Utility: Reproducibility
# -----------------------------
def seed_all(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------------------
# EEG helpers (based on CBraMod test script)
# -----------------------------
def downsample_eeg(eeg_data: np.ndarray, original_fs: int = 1000, target_fs: int = 200) -> np.ndarray:
    factor = original_fs // target_fs
    x = scipy.signal.decimate(eeg_data, factor, axis=1, ftype='iir', zero_phase=True)
    return x.copy()


def patchify_eeg(eeg: np.ndarray, patch_size: int = 200) -> Tuple[np.ndarray, int]:
    channels, total_time = eeg.shape
    num_patches = total_time // patch_size
    if num_patches == 0:
        raise ValueError(f"EEG too short: {total_time} points < {patch_size}")
    usable = num_patches * patch_size
    eeg_trim = eeg[:, :usable]
    patches = eeg_trim.reshape(channels, num_patches, patch_size)
    return patches, num_patches


def normalize_channelwise(x: torch.Tensor) -> torch.Tensor:
    """
    Normalize per-channel over time/patch dims.
    Accepts (B,C,P,S) or (1,C,P,S) or (C,P,S) (auto-expands).
    """
    if x.dim() == 3:  # (C,P,S)
        x = x.unsqueeze(0)
    assert x.dim() == 4, f"Expected 4D tensor (B,C,P,S), got {tuple(x.shape)}"
    # Max over (P,S) per (B,C)
    max_abs = torch.amax(torch.abs(x), dim=(2, 3), keepdim=True)
    max_abs = torch.where(max_abs > 0, max_abs, torch.ones_like(max_abs))
    return x / max_abs


# -----------------------------
# fMRI helpers (based on BrainLM test script)
# -----------------------------
def extract_a424_from_native(bold_path: str, atlas_path: str) -> Tuple[np.ndarray, nib.Nifti1Image, np.ndarray, np.ndarray]:
    """
    Load BOLD and atlas in BOLD native space and extract A424 parcel time series.
    Always returns a fixed parcel dimension V=424 by enforcing parcel ids 1..424
    and zero-filling parcels with no voxels after resampling.

    Returns:
      parcel_ts: (T, 424) float64
      bold_img: nib.Nifti1Image
      atlas_data: np.ndarray (resampled atlas, same space as bold)
      parcel_ids: np.ndarray shape (424,) with values 1..424
    """
    bold_img = nib.load(bold_path)
    bold_data = bold_img.get_fdata()  # (X,Y,Z,T)
    atlas_img = nib.load(atlas_path)
    try:
        atlas_resampled = resample_to_img(
            atlas_img, bold_img, interpolation='nearest', force_resample=True, copy_header=True
        )
    except TypeError:
        # older nilearn signature
        atlas_resampled = resample_to_img(atlas_img, bold_img, interpolation='nearest')

    atlas_data = atlas_resampled.get_fdata()  # (X,Y,Z)
    T = bold_data.shape[-1]

    # Flatten for masked averaging
    bold_2d = bold_data.reshape(-1, T)       # (vox, T)
    atlas_flat = atlas_data.reshape(-1)      # (vox,)

    # FIX: enforce constant parcel ids 1..424
    parcel_ids = np.arange(1, 425, dtype=int)

    # Preallocate output (T, 424)
    parcel_ts = np.zeros((T, parcel_ids.size), dtype=np.float64)

    # Mean over voxels for each parcel id; zero if none present
    for i, pid in enumerate(parcel_ids):
        idx = np.where(atlas_flat == pid)[0]
        if idx.size > 0:
            parcel_ts[:, i] = np.nanmean(bold_2d[idx, :], axis=0)
        # else: keep zeros

    # Clean any inf/nan
    parcel_ts = np.nan_to_num(parcel_ts, nan=0.0, posinf=0.0, neginf=0.0)

    return parcel_ts, bold_img, atlas_data, parcel_ids


def pad_timepoints_for_brainlm(parcel_ts: np.ndarray, patch_size: int = 20) -> Tuple[np.ndarray, int]:
    num_timepoints, _ = parcel_ts.shape
    remainder = num_timepoints % patch_size
    if remainder == 0:
        return parcel_ts, num_timepoints
    pad = patch_size - remainder
    padded = np.zeros((num_timepoints + pad, parcel_ts.shape[1]))
    padded[:num_timepoints] = parcel_ts
    return padded, num_timepoints + pad


def pad_timepoints_for_brainlm_torch(signal_vectors_btv: torch.Tensor, patch_size: int = 20) -> torch.Tensor:
    """
    Pad along time dimension T to be divisible by patch_size.
    signal_vectors_btv: (B, T, V)
    returns: (B, T_pad, V)
    """
    B, T, V = signal_vectors_btv.shape
    remainder = T % patch_size
    if remainder == 0:
        return signal_vectors_btv
    padding_needed = patch_size - remainder
    padded = torch.zeros((B, T + padding_needed, V), dtype=signal_vectors_btv.dtype, device=signal_vectors_btv.device)
    padded[:, :T, :] = signal_vectors_btv
    return padded


# -----------------------------
# fMRI normalization helpers
# -----------------------------
def normalize_fmri_time_series(fmri_btv: torch.Tensor, mode: str = 'zscore', eps: float = 1e-8) -> torch.Tensor:
    """
    Normalize fMRI per parcel over time.
    fmri_btv: (B, T, V)
    mode: 'zscore' | 'psc' | 'mad' | 'none'
    Returns tensor of same shape, float32.
    """
    if mode == 'none':
        return fmri_btv.to(dtype=torch.float32)

    x = fmri_btv.to(dtype=torch.float32)
    if mode == 'zscore':
        mean_t = x.mean(dim=1, keepdim=True)
        std_t = x.std(dim=1, keepdim=True, unbiased=False)
        return (x - mean_t) / (std_t + eps)
    elif mode == 'psc':
        mean_t = x.mean(dim=1, keepdim=True)
        return 100.0 * (x - mean_t) / (mean_t + eps)
    elif mode == 'mad':
        median_t = x.median(dim=1, keepdim=True).values
        mad = (x - median_t).abs().median(dim=1, keepdim=True).values
        scaled_mad = mad * 1.4826
        return (x - median_t) / (scaled_mad + eps)
    else:
        raise ValueError(f"Unsupported fmri normalization mode: {mode}")


# -----------------------------
# Feature extractors (frozen)
# -----------------------------
class FrozenCBraMod(nn.Module):
    def __init__(self, in_dim: int, d_model: int, seq_len: int, n_layer: int, nhead: int, dim_feedforward: int,
                 weights_path: Path, device: torch.device) -> None:
        super().__init__()
        self.model = CBraMod(
            in_dim=in_dim,
            out_dim=in_dim,
            d_model=d_model,
            seq_len=seq_len,
            n_layer=n_layer,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
        )
        # Load weights robustly
        checkpoint = None
        try:
            try:
                checkpoint = torch.load(str(weights_path), map_location=device, weights_only=False)
            except TypeError:
                checkpoint = torch.load(str(weights_path), map_location=device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['state_dict'], strict=False)
            else:
                self.model.load_state_dict(checkpoint, strict=False)
        except Exception as e:
            raise RuntimeError(f"Failed to load CBraMod weights from {weights_path}: {e}")
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.to(device)
        self.device = device

    @torch.no_grad()
    def extract_latents(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, channels, patch_num, patch_size)
        returns: (L, B, seq_len, channels, input_dim)
        """
        patch_emb = self.model.patch_embedding(x)
        cur = patch_emb
        outs: List[torch.Tensor] = []
        for layer in self.model.encoder.layers:
            cur = layer(cur)
            outs.append(cur)
        return torch.stack(outs, dim=0)


class FrozenBrainLM(nn.Module):
    def __init__(self, model_dir: Path, device: torch.device) -> None:
        super().__init__()
        config_path = model_dir / "config.json"
        weights_path = model_dir / "pytorch_model.bin"
        with open(config_path, 'r') as f:
            cfg = json.load(f)
        config = BrainLMConfig(**cfg)
        self.model = BrainLMForPretraining(config)
        try:
            try:
                checkpoint = torch.load(str(weights_path), map_location=device, weights_only=True)
            except TypeError:
                checkpoint = torch.load(str(weights_path), map_location=device)
            self.model.load_state_dict(checkpoint, strict=False)
        except Exception as e:
            raise RuntimeError(f"Failed to load BrainLM weights from {weights_path}: {e}")
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.to(device)
        self.device = device

    @torch.no_grad()
    def extract_latents(self, signal_vectors: torch.Tensor, xyz_vectors: torch.Tensor, noise: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Inputs:
          signal_vectors: (B, num_voxels, timepoints)
          xyz_vectors:    (B, num_voxels, 3)
          noise:          (B, num_voxels * (timepoints // 20)) or None
        Returns:
          combined_latent: (L, B, T_tokens, hidden_size)
        """
        embeddings, mask, ids_restore = self.model.vit.embeddings(
            signal_vectors=signal_vectors, xyz_vectors=xyz_vectors, noise=noise
        )
        encoder_outputs = self.model.vit.encoder(
            hidden_states=embeddings, output_hidden_states=True, return_dict=True
        )
        all_hidden_states = encoder_outputs.hidden_states  # tuple of (B,T,D)
        return torch.stack(list(all_hidden_states), dim=0)


# -----------------------------
# Translator model (trainable)
# -----------------------------
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
        d_model: int = 256,
        n_heads: int = 8,
        d_ff: int = 1024,
        dropout: float = 0.1,
        debug: bool = False,
    ) -> None:
        super().__init__()
        self.debug = debug

        # Adapters
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

        # Encoders & fusion
        self.eeg_encoder = HierarchicalEncoder(d_model, n_heads, d_ff, dropout, n_layers_per_stack=2)
        self.fmri_encoder = HierarchicalEncoder(d_model, n_heads, d_ff, dropout, n_layers_per_stack=2)
        self.cross_attn = CrossAttentionLayer(d_model, n_heads, dropout)
        self.compressor = BidirectionalAdaptiveCompressor()

        # Decoders (latent-to-signal adapters)
        self.eeg_decoder = EEGDecodingAdapter(
            channels=eeg_channels,
            patch_num=eeg_patch_num,
            n_layers=eeg_n_layers,
            patch_size=eeg_input_dim,
            d_model=d_model,
        )
        # Lightweight fMRI signal head
        self.fmri_signal_seq = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(d_model * 2, d_model), nn.LayerNorm(d_model)
        )
        self.fmri_token_proj = nn.Linear(d_model, 1)

    def _dbg(self, msg: str) -> None:
        if self.debug:
            print(msg, flush=True)

    def forward(
        self,
        eeg_latents: Optional[torch.Tensor],  # (L,B,S,C,D)
        fmri_latents: Optional[torch.Tensor], # (L,B,T,H)
        modality_mask: torch.Tensor,          # (B,2) booleans
        fmri_target_T: Optional[int] = None,
        fmri_target_V: Optional[int] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        self._dbg("\n[Forward] start")
        if eeg_latents is not None:
            self._dbg(f"  EEG latents in: {tuple(eeg_latents.shape)} (L,B,S,C,D)")
            eeg_adapted = self.adapter_eeg(eeg_latents)
            self._dbg(f"  EEG after adapter: {tuple(eeg_adapted.shape)} (B, N_eeg_tokens, D)")
            eeg_lower = eeg_adapted
            eeg_higher = eeg_adapted
        else:
            self._dbg("  EEG absent")
            eeg_lower = eeg_higher = None

        if fmri_latents is not None:
            self._dbg(f"  fMRI latents in: {tuple(fmri_latents.shape)} (L,B,T_tokens,H)")
            fmri_adapted = self.adapter_fmri(fmri_latents)
            self._dbg(f"  fMRI after adapter: {tuple(fmri_adapted.shape)} (B, N_fmri_tokens, D=256)")
            fmri_lower = fmri_adapted
            fmri_higher = fmri_adapted
        else:
            self._dbg("  fMRI absent")
            fmri_lower = fmri_higher = None

        # Masking for missing modalities
        if eeg_lower is None:
            eeg_lower = torch.zeros_like(fmri_higher)
            eeg_higher = torch.zeros_like(fmri_higher)
            self._dbg("  EEG pathway zeroed to match fMRI shape")
        if fmri_lower is None:
            fmri_lower = torch.zeros_like(eeg_higher)
            fmri_higher = torch.zeros_like(eeg_higher)
            self._dbg("  fMRI pathway zeroed to match EEG shape")

        eeg_lower_enc, eeg_higher_enc = self.eeg_encoder(eeg_lower, eeg_higher)
        fmri_lower_enc, fmri_higher_enc = self.fmri_encoder(fmri_lower, fmri_higher)
        self._dbg(f"  EEG enc higher: {tuple(eeg_higher_enc.shape)}  | fMRI enc higher: {tuple(fmri_higher_enc.shape)}")

        eeg_compressed, fmri_compressed, _ = self.compressor(eeg_higher_enc, fmri_higher_enc)
        self._dbg(f"  After compressor -> EEG: {tuple(eeg_compressed.shape)}  fMRI: {tuple(fmri_compressed.shape)}")

        fused_output = self.cross_attn(eeg_compressed, fmri_compressed)
        self._dbg(f"  Fused output: {tuple(fused_output.shape)} (B, Tfused, D)")

        # EEG decode
        eeg_reconstructed_layers = self.eeg_decoder(fused_output)  # (L,B,P,C,S)
        self._dbg(f"  EEG decoder raw: {tuple(eeg_reconstructed_layers.shape)} (L,B,P,C,S)")
        # Mean over layers -> (B,P,C,S), then permute to (B,C,P,S)
        eeg_signal = eeg_reconstructed_layers.mean(dim=0).permute(0, 2, 1, 3).contiguous()
        self._dbg(f"  EEG signal out: {tuple(eeg_signal.shape)} (B,C,P,S)")

        # fMRI decode
        fmri_signal = None
        if fmri_target_T is not None and fmri_target_V is not None:
            x = self.fmri_signal_seq(fused_output)           # (B, Tfused, D)
            self._dbg(f"  fMRI seq head: {tuple(x.shape)} (B,Tfused,D)")
            x = self.fmri_token_proj(x)                      # (B, Tfused, 1)
            self._dbg(f"  fMRI token proj: {tuple(x.shape)} (B,Tfused,1)")
            xt = x.transpose(1, 2)                           # (B, 1, Tfused)
            target_len = fmri_target_T * fmri_target_V
            xt = nn.functional.interpolate(xt, size=target_len, mode='linear', align_corners=False)
            y = xt.transpose(1, 2).squeeze(-1)               # (B, target_len)
            fmri_signal = y.view(y.shape[0], fmri_target_T, fmri_target_V)  # (B,T,V)
            self._dbg(f"  fMRI signal out: {tuple(fmri_signal.shape)} (B,T,V)")
        else:
            self._dbg("  fMRI target T/V not provided; skipping fMRI reconstruction")

        self._dbg("[Forward] end\n")
        return eeg_signal, fmri_signal


# -----------------------------
# Datasets
# -----------------------------
class EEGWindowsDataset(Dataset):
    def __init__(
        self,
        eeg_mat_files: List[Path],
        original_fs: int,
        target_fs: int,
        window_sec: int,
        channels_limit: int = 34,  # default to 34 EEG channels
        device: str = 'cpu',
    ) -> None:
        self.eeg_mat_files = eeg_mat_files
        self.original_fs = original_fs
        self.target_fs = target_fs
        self.window_sec = window_sec
        self.channels_limit = channels_limit
        self.device = torch.device(device)
        self.index: List[Tuple[int, int, int]] = []  # (file_idx, start_patch, num_patches)

        for i, fpath in enumerate(self.eeg_mat_files):
            try:
                mat = sio.loadmat(str(fpath))
                eeg_key = None
                for key in ['data_reref', 'eeg', 'data', 'EEG', 'eegdata']:
                    if key in mat:
                        eeg_key = key
                        break
                if eeg_key is None:
                    continue
                data = mat[eeg_key]
                if data.shape[0] > data.shape[1]:
                    data = data.T
                ds = downsample_eeg(data, self.original_fs, self.target_fs)
                if ds.shape[0] > channels_limit:
                    ds = ds[:channels_limit]
                patches, num_patches = patchify_eeg(ds, patch_size=self.target_fs)
                window_patches = window_sec  # since 1s per patch
                for start_patch in range(0, max(1, num_patches - window_patches + 1)):
                    self.index.append((i, start_patch, window_patches))
            except Exception:
                continue

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        file_idx, start_patch, window_patches = self.index[idx]
        fpath = self.eeg_mat_files[file_idx]
        mat = sio.loadmat(str(fpath))
        eeg_key = None
        for key in ['data_reref', 'eeg', 'data', 'EEG', 'eegdata']:
            if key in mat:
                eeg_key = key
                break
        data = mat[eeg_key]
        if data.shape[0] > data.shape[1]:
            data = data.T
        ds = downsample_eeg(data, self.original_fs, self.target_fs)
        if ds.shape[0] > self.channels_limit:
            ds = ds[:self.channels_limit]
        patches, _ = patchify_eeg(ds, patch_size=self.target_fs)
        window = patches[:, start_patch:start_patch + window_patches, :].copy()
        x = torch.from_numpy(window).unsqueeze(0).float().to(self.device)  # (1,C,P,S)
        x = normalize_channelwise(x)
        return {
            'eeg_window': x.squeeze(0).cpu(),
            'meta_file': str(fpath),
            'meta_start_patch': start_patch,
        }


class FMRIDataset(Dataset):
    def __init__(
        self,
        bold_files: List[Path],
        a424_label_nii: Path,
        window_sec: int,
        tr: float = 2.0,
    ) -> None:
        self.bold_files = bold_files
        self.a424_label_nii = a424_label_nii
        self.window_sec = window_sec
        self.tr = tr
        self.index: List[Tuple[int, int, int]] = []  # (file_idx, start_t, window_T)
        for i, bpath in enumerate(self.bold_files):
            try:
                parcel_ts, _, _, _ = extract_a424_from_native(str(bpath), str(a424_label_nii))
                T = parcel_ts.shape[0]
                window_T = int(round(window_sec / tr))
                if T < window_T:
                    continue
                for start_t in range(0, T - window_T + 1, window_T):
                    self.index.append((i, start_t, window_T))
            except Exception:
                continue

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        file_idx, start_t, window_T = self.index[idx]
        bpath = self.bold_files[file_idx]
        parcel_ts, _, _, _ = extract_a424_from_native(str(bpath), str(self.a424_label_nii))
        window = parcel_ts[start_t:start_t + window_T, :]  # (T, V)
        return {
            'fmri_window': torch.from_numpy(window.astype(np.float32)),  # (T,V)
            'meta_file': str(bpath),
            'meta_start_t': start_t,
        }


# -----------------------------
# Training utilities
# -----------------------------
@dataclass
class TrainConfig:
    eeg_root: Path
    fmri_root: Path
    a424_label_nii: Path
    cbramod_weights: Path
    brainlm_model_dir: Path
    output_dir: Path
    device: str = 'cpu'
    seed: int = 42
    original_fs: int = 1000
    target_fs: int = 200
    window_sec: int = 30
    batch_size: int = 2
    lr: float = 1e-4
    weight_decay: float = 1e-4
    num_epochs: int = 5
    num_workers: int = 0
    train_ratio: float = 0.7
    val_ratio: float = 0.1
    test_ratio: float = 0.2
    amp: bool = False
    debug: bool = False
    fmri_norm: str = 'zscore'  # 'zscore' | 'psc' | 'mad' | 'none'
    # wandb
    wandb_project: Optional[str] = None
    wandb_run_name: Optional[str] = None
    wandb_off: bool = False
    # loss weights
    eeg_loss_w: float = 1.0
    fmri_loss_w: float = 1.0


def find_eeg_files(root: Path) -> List[Path]:
    patterns = [
        str(root / "**" / "EEG_rereferenced.mat"),
        str(root / "**" / "*.mat"),
    ]
    files: List[Path] = []
    for pat in patterns:
        files.extend([Path(p) for p in glob.glob(pat, recursive=True)])
    files = sorted(set(files), key=lambda p: ("EEG_rereferenced.mat" not in p.name, str(p).lower()))
    return files


def find_bold_files(root: Path) -> List[Path]:
    patterns = [
        str(root / "**" / "*bold.nii.gz"),
        str(root / "**" / "*bold.nii"),
    ]
    files: List[Path] = []
    for pat in patterns:
        files.extend([Path(p) for p in glob.glob(pat, recursive=True)])
    files = sorted(set(files), key=lambda p: str(p).lower())
    return files


def split_files(files: List[Path], ratios: Tuple[float, float, float], seed: int) -> Tuple[List[Path], List[Path], List[Path]]:
    n = len(files)
    if n == 0:
        return [], [], []
    rng = np.random.default_rng(seed)
    indices = np.arange(n)
    rng.shuffle(indices)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]
    idx_to_file = {i: files[i] for i in range(n)}
    return [idx_to_file[i] for i in train_idx], [idx_to_file[i] for i in val_idx], [idx_to_file[i] for i in test_idx]


def collate_eeg(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    eeg_windows = torch.stack([b['eeg_window'] for b in batch], dim=0)  # (B,C,P,S)
    return {'eeg_window': eeg_windows}


def collate_fmri(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    fmri_windows = torch.stack([b['fmri_window'] for b in batch], dim=0)  # (B,T,V)
    return {'fmri_window': fmri_windows}


def train_loop(cfg: TrainConfig) -> None:
    def dbg(msg: str) -> None:
        if cfg.debug:
            print(msg, flush=True)

    seed_all(cfg.seed)
    device = torch.device(cfg.device)
    os.makedirs(cfg.output_dir, exist_ok=True)

    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision('high')
        except Exception:
            pass

    # Discover files and split
    eeg_files = find_eeg_files(cfg.eeg_root)
    fmri_files = find_bold_files(cfg.fmri_root)

    eeg_train, eeg_val, eeg_test = split_files(eeg_files, (cfg.train_ratio, cfg.val_ratio, cfg.test_ratio), cfg.seed)
    fmri_train, fmri_val, fmri_test = split_files(fmri_files, (cfg.train_ratio, cfg.val_ratio, cfg.test_ratio), cfg.seed)

    # Datasets / Loaders
    ds_eeg_train = EEGWindowsDataset(eeg_train, cfg.original_fs, cfg.target_fs, cfg.window_sec, channels_limit=34, device='cpu')
    ds_eeg_val = EEGWindowsDataset(eeg_val, cfg.original_fs, cfg.target_fs, cfg.window_sec, channels_limit=34, device='cpu')
    ds_eeg_test = EEGWindowsDataset(eeg_test, cfg.original_fs, cfg.target_fs, cfg.window_sec, channels_limit=34, device='cpu')

    ds_fmri_train = FMRIDataset(fmri_train, cfg.a424_label_nii, cfg.window_sec, tr=2.0)
    ds_fmri_val = FMRIDataset(fmri_val, cfg.a424_label_nii, cfg.window_sec, tr=2.0)
    ds_fmri_test = FMRIDataset(fmri_test, cfg.a424_label_nii, cfg.window_sec, tr=2.0)

    pin = device.type == 'cuda'
    dl_eeg_train = DataLoader(ds_eeg_train, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, collate_fn=collate_eeg, pin_memory=pin)
    dl_eeg_val = DataLoader(ds_eeg_val, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, collate_fn=collate_eeg, pin_memory=pin)
    dl_eeg_test = DataLoader(ds_eeg_test, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, collate_fn=collate_eeg, pin_memory=pin)

    dl_fmri_train = DataLoader(ds_fmri_train, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, collate_fn=collate_fmri, pin_memory=pin)
    dl_fmri_val = DataLoader(ds_fmri_val, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, collate_fn=collate_fmri, pin_memory=pin)
    dl_fmri_test = DataLoader(ds_fmri_test, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, collate_fn=collate_fmri, pin_memory=pin)

    # Frozen feature extractors
    seq_len_eeg = cfg.window_sec
    frozen_cbramod = FrozenCBraMod(
        in_dim=cfg.target_fs,
        d_model=cfg.target_fs,
        seq_len=seq_len_eeg,
        n_layer=12,
        nhead=8,
        dim_feedforward=800,
        weights_path=cfg.cbramod_weights,
        device=device,
    )
    frozen_brainlm = FrozenBrainLM(cfg.brainlm_model_dir, device=device)

    # Translator model (trainable)
    translator = TranslatorModel(
        eeg_channels=34,  # use 34 EEG channels
        eeg_patch_num=seq_len_eeg,
        eeg_n_layers=12,
        eeg_input_dim=cfg.target_fs,
        fmri_n_layers=5,
        fmri_hidden_size=256,
        fmri_tokens_target=424 * int(round(cfg.window_sec / 2.0)),
        d_model=256,
        n_heads=8,
        d_ff=1024,
        dropout=0.1,
        debug=cfg.debug if 'cffg' in locals() else cfg.debug,
    ).to(device)

    optim = torch.optim.AdamW(
        [p for p in translator.parameters() if p.requires_grad],
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    mse_loss = nn.MSELoss()

    # wandb init
    run = None
    if not cfg.wandb_off:
        run = wandb.init(
            project=cfg.wandb_project or "oddball_eeg_fmri",
            name=cfg.wandb_run_name,
            config={**vars(cfg), "model": "Translator+CBraMod+BrainLM"},
        )
        wandb.watch(translator, log="gradients", log_freq=200)

    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.amp and device.type == 'cuda'))
    global_step = 0  # for step-wise logging

    def run_epoch(mode: str, dl_eeg, dl_fmri, epoch: int) -> Dict[str, float]:
        nonlocal global_step
        is_train = mode == 'train'
        translator.train(is_train)
        total_eeg_loss = 0.0
        total_fmri_loss = 0.0
        steps = 0

        eeg_iter = iter(dl_eeg)
        fmri_iter = iter(dl_fmri)
        max_steps = max(len(dl_eeg), len(dl_fmri))

        for step in range(max_steps):
            eeg_batch = None
            fmri_batch = None
            try:
                eeg_batch = next(eeg_iter)
            except StopIteration:
                pass
            try:
                fmri_batch = next(fmri_iter)
            except StopIteration:
                pass

            eeg_latents_t = None
            fmri_latents_t = None

            if eeg_batch is not None:
                x_eeg = eeg_batch['eeg_window'].to(device, non_blocking=True)  # (B_eeg,C,P,S)
                with torch.no_grad():
                    eeg_latents = frozen_cbramod.extract_latents(x_eeg)
                eeg_latents_t = eeg_latents  # (L,B_eeg,S,C,D)
            if fmri_batch is not None:
                fmri_t = fmri_batch['fmri_window'].to(device, non_blocking=True)  # (B_fmri,T,V)
                # Normalize per parcel over time BEFORE both target and latent extraction
                fmri_t = normalize_fmri_time_series(fmri_t, mode=cfg.fmri_norm)
                Bf, Tf, Vf = fmri_t.shape
                fmri_padded = pad_timepoints_for_brainlm_torch(fmri_t, patch_size=20)  # (B,Tp,V)
                Tp = fmri_padded.shape[1]
                signal_vectors = fmri_padded.permute(0, 2, 1).contiguous()
                if getattr(frozen_brainlm.model.config, 'num_brain_voxels', None) != Vf:
                    frozen_brainlm.model.config.num_brain_voxels = Vf
                    if hasattr(frozen_brainlm.model.vit, 'embeddings'):
                        frozen_brainlm.model.vit.embeddings.num_brain_voxels = Vf
                    if hasattr(frozen_brainlm.model, 'decoder'):
                        frozen_brainlm.model.decoder.num_brain_voxels = Vf
                patch_size = 20
                P = Tp // patch_size
                noise = torch.rand(Bf, Vf * P, device=device)
                xyz = torch.zeros(Bf, Vf, 3, device=device)
                with torch.no_grad():
                    fmri_latents = frozen_brainlm.extract_latents(signal_vectors, xyz, noise)
                fmri_latents_t = fmri_latents  # (L,B,T_tokens,H)

            if eeg_latents_t is None and fmri_latents_t is None:
                continue

            # Align batch sizes by clipping to the smaller batch
            Be = int(eeg_latents_t.shape[1]) if eeg_latents_t is not None else None
            Bf2 = int(fmri_latents_t.shape[1]) if fmri_latents_t is not None else None
            if Be is not None and Bf2 is not None:
                Bmin = min(Be, Bf2)
            elif Be is not None:
                Bmin = Be
            elif Bf2 is not None:
                Bmin = Bf2
            else:
                continue

            if eeg_latents_t is not None and Be != Bmin:
                eeg_latents_t = eeg_latents_t[:, :Bmin]
                x_eeg = x_eeg[:Bmin]
            if fmri_latents_t is not None and Bf2 != Bmin:
                fmri_latents_t = fmri_latents_t[:, :Bmin]
                fmri_t = fmri_t[:Bmin]

            # Build modality mask AFTER clipping
            modality_mask = torch.zeros((Bmin, 2), dtype=torch.bool, device=device)
            if eeg_latents_t is not None:
                modality_mask[:, 0] = True
            if fmri_latents_t is not None:
                modality_mask[:, 1] = True

            if cfg.debug:
                print("\n[Batch Debug]")
                print(f"  step={step} mode={mode} Bmin={Bmin}")
                print(f"  EEG present? {eeg_latents_t is not None}  fMRI present? {fmri_latents_t is not None}")
                if eeg_latents_t is not None:
                    print(f"  EEG latents_t: {tuple(eeg_latents_t.shape)}  target x_eeg: {tuple(x_eeg.shape)}")
                if fmri_latents_t is not None:
                    print(f"  fMRI latents_t: {tuple(fmri_latents_t.shape)}  target fmri_t: {tuple(fmri_t.shape)}")
                print(f"  Modality mask: shape={tuple(modality_mask.shape)} first_row={modality_mask[0].tolist()}")

            if is_train:
                optim.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda', enabled=(cfg.amp and device.type == 'cuda')):
                # Recompute fmri_T_arg / fmri_V_arg AFTER clipping
                fmri_T_arg = int(fmri_t.shape[1]) if fmri_latents_t is not None else None
                fmri_V_arg = int(fmri_t.shape[2]) if fmri_latents_t is not None else None
                recon_eeg, recon_fmri = translator(
                    eeg_latents_t, fmri_latents_t, modality_mask,
                    fmri_target_T=fmri_T_arg, fmri_target_V=fmri_V_arg
                )

            # Accumulate modality losses and apply weights
            loss = 0.0
            if eeg_latents_t is not None and recon_eeg is not None:
                if cfg.debug:
                    print(f"  recon_eeg: {tuple(recon_eeg.shape)} vs x_eeg: {tuple(x_eeg.shape)}")
                loss_eeg = mse_loss(recon_eeg, x_eeg)
                total_eeg_loss += float(loss_eeg.detach().cpu().item())
                loss = loss + (cfg.eeg_loss_w * loss_eeg)
            if fmri_latents_t is not None and recon_fmri is not None:
                if cfg.debug:
                    print(f"  recon_fmri: {tuple(recon_fmri.shape)} vs fmri_t: {tuple(fmri_t.shape)}")
                loss_fmri = mse_loss(recon_fmri, fmri_t)
                total_fmri_loss += float(loss_fmri.detach().cpu().item())
                loss = loss + (cfg.fmri_loss_w * loss_fmri)

            # per-step wandb logs
            if is_train and not cfg.wandb_off:
                wandb.log({
                    "train/step_weighted_total_loss": float(loss.detach().cpu()),
                    "train/step_eeg_loss": float(loss_eeg.detach().cpu()) if eeg_latents_t is not None and recon_eeg is not None else 0.0,
                    "train/step_fmri_loss": float(loss_fmri.detach().cpu()) if fmri_latents_t is not None and recon_fmri is not None else 0.0,
                    "train/eeg_loss_w": cfg.eeg_loss_w,
                    "train/fmri_loss_w": cfg.fmri_loss_w,
                }, step=global_step)

            if is_train:
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(translator.parameters(), 1.0)
                scaler.step(optim)
                scaler.update()
                global_step += 1

            steps += 1

        scalars = {
            f'{mode}_eeg_loss': total_eeg_loss / max(1, steps),
            f'{mode}_fmri_loss': total_fmri_loss / max(1, steps),
        }
        return scalars

    best_val = float('inf')
    best_path = cfg.output_dir / 'translator_best.pt'

    for epoch in range(1, cfg.num_epochs + 1):
        t0 = time.time()
        tr = run_epoch('train', dl_eeg_train, dl_fmri_train, epoch)
        va = run_epoch('val', dl_eeg_val, dl_fmri_val, epoch)
        dur = time.time() - t0
        print(f"Epoch {epoch:02d} | {dur:.1f}s | train EEG {tr['train_eeg_loss']:.6f} FMRI {tr['train_fmri_loss']:.6f} | "
              f"val EEG {va['val_eeg_loss']:.6f} FMRI {va['val_fmri_loss']:.6f}")

        # per-epoch wandb logs
        if not cfg.wandb_off:
            wandb.log({
                "epoch": epoch,
                "train/eeg_loss": tr['train_eeg_loss'],
                "train/fmri_loss": tr['train_fmri_loss'],
                "train/total_loss": tr['train_eeg_loss'] + tr['train_fmri_loss'],
                "val/eeg_loss": va['val_eeg_loss'],
                "val/fmri_loss": va['val_fmri_loss'],
                "val/total_loss": va['val_eeg_loss'] + va['val_fmri_loss'],
                "time/epoch_seconds": dur,
                }, step=global_step)

        val_score = va['val_eeg_loss'] + va['val_fmri_loss']
        if val_score < best_val:
            best_val = val_score
            torch.save({'model': translator.state_dict(), 'epoch': epoch, 'val_score': best_val}, best_path)
            print(f"  ✅ Saved best model to {best_path}")
            if not cfg.wandb_off:
                wandb.summary["best_val_total_loss"] = best_val
                try:
                    wandb.save(str(best_path))
                except Exception:
                    pass

    # Test with best model if saved
    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device)
        translator.load_state_dict(ckpt['model'])

    te = run_epoch('test', dl_eeg_test, dl_fmri_test, epoch=0)
    with open(cfg.output_dir / 'metrics.json', 'w') as f:
        json.dump(te, f, indent=2)
    print(f"Test metrics: {te}")

    if not cfg.wandb_off:
        wandb.log({
            "test/eeg_loss": te['test_eeg_loss'],
            "test/fmri_loss": te['test_fmri_loss'],
            "test/total_loss": te['test_eeg_loss'] + te['test_fmri_loss'],
        }, step=global_step)
        wandb.finish()


def main() -> None:
    parser = argparse.ArgumentParser(description='Train translator-decoder on Oddball EEG and fMRI for reconstruction')
    parser.add_argument('--eeg_root', type=str, required=True, help='Path to Oddball EEG root (e.g., Oddball/ds116_eeg)')
    parser.add_argument('--fmri_root', type=str, required=True, help='Path to Oddball fMRI root (e.g., Oddball/ds000116)')
    parser.add_argument('--a424_label_nii', type=str, required=True, help='Path to A424 atlas NIfTI in BOLD native space')
    parser.add_argument('--cbramod_weights', type=str, default=str(CBRAMOD_DIR / 'pretrained_weights' / 'pretrained_weights.pth'))
    parser.add_argument('--brainlm_model_dir', type=str, required=True, help='Path to BrainLM pretrained checkpoint dir containing config.json and pytorch_model.bin')
    parser.add_argument('--output_dir', type=str, default=str(THIS_DIR / 'translator_runs'))
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--window_sec', type=int, default=30)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--amp', action='store_true', help='Enable mixed precision training on CUDA')
    parser.add_argument('--debug', action='store_true', help='Enable verbose debug prints for shapes/flow')
    parser.add_argument('--fmri_norm', type=str, default='zscore', choices=['zscore','psc','mad','none'], help='Per-parcel temporal normalization for fMRI')
    # wandb
    parser.add_argument('--wandb_project', type=str, default=None, help='W&B project name')
    parser.add_argument('--wandb_run_name', type=str, default=None, help='W&B run name')
    parser.add_argument('--wandb_off', action='store_true', help='Disable W&B logging')
    # loss weights
    parser.add_argument('--eeg_loss_w', type=float, default=1.0, help='Weight for EEG reconstruction loss')
    parser.add_argument('--fmri_loss_w', type=float, default=3.0, help='Weight for fMRI reconstruction loss')

    args = parser.parse_args()

    cfg = TrainConfig(
        eeg_root=Path(args.eeg_root),
        fmri_root=Path(args.fmri_root),
        a424_label_nii=Path(args.a424_label_nii),
        cbramod_weights=Path(args.cbramod_weights),
        brainlm_model_dir=Path(args.brainlm_model_dir),
        output_dir=Path(args.output_dir),
        device=args.device,
        seed=args.seed,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        window_sec=args.window_sec,
        num_workers=args.num_workers,
        amp=args.amp,
        fmri_norm=args.fmri_norm,
        debug=args.debug,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        wandb_off=args.wandb_off,
        eeg_loss_w=args.eeg_loss_w,
        fmri_loss_w=args.fmri_loss_w,
    )

    train_loop(cfg)


if __name__ == '__main__':
    main()
