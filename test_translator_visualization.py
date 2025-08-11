#!/usr/bin/env python3
"""
Visualize original vs reconstructed EEG and fMRI (DiFuMo).

What it does
------------
- Loads one batch from PairedAlignedDataset
- Runs frozen CBraMod + BrainLM + TranslatorModel to get reconstructions
- Plots EEG overlays (original vs recon) for selected channels
- If --use_difumo and V==difumo_dim:
    - Uses NiftiMapsMasker.inverse_transform to write 4D NIfTI for:
        * target fMRI (DiFuMo component time series)
        * reconstructed fMRI (DiFuMo component time series)
    - Saves a few preview PNGs (orthogonal slices) for several timepoints

Notes
-----
- Assumes your test/train model definitions from the repo (dynamic fmri_voxel_embed).
- If your dataset windows aren’t in DiFuMo space (i.e., V != difumo_dim),
  the DiFuMo inverse_transform won’t match; in that case set --use_difumo off
  (you’ll still get EEG plots). We can add an A424→DiFuMo mapper if needed.

Example
-------
python visualize_reconstructions.py \
  --eeg_root ../Oddball/ds116_eeg \
  --fmri_root ../Oddball/ds000116 \
  --a424_label_nii ../BrainLM/A424_resampled_to_bold.nii.gz \
  --cbramod_weights D:/.../CBraMod/pretrained_weights.pth \
  --brainlm_model_dir D:/.../BrainLM/pretrained_models/2023-06-06-22_15_00-checkpoint-1400 \
  --checkpoint translator_runs/odd_both_single_partial_run/checkpoint_best.pt \
  --use_difumo --difumo_dim 256 --out_dir viz_out --device cuda
"""

from __future__ import annotations

import os
import json
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

# nilearn for DiFuMo maps & inverse_transform
from nilearn.maskers import NiftiMapsMasker
from nilearn.image import resample_to_img
from nilearn import plotting
try:
    from nilearn.datasets import fetch_atlas_difumo
    _CAN_FETCH = True
except Exception:
    _CAN_FETCH = False

# ---------- Repo-local imports ----------
THIS_DIR = Path(__file__).parent
REPO_ROOT = THIS_DIR.parent
CBRAMOD_DIR = REPO_ROOT / "CBraMod"
BRAINLM_DIR = REPO_ROOT / "BrainLM"

import sys
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(THIS_DIR / "CBraMod"))
sys.path.insert(0, str(THIS_DIR / "BrainLM"))
sys.path.append(str(CBRAMOD_DIR))
sys.path.append(str(BRAINLM_DIR))

from module import (  # type: ignore
    BidirectionalAdaptiveCompressor,
    ConvEEGInputAdapter,
    fMRIInputAdapterConv1d,
    HierarchicalEncoder,
    CrossAttentionLayer,
    EEGDecodingAdapter,
)
from models.cbramod import CBraMod  # type: ignore
from brainlm_mae.modeling_brainlm import BrainLMForPretraining  # type: ignore
from brainlm_mae.configuration_brainlm import BrainLMConfig     # type: ignore
from data_oddball import (  # type: ignore
    pad_timepoints_for_brainlm_torch,
    load_a424_coords,
    PairedAlignedDataset,
    collate_paired,
    find_eeg_files,
    find_bold_files,
    _parse_sr_from_path,
)

# -----------------------------
# Frozen feature extractors
# -----------------------------
class FrozenCBraMod(nn.Module):
    def __init__(self, in_dim: int, d_model: int, seq_len: int, n_layer: int, nhead: int, dim_feedforward: int,
                 weights_path: Path, device: torch.device) -> None:
        super().__init__()
        self.model = CBraMod(
            in_dim=in_dim, out_dim=in_dim, d_model=d_model,
            seq_len=seq_len, n_layer=n_layer, nhead=nhead, dim_feedforward=dim_feedforward,
        )
        ckpt = torch.load(str(weights_path), map_location=device)
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            self.model.load_state_dict(ckpt['model_state_dict'], strict=False)
        elif isinstance(ckpt, dict) and 'state_dict' in ckpt:
            self.model.load_state_dict(ckpt['state_dict'], strict=False)
        else:
            self.model.load_state_dict(ckpt, strict=False)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.to(device)

    @torch.no_grad()
    def extract_latents(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,C,P,S) → (L,B,P,C,D)
        patch_emb = self.model.patch_embedding(x)
        cur = patch_emb
        outs = []
        for layer in self.model.encoder.layers:
            cur = layer(cur)
            outs.append(cur)
        return torch.stack(outs, dim=0)

class FrozenBrainLM(nn.Module):
    def __init__(self, model_dir: Path, device: torch.device) -> None:
        super().__init__()
        with open(model_dir / "config.json", "r") as f:
            cfg = json.load(f)
        config = BrainLMConfig(**cfg)
        self.model = BrainLMForPretraining(config)
        ckpt = torch.load(str(model_dir / "pytorch_model.bin"), map_location=device)
        self.model.load_state_dict(ckpt, strict=False)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.to(device)

    @torch.no_grad()
    def extract_latents(self, signal_vectors: torch.Tensor, xyz_vectors: torch.Tensor, noise: Optional[torch.Tensor]) -> torch.Tensor:
        embeddings, mask, ids_restore = self.model.vit.embeddings(
            signal_vectors=signal_vectors, xyz_vectors=xyz_vectors, noise=noise
        )
        enc = self.model.vit.encoder(hidden_states=embeddings, output_hidden_states=True, return_dict=True)
        return torch.stack(list(enc.hidden_states), dim=0)  # (L,B,Ttok,H)

# -----------------------------
# Translator (matching train/test)
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
    ) -> None:
        super().__init__()
        self.adapter_eeg = ConvEEGInputAdapter(
            seq_len=eeg_patch_num, n_layers=eeg_n_layers, channels=eeg_channels,
            input_dim=eeg_input_dim, output_dim=d_model,
        )
        self.adapter_fmri = fMRIInputAdapterConv1d(
            seq_len=fmri_tokens_target, n_layers=fmri_n_layers,
            input_dim=fmri_hidden_size, output_dim=d_model, target_seq_len=512,
        )
        self.eeg_encoder = HierarchicalEncoder(d_model, n_heads, d_ff, dropout, n_layers_per_stack=2)
        self.fmri_encoder = HierarchicalEncoder(d_model, n_heads, d_ff, dropout, n_layers_per_stack=2)
        self.cross_attn = CrossAttentionLayer(d_model, n_heads, dropout)
        self.compressor = BidirectionalAdaptiveCompressor()

        self.eeg_decoder = EEGDecodingAdapter(
            channels=eeg_channels, patch_num=eeg_patch_num, n_layers=eeg_n_layers,
            patch_size=eeg_input_dim, d_model=d_model,
        )

        self.fmri_proj = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(d_model * 2, d_model), nn.LayerNorm(d_model)
        )
        self.fmri_depthwise = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=d_model)
        self.fmri_voxel_embed: Optional[nn.Embedding] = None

    def forward(self, eeg_latents, fmri_latents, fmri_target_T: int, fmri_target_V: int):
        eeg_adapt = self.adapter_eeg(eeg_latents)     # (B, Neeg, D)
        fmri_adapt = self.adapter_fmri(fmri_latents)  # (B, Nfmri, D)
        _, eeg_hi = self.eeg_encoder(eeg_adapt, eeg_adapt)
        _, fmr_hi = self.fmri_encoder(fmri_adapt, fmri_adapt)
        eeg_c, fmr_c, _ = self.compressor(eeg_hi, fmr_hi)
        fused = self.cross_attn(eeg_c, fmr_c)         # (B, Tfused, D)

        eeg_layers = self.eeg_decoder(fused)          # (L,B,P,C,S)
        eeg_signal = eeg_layers.mean(dim=0).permute(0, 2, 1, 3).contiguous()  # (B,C,P,S)

        H = self.fmri_proj(fused)                     # (B, Tfused, D)
        Hc = self.fmri_depthwise(H.transpose(1, 2)).transpose(1, 2)  # (B, Tfused, D)
        if Hc.shape[-2] != fmri_target_T:
            Hc = nn.functional.interpolate(Hc.transpose(1,2), size=fmri_target_T, mode='linear', align_corners=False).transpose(1,2)
        if (self.fmri_voxel_embed is None) or (self.fmri_voxel_embed.num_embeddings != fmri_target_V) or (self.fmri_voxel_embed.embedding_dim != Hc.shape[-1]):
            self.fmri_voxel_embed = nn.Embedding(fmri_target_V, Hc.shape[-1]).to(Hc.device)
        E = self.fmri_voxel_embed.weight
        fmri_signal = torch.matmul(Hc, E.t())         # (B,T,V)
        return eeg_signal, fmri_signal

# -----------------------------
# Helpers: EEG grouping
# -----------------------------
def group_eeg_latents_seconds(eeg_latents: torch.Tensor, seconds_per_token: int) -> torch.Tensor:
    L, B, P, C, D = eeg_latents.shape
    g = max(1, min(seconds_per_token, P))
    P_grp = max(1, P // g)
    return eeg_latents[:, :, :P_grp*g].reshape(L, B, P_grp, g, C, D).mean(dim=3)

def group_eeg_signal_seconds(x_eeg_sig: torch.Tensor, seconds_per_token: int) -> torch.Tensor:
    B, C, P, S = x_eeg_sig.shape
    g = max(1, min(seconds_per_token, P))
    P_grp = max(1, P // g)
    return x_eeg_sig[:, :, :P_grp*g, :].reshape(B, C, P_grp, g, S).mean(dim=3)

# -----------------------------
# Viz utils
# -----------------------------
def plot_eeg_overlays(x_true: torch.Tensor, x_recon: torch.Tensor, out_dir: Path, channels: List[int] = [0,1,2,3], max_seconds: Optional[int]=None):
    """
    x_*: (B,C,Pe,S) grouped EEG. We'll plot first sample, a few channels, flattening Pe*S to seconds.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    b = 0
    C = x_true.shape[1]
    chans = [c for c in channels if c < C]
    true_flat = x_true[b].permute(0,1,2).reshape(C, -1).cpu().numpy()  # (C, Pe*S)
    recon_flat = x_recon[b].permute(0,1,2).reshape(C, -1).cpu().numpy()

    for c in chans:
        T = true_flat.shape[1]
        if max_seconds is not None:
            T = min(T, max_seconds)
        plt.figure(figsize=(10, 3))
        plt.plot(true_flat[c, :T], label="EEG true")
        plt.plot(recon_flat[c, :T], label="EEG recon", alpha=0.8)
        plt.title(f"EEG Channel {c} — true vs recon")
        plt.xlabel("Timepoints")
        plt.ylabel("Amplitude (z-sc)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"eeg_overlay_ch{c:02d}.png", dpi=150)
        plt.close()

def save_difumo_niftis(
    difumo_maps: Path,
    ts_true: torch.Tensor,    # (B,T,V) with V == n_components
    ts_recon: torch.Tensor,   # (B,T,V)
    out_dir: Path,
    tr: float = 2.0,
    preview_times: List[int] = [0, -1, None],  # first, last, mid
) -> bool:
    """
    Writes 4D NIfTIs for first batch element (B0) using DiFuMo maps.
    Also saves a few stat-map previews at selected timepoints.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    masker = NiftiMapsMasker(maps_img=str(difumo_maps), standardize=False)
    masker.fit()

    # Validate component dimension matches time series V
    try:
        import nibabel as nib
        maps_img = nib.load(str(difumo_maps))
        n_components = maps_img.shape[-1] if maps_img.ndim == 4 else 1
    except Exception:
        n_components = ts_true.shape[-1]
    if int(ts_true.shape[-1]) != int(n_components):
        print(f"[SKIP] DiFuMo inverse_transform skipped: ts V={int(ts_true.shape[-1])} != maps comps={int(n_components)}")
        return False

    b0_true = ts_true[0].cpu().numpy()   # (T,V)
    b0_reco = ts_recon[0].cpu().numpy()  # (T,V)

    # inverse_transform expects (n_samples, n_components); we treat samples as time
    img_true = masker.inverse_transform(b0_true)   # 4D
    img_reco = masker.inverse_transform(b0_reco)

    # Set TR if possible
    try:
        hdr = img_true.header
        zooms = list(hdr.get_zooms())
        if len(zooms) == 4:
            zooms = zooms[:3] + [tr]
            hdr.set_zooms(zooms)
        hdr2 = img_reco.header
        zooms2 = list(hdr2.get_zooms())
        if len(zooms2) == 4:
            zooms2 = zooms2[:3] + [tr]
            hdr2.set_zooms(zooms2)
    except Exception:
        pass

    import nibabel as nib
    nib.save(img_true, str(out_dir / "difumo_target_4D.nii.gz"))
    nib.save(img_reco, str(out_dir / "difumo_recon_4D.nii.gz"))

    # Previews
    # Convert negative or None indices
    T = b0_true.shape[0]
    times = []
    for t in preview_times:
        if t is None:
            times.append(T // 2)
        elif t < 0:
            times.append(max(0, T + t))
        else:
            times.append(min(T-1, t))

    for t in times:
        # build single-volume images by taking time slice
        vol_true = masker.inverse_transform(b0_true[t:t+1, :])  # 3D
        vol_reco = masker.inverse_transform(b0_reco[t:t+1, :])

        # Save PNG previews
        fig = plotting.plot_stat_map(vol_true, display_mode='ortho', title=f"DiFuMo TRUE @t={t}", annotate=False)
        fig.savefig(str(out_dir / f"difumo_true_t{t:04d}.png"), dpi=150); fig.close()
        fig = plotting.plot_stat_map(vol_reco, display_mode='ortho', title=f"DiFuMo RECON @t={t}", annotate=False)
        fig.savefig(str(out_dir / f"difumo_recon_t{t:04d}.png"), dpi=150); fig.close()
    return True


def build_a424_to_difumo_weights(a424_label_nii: Path, difumo_maps_nii: Path) -> np.ndarray:
    """
    Build a 424 x K weight matrix mapping A424 parcel time series to DiFuMo component time series
    by averaging each DiFuMo map within each A424 parcel.
    """
    import nibabel as nib
    atlas_img = nib.load(str(a424_label_nii))
    maps_img = nib.load(str(difumo_maps_nii))
    # Resample DiFuMo maps to atlas grid
    try:
        maps_rs = resample_to_img(maps_img, atlas_img, interpolation='continuous', force_resample=True, copy_header=True)
    except TypeError:
        maps_rs = resample_to_img(maps_img, atlas_img, interpolation='continuous')
    atlas = atlas_img.get_fdata()
    maps = maps_rs.get_fdata()
    if maps.ndim == 3:
        maps = maps[..., None]
    K = maps.shape[-1]
    W = np.zeros((424, K), dtype=np.float64)
    # For each parcel, compute mean of each component within parcel voxels
    for i in range(424):
        roi = i + 1
        mask = (atlas == roi)
        if not np.any(mask):
            continue
        vals = maps[mask, :]  # (N, K)
        # Mean absolute value is more robust than signed mean
        W[i, :] = np.mean(np.abs(vals), axis=0)
    # Normalize rows to sum to 1 to form a convex combination
    row_sums = W.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0.0] = 1.0
    W = W / row_sums
    return W


def save_difumo_timeseries(ts_true_k: torch.Tensor, ts_recon_k: torch.Tensor, out_dir: Path) -> None:
    """
    Save DiFuMo component time series (B,T,K) as .npy for true and recon.
    Also save small line overlays for first few components.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(str(out_dir / 'difumo_time_series_true.npy'), ts_true_k.detach().cpu().numpy())
    np.save(str(out_dir / 'difumo_time_series_recon.npy'), ts_recon_k.detach().cpu().numpy())

    # Plot first 10 components for B0
    b0_true = ts_true_k[0].detach().cpu().numpy()
    b0_reco = ts_recon_k[0].detach().cpu().numpy()
    import matplotlib.pyplot as plt
    nplot = min(10, b0_true.shape[1])
    for k in range(nplot):
        plt.figure(figsize=(10,3))
        plt.plot(b0_true[:, k], label='true')
        plt.plot(b0_reco[:, k], label='recon', alpha=0.8)
        plt.title(f'DiFuMo component {k} — true vs recon')
        plt.xlabel('Time (volumes)'); plt.ylabel('Amplitude (z-sc)')
        plt.legend(); plt.tight_layout()
        plt.savefig(out_dir / f'difumo_overlay_comp{k:03d}.png', dpi=150)
        plt.close()


def plot_fmri_overlays(ts_true: torch.Tensor, ts_recon: torch.Tensor, out_dir: Path, rois: List[int] = [0,1,2,3,4]):
    """
    Save line plots of fMRI time series for selected ROIs from the first batch element.
    ts_*: (B,T,V)
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    b0_true = ts_true[0].detach().cpu().numpy()  # (T,V)
    b0_reco = ts_recon[0].detach().cpu().numpy()
    T, V = b0_true.shape
    sel = [r for r in rois if 0 <= r < V]
    import matplotlib.pyplot as plt
    for r in sel:
        plt.figure(figsize=(10,3))
        plt.plot(b0_true[:, r], label='fMRI true')
        plt.plot(b0_reco[:, r], label='fMRI recon', alpha=0.8)
        plt.title(f'fMRI ROI {r} — true vs recon')
        plt.xlabel('Time (volumes)')
        plt.ylabel('Z-scored signal')
        plt.legend(); plt.tight_layout()
        plt.savefig(out_dir / f'fmri_overlay_roi{r:03d}.png', dpi=150)
        plt.close()


def plot_fmri_heatmaps(ts_true: torch.Tensor, ts_recon: torch.Tensor, out_dir: Path):
    """
    Save heatmaps (T x V) for first batch element: target and recon.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    b0_true = ts_true[0].detach().cpu().numpy()  # (T,V)
    b0_reco = ts_recon[0].detach().cpu().numpy()
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12,4))
    plt.imshow(b0_true.T, aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar(); plt.title('fMRI target (T x V)'); plt.xlabel('Time'); plt.ylabel('ROI')
    plt.tight_layout(); plt.savefig(out_dir / 'fmri_target_heatmap.png', dpi=150); plt.close()
    plt.figure(figsize=(12,4))
    plt.imshow(b0_reco.T, aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar(); plt.title('fMRI recon (T x V)'); plt.xlabel('Time'); plt.ylabel('ROI')
    plt.tight_layout(); plt.savefig(out_dir / 'fmri_recon_heatmap.png', dpi=150); plt.close()

# -----------------------------
# Config and main
# -----------------------------
@dataclass
class VizConfig:
    eeg_root: Path
    fmri_root: Path
    a424_label_nii: Path
    cbramod_weights: Path
    brainlm_model_dir: Path
    checkpoint: Path
    out_dir: Path
    device: str = "cuda"
    seed: int = 42
    fmri_norm: str = "zscore"
    window_sec: int = 30
    original_fs: int = 1000
    target_fs: int = 200
    stride_sec: Optional[int] = None
    channels_limit: int = 34
    batch_size: int = 4
    num_workers: int = 0
    eeg_seconds_per_token: int = 40
    use_difumo: bool = False
    difumo_dim: int = 256
    difumo_maps_nii: Optional[Path] = None
    tr: float = 2.0
    eeg_channels_to_plot: str = "0,1,2,3"

def build_models(cfg: VizConfig, device: torch.device):
    # Frozen extractors
    seq_len_eeg = cfg.window_sec
    frozen_eeg = FrozenCBraMod(
        in_dim=cfg.target_fs, d_model=cfg.target_fs, seq_len=seq_len_eeg,
        n_layer=12, nhead=8, dim_feedforward=800,
        weights_path=cfg.cbramod_weights, device=device
    )
    frozen_fmri = FrozenBrainLM(cfg.brainlm_model_dir, device=device)

    # Translator (must match training)
    eeg_group = max(1, int(cfg.eeg_seconds_per_token))
    eeg_patch_num_grouped = max(1, int(cfg.window_sec) // eeg_group)
    translator = TranslatorModel(
        eeg_channels=cfg.channels_limit,
        eeg_patch_num=eeg_patch_num_grouped,
        eeg_n_layers=12,
        eeg_input_dim=cfg.target_fs,
        fmri_n_layers=5,
        fmri_hidden_size=256,
        fmri_tokens_target=424 * int(round(cfg.window_sec / 2.0)),  # matches your train/test
        d_model=256,
        n_heads=8,
        d_ff=1024,
        dropout=0.1,
    ).to(device)

    # Load translator weights (filter unexpected keys like fmri_voxel_embed.weight)
    try:
        ckpt = torch.load(str(cfg.checkpoint), map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(str(cfg.checkpoint), map_location=device)
    state = ckpt.get("translator_state", ckpt)
    current = translator.state_dict()
    filtered = {k: v for k, v in state.items()
                if k in current and getattr(current[k], 'shape', None) == getattr(v, 'shape', None)}
    translator.load_state_dict(filtered, strict=False)
    translator.eval()

    return frozen_eeg, frozen_fmri, translator

def maybe_get_difumo_maps(cfg: VizConfig) -> Optional[Path]:
    if not cfg.use_difumo:
        return None
    if cfg.difumo_maps_nii and Path(cfg.difumo_maps_nii).exists():
        return cfg.difumo_maps_nii
    if not _CAN_FETCH:
        raise FileNotFoundError("DiFuMo maps not provided and nilearn fetch not available. Pass --difumo_maps_nii.")
    atlas = fetch_atlas_difumo(dimension=cfg.difumo_dim)
    return Path(atlas.maps)

def _rand_block_mask_time(B: int, L: int, vis_frac: float, device) -> torch.Tensor:
    """Return (B,L) True=MASKED with a single visible contiguous block of length vis_frac*L."""
    M = torch.ones(B, L, dtype=torch.bool, device=device)
    vis = max(1, int(round(vis_frac * L))); vis = min(vis, L)
    if L > vis:
        starts = torch.randint(0, L - vis + 1, (B,), device=device)
    else:
        starts = torch.zeros(B, dtype=torch.long, device=device)
    for i in range(B):
        M[i, starts[i]:starts[i]+vis] = False
    return M


def run_once(cfg: VizConfig, subset: str = "test", mode: str = "both", partial_visible_frac: float = 0.5):
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

    device = torch.device(cfg.device)
    out_dir = cfg.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build subject-run split (fixed subjects from config if provided; else 70/10/20)
    eeg_files = find_eeg_files(cfg.eeg_root)
    fmri_files = find_bold_files(cfg.fmri_root)
    key_to_eeg, key_to_fmri = {}, {}
    for p in eeg_files:
        sr = _parse_sr_from_path(p)
        if all(sr):
            key_to_eeg[(sr[0], sr[1])] = p
    for p in fmri_files:
        sr = _parse_sr_from_path(p)
        if all(sr):
            key_to_fmri[(sr[0], sr[1])] = p
    inter_keys = sorted(set(key_to_eeg.keys()) & set(key_to_fmri.keys()))
    if len(inter_keys) == 0:
        raise RuntimeError("No aligned (subject,run) pairs found; check paths.")
    def to_set(xs):
        return set(str(int(s)) for s in (xs or []))
    # Try to read fixed split from a sidecar YAML if present (same base path as checkpoint)
    fixed_train, fixed_val, fixed_test = set(), set(), set()
    try:
        sidecar = Path(str(cfg.checkpoint) + '.splits.yaml')
        if sidecar.exists():
            import yaml  # type: ignore
            with open(sidecar, 'r') as f:
                d = yaml.safe_load(f) or {}
            fixed_train = to_set(d.get('train_subjects'))
            fixed_val   = to_set(d.get('val_subjects'))
            fixed_test  = to_set(d.get('test_subjects'))
    except Exception:
        pass

    if fixed_train or fixed_val or fixed_test:
        train_keys = tuple(k for k in inter_keys if k[0] in fixed_train)
        val_keys   = tuple(k for k in inter_keys if k[0] in fixed_val)
        test_keys  = tuple(k for k in inter_keys if k[0] in fixed_test)
    else:
        rng = np.random.default_rng(cfg.seed)
        indices = np.arange(len(inter_keys))
        rng.shuffle(indices)
        n_train = int(len(indices) * 0.7)
        n_val = int(len(indices) * 0.1)
        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]
        test_idx = indices[n_train + n_val:]
        train_keys = tuple(inter_keys[i] for i in train_idx)
        val_keys = tuple(inter_keys[i] for i in val_idx)
        test_keys = tuple(inter_keys[i] for i in test_idx)

    include_sr = None
    if subset == 'train':
        include_sr = train_keys
    elif subset == 'val':
        include_sr = val_keys
    elif subset == 'test':
        include_sr = test_keys
    elif subset == 'all':
        include_sr = None

    print("include_sr", include_sr)
    # Data restricted to chosen subset
    ds = PairedAlignedDataset(
        eeg_root=cfg.eeg_root,
        fmri_root=cfg.fmri_root,
        a424_label_nii=cfg.a424_label_nii,
        window_sec=cfg.window_sec,
        original_fs=cfg.original_fs,
        target_fs=cfg.target_fs,
        tr=cfg.tr,
        channels_limit=cfg.channels_limit,
        fmri_norm=cfg.fmri_norm,
        stride_sec=cfg.stride_sec,
        device='cpu',
        include_sr_keys=include_sr,
    )
    if len(ds) == 0:
        raise RuntimeError("No samples in dataset; check paths.")

    dl = DataLoader(
        ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, collate_fn=collate_paired, pin_memory=(device.type=='cuda')
    )

    frozen_eeg, frozen_fmri, translator = build_models(cfg, device)
    difumo_maps = maybe_get_difumo_maps(cfg) if cfg.use_difumo else None

    # Get one batch
    batch = next(iter(dl))
    x_eeg = batch['eeg_window'].to(device)   # (B,C,P,S)
    fmri_t = batch['fmri_window'].to(device) # (B,T,V)
    B, C, P, S = x_eeg.shape
    _, T, V = fmri_t.shape

    # Warn if DiFuMo requested but V mismatch
    if cfg.use_difumo and V != cfg.difumo_dim:
        print(f"[WARN] V={V} in batch but DiFuMo dim={cfg.difumo_dim}. "
              f"Inverse transform will be invalid. "
              f"Make sure your dataset windows are in DiFuMo space.")

    # Observed inputs based on tri-mix mode
    x_eeg_obs = x_eeg.clone()
    fmri_obs  = fmri_t.clone()
    if mode == "both":
        pass
    elif mode == "eeg2fmri":
        fmri_obs[:] = 0.0
    elif mode == "fmri2eeg":
        x_eeg_obs[:] = 0.0
    elif mode == "partial_eeg":
        M_eeg_time = _rand_block_mask_time(B, P, partial_visible_frac, device)
        for b in range(B):
            x_eeg_obs[b, :, M_eeg_time[b], :] = 0.0
    elif mode == "partial_fmri":
        M_t = _rand_block_mask_time(B, T, partial_visible_frac, device)
        M_fmri = M_t.unsqueeze(-1).expand(-1, T, V)
        fmri_obs[M_fmri] = 0.0
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Frozen latents
    with torch.no_grad():
        eeg_latents = frozen_eeg.extract_latents(x_eeg_obs)  # (L,B,P,C,D)

        fmri_padded = pad_timepoints_for_brainlm_torch(fmri_obs, patch_size=20)  # (B,Tp,V)
        signal_vectors = fmri_padded.permute(0,2,1).contiguous()                 # (B,V,Tp)

        # coords (normalized A424 if V==424 else zeros)
        try:
            cand = [
                THIS_DIR / 'BrainLM' / 'resources' / 'atlases' / 'A424_Coordinates.dat',
                THIS_DIR / 'resources' / 'atlases' / 'A424_Coordinates.dat',
                REPO_ROOT / 'BrainLM' / 'toolkit' / 'atlases' / 'A424_Coordinates.dat',
            ]
            for pth in cand:
                if pth.exists():
                    a424_dat = pth
                    break
            else:
                a424_dat = cand[-1]
            coords_np = load_a424_coords(a424_dat)
            coords_np = coords_np / (np.max(np.abs(coords_np)) or 1.0)
            if V == 424:
                xyz = torch.from_numpy(coords_np).to(device=device, dtype=torch.float32)[None, ...].repeat(B,1,1)
            else:
                xyz = torch.zeros(B, V, 3, device=device)
        except Exception:
            xyz = torch.zeros(B, V, 3, device=device)

        fmri_latents = frozen_fmri.extract_latents(signal_vectors, xyz, noise=None)  # (L,B,Ttok,H)

    # Group EEG to match decoder
    eeg_latents_t = group_eeg_latents_seconds(eeg_latents, cfg.eeg_seconds_per_token)
    x_eeg_grp     = group_eeg_signal_seconds(x_eeg, cfg.eeg_seconds_per_token)

    with torch.no_grad():
        recon_eeg, recon_fmri = translator(
            eeg_latents_t, fmri_latents,
            fmri_target_T=int(fmri_t.shape[1]),
            fmri_target_V=int(fmri_t.shape[2]),
        )

    # ---- Save EEG overlays
    chans = [int(c.strip()) for c in cfg.eeg_channels_to_plot.split(",") if c.strip().isdigit()]
    plot_eeg_overlays(x_eeg_grp, recon_eeg, out_dir / "eeg_plots", channels=chans, max_seconds=None)
    print(f"[OK] EEG overlays saved to {out_dir/'eeg_plots'}")

    # ---- DiFuMo component time series + NIfTIs (+ previews) if requested; keep generic plots too
    if cfg.use_difumo:
        if difumo_maps is None or not Path(difumo_maps).exists():
            raise FileNotFoundError("DiFuMo maps not found. Pass --difumo_maps_nii or allow nilearn to fetch.")
        ok = False
        if fmri_t.shape[-1] == cfg.difumo_dim:
            # Already in DiFuMo space: save time series and NIfTIs
            save_difumo_timeseries(fmri_t, recon_fmri, out_dir / 'difumo_timeseries')
            ok = save_difumo_niftis(
                difumo_maps=Path(difumo_maps),
                ts_true=fmri_t, ts_recon=recon_fmri,
                out_dir=out_dir / "difumo_niftis",
                tr=cfg.tr,
                preview_times=[0, None, -1],
            )
        else:
            # Build A424→DiFuMo projection and convert both target and recon to DiFuMo components
            try:
                W = build_a424_to_difumo_weights(cfg.a424_label_nii, Path(difumo_maps))  # (424,K)
                if W.shape[1] != cfg.difumo_dim:
                    print(f"[WARN] DiFuMo maps dim={W.shape[1]} != requested {cfg.difumo_dim}; proceeding with maps dim")
                # Project (B,T,424) x (424,K) -> (B,T,K)
                fmri_true_k = torch.tensor(W, dtype=fmri_t.dtype, device=fmri_t.device)  # (424,K)
                fmri_true_d = torch.matmul(fmri_t, fmri_true_k)        # (B,T,K)
                fmri_reco_d = torch.matmul(recon_fmri, fmri_true_k)    # (B,T,K)
                save_difumo_timeseries(fmri_true_d, fmri_reco_d, out_dir / 'difumo_timeseries')
                ok = save_difumo_niftis(
                    difumo_maps=Path(difumo_maps),
                    ts_true=fmri_true_d, ts_recon=fmri_reco_d,
                    out_dir=out_dir / "difumo_niftis",
                    tr=cfg.tr,
                    preview_times=[0, None, -1],
                )
            except Exception as e:
                print(f"[WARN] A424→DiFuMo projection failed: {e}")
        if ok:
            print(f"[OK] DiFuMo NIfTIs + previews saved to {out_dir/'difumo_niftis'}")
        else:
            # Still save generic fMRI plots
            plot_fmri_overlays(fmri_t, recon_fmri, out_dir / 'fmri_plots')
            plot_fmri_heatmaps(fmri_t, recon_fmri, out_dir / 'fmri_plots')
            print(f"[OK] Saved generic fMRI plots to {out_dir/'fmri_plots'}")
    else:
        plot_fmri_overlays(fmri_t, recon_fmri, out_dir / 'fmri_plots')
        plot_fmri_heatmaps(fmri_t, recon_fmri, out_dir / 'fmri_plots')
        print(f"[OK] Saved generic fMRI plots to {out_dir/'fmri_plots'}")

def main():
    ap = argparse.ArgumentParser(description="Visualize original vs reconstructed EEG and fMRI (DiFuMo), with tri-mix masking modes.")
    ap.add_argument('--eeg_root', type=str, required=True)
    ap.add_argument('--fmri_root', type=str, required=True)
    ap.add_argument('--a424_label_nii', type=str, required=True)
    ap.add_argument('--cbramod_weights', type=str, required=True)
    ap.add_argument('--brainlm_model_dir', type=str, required=True)
    ap.add_argument('--checkpoint', type=str, required=True)
    ap.add_argument('--out_dir', type=str, default=str(THIS_DIR / "viz_out"))
    ap.add_argument('--device', type=str, default='cuda')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--fmri_norm', type=str, default='zscore', choices=['zscore','psc','mad','none'])
    ap.add_argument('--window_sec', type=int, default=30)
    ap.add_argument('--original_fs', type=int, default=1000)
    ap.add_argument('--target_fs', type=int, default=200)
    ap.add_argument('--stride_sec', type=int, default=None)
    ap.add_argument('--channels_limit', type=int, default=34)
    ap.add_argument('--batch_size', type=int, default=4)
    ap.add_argument('--num_workers', type=int, default=0)
    ap.add_argument('--eeg_seconds_per_token', type=int, default=40)
    ap.add_argument('--tr', type=float, default=2.0)
    ap.add_argument('--eeg_channels_to_plot', type=str, default="0,1,2,3")
    ap.add_argument('--mode', type=str, default='both', choices=['both','eeg2fmri','fmri2eeg','partial_eeg','partial_fmri'])
    ap.add_argument('--partial_visible_frac', type=float, default=0.5)
    ap.add_argument('--subset', type=str, default='test', choices=['train','val','test','all'], help='Which subject-run split to visualize')

    # DiFuMo controls
    ap.add_argument('--use_difumo', action='store_true')
    ap.add_argument('--difumo_dim', type=int, default=256, choices=[64,128,256,512,1024])
    ap.add_argument('--difumo_maps_nii', type=str, default=None, help="Local path to DiFuMo maps NIfTI. If absent and internet available, nilearn will fetch.")

    args = ap.parse_args()
    cfg = VizConfig(
        eeg_root=Path(args.eeg_root),
        fmri_root=Path(args.fmri_root),
        a424_label_nii=Path(args.a424_label_nii),
        cbramod_weights=Path(args.cbramod_weights),
        brainlm_model_dir=Path(args.brainlm_model_dir),
        checkpoint=Path(args.checkpoint),
        out_dir=Path(args.out_dir),
        device=args.device,
        seed=args.seed,
        fmri_norm=args.fmri_norm,
        window_sec=args.window_sec,
        original_fs=args.original_fs,
        target_fs=args.target_fs,
        stride_sec=args.stride_sec,
        channels_limit=args.channels_limit,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        eeg_seconds_per_token=args.eeg_seconds_per_token,
        use_difumo=bool(args.use_difumo),
        difumo_dim=int(args.difumo_dim),
        difumo_maps_nii=(Path(args.difumo_maps_nii) if args.difumo_maps_nii else None),
        tr=float(args.tr),
        eeg_channels_to_plot=args.eeg_channels_to_plot,
    )

    # Basic checks
    for p in [cfg.eeg_root, cfg.fmri_root, cfg.a424_label_nii, cfg.cbramod_weights, cfg.brainlm_model_dir, cfg.checkpoint]:
        if p is None or not Path(p).exists():
            raise FileNotFoundError(f"Missing path: {p}")

    if cfg.use_difumo and cfg.difumo_maps_nii is not None and not cfg.difumo_maps_nii.exists():
        raise FileNotFoundError(f"DiFuMo maps not found: {cfg.difumo_maps_nii}")

    run_once(cfg, subset=args.subset, mode=args.mode, partial_visible_frac=args.partial_visible_frac)

if __name__ == "__main__":
    main()
# python test_translator_paired.py --config configs\translator.yaml --mode fmri2eeg --device cuda
#  python test_translator_paired.py --config configs\translator.yaml --mode eeg2fmri --device cuda

# --mode: both | eeg2fmri | fmri2eeg | partial_eeg | partial_fmri
# --partial_visible_frac: fraction of time visible in partial modes (default 0.5)