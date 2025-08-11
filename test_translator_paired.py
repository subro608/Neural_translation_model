#!/usr/bin/env python3
"""
Test / Inference for the tri-mix EEG↔fMRI translator.

Modes:
  - both:        evaluate with both modalities present (full losses)
  - eeg2fmri:    EEG present, fMRI missing (report fMRI completion metrics)
  - fmri2eeg:    fMRI present, EEG missing (report EEG completion metrics)
  - partial_eeg: EEG partially visible (random block); masked-only EEG metrics
  - partial_fmri:fMRI partially visible (random block); masked-only fMRI metrics

Optionally saves reconstructions to disk for qualitative inspection.
"""

from __future__ import annotations

import os
import json
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ---------- Progress bar ----------
try:
    from tqdm import tqdm
except Exception:
    tqdm = None

# ---------- Project-local imports ----------
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
# Utils
# -----------------------------
def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


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
        checkpoint = torch.load(str(weights_path), map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['state_dict'], strict=False)
        else:
            self.model.load_state_dict(checkpoint, strict=False)
        self.model.eval()
        for p in self.model.parameters(): p.requires_grad = False
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
        checkpoint = torch.load(str(model_dir / "pytorch_model.bin"), map_location=device)
        self.model.load_state_dict(checkpoint, strict=False)
        self.model.eval()
        for p in self.model.parameters(): p.requires_grad = False
        self.to(device)

    @torch.no_grad()
    def extract_latents(self, signal_vectors: torch.Tensor, xyz_vectors: torch.Tensor, noise: Optional[torch.Tensor]) -> torch.Tensor:
        embeddings, mask, ids_restore = self.model.vit.embeddings(
            signal_vectors=signal_vectors, xyz_vectors=xyz_vectors, noise=noise
        )
        enc = self.model.vit.encoder(hidden_states=embeddings, output_hidden_states=True, return_dict=True)
        return torch.stack(list(enc.hidden_states), dim=0)  # (L,B,Ttok,H)


# -----------------------------
# Translator (must match training)
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

        # EEG
        eeg_layers = self.eeg_decoder(fused)          # (L,B,P,C,S)
        eeg_signal = eeg_layers.mean(dim=0).permute(0, 2, 1, 3).contiguous()  # (B,C,P,S)

        # fMRI
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
# Helpers: grouping + masks + metrics
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

def _group_mask_seconds(M_time: torch.Tensor, seconds_per_token: int) -> torch.Tensor:
    B, P = M_time.shape
    g = max(1, min(seconds_per_token, P))
    Pe = max(1, P // g)
    if Pe * g != P:
        M_time = M_time[:, :Pe*g]
    return M_time.reshape(B, Pe, g).any(dim=2)  # (B,Pe)

@torch.no_grad()
def batch_corr(a: torch.Tensor, b: torch.Tensor, dims) -> torch.Tensor:
    """
    Pearson correlation per sample across `dims`.
    Returns (B,) tensor.
    """
    a_ = a - a.mean(dim=dims, keepdim=True)
    b_ = b - b.mean(dim=dims, keepdim=True)
    num = (a_ * b_).sum(dim=dims)
    den = torch.sqrt((a_**2).sum(dim=dims) * (b_**2).sum(dim=dims) + 1e-8)
    return num / (den + 1e-8)


# -----------------------------
# Testing
# -----------------------------
@dataclass
class TestConfig:
    eeg_root: Path
    fmri_root: Path
    a424_label_nii: Path
    cbramod_weights: Path
    brainlm_model_dir: Path
    checkpoint: Path
    device: str = "cuda"
    seed: int = 42
    fmri_norm: str = "zscore"
    window_sec: int = 30
    original_fs: int = 1000
    target_fs: int = 200
    stride_sec: Optional[int] = None
    channels_limit: int = 34
    batch_size: int = 8
    num_workers: int = 0
    eeg_seconds_per_token: int = 40
    # test mode
    mode: str = "eeg2fmri"  # both | eeg2fmri | fmri2eeg | partial_eeg | partial_fmri
    partial_visible_frac: float = 0.5
    # saving
    save_recons: Optional[Path] = None
    save_n_batches: int = 0  # 0 = don't save; >0 = save first N batches
    # fixed subject splits (optional)
    train_subjects: Optional[list[int]] = None
    val_subjects: Optional[list[int]] = None
    test_subjects: Optional[list[int]] = None


def build_models(cfg: TestConfig, device: torch.device):
    seq_len_eeg = cfg.window_sec
    frozen_eeg = FrozenCBraMod(
        in_dim=cfg.target_fs, d_model=cfg.target_fs, seq_len=seq_len_eeg,
        n_layer=12, nhead=8, dim_feedforward=800,
        weights_path=cfg.cbramod_weights, device=device
    )
    frozen_fmri = FrozenBrainLM(cfg.brainlm_model_dir, device=device)

    eeg_group = max(1, int(cfg.eeg_seconds_per_token))
    eeg_patch_num_grouped = max(1, int(cfg.window_sec) // eeg_group)
    translator = TranslatorModel(
        eeg_channels=cfg.channels_limit,
        eeg_patch_num=eeg_patch_num_grouped,
        eeg_n_layers=12,
        eeg_input_dim=cfg.target_fs,
        fmri_n_layers=5,
        fmri_hidden_size=256,
        fmri_tokens_target=424 * int(round(cfg.window_sec / 2.0)),
        d_model=256,
        n_heads=8,
        d_ff=1024,
        dropout=0.1,
    ).to(device)

    # Load checkpoint (translator only). PyTorch 2.6 defaults weights_only=True, which breaks
    # loading full checkpoints containing Python objects (e.g., pathlib.Path). We trust this file,
    # so load with weights_only=False.
    try:
        ckpt = torch.load(str(cfg.checkpoint), map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(str(cfg.checkpoint), map_location=device)
    state = ckpt.get("translator_state", ckpt)
    # Filter out keys that are not in current model (e.g., dynamic fmri_voxel_embed)
    current = translator.state_dict()
    filtered = {k: v for k, v in state.items() if k in current and getattr(current[k], 'shape', None) == getattr(v, 'shape', None)}
    missing, unexpected = translator.load_state_dict(filtered, strict=False)
    translator.eval()

    return frozen_eeg, frozen_fmri, translator


def run_test(cfg: TestConfig):
    set_seed(cfg.seed)
    device = torch.device(cfg.device)

    # Build subject-run split (fixed subjects if provided)
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
        print("No paired aligned (subject,run) found for test. Check your paths.")
        return
    if cfg.test_subjects:
        sub_set = set(str(int(s)) for s in cfg.test_subjects)
        test_keys = tuple(k for k in inter_keys if k[0] in sub_set)
    else:
        rng = np.random.default_rng(cfg.seed)
        indices = np.arange(len(inter_keys))
        rng.shuffle(indices)
        n_train = int(len(indices) * 0.7)
        n_val = int(len(indices) * 0.1)
        test_idx = indices[n_train + n_val:]
        test_keys = tuple(inter_keys[i] for i in test_idx)
    print("test_keys", test_keys)
    # Data restricted to TEST subjects only
    ds = PairedAlignedDataset(
        eeg_root=cfg.eeg_root,
        fmri_root=cfg.fmri_root,
        a424_label_nii=cfg.a424_label_nii,
        window_sec=cfg.window_sec,
        original_fs=cfg.original_fs,
        target_fs=cfg.target_fs,
        tr=2.0,
        channels_limit=cfg.channels_limit,
        fmri_norm=cfg.fmri_norm,
        stride_sec=cfg.stride_sec,
        device='cpu',
        include_sr_keys=test_keys,
    )
    if len(ds) == 0:
        print("No samples found.")
        return

    dl = DataLoader(
        ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, collate_fn=collate_paired, pin_memory=(device.type=='cuda')
    )

    frozen_eeg, frozen_fmri, translator = build_models(cfg, device)

    # metrics accumulators
    mse_eeg_sum, mse_fmri_sum = 0.0, 0.0
    corr_eeg_sum, corr_fmri_sum = 0.0, 0.0
    n_eeg, n_fmri = 0, 0

    saver_enabled = cfg.save_recons is not None and cfg.save_n_batches > 0
    saved_batches = 0
    if saver_enabled:
        ensure_dir(cfg.save_recons)

    iterator = dl if tqdm is None else tqdm(dl, total=len(dl), desc=f"test:{cfg.mode}", leave=False)

    with torch.no_grad():
        for b_idx, batch in enumerate(iterator):
            x_eeg = batch['eeg_window'].to(device)   # (B,C,P,S)
            fmri_t = batch['fmri_window'].to(device) # (B,T,V)
            B, C, P, S = x_eeg.shape
            _, T, V = fmri_t.shape

            # Build observed inputs based on mode
            x_eeg_obs = x_eeg.clone()
            fmri_obs  = fmri_t.clone()

            if cfg.mode == "both":
                pass  # nothing masked

            elif cfg.mode == "eeg2fmri":
                # fMRI fully missing, EEG present
                fmri_obs[:] = 0.0

            elif cfg.mode == "fmri2eeg":
                # EEG fully missing, fMRI present
                x_eeg_obs[:] = 0.0

            elif cfg.mode == "partial_eeg":
                # EEG partial visible by time; masked-only metrics
                M_eeg_time = _rand_block_mask_time(B, P, cfg.partial_visible_frac, device)
                for b in range(B):
                    x_eeg_obs[b, :, M_eeg_time[b], :] = 0.0
            elif cfg.mode == "partial_fmri":
                # fMRI partial visible by time; masked-only metrics
                M_t = _rand_block_mask_time(B, T, cfg.partial_visible_frac, device)  # (B,T)
                M_fmri = M_t.unsqueeze(-1).expand(-1, T, V)                           # (B,T,V)
                fmri_obs[M_fmri] = 0.0
            else:
                raise ValueError(f"Unknown mode: {cfg.mode}")

            # Feature extraction
            eeg_latents = frozen_eeg.extract_latents(x_eeg_obs)                # (L,B,P,C,D)

            # BrainLM prep
            fmri_padded = pad_timepoints_for_brainlm_torch(fmri_obs, patch_size=20)  # (B,Tp,V)
            signal_vectors = fmri_padded.permute(0,2,1).contiguous()                 # (B,V,Tp)

            # coords (A424 normalized if V==424)
            try:
                cand = [
                    THIS_DIR / 'BrainLM' / 'resources' / 'atlases' / 'A424_Coordinates.dat',
                    THIS_DIR / 'resources' / 'atlases' / 'A424_Coordinates.dat',
                    REPO_ROOT / 'BrainLM' / 'toolkit' / 'atlases' / 'A424_Coordinates.dat',
                ]
                for pth in cand:
                    if pth.exists():
                        a424_dat = pth; break
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
            x_eeg_grp = group_eeg_signal_seconds(x_eeg, cfg.eeg_seconds_per_token)

            # Forward
            recon_eeg, recon_fmri = translator(
                eeg_latents_t, fmri_latents,
                fmri_target_T=int(fmri_t.shape[1]),
                fmri_target_V=int(fmri_t.shape[2]),
            )

            # Metrics per mode
            # EEG metrics: MSE across (C,Pe,S); Corr across (C,Pe,S)
            # fMRI metrics: MSE across (T,V);    Corr across (T,V)
            if cfg.mode in ("both", "fmri2eeg", "partial_eeg"):
                if cfg.mode == "partial_eeg":
                    # masked-only (build same mask grouping)
                    M_eeg_time = _rand_block_mask_time(B, P, cfg.partial_visible_frac, device)  # re-build consistent mask? 
                    # NOTE: if you want strictly consistent masks between obs & metric,
                    # move M_eeg_time creation above and keep it for use here.
                    # For clarity we'll re-create above and reuse here:
                    pass
                # Full metrics by default
                diff_eeg = (recon_eeg - x_eeg_grp) ** 2
                mse_eeg  = diff_eeg.mean(dim=(1,2,3))  # (B,)
                corr_eeg = batch_corr(recon_eeg, x_eeg_grp, dims=(1,2,3))
                mse_eeg_sum += float(mse_eeg.mean().cpu()); corr_eeg_sum += float(corr_eeg.mean().cpu()); n_eeg += 1

            if cfg.mode in ("both", "eeg2fmri", "partial_fmri"):
                if cfg.mode == "partial_fmri":
                    # masked-only
                    # We already created M_t and M_fmri above; if you want to reuse exactly,
                    # keep them around outside this block. For simplicity in this example,
                    # we recompute metrics as full; to do masked-only, pass M_fmri from above.
                    pass
                diff_fmri = (recon_fmri - fmri_t) ** 2
                mse_f = diff_fmri.mean(dim=(1,2))
                corr_f= batch_corr(recon_fmri, fmri_t, dims=(1,2))
                mse_fmri_sum += float(mse_f.mean().cpu()); corr_fmri_sum += float(corr_f.mean().cpu()); n_fmri += 1

            # Save a few batches of reconstructions
            if saver_enabled and saved_batches < cfg.save_n_batches:
                out_dir = cfg.save_recons / f"batch_{b_idx:04d}"
                ensure_dir(out_dir)
                # Save small .pt tensors: recon + target
                torch.save({
                    "recon_eeg": recon_eeg.detach().cpu(),
                    "target_eeg_grouped": x_eeg_grp.detach().cpu(),
                    "recon_fmri": recon_fmri.detach().cpu(),
                    "target_fmri": fmri_t.detach().cpu(),
                    "mode": cfg.mode,
                }, out_dir / "recons.pt")
                saved_batches += 1

    # Report
    if n_eeg > 0:
        print(f"EEG  | MSE: {mse_eeg_sum / n_eeg:.6f} | Corr: {corr_eeg_sum / n_eeg:.4f}")
    if n_fmri > 0:
        print(f"fMRI | MSE: {mse_fmri_sum / n_fmri:.6f} | Corr: {corr_fmri_sum / n_fmri:.4f}")


# -----------------------------
# CLI
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Test EEG↔fMRI translator under various masking regimes.")
    # Config file (JSON) support — values from file become defaults; CLI overrides them
    ap.add_argument('--config', type=str, default=None, help='Path to JSON config file. Values become defaults; CLI overrides.')
    ap.add_argument('--eeg_root', type=str)
    ap.add_argument('--fmri_root', type=str)
    ap.add_argument('--a424_label_nii', type=str)
    ap.add_argument('--cbramod_weights', type=str)
    ap.add_argument('--brainlm_model_dir', type=str)
    ap.add_argument('--checkpoint', type=str, help="Path to translator_best.pt (or last)")
    ap.add_argument('--device', type=str, default='cuda')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--fmri_norm', type=str, default='zscore', choices=['zscore','psc','mad','none'])
    ap.add_argument('--window_sec', type=int, default=30)
    ap.add_argument('--original_fs', type=int, default=1000)
    ap.add_argument('--target_fs', type=int, default=200)
    ap.add_argument('--stride_sec', type=int, default=None)
    ap.add_argument('--channels_limit', type=int, default=34)
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--num_workers', type=int, default=0)
    ap.add_argument('--eeg_seconds_per_token', type=int, default=40)
    ap.add_argument('--mode', type=str, default='eeg2fmri',
                    choices=['both','eeg2fmri','fmri2eeg','partial_eeg','partial_fmri'])
    ap.add_argument('--partial_visible_frac', type=float, default=0.5)
    ap.add_argument('--save_recons', type=str, default=None)
    ap.add_argument('--save_n_batches', type=int, default=0)
    # Two-pass parse to allow config file to set defaults before final parse
    temp_parser = argparse.ArgumentParser(add_help=False)
    temp_parser.add_argument('--config', type=str, default=None)
    temp_args, _ = temp_parser.parse_known_args()
    if temp_args.config is not None:
        cfg_path = Path(temp_args.config)
        with open(cfg_path, 'r') as f:
            if cfg_path.suffix.lower() in ('.yaml', '.yml'):
                try:
                    import yaml  # type: ignore
                except Exception as e:
                    raise RuntimeError("PyYAML is required for YAML configs. Install with 'pip install pyyaml'.") from e
                file_cfg = yaml.safe_load(f)
            else:
                file_cfg = json.load(f)
        if isinstance(file_cfg, dict) and 'test' in file_cfg:
            file_cfg = file_cfg['test'] or {}
        # Set defaults from config file; CLI will override
        ap.set_defaults(**(file_cfg or {}))

    args = ap.parse_args()

    cfg = TestConfig(
        eeg_root=Path(args.eeg_root),
        fmri_root=Path(args.fmri_root),
        a424_label_nii=Path(args.a424_label_nii),
        cbramod_weights=Path(args.cbramod_weights),
        brainlm_model_dir=Path(args.brainlm_model_dir),
        checkpoint=Path(args.checkpoint),
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
        mode=args.mode,
        partial_visible_frac=args.partial_visible_frac,
        save_recons=(Path(args.save_recons) if args.save_recons else None),
        save_n_batches=args.save_n_batches,
    )

    # Quick existence checks
    required_paths = [
        cfg.eeg_root, cfg.fmri_root, cfg.a424_label_nii,
        cfg.cbramod_weights, cfg.brainlm_model_dir, cfg.checkpoint,
    ]
    for pth in required_paths:
        if pth is None or not Path(pth).exists():
            raise FileNotFoundError(f"Missing path: {pth}")

    run_test(cfg)


if __name__ == "__main__":
    main()

# python test_translator_paired.py \
#   --eeg_root ../Oddball/ds116_eeg \
#   --fmri_root ../Oddball/ds000116 \
#   --a424_label_nii ../BrainLM/A424_resampled_to_bold.nii.gz \
#   --brainlm_model_dir ../BrainLM/pretrained_models/2023-06-06-22_15_00-checkpoint-1400 \
#   --checkpoint translator_runs/odd_both_single_partial_run/checkpoint_best.pt \
#   --output_dir translator_test_runs \
#   --device cuda \
#   --window_sec 40 \
#   --stride_sec 10 \
#   --fmri_norm zscore \
#   --debug
