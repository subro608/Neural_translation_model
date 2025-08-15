#!/usr/bin/env python3
"""
Visualize original vs reconstructed EEG and fMRI (A424), parity with training.

Key alignments with training:
- Fixed subject splits via collect_common_sr_keys / fixed_subject_keys (no shuffle)
- Translator built with fmri_tokens_target = T * V (NOT Ttok)
- Same fMRI decode path: fMRIDecodingAdapter + token-scalar head + tanh + learnable affine
"""

from __future__ import annotations

import os
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

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
    fMRIDecodingAdapter,
)

from models.cbramod import CBraMod  # type: ignore
from brainlm_mae.modeling_brainlm import BrainLMForPretraining  # type: ignore
from brainlm_mae.configuration_brainlm import BrainLMConfig     # type: ignore

from data_oddball import (  # type: ignore
    pad_timepoints_for_brainlm_torch,
    load_a424_coords,
    PairedAlignedDataset,
    collate_paired,
    collect_common_sr_keys,
    fixed_subject_keys,
)

# ---------- Hardcoded configuration (adjust as needed) ----------
EEG_ROOT = r"D:\Neuroinformatics_research_2025\Oddball\ds116_eeg"
FMRI_ROOT = r"D:\Neuroinformatics_research_2025\Oddball\ds000116"
A424_LABEL_NII = r"D:\Neuroinformatics_research_2025\BrainLM\A424_resampled_to_bold.nii.gz"
CBRAMOD_WEIGHTS = r"D:\Neuroinformatics_research_2025\MNI_templates\CBraMod\pretrained_weights\pretrained_weights.pth"
BRAINLM_MODEL_DIR = r"D:\Neuroinformatics_research_2025\MNI_templates\BrainLM\pretrained_models\2023-06-06-22_15_00-checkpoint-1400"
CHECKPOINT = r"D:\Neuroinformatics_research_2025\Multi_modal_NTM\oddball_Neural_translation_run3_aug13_data_split_eeg1_fmri6_bs16\translator_best.pt"
OUT_DIR = r"D:\Neuroinformatics_research_2025\Multi_modal_NTM\viz_out_test_aug13_eeg2fmri_eeg1_fmri6"

# Optional: if you want to auto-read the exact splits saved by training:
TRAIN_OUTPUT_DIR = r"D:\Neuroinformatics_research_2025\Multi_modal_NTM\oddball_Neural_translation_run3_aug13_data_split_eeg1_fmri6_bs16"
DEVICE = "cuda"
SEED = 42
FMRI_NORM = "zscore"
WINDOW_SEC = 40
ORIGINAL_FS = 1000
TARGET_FS = 200
STRIDE_SEC = 10
CHANNELS_LIMIT = 34
BATCH_SIZE = 8
NUM_WORKERS = 0
EEG_SECONDS_PER_TOKEN = 1
TR = 2.0

SUBSET = "test"  # train | val | test | all
MODE = "eeg2fmri"   # both | eeg2fmri | fmri2eeg | partial_eeg | partial_fmri
PARTIAL_VISIBLE_FRAC = 0.5

EEG_CHANNELS_TO_PLOT = "0,1,2,3"
FMRI_ROIS_TO_PLOT = [0, 1, 2, 3, 4]

# Fixed subjects (mirror translator.yaml)
FIXED_TRAIN_SUBJECTS = [1,2,3,4,5,6,7,8,9,10,11]
FIXED_VAL_SUBJECTS   = [12]
FIXED_TEST_SUBJECTS  = [13,14,15,17]

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
        try:
            try:
                ckpt = torch.load(str(weights_path), map_location=device, weights_only=False)
            except TypeError:
                ckpt = torch.load(str(weights_path), map_location=device)
            if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
                self.model.load_state_dict(ckpt['model_state_dict'], strict=False)
            elif isinstance(ckpt, dict) and 'state_dict' in ckpt:
                self.model.load_state_dict(ckpt['state_dict'], strict=False)
            else:
                self.model.load_state_dict(ckpt, strict=False)
        except Exception as e:
            raise RuntimeError(f"Failed to load CBraMod weights from {weights_path}: {e}")
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
        try:
            try:
                ckpt = torch.load(str(model_dir / "pytorch_model.bin"), map_location=device, weights_only=True)
            except TypeError:
                ckpt = torch.load(str(model_dir / "pytorch_model.bin"), map_location=device)
            self.model.load_state_dict(ckpt, strict=False)
        except Exception as e:
            raise RuntimeError(f"Failed to load BrainLM weights from {model_dir}: {e}")
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.to(device)

    @torch.no_grad()
    def extract_latents(self, signal_vectors: torch.Tensor, xyz: torch.Tensor) -> torch.Tensor:
        embeddings, _, _ = self.model.vit.embeddings(
            signal_vectors=signal_vectors, xyz_vectors=xyz, noise=None
        )
        enc = self.model.vit.encoder(hidden_states=embeddings, output_hidden_states=True, return_dict=True)
        return torch.stack(list(enc.hidden_states), dim=0)  # (L,B,Ttok,H)

# -----------------------------
# Translator (matches training fMRI path)
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
        fmri_target_T: int,
        fmri_target_V: int,
        d_model: int = 256,
        n_heads: int = 8,
        d_ff: int = 1024,
        dropout: float = 0.1,
        voxel_count: int = 424,
        debug: bool = False,
    ) -> None:
        super().__init__()
        self.debug = debug
        self._voxel_count = int(voxel_count)
        self._fmri_n_layers = int(fmri_n_layers)
        self._fmri_hidden_size = int(fmri_hidden_size)
        self._d_model = int(d_model)

        # Adapters
        self.adapter_eeg = ConvEEGInputAdapter(
            seq_len=eeg_patch_num, n_layers=eeg_n_layers, channels=eeg_channels,
            input_dim=eeg_input_dim, output_dim=d_model,
        )
        self.adapter_fmri = fMRIInputAdapterConv1d(
            seq_len=fmri_tokens_target, n_layers=fmri_n_layers,
            input_dim=fmri_hidden_size, output_dim=d_model, target_seq_len=512,
        )

        # Encoders & fusion
        self.eeg_encoder = HierarchicalEncoder(d_model, n_heads, d_ff, dropout, n_layers_per_stack=2)
        self.fmri_encoder = HierarchicalEncoder(d_model, n_heads, d_ff, dropout, n_layers_per_stack=2)
        self.cross_attn = CrossAttentionLayer(d_model, n_heads, dropout)
        self.compressor = BidirectionalAdaptiveCompressor()

        # Decoders
        self.eeg_decoder = EEGDecodingAdapter(
            channels=eeg_channels, patch_num=eeg_patch_num, n_layers=eeg_n_layers,
            patch_size=eeg_input_dim, d_model=d_model,
        )
        self.fmri_decoder = fMRIDecodingAdapter(
            num_voxels=fmri_target_V,
            timepoints_per_voxel=fmri_target_T,
            n_layers=fmri_n_layers,
            hidden_size=fmri_hidden_size,
            d_model=d_model,
            max_target_tokens=100_000,
            downsample_to_cap=True,
        )
        self.fmri_token_head = nn.Linear(self._fmri_hidden_size, 1)
        self.fmri_out_scale = nn.Parameter(torch.tensor(1.0))
        self.fmri_out_bias  = nn.Parameter(torch.tensor(0.0))

    def _dbg(self, msg: str) -> None:
        if self.debug:
            print(msg, flush=True)

    def forward(self, eeg_latents, fmri_latents, fmri_target_T: int, fmri_target_V: int):
        if int(fmri_target_V) != self._voxel_count:
            raise ValueError(f"TranslatorModel expects V={self._voxel_count}, got V={fmri_target_V}")

        # Adapt & encode
        eeg_adapt = self.adapter_eeg(eeg_latents)     # (B, Neeg, D)
        fmri_adapt = self.adapter_fmri(fmri_latents)  # (B, Nfmri, D)
        _, eeg_hi = self.eeg_encoder(eeg_adapt, eeg_adapt)
        _, fmr_hi = self.fmri_encoder(fmri_adapt, fmri_adapt)
        eeg_c, fmr_c, _ = self.compressor(eeg_hi, fmr_hi)
        fused = self.cross_attn(eeg_c, fmr_c)         # (B, Tfused, D)

        # EEG decode to signal
        eeg_layers = self.eeg_decoder(fused)          # (L,B,P,C,S)
        eeg_signal = eeg_layers.mean(dim=0).permute(0, 2, 1, 3).contiguous()  # (B,C,P,S)

        # fMRI decode to (B,T,V)
        dec_tokens, eff_len, _ = self.fmri_decoder(fused)  # (L,B,eff_len,H)
        dec_mean = dec_tokens.mean(dim=0)                  # (B,eff_len,H)
        token_scalar = self.fmri_token_head(dec_mean)      # (B,eff_len,1)
        token_scalar = token_scalar.transpose(1, 2)        # (B,1,eff_len)

        target_tokens = int(fmri_target_T) * int(fmri_target_V)
        if eff_len != target_tokens:
            if eff_len < target_tokens:
                token_scalar = nn.functional.interpolate(token_scalar, size=target_tokens, mode="linear", align_corners=False)
            else:
                token_scalar = nn.functional.adaptive_avg_pool1d(token_scalar, target_tokens)

        token_scalar = token_scalar.transpose(1, 2).contiguous().squeeze(-1)  # (B,target_tokens)
        fmri_signal = token_scalar.view(-1, int(fmri_target_T), int(fmri_target_V))  # (B,T,V)
        fmri_signal = torch.tanh(fmri_signal)
        fmri_signal = self.fmri_out_scale * fmri_signal + self.fmri_out_bias
        return eeg_signal, fmri_signal

# -----------------------------
# Helpers: EEG grouping & masks
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

# -----------------------------
# Viz utils
# -----------------------------
def _contiguous_true_segments(bool_arr: np.ndarray) -> List[Tuple[int,int]]:
    idx = np.flatnonzero(bool_arr)
    if idx.size == 0:
        return []
    splits = np.where(np.diff(idx) != 1)[0] + 1
    segs = np.split(idx, splits)
    return [(int(seg[0]), int(seg[-1])) for seg in segs]

def plot_eeg_overlays(x_true: torch.Tensor, x_recon: torch.Tensor, out_dir: Path, channels: List[int] = [0,1,2,3], max_seconds: Optional[int]=None):
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

def plot_fmri_overlays(ts_true: torch.Tensor, ts_recon: torch.Tensor, out_dir: Path, rois: List[int] = [0,1,2,3,4],
                       mask_bool_T: Optional[np.ndarray] = None, masked_only: bool = False, tr: float = 2.0):
    out_dir.mkdir(parents=True, exist_ok=True)
    b0_true = ts_true[0].detach().cpu().numpy()  # (T,V)
    b0_reco = ts_recon[0].detach().cpu().numpy()
    T, V = b0_true.shape
    sel = [r for r in rois if 0 <= r < V]
    t = np.arange(T) * tr
    for r in sel:
        plt.figure(figsize=(10,3))
        plt.plot(t, b0_true[:, r], label='fMRI true')
        if masked_only and mask_bool_T is not None:
            recon_curve = b0_reco[:, r].copy()
            recon_curve[~mask_bool_T] = np.nan
            plt.plot(t, recon_curve, label='fMRI recon (masked only)', alpha=0.9)
            idx = np.flatnonzero(mask_bool_T)
            if idx.size:
                splits = np.where(np.diff(idx) != 1)[0] + 1
                for seg in np.split(idx, splits):
                    plt.axvspan(t[seg[0]], t[seg[-1]] + tr, alpha=0.15)
        else:
            plt.plot(t, b0_reco[:, r], label='fMRI recon', alpha=0.8)
        plt.title(f'fMRI ROI {r} — true vs recon')
        plt.xlabel('Time (s)'); plt.ylabel('Z-scored signal')
        plt.legend(); plt.tight_layout()
        plt.savefig(out_dir / f'fmri_overlay_roi{r:03d}.png', dpi=150)
        plt.close()

def plot_fmri_heatmaps(ts_true: torch.Tensor, ts_recon: torch.Tensor, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    b0_true = ts_true[0].detach().cpu().numpy()  # (T,V)
    b0_reco = ts_recon[0].detach().cpu().numpy()
    plt.figure(figsize=(12,4))
    plt.imshow(b0_true.T, aspect='auto', origin='lower')
    plt.colorbar(); plt.title('fMRI target (T x V)'); plt.xlabel('Time'); plt.ylabel('ROI')
    plt.tight_layout(); plt.savefig(out_dir / 'fmri_target_heatmap.png', dpi=150); plt.close()
    plt.figure(figsize=(12,4))
    plt.imshow(b0_reco.T, aspect='auto', origin='lower')
    plt.colorbar(); plt.title('fMRI recon (T x V)'); plt.xlabel('Time'); plt.ylabel('ROI')
    plt.tight_layout(); plt.savefig(out_dir / 'fmri_recon_heatmap.png', dpi=150); plt.close()

def plot_fmri_masked_segment_zooms(ts_true: torch.Tensor, ts_recon: torch.Tensor, out_dir: Path,
                                   rois: List[int], mask_bool_T: np.ndarray, tr: float = 2.0):
    out_dir.mkdir(parents=True, exist_ok=True)
    b0_true = ts_true[0].detach().cpu().numpy()  # (T,V)
    b0_reco = ts_recon[0].detach().cpu().numpy()
    T, V = b0_true.shape
    t = np.arange(T) * tr
    segments = _contiguous_true_segments(mask_bool_T)
    if not segments:
        return
    sel = [r for r in rois if 0 <= r < V]
    for r in sel:
        for si, (s, e) in enumerate(segments):
            sl = slice(s, e + 1)
            plt.figure(figsize=(9,3))
            plt.plot(t[sl], b0_true[sl, r], label='fMRI true')
            plt.plot(t[sl], b0_reco[sl, r], label='fMRI recon (masked seg)', alpha=0.9)
            plt.title(f'ROI {r} — masked segment {si+1} [{s}:{e}]')
            plt.xlabel('Time (s)'); plt.ylabel('Z-scored signal')
            plt.legend(); plt.tight_layout()
            plt.savefig(out_dir / f'fmri_masked_zoom_roi{r:03d}_seg{si+1:02d}.png', dpi=150)
            plt.close()

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
    window_sec: int = 40
    original_fs: int = 1000
    target_fs: int = 200
    stride_sec: Optional[int] = None
    channels_limit: int = 34
    batch_size: int = 4
    num_workers: int = 0
    eeg_seconds_per_token: int = 40
    tr: float = 2.0
    eeg_channels_to_plot: str = "0,1,2,3"

def build_frozen_models(cfg: VizConfig, device: torch.device):
    seq_len_eeg = cfg.window_sec
    frozen_eeg = FrozenCBraMod(
        in_dim=cfg.target_fs, d_model=cfg.target_fs, seq_len=seq_len_eeg,
        n_layer=12, nhead=8, dim_feedforward=800,
        weights_path=cfg.cbramod_weights, device=device
    )
    frozen_fmri = FrozenBrainLM(cfg.brainlm_model_dir, device=device)
    return frozen_eeg, frozen_fmri

def build_translator(
    cfg: VizConfig,
    device: torch.device,
    fmri_tokens_target: int,
    fmri_target_T: int,
    fmri_target_V: int,
    fmri_n_layers: int,
    fmri_hidden_size: int
) -> TranslatorModel:
    eeg_group = max(1, int(cfg.eeg_seconds_per_token))
    eeg_patch_num_grouped = max(1, int(cfg.window_sec) // eeg_group)
    translator = TranslatorModel(
        eeg_channels=cfg.channels_limit,
        eeg_patch_num=eeg_patch_num_grouped,
        eeg_n_layers=12,
        eeg_input_dim=cfg.target_fs,
        fmri_n_layers=fmri_n_layers,
        fmri_hidden_size=fmri_hidden_size,
        fmri_tokens_target=fmri_tokens_target,  # MUST be T * V (as in training)
        fmri_target_T=fmri_target_T,
        fmri_target_V=fmri_target_V,
        d_model=256, n_heads=8, d_ff=1024, dropout=0.1,
        voxel_count=fmri_target_V,
        debug=False,
    ).to(device)

    # Load translator weights (filter unexpected keys)
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
    return translator

def _load_fixed_subjects_from_json(output_dir: str) -> Optional[Tuple[List[int], List[int], List[int]]]:
    try:
        p = Path(output_dir) / "subject_splits.json"
        if not p.exists():
            return None
        with open(p, "r") as f:
            j = json.load(f)
        tr = [int(s) for s in j.get("train_subjects", [])]
        va = [int(s) for s in j.get("val_subjects", [])]
        te = [int(s) for s in j.get("test_subjects", [])]
        if not tr or not va or not te:
            return None
        return tr, va, te
    except Exception:
        return None

def run_once(cfg: VizConfig, subset: str = "test", mode: str = "both", partial_visible_frac: float = 0.5):
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

    device = torch.device(cfg.device)
    out_dir = cfg.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # ----- Build subject-run split exactly like training (no shuffle) -----
    inter_keys = collect_common_sr_keys(cfg.eeg_root, cfg.fmri_root)
    if len(inter_keys) == 0:
        raise RuntimeError("No aligned (subject, run) pairs found; check paths.")

    # Prefer subjects saved by training; otherwise fall back to translator.yaml lists
    loaded = _load_fixed_subjects_from_json(TRAIN_OUTPUT_DIR)
    if loaded is not None:
        train_subj, val_subj, test_subj = loaded
    else:
        train_subj, val_subj, test_subj = FIXED_TRAIN_SUBJECTS, FIXED_VAL_SUBJECTS, FIXED_TEST_SUBJECTS

    train_keys, val_keys, test_keys = fixed_subject_keys(
        cfg.eeg_root, cfg.fmri_root, train_subj, val_subj, test_subj
    )

    include_sr = None
    if subset == 'train':
        include_sr = train_keys
    elif subset == 'val':
        include_sr = val_keys
    elif subset == 'test':
        include_sr = test_keys
    elif subset == 'all':
        include_sr = None
    else:
        raise ValueError(f"Unknown subset: {subset}")

    # Dataset (+ chosen subset)
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
        raise RuntimeError("No samples in dataset; check paths/splits.")

    dl = DataLoader(
        ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, collate_fn=collate_paired, pin_memory=(device.type=='cuda')
    )

    frozen_eeg, frozen_fmri = build_frozen_models(cfg, device)

    # Get one batch
    batch = next(iter(dl))
    x_eeg = batch['eeg_window'].to(device)   # (B,C,P,S)
    fmri_t = batch['fmri_window'].to(device) # (B,T,V)
    B, C, P, S = x_eeg.shape
    _, T, V = fmri_t.shape

    # Observed inputs based on viz mode
    x_eeg_obs = x_eeg.clone()
    fmri_obs  = fmri_t.clone()
    M_eeg_time = None
    M_t = None
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

    # ---- BrainLM patching & coords ----
    patch = int(getattr(frozen_fmri.model.config, "timepoint_patching_size", 20))
    fmri_padded = pad_timepoints_for_brainlm_torch(fmri_obs, patch_size=patch)   # (B,Tp,V)
    signal_vectors = fmri_padded.permute(0,2,1).contiguous()                     # (B,V,Tp)

    # Sync voxel count to current V (defensive)
    frozen_fmri.model.config.num_brain_voxels = int(V)
    if hasattr(frozen_fmri.model, "vit") and hasattr(frozen_fmri.model.vit, "embeddings"):
        frozen_fmri.model.vit.embeddings.num_brain_voxels = int(V)
    if hasattr(frozen_fmri.model, "decoder"):
        frozen_fmri.model.decoder.num_brain_voxels = int(V)

    # xyz coords (normalized A424 if V==424 else zeros)
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

    # ---- Frozen latents ----
    with torch.no_grad():
        eeg_latents = frozen_eeg.extract_latents(x_eeg_obs)     # (L,B,P,C,D)
        fmri_latents = frozen_fmri.extract_latents(signal_vectors, xyz)  # (L,B,Ttok,H)

    # Shapes to drive translator (derived from BrainLM config)
    fmri_n_layers = int(getattr(frozen_fmri.model.config, "num_hidden_layers", 4)) + 1
    fmri_hidden_size = int(getattr(frozen_fmri.model.config, "hidden_size", 256))

    # ---- Translator (built AFTER we know T and V)
    # IMPORTANT: fmri_tokens_target MUST be T * V (same as training), not Ttok
    translator = build_translator(
        cfg, device,
        fmri_tokens_target=int(fmri_t.shape[1]) * int(fmri_t.shape[2]),
        fmri_target_T=int(fmri_t.shape[1]),
        fmri_target_V=int(fmri_t.shape[2]),
        fmri_n_layers=fmri_n_layers,
        fmri_hidden_size=fmri_hidden_size
    )

    # Group EEG to match decoder/grouping
    eeg_latents_t = group_eeg_latents_seconds(eeg_latents, cfg.eeg_seconds_per_token)
    x_eeg_grp     = group_eeg_signal_seconds(x_eeg, cfg.eeg_seconds_per_token)

    with torch.no_grad():
        recon_eeg, recon_fmri = translator(
            eeg_latents_t, fmri_latents,
            fmri_target_T=int(fmri_t.shape[1]),
            fmri_target_V=int(fmri_t.shape[2]),
        )

    # ---- DEBUG: fMRI stats
    try:
        def _summ(name: str, arr: np.ndarray) -> str:
            return f"{name}: shape={arr.shape} min={np.nanmin(arr):.4f} max={np.nanmax(arr):.4f} mean={np.nanmean(arr):.4f} std={np.nanstd(arr):.4f}"
        fmri_true_np = fmri_t[0].detach().cpu().numpy()
        fmri_in_np   = fmri_obs[0].detach().cpu().numpy()
        fmri_rec_np  = recon_fmri[0].detach().cpu().numpy()
        print("[DEBUG] fMRI normalization check (first batch sample):")
        print("  " + _summ("target_true", fmri_true_np))
        print("  " + _summ("input_obs  ", fmri_in_np))
        print("  " + _summ("reconstructed", fmri_rec_np))
    except Exception as e:
        print(f"[WARN] fMRI debug stats failed: {e}")

    # ---- Save EEG overlays
    chans = [int(c.strip()) for c in cfg.eeg_channels_to_plot.split(",") if c.strip().isdigit()]
    plot_eeg_overlays(x_eeg_grp, recon_eeg, out_dir / "eeg_plots", channels=chans, max_seconds=None)
    print(f"[OK] EEG overlays saved to {out_dir/'eeg_plots'}")

    # ---- fMRI plots
    plot_fmri_overlays(fmri_t, recon_fmri, out_dir / 'fmri_plots', rois=FMRI_ROIS_TO_PLOT, tr=cfg.tr)
    plot_fmri_heatmaps(fmri_t, recon_fmri, out_dir / 'fmri_plots')
    print(f"[OK] Saved generic fMRI plots to {out_dir/'fmri_plots'}")

    # ---- fMRI masked-only overlays + zoomed segments
    if mode == "partial_fmri" and M_t is not None:
        mask_bool_T = M_t[0].detach().cpu().numpy().astype(bool)
        plot_fmri_overlays(
            fmri_t, recon_fmri, out_dir / 'fmri_plots_masked_only',
            rois=FMRI_ROIS_TO_PLOT, mask_bool_T=mask_bool_T, masked_only=True, tr=cfg.tr
        )
        plot_fmri_masked_segment_zooms(
            fmri_t, recon_fmri, out_dir / 'fmri_plots_masked_zooms',
            rois=FMRI_ROIS_TO_PLOT, mask_bool_T=mask_bool_T, tr=cfg.tr
        )
        print(f"[OK] Saved fMRI masked-only overlays to {out_dir/'fmri_plots_masked_only'}")
        print(f"[OK] Saved fMRI per-segment zooms to {out_dir/'fmri_plots_masked_zooms'}")

def main():
    cfg = VizConfig(
        eeg_root=Path(EEG_ROOT),
        fmri_root=Path(FMRI_ROOT),
        a424_label_nii=Path(A424_LABEL_NII),
        cbramod_weights=Path(CBRAMOD_WEIGHTS),
        brainlm_model_dir=Path(BRAINLM_MODEL_DIR),
        checkpoint=Path(CHECKPOINT),
        out_dir=Path(OUT_DIR),
        device=DEVICE,
        seed=SEED,
        fmri_norm=FMRI_NORM,
        window_sec=WINDOW_SEC,
        original_fs=ORIGINAL_FS,
        target_fs=TARGET_FS,
        stride_sec=STRIDE_SEC,
        channels_limit=CHANNELS_LIMIT,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        eeg_seconds_per_token=EEG_SECONDS_PER_TOKEN,
        tr=TR,
        eeg_channels_to_plot=EEG_CHANNELS_TO_PLOT,
    )
    for p in [cfg.eeg_root, cfg.fmri_root, cfg.a424_label_nii, cfg.cbramod_weights, cfg.brainlm_model_dir, cfg.checkpoint]:
        if p is None or not Path(p).exists():
            raise FileNotFoundError(f"Missing path: {p}")
    run_once(cfg, subset=SUBSET, mode=MODE, partial_visible_frac=PARTIAL_VISIBLE_FRAC)

if __name__ == "__main__":
    main()
