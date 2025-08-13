#!/usr/bin/env python3
"""
Train the adapter_main_translatordecoder using paired, aligned Oddball EEG+fMRI windows,
with tri-mix training:
  - Both modalities available
  - Single-modality (one side missing)
  - Partial-modality (randomly EEG or fMRI partially visible, random visible fraction)

Saves FULL checkpoints (translator + optimizer + scaler + config + rng + metrics)
and optionally resumes exactly via --resume.

Changes vs previous version:
- fMRIDecodingAdapter is now instantiated in TranslatorModel.__init__ (not in forward),
  so it’s trainable and checkpointed.
- fmri_n_layers and fmri_hidden_size are derived from BrainLM config for robustness.
- fmri_target_T and fmri_target_V are passed when building the translator so the decoder
  is shaped correctly from the start.

New in this version:
- Uses fixed, deterministic subject intake via data_oddball.fixed_subject_keys / collect_common_sr_keys
- No random shuffling when forming splits; default subject split is deterministic by subject ID order
- DataLoader shuffle disabled (train/val) for full determinism of sample order
"""

from __future__ import annotations

import os
import sys
import json
import time
import math
import argparse
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Tuple, List, Set

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# wandb (optional)
import wandb
try:
    from tqdm import tqdm
except Exception:
    tqdm = None

# ---------- Local repo layout ----------
THIS_DIR = Path(__file__).parent
REPO_ROOT = THIS_DIR.parent
CBRAMOD_DIR = REPO_ROOT / "CBraMod"
BRAINLM_DIR = REPO_ROOT / "BrainLM"

# Prefer local copies inside project if present
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(THIS_DIR / "CBraMod"))
sys.path.insert(0, str(THIS_DIR / "BrainLM"))
sys.path.append(str(CBRAMOD_DIR))
sys.path.append(str(BRAINLM_DIR))

# Adapters & core
from module import (  # type: ignore
    BidirectionalAdaptiveCompressor,
    ConvEEGInputAdapter,
    fMRIInputAdapterConv1d,
    HierarchicalEncoder,
    CrossAttentionLayer,
    EEGDecodingAdapter,
    fMRIDecodingAdapter,
)

# Models
from models.cbramod import CBraMod  # type: ignore
from brainlm_mae.modeling_brainlm import BrainLMForPretraining  # type: ignore
from brainlm_mae.configuration_brainlm import BrainLMConfig     # type: ignore

# Data (updated helpers)
from data_oddball import (  # type: ignore
    pad_timepoints_for_brainlm_torch,
    load_a424_coords,
    PairedAlignedDataset,
    collate_paired,
    collect_common_sr_keys,
    fixed_subject_keys,
)

# -----------------------------
# Utility
# -----------------------------
def seed_all(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# -----------------------------
# Frozen feature extractors
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
        x: (B, channels, patch_num, patch_size)  -> returns (L, B, seq_len, channels, input_dim)
        """
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
          (L, B, T_tokens, hidden_size)
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
# Trainable translator
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

        # Decoders
        self.eeg_decoder = EEGDecodingAdapter(
            channels=eeg_channels,
            patch_num=eeg_patch_num,
            n_layers=eeg_n_layers,
            patch_size=eeg_input_dim,
            d_model=d_model,
        )

        # fMRI decoder (token sequence) + head to scalarize tokens and reshape to (T,V)
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

    def forward(
        self,
        eeg_latents: torch.Tensor,        # (L,B,S,C,D)
        fmri_latents: torch.Tensor,       # (L,B,T,H)
        fmri_target_T: int,
        fmri_target_V: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self._dbg("\n[Forward] start")
        if fmri_target_V != self._voxel_count:
            raise ValueError(f"TranslatorModel expects V={self._voxel_count}, got V={fmri_target_V}")

        eeg_adapted = self.adapter_eeg(eeg_latents)    # (B, N_eeg_tokens, D)
        fmri_adapted = self.adapter_fmri(fmri_latents) # (B, N_fmri_tokens, D)

        _, eeg_higher = self.eeg_encoder(eeg_adapted, eeg_adapted)
        _, fmri_higher = self.fmri_encoder(fmri_adapted, fmri_adapted)

        eeg_c, fmri_c, _ = self.compressor(eeg_higher, fmri_higher)
        fused = self.cross_attn(eeg_c, fmri_c)         # (B, Tfused, D)

        # EEG decode
        eeg_layers = self.eeg_decoder(fused)           # (L,B,P,C,S)
        eeg_signal = eeg_layers.mean(dim=0).permute(0, 2, 1, 3).contiguous()  # (B,C,P,S)

        # fMRI decode via fMRIDecodingAdapter → (L,B,eff_len,H)
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
        self._dbg("[Forward] end\n")
        return eeg_signal, fmri_signal

# -----------------------------
# Config
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
    stride_sec: Optional[int] = None
    batch_size: int = 8
    lr: float = 1e-4
    weight_decay: float = 1e-4
    num_epochs: int = 5
    num_workers: int = 0
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
    # EEG token grouping
    eeg_seconds_per_token: int = 40
    # resume
    resume: Optional[Path] = None
    # fixed subject splits (optional). Use subject numeric IDs.
    train_subjects: Optional[List[int]] = None
    val_subjects: Optional[List[int]] = None
    test_subjects: Optional[List[int]] = None

# -----------------------------
# Checkpoints
# -----------------------------
def save_checkpoint(path: Path, translator: TranslatorModel, optim: torch.optim.Optimizer,
                    scaler: torch.amp.GradScaler, cfg: TrainConfig, epoch: int,
                    global_step: int, best_val: float) -> None:
    # Convert any Path objects in config to strings for safe serialization (PyTorch 2.6 safe loader)
    def _to_jsonable(obj):
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, dict):
            return {k: _to_jsonable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple, set)):
            return [_to_jsonable(v) for v in obj]
        return obj

    cfg_dict = _to_jsonable(asdict(cfg))

    ckpt = {
        "translator_state": translator.state_dict(),
        "optimizer_state": optim.state_dict(),
        "scaler_state": scaler.state_dict(),
        "config": cfg_dict,
        "epoch": epoch,
        "global_step": global_step,
        "best_val": best_val,
        "rng_state": {
            "torch": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "numpy": np.random.get_state(),
        },
        "versions": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda if torch.cuda.is_available() else None,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, path)

def load_checkpoint(path: Path, translator: TranslatorModel, optim: torch.optim.Optimizer,
                    scaler: torch.amp.GradScaler, device: torch.device):
    # PyTorch 2.6 safe loader fallback
    try:
        ckpt = torch.load(path, map_location=device)
    except Exception:
        ckpt = torch.load(path, map_location=device, weights_only=False)

    tsd = ckpt.get("translator_state", {})
    # Allow missing keys (e.g., older ckpts)
    translator.load_state_dict(tsd, strict=False)

    if "optimizer_state" in ckpt:
        try:
            optim.load_state_dict(ckpt["optimizer_state"])
        except ValueError as e:
            print(f"[WARN] Optimizer state incompatible with current model parameters: {e}\n"
                  f"       Proceeding with a fresh optimizer state (weights are loaded).")
        except Exception as e:
            print(f"[WARN] Failed to load optimizer state: {e}. Proceeding with a fresh optimizer state.")
    if "scaler_state" in ckpt and isinstance(scaler, torch.amp.GradScaler):
        try:
            scaler.load_state_dict(ckpt["scaler_state"])
        except Exception:
            pass

    epoch = ckpt.get("epoch", 0)
    global_step = ckpt.get("global_step", 0)
    best_val = ckpt.get("best_val", float("inf"))

    # Restore RNG (best-effort)
    rng = ckpt.get("rng_state", {})
    try:
        if "torch" in rng and rng["torch"] is not None:
            torch.set_rng_state(rng["torch"])
        if torch.cuda.is_available() and "torch_cuda" in rng and rng["torch_cuda"]:
            torch.cuda.set_rng_state_all(rng["torch_cuda"])
        if "numpy" in rng and rng["numpy"]:
            np.random.set_state(rng["numpy"])
    except Exception:
        pass
    return epoch, global_step, best_val

# -----------------------------
# Mask builders
# -----------------------------
def _rand_block_mask_time_per_sample(B: int, L: int, frac_vis_each: torch.Tensor, device) -> torch.Tensor:
    """
    Build boolean masks (B, L) where True==MASKED.
    Each sample i keeps a contiguous visible block of length int(frac_vis_each[i]*L) at a random start.
    """
    M = torch.ones(B, L, dtype=torch.bool, device=device)  # start masked
    for i in range(B):
        vis = max(1, int(round(float(frac_vis_each[i]) * L)))
        vis = min(vis, L)
        if L > vis:
            start = int(torch.randint(0, L - vis + 1, (1,), device=device))
        else:
            start = 0
        M[i, start:start+vis] = False  # False = visible
    return M  # True = masked

def _group_mask_seconds(M_time: torch.Tensor, seconds_per_token: int) -> torch.Tensor:
    """
    M_time: (B, P) boolean (True=masked) at 1-second resolution
    Returns M_grp: (B, Pe) where Pe = P // seconds_per_token.
    Group is masked if ANY second in the group is masked.
    """
    B, P = M_time.shape
    g = max(1, min(seconds_per_token, P))
    Pe = max(1, P // g)
    if Pe * g != P:
        M_time = M_time[:, :Pe*g]
    M_trim = M_time.reshape(B, Pe, g)
    return M_trim.any(dim=2)  # (B, Pe)

# -----------------------------
# EEG grouping helpers
# -----------------------------
def group_eeg_latents_seconds(eeg_latents: torch.Tensor, seconds_per_token: int) -> torch.Tensor:
    # eeg_latents: (L,B,P,C,D) where P=window_sec (1s tokens)
    L, B, P, C, D = eeg_latents.shape
    g = max(1, min(seconds_per_token, P))
    P_grp = max(1, P // g)
    x = eeg_latents[:, :, :P_grp * g].reshape(L, B, P_grp, g, C, D).mean(dim=3)
    return x

def group_eeg_signal_seconds(x_eeg_sig: torch.Tensor, seconds_per_token: int) -> torch.Tensor:
    # x_eeg_sig: (B,C,P,S)
    B_, C_, P_, S_ = x_eeg_sig.shape
    g = max(1, min(seconds_per_token, P_))
    P_grp = max(1, P_ // g)
    x = x_eeg_sig[:, :, :P_grp * g, :].reshape(B_, C_, P_grp, g, S_).mean(dim=3)
    return x

# -----------------------------
# Training loop with tri-mix
# -----------------------------
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

    # ---------- Deterministic subject-run splits (no shuffling) ----------
    inter_keys = collect_common_sr_keys(cfg.eeg_root, cfg.fmri_root)  # sorted by (subject, run)
    if len(inter_keys) == 0:
        print("No paired aligned (subject,run) found. Check your paths.")
        return

    # If user supplies explicit subject lists, use them exactly
    if cfg.train_subjects or cfg.val_subjects or cfg.test_subjects:
        train_keys, val_keys, test_keys = fixed_subject_keys(
            cfg.eeg_root,
            cfg.fmri_root,
            cfg.train_subjects or [],
            cfg.val_subjects or [],
            cfg.test_subjects or [],
        )
    else:
        # Deterministic default split by SUBJECT (70/10/20), ordered by subject ID
        subjects_sorted = sorted({k[0] for k in inter_keys}, key=lambda s: int(s))
        n_subj = len(subjects_sorted)
        if n_subj < 3:
            # Require user to specify lists if we can't make 3 non-empty splits deterministically
            raise RuntimeError(f"Found {n_subj} subjects; please provide --train_subjects/--val_subjects/--test_subjects.")
        n_train = max(1, int(n_subj * 0.7))
        n_val   = max(1, int(n_subj * 0.1))
        # ensure sum==n_subj and all non-empty
        if n_train + n_val >= n_subj:
            n_val = 1
            n_train = max(1, n_subj - n_val - 1)
        n_test = n_subj - n_train - n_val

        train_subj = set(subjects_sorted[:n_train])
        val_subj   = set(subjects_sorted[n_train:n_train+n_val])
        test_subj  = set(subjects_sorted[n_train+n_val:])

        # Build keys deterministically
        train_keys = tuple(k for k in inter_keys if k[0] in train_subj)
        val_keys   = tuple(k for k in inter_keys if k[0] in val_subj)
        test_keys  = tuple(k for k in inter_keys if k[0] in test_subj)

    # --- Sanity checks: subjects & keys (no leakage) ---
    def _assert_disjoint_sets(a: Set[str], b: Set[str], aname="A", bname="B"):
        inter = set(a) & set(b)
        if inter:
            raise ValueError(f"Leakage: {aname} ∩ {bname} = {sorted(inter)}")

    def _subjects_from_keys(keys: Tuple[Tuple[str, str], ...]):
        return {str(k[0]) for k in keys}

    # Subjects present in each split (as derived from keys)
    train_subj_from_keys = _subjects_from_keys(train_keys)
    val_subj_from_keys   = _subjects_from_keys(val_keys)
    test_subj_from_keys  = _subjects_from_keys(test_keys)

    # If user provided explicit lists, enforce membership
    if cfg.train_subjects or cfg.val_subjects or cfg.test_subjects:
        exp_train = set(str(int(s)) for s in (cfg.train_subjects or []))
        exp_val   = set(str(int(s)) for s in (cfg.val_subjects or []))
        exp_test  = set(str(int(s)) for s in (cfg.test_subjects or []))
        assert train_subj_from_keys.issubset(exp_train), \
            f"TRAIN keys include subjects not in train_subjects: {sorted(train_subj_from_keys - exp_train)}"
        assert val_subj_from_keys.issubset(exp_val), \
            f"VAL keys include subjects not in val_subjects: {sorted(val_subj_from_keys - exp_val)}"
        assert test_subj_from_keys.issubset(exp_test), \
            f"TEST keys include subjects not in test_subjects: {sorted(test_subj_from_keys - exp_test)}"

    # Subject-level disjointness
    _assert_disjoint_sets(train_subj_from_keys, val_subj_from_keys, "train", "val")
    _assert_disjoint_sets(train_subj_from_keys, test_subj_from_keys, "train", "test")
    _assert_disjoint_sets(val_subj_from_keys,   test_subj_from_keys, "val",   "test")

    # Key-level disjointness
    train_key_set, val_key_set, test_key_set = set(train_keys), set(val_keys), set(test_keys)
    assert train_key_set.isdisjoint(val_key_set),  "Key leakage: train ∩ val is non-empty"
    assert train_key_set.isdisjoint(test_key_set), "Key leakage: train ∩ test is non-empty"
    assert val_key_set.isdisjoint(test_key_set),   "Key leakage: val ∩ test is non-empty"

    # Non-empty splits
    if not train_keys or not val_keys or not test_keys:
        raise RuntimeError(
            f"Empty split: sizes -> train={len(train_keys)} val={len(val_keys)} test={len(test_keys)}"
        )

    # Helpful prints
    print(f"[splits] subjects: train={sorted(list(train_subj_from_keys), key=int)} "
          f"val={sorted(list(val_subj_from_keys), key=int)} test={sorted(list(test_subj_from_keys), key=int)}")
    print(f"[splits] keys:     train={len(train_keys)} val={len(val_keys)} test={len(test_keys)}")

    # Save split subjects for later auditing in test
    try:
        split_path = Path(cfg.output_dir) / "subject_splits.json"
        split_path.parent.mkdir(parents=True, exist_ok=True)
        with open(split_path, "w") as f:
            json.dump({
                "train_subjects": sorted(list(train_subj_from_keys), key=int),
                "val_subjects":   sorted(list(val_subj_from_keys), key=int),
                "test_subjects":  sorted(list(test_subj_from_keys), key=int),
            }, f, indent=2)
        print(f"[splits] saved -> {split_path}")
    except Exception as e:
        print(f"[splits] WARN: couldn't save subject_splits.json ({e})")

    # Optional: print concrete keys (can be verbose)
    print("train_keys", train_keys, "val_keys", val_keys, "test_keys", test_keys)

    # DataLoaders (no shuffle for determinism)
    pin = device.type == 'cuda'
    dl_train_kwargs = dict(
        dataset=PairedAlignedDataset(
            eeg_root=cfg.eeg_root,
            fmri_root=cfg.fmri_root,
            a424_label_nii=cfg.a424_label_nii,
            window_sec=cfg.window_sec,
            original_fs=cfg.original_fs,
            target_fs=cfg.target_fs,
            tr=2.0,
            channels_limit=34,
            fmri_norm=cfg.fmri_norm,
            stride_sec=cfg.stride_sec,
            device='cpu',
            include_sr_keys=train_keys,
        ),
        batch_size=cfg.batch_size, shuffle=False,  # <<< no shuffling
        num_workers=cfg.num_workers,
        collate_fn=collate_paired, pin_memory=pin, drop_last=False
    )
    dl_val_kwargs = dict(
        dataset=PairedAlignedDataset(
            eeg_root=cfg.eeg_root,
            fmri_root=cfg.fmri_root,
            a424_label_nii=cfg.a424_label_nii,
            window_sec=cfg.window_sec,
            original_fs=cfg.original_fs,
            target_fs=cfg.target_fs,
            tr=2.0,
            channels_limit=34,
            fmri_norm=cfg.fmri_norm,
            stride_sec=cfg.stride_sec,
            device='cpu',
            include_sr_keys=val_keys,
        ),
        batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=collate_paired, pin_memory=pin, drop_last=False
    )
    if cfg.num_workers > 0:
        dl_train_kwargs.update(persistent_workers=True, prefetch_factor=2)
        dl_val_kwargs.update(persistent_workers=True, prefetch_factor=2)

    dl_train = DataLoader(**dl_train_kwargs)
    dl_val = DataLoader(**dl_val_kwargs)

    if len(dl_train.dataset) == 0 or len(dl_val.dataset) == 0:
        print("Empty train or val split. Adjust subject lists or check data.")
        return

    # Frozen feature extractors
    seq_len_eeg = cfg.window_sec  # EEG: 1s tokens
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

    # ----- Derive BrainLM properties robustly
    fmri_n_layers = int(getattr(frozen_brainlm.model.config, "num_hidden_layers", 4)) + 1  # +1 for embeddings output
    fmri_hidden_size = int(getattr(frozen_brainlm.model.config, "hidden_size", 256))

    # ----- Fix fMRI decoder target shape at build time
    fmri_target_V = 424
    fmri_target_T = int(round(cfg.window_sec / 2.0))  # TR=2s → frames per window

    # Translator (constant V=424)
    eeg_group = max(1, int(cfg.eeg_seconds_per_token))
    eeg_patch_num_grouped = max(1, int(cfg.window_sec) // eeg_group)
    translator = TranslatorModel(
        eeg_channels=34,
        eeg_patch_num=eeg_patch_num_grouped,
        eeg_n_layers=12,
        eeg_input_dim=cfg.target_fs,
        fmri_n_layers=fmri_n_layers,
        fmri_hidden_size=fmri_hidden_size,
        fmri_tokens_target=fmri_target_V * fmri_target_T,  # informational
        fmri_target_T=fmri_target_T,
        fmri_target_V=fmri_target_V,
        d_model=256,
        n_heads=8,
        d_ff=1024,
        dropout=0.1,
        voxel_count=fmri_target_V,
        debug=cfg.debug,
    ).to(device)

    # Optimizer
    optim = torch.optim.AdamW(
        [p for p in translator.parameters() if p.requires_grad],
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    mse_loss = nn.MSELoss()
    scaler = torch.amp.GradScaler('cuda', enabled=(cfg.amp and device.type == 'cuda'))

    # Scheduler: warmup (5%) + cosine to 10% of base LR
    total_steps = max(1, cfg.num_epochs * len(dl_train))
    warmup_steps = max(1, int(0.05 * total_steps))

    def lr_lambda(step: int):
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))  # cosine 1.0 -> 0.1

    scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)

    # wandb
    run = None
    if not cfg.wandb_off:
        run = wandb.init(
            project=cfg.wandb_project or "oddball_eeg_fmri_paired",
            name=cfg.wandb_run_name,
            config={**asdict(cfg), "model": "Translator+CBraMod(frozen)+BrainLM(frozen)"},
        )
        try:
            wandb.watch(translator, log="gradients", log_freq=200)
        except Exception:
            pass

    # Resume (auto-detect last checkpoint if available)
    start_epoch = 1
    global_step = 0
    best_val = float('inf')
    best_path = Path(cfg.output_dir) / 'translator_best.pt'
    last_path = Path(cfg.output_dir) / 'translator_last.pt'
    resume_path: Optional[Path] = None
    if cfg.resume is not None and Path(cfg.resume).exists():
        resume_path = Path(cfg.resume)
    elif last_path.exists():
        resume_path = last_path
    if resume_path is not None:
        print(f"Resuming from {resume_path}")
        last_epoch, global_step, best_val = load_checkpoint(resume_path, translator, optim, scaler, device)
        start_epoch = max(1, int(last_epoch) + 1)

    # ---- Fixed tri-mix settings (no flags) ----
    TRI_RATIOS = (0.34, 0.33, 0.33)        # (both, single, partial)
    PARTIAL_VISIBLE_RANGE = (0.30, 0.80)   # random visible fraction per partial sample
    SINGLE_SIDE = "random"                 # "random" | "eeg_only" | "fmri_only"

    def run_epoch(mode: str, epoch: int):
        nonlocal global_step, best_val
        is_train = (mode == 'train')
        translator.train(is_train)

        total_eeg_loss = 0.0
        total_fmri_loss = 0.0
        steps = 0

        rsum = sum(TRI_RATIOS)
        r_both, r_single, r_partial = [r/rsum for r in TRI_RATIOS]

        data_iter = dl_train if is_train else dl_val
        iterator = data_iter if tqdm is None else tqdm(data_iter, total=len(data_iter), desc=f"{mode} {epoch}/{cfg.num_epochs}", leave=False)

        for i, batch in enumerate(iterator):
            x_eeg = batch['eeg_window'].to(device, non_blocking=True)   # (B,C,P,S)
            fmri_t = batch['fmri_window'].to(device, non_blocking=True) # (B,T,V)

            B, _, P, _ = x_eeg.shape
            _, T, V = fmri_t.shape

            # ----- assign per-sample conditions: 0=both, 1=single, 2=partial
            nb_both   = int(round(B * r_both))
            nb_single = int(round(B * r_single))
            nb_part   = max(0, B - nb_both - nb_single)

            cond = torch.empty(B, dtype=torch.int64, device=device)
            perm = torch.randperm(B, device=device)
            cond[perm[:nb_both]] = 0
            cond[perm[nb_both:nb_both+nb_single]] = 1
            cond[perm[nb_both+nb_single:]] = 2

            # single: which side remains?
            if SINGLE_SIDE == 'random':
                keep_eeg = (torch.rand(B, device=device) < 0.5)
            else:
                keep_eeg = torch.full((B,), SINGLE_SIDE == 'eeg_only', dtype=torch.bool, device=device)
            miss_eeg   = (cond==1) & (~keep_eeg)   # EEG missing
            miss_fmri  = (cond==1) & ( keep_eeg)   # fMRI missing
            is_partial = (cond==2)

            # partial: which modality?
            partial_eeg = torch.zeros(B, dtype=torch.bool, device=device)
            partial_fmr = torch.zeros(B, dtype=torch.bool, device=device)
            if is_partial.any():
                choose_eeg = (torch.rand(B, device=device) < 0.5)
                partial_eeg = is_partial & choose_eeg
                partial_fmr = is_partial & (~choose_eeg)

            # random visible fractions (unused entries ignored)
            lo, hi = PARTIAL_VISIBLE_RANGE
            frac_vis = torch.empty(B, device=device).uniform_(lo, hi)

            # EEG time masks (B,P) for partial_eeg
            M_eeg_time = torch.zeros(B, P, dtype=torch.bool, device=device)
            if partial_eeg.any():
                idx = torch.nonzero(partial_eeg, as_tuple=False).squeeze(1)
                M_eeg_time[idx] = _rand_block_mask_time_per_sample(
                    int(idx.numel()), P, frac_vis[idx], device
                )

            # fMRI time masks (B,T,V) for partial_fmr
            M_fmri = torch.zeros(B, T, V, dtype=torch.bool, device=device)
            if partial_fmr.any():
                idx = torch.nonzero(partial_fmr, as_tuple=False).squeeze(1)
                Mt = _rand_block_mask_time_per_sample(
                    int(idx.numel()), T, frac_vis[idx], device
                )  # (b_p, T)
                M_fmri[idx, :, :] = Mt.unsqueeze(-1).expand(-1, T, V)

            # ----- observed inputs to encoders (no leakage)
            x_eeg_obs = x_eeg.clone()
            x_eeg_obs[miss_eeg] = 0.0
            # Vectorized masking for EEG partial
            if partial_eeg.any():
                be = torch.nonzero(partial_eeg, as_tuple=False).squeeze(1)  # (b_p,)
                Mm = M_eeg_time[be].unsqueeze(1).unsqueeze(-1)              # (b_p,1,P,1)
                x_eeg_obs[be] = x_eeg_obs[be].masked_fill(Mm, 0.0)

            fmri_obs = fmri_t.clone()
            fmri_obs[miss_fmri] = 0.0
            fmri_obs[M_fmri] = 0.0

            # ----- frozen latents
            with torch.no_grad():
                eeg_latents = frozen_cbramod.extract_latents(x_eeg_obs)  # (L,B,P,C,D)

                fmri_padded = pad_timepoints_for_brainlm_torch(fmri_obs, patch_size=20)  # (B,Tp,V)
                signal_vectors = fmri_padded.permute(0, 2, 1).contiguous()               # (B,V, Tp)

                # coords (A424 normalized if V==424, else zeros) with robust search
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
                    max_abs = float(np.max(np.abs(coords_np))) or 1.0
                    coords_np = coords_np / max_abs
                    if V == 424:
                        xyz = torch.from_numpy(coords_np).to(device=device, dtype=torch.float32).unsqueeze(0).repeat(B, 1, 1)
                    else:
                        xyz = torch.zeros(B, V, 3, device=device)
                except Exception:
                    xyz = torch.zeros(B, V, 3, device=device)

                fmri_latents = frozen_brainlm.extract_latents(signal_vectors, xyz, noise=None)  # (L,B,Ttok,H)

            # extra safety: zero latents when fully missing
            if miss_eeg.any():
                eeg_latents[:, miss_eeg, :, :, :] = 0.0
            if miss_fmri.any():
                fmri_latents[:, miss_fmri, :, :] = 0.0

            # ----- group EEG and forward
            eeg_latents_t = group_eeg_latents_seconds(eeg_latents, cfg.eeg_seconds_per_token)  # (L,B,Pe,C,D)
            x_eeg_grp     = group_eeg_signal_seconds(x_eeg,  cfg.eeg_seconds_per_token)        # (B,C,Pe,S)

            if is_train:
                optim.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda', enabled=(cfg.amp and device.type == 'cuda')):
                recon_eeg, recon_fmri = translator(
                    eeg_latents_t, fmri_latents,
                    fmri_target_T=int(fmri_t.shape[1]),
                    fmri_target_V=int(fmri_t.shape[2]),
                )

                # ----- losses
                # EEG: masked-only for partial_eeg (grouped mask)
                eeg_diff = (recon_eeg - x_eeg_grp) ** 2                     # (B,C,Pe,S)
                eeg_diff_cs = eeg_diff.mean(dim=(1,3))                      # (B,Pe)
                M_eeg_grp = _group_mask_seconds(M_eeg_time, cfg.eeg_seconds_per_token)  # (B,Pe)

                eeg_full = eeg_diff_cs.mean(dim=1)                          # (B,)
                masked_sum_e = (eeg_diff_cs * M_eeg_grp.float()).sum(dim=1)
                masked_cnt_e = M_eeg_grp.sum(dim=1).clamp_min(1)
                eeg_mask_only = masked_sum_e / masked_cnt_e
                eeg_mse = torch.where(partial_eeg, eeg_mask_only, eeg_full)

                # fMRI: masked-only for partial_fmr
                fmri_diff = (recon_fmri - fmri_t) ** 2                      # (B,T,V)
                fmri_full = fmri_diff.flatten(1).mean(dim=1)                # (B,)
                masked_sum_f = (fmri_diff * M_fmri.float()).flatten(1).sum(dim=1)
                masked_cnt_f = M_fmri.flatten(1).sum(dim=1).clamp_min(1)
                fmri_mask_only = masked_sum_f / masked_cnt_f
                fmri_mse = torch.where(partial_fmr, fmri_mask_only, fmri_full)

                # emphasize cross-modal when single; otherwise equal
                alpha_miss, beta_pres = 1.0, 1.0
                w_eeg = torch.where(miss_eeg, torch.as_tensor(alpha_miss, device=device),
                                              torch.as_tensor(beta_pres,  device=device))
                w_fmr = torch.where(miss_fmri, torch.as_tensor(alpha_miss, device=device),
                                              torch.as_tensor(beta_pres,  device=device))

                loss_eeg  = (w_eeg * eeg_mse).mean()
                loss_fmri = (w_fmr * fmri_mse).mean()
                loss = cfg.eeg_loss_w * loss_eeg + cfg.fmri_loss_w * loss_fmri

            if is_train:
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(translator.parameters(), 1.0)
                scaler.step(optim)
                scaler.update()
                scheduler.step()
                global_step += 1

            total_eeg_loss += float(loss_eeg.detach().cpu())
            total_fmri_loss += float(loss_fmri.detach().cpu())
            steps += 1

            # progress bar
            if tqdm is not None:
                iterator.set_postfix_str(f"loss:{float(loss.detach().cpu()):.3f} lr:{optim.param_groups[0]['lr']:.2e}")

            if is_train and not cfg.wandb_off:
                wandb.log({
                    "train/step_total_loss": float(loss.detach().cpu()),
                    "train/step_eeg_loss": float(loss_eeg.detach().cpu()),
                    "train/step_fmri_loss": float(loss_fmri.detach().cpu()),
                    "train/lr": float(optim.param_groups[0]['lr']),
                    "train/frac_both":   float((cond==0).float().mean().cpu()),
                    "train/frac_single": float((cond==1).float().mean().cpu()),
                    "train/frac_partial":float((cond==2).float().mean().cpu()),
                }, step=global_step)

        scalars = {
            f'{mode}_eeg_loss': total_eeg_loss / max(1, steps),
            f'{mode}_fmri_loss': total_fmri_loss / max(1, steps),
        }
        return scalars

    # ---- epochs
    for epoch in range(start_epoch, cfg.num_epochs + 1):
        t0 = time.time()
        tr = run_epoch('train', epoch)
        va = run_epoch('val', epoch)
        dur = time.time() - t0

        print(f"Epoch {epoch:02d} | {dur:.1f}s | "
              f"train EEG {tr['train_eeg_loss']:.6f} FMRI {tr['train_fmri_loss']:.6f} | "
              f"val EEG {va['val_eeg_loss']:.6f} FMRI {va['val_fmri_loss']:.6f}")

        # Save "last"
        last_path = Path(cfg.output_dir) / 'translator_last.pt'
        best_path = Path(cfg.output_dir) / 'translator_best.pt'
        save_checkpoint(last_path, translator, optim, scaler, cfg, epoch, global_step, best_val)

        # Best by sum
        val_score = va['val_eeg_loss'] + va['val_fmri_loss']
        if val_score < best_val:
            best_val = val_score
            save_checkpoint(best_path, translator, optim, scaler, cfg, epoch, global_step, best_val)
            print(f"  ✅ Saved BEST checkpoint to {best_path}")
            if not cfg.wandb_off:
                try:
                    wandb.summary["best_val_total_loss"] = best_val
                except Exception:
                    pass

        if not cfg.wandb_off:
            wandb.log({
                "epoch": epoch,
                "time/epoch_seconds": dur,
                "train/eeg_loss": tr['train_eeg_loss'],
                "train/fmri_loss": tr['train_fmri_loss'],
                "val/eeg_loss": va['val_eeg_loss'],
                "val/fmri_loss": va['val_fmri_loss'],
                "val/total_loss": val_score,
                "best/val_total_loss": best_val,
            }, step=global_step)

    if not cfg.wandb_off:
        wandb.finish()

def dry_run_debug(
    device: torch.device,
    window_sec: int = 30,
    target_fs: int = 200,
    eeg_seconds_per_token: int = 40,
    eeg_channels: int = 34,
    fmri_voxels: int = 424,
    fmri_hidden_size: int = 256,
    fmri_n_layers: int = 5,
    batch_size: int = 2,
    d_model: int = 256,
    n_heads: int = 8,
    d_ff: int = 1024,
    dropout: float = 0.1,
):
    """
    Run a synthetic forward pass through the exact module stack and print shapes.
    Mirrors training-time shapes & grouping.
    """

    def shp(x): return tuple(int(d) for d in x.shape)

    # -------- Derived dims (match training defaults) --------
    P = int(window_sec)                # EEG time tokens at 1s resolution
    S = int(target_fs)                 # EEG samples per second
    T = int(round(window_sec / 2.0))   # fMRI time points (TR = 2s)
    V = int(fmri_voxels)               # voxels
    g = max(1, min(eeg_seconds_per_token, P))
    Pe = max(1, P // g)                # grouped EEG tokens
    L_eeg = 12                         # matches FrozenCBraMod n_layer
    L_fmri = int(fmri_n_layers)        # ~ num_hidden_layers + 1 used in training
    Ttok_dummy = 512                   # Brain tokens fed to fMRI adapter

    print("\n=== DRY-RUN DEBUG (synthetic tensors) ===")
    print(f"Batch size B        : {batch_size}")
    print(f"EEG -> C,P,S        : {eeg_channels}, {P}, {S}")
    print(f"EEG grouping g, Pe  : {g}, {Pe}")
    print(f"fMRI -> T,V         : {T}, {V}")
    print(f"BrainLM dummy tokens: {Ttok_dummy} (len of token seq into fMRI adapter)")
    print(f"Latent layers (EEG,fMRI): {L_eeg}, {L_fmri}")
    print(f"d_model, n_heads, d_ff : {d_model}, {n_heads}, {d_ff}\n")

    # -------- Synthetic latents (already GROUPED for EEG) --------
    eeg_latents = torch.randn(L_eeg, batch_size, Pe, eeg_channels, S, device=device)
    fmri_latents = torch.randn(L_fmri, batch_size, Ttok_dummy, fmri_hidden_size, device=device)

    translator = TranslatorModel(
        eeg_channels=eeg_channels,
        eeg_patch_num=Pe,
        eeg_n_layers=12,
        eeg_input_dim=S,
        fmri_n_layers=L_fmri,
        fmri_hidden_size=fmri_hidden_size,
        fmri_tokens_target=V * T,   # informational
        fmri_target_T=T,
        fmri_target_V=V,
        d_model=d_model,
        n_heads=n_heads,
        d_ff=d_ff,
        dropout=dropout,
        voxel_count=V,
        debug=True,
    ).to(device)
    translator.eval()

    total_params = sum(p.numel() for p in translator.parameters())
    trainable_params = sum(p.numel() for p in translator.parameters() if p.requires_grad)
    print(f"Translator params (total/trainable): {total_params:,} / {trainable_params:,}\n")

    with torch.no_grad():
        eeg_adapt = translator.adapter_eeg(eeg_latents)      # (B, Neeg, D)
        fmri_adapt = translator.adapter_fmri(fmri_latents)   # (B, Nfmri, D)
        print("[adapter_eeg]  ->", shp(eeg_adapt))
        print("[adapter_fmri] ->", shp(fmri_adapt))

        _, eeg_hi = translator.eeg_encoder(eeg_adapt, eeg_adapt)
        _, fmr_hi = translator.fmri_encoder(fmri_adapt, fmri_adapt)
        print("[eeg_encoder]  ->", shp(eeg_hi))
        print("[fmri_encoder] ->", shp(fmr_hi))

        eeg_c, fmr_c, _ = translator.compressor(eeg_hi, fmr_hi)
        print("[compressor] eeg_c:", shp(eeg_c), " fmri_c:", shp(fmr_c))
        fused = translator.cross_attn(eeg_c, fmr_c)
        print("[cross_attn] fused:", shp(fused))

        eeg_layers = translator.eeg_decoder(fused)           # (L,B,P,C,S)
        print("[eeg_decoder] layers:", shp(eeg_layers))
        eeg_signal = eeg_layers.mean(dim=0).permute(0, 2, 1, 3).contiguous()  # (B,C,P,S)
        print("[eeg_signal] ->", shp(eeg_signal), "(expect (B,C,Pe,S) =",
              (batch_size, eeg_channels, Pe, S), ")")

        dec_tokens, eff_len, _ = translator.fmri_decoder(fused)  # (L,B,eff_len,H)
        print("[fmri_decoder] tokens:", shp(dec_tokens), "eff_len:", int(eff_len))
        dec_mean = dec_tokens.mean(dim=0)                        # (B,eff_len,H)
        token_scalar = translator.fmri_token_head(dec_mean)      # (B,eff_len,1)
        print("[token_head]  ->", shp(token_scalar))
        token_scalar = token_scalar.transpose(1, 2)              # (B,1,eff_len)

        target_tokens = T * V
        if eff_len != target_tokens:
            if eff_len < target_tokens:
                token_scalar = nn.functional.interpolate(token_scalar, size=target_tokens,
                                                        mode="linear", align_corners=False)
                print(f"[resample] upsample  {int(eff_len)} -> {target_tokens}")
            else:
                token_scalar = nn.functional.adaptive_avg_pool1d(token_scalar, target_tokens)
                print(f"[resample] downsample {int(eff_len)} -> {target_tokens}")

        token_scalar = token_scalar.transpose(1, 2).contiguous().squeeze(-1)  # (B,target_tokens)
        fmri_signal = token_scalar.view(batch_size, T, V)                     # (B,T,V)
        fmri_signal = torch.tanh(fmri_signal)
        fmri_signal = translator.fmri_out_scale * fmri_signal + translator.fmri_out_bias
        print("[fmri_signal] ->", shp(fmri_signal), "(expect (B,T,V) =",
              (batch_size, T, V), ")")

    print("\n✅ DRY-RUN DONE\n")

# -----------------------------
# CLI
# -----------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description='Train translator-decoder on paired Oddball EEG+fMRI with tri-mix masking')
    # Config file (YAML/JSON) support — values from file become defaults; CLI overrides
    parser.add_argument('--config', type=str, default=None, help='Path to YAML/JSON config. If YAML has sections, uses "train".')
    parser.add_argument('--eeg_root', type=str)
    parser.add_argument('--fmri_root', type=str)
    parser.add_argument('--a424_label_nii', type=str, default=str(THIS_DIR / 'BrainLM' / 'resources' / 'atlases' / 'A424_resampled_to_bold.nii.gz'))
    # Set your local defaults or change here:
    parser.add_argument('--cbramod_weights', type=str, default=r"D:\Neuroinformatics_research_2025\MNI_templates\CBraMod\pretrained_weights\pretrained_weights.pth")
    parser.add_argument('--brainlm_model_dir', type=str, default=r"D:\Neuroinformatics_research_2025\MNI_templates\BrainLM\pretrained_models\2023-06-06-22_15_00-checkpoint-1400")
    parser.add_argument('--output_dir', type=str, default=str(THIS_DIR / 'translator_runs'))
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--window_sec', type=int, default=30)
    parser.add_argument('--stride_sec', type=int, default=None, help='Stride in seconds between windows (controls overlap)')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--fmri_norm', type=str, default='zscore', choices=['zscore','psc','mad','none'])
    # wandb
    parser.add_argument('--wandb_project', type=str, default=None)
    parser.add_argument('--wandb_run_name', type=str, default=None)
    parser.add_argument('--wandb_off', action='store_true')
    # loss weights
    parser.add_argument('--eeg_loss_w', type=float, default=1.0)
    parser.add_argument('--fmri_loss_w', type=float, default=1.0)
    # EEG token grouping
    parser.add_argument('--eeg_seconds_per_token', type=int, default=40)
    # resume
    parser.add_argument('--resume', type=str, default=None, help='Path to a full checkpoint to resume training')
    # explicit subject splits (can be provided via YAML or CLI)
    parser.add_argument('--train_subjects', type=int, nargs='*', default=None,
                        help='Explicit train subject IDs (e.g., --train_subjects 1 2 3). Overrides auto split.')
    parser.add_argument('--val_subjects', type=int, nargs='*', default=None,
                        help='Explicit val subject IDs (e.g., --val_subjects 13). Overrides auto split.')
    parser.add_argument('--test_subjects', type=int, nargs='*', default=None,
                        help='Explicit test subject IDs (e.g., --test_subjects 14 15 17). Overrides auto split.')
    parser.add_argument('--dry_run', action='store_true',
                        help='Run a synthetic forward pass and print shapes; no data or training.')

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
        if isinstance(file_cfg, dict) and 'train' in file_cfg:
            file_cfg = file_cfg['train'] or {}
        parser.set_defaults(**(file_cfg or {}))

    args = parser.parse_args()

    # Strict path checks: fail fast if missing
    if not Path(args.a424_label_nii).exists():
        raise FileNotFoundError(f"A424 atlas not found: {args.a424_label_nii}")
    if not Path(args.cbramod_weights).exists():
        raise FileNotFoundError(f"CBraMod weights not found: {args.cbramod_weights}")
    if not Path(args.brainlm_model_dir).exists():
        raise FileNotFoundError(f"BrainLM checkpoint dir not found: {args.brainlm_model_dir}")

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
        stride_sec=args.stride_sec,
        num_workers=args.num_workers,
        amp=args.amp,
        fmri_norm=args.fmri_norm,
        debug=args.debug,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        wandb_off=args.wandb_off,
        eeg_loss_w=args.eeg_loss_w,
        fmri_loss_w=args.fmri_loss_w,
        eeg_seconds_per_token=args.eeg_seconds_per_token,
        resume=Path(args.resume) if args.resume else None,
        train_subjects=args.train_subjects,
        val_subjects=args.val_subjects,
        test_subjects=args.test_subjects,
    )
    if args.dry_run:
        dev = torch.device(args.device)
        dry_run_debug(
            device=dev,
            window_sec=cfg.window_sec,
            target_fs=cfg.target_fs,
            eeg_seconds_per_token=cfg.eeg_seconds_per_token,
            eeg_channels=34,              # matches translator build
            fmri_voxels=424,              # fixed in translator
            fmri_hidden_size=256,
            fmri_n_layers=5,
            batch_size=2,                 # tweak to taste
            d_model=256, n_heads=8, d_ff=1024, dropout=0.1,
        )
        return

    train_loop(cfg)

if __name__ == '__main__':
    main()
