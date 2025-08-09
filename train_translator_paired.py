#!/usr/bin/env python3
"""
Train the adapter_main_translatordecoder using paired, aligned Oddball EEG+fMRI windows.

Key changes vs earlier version
- Uses a SINGLE PairedAlignedDataset / DataLoader -> every batch always has both EEG and fMRI from same (sub, run, time)
- Overlap control via --stride_sec (e.g., window_sec=40, stride_sec=10 -> 30s overlap)
- Saves FULL training checkpoints (translator + optimizer + scaler + config + rng + metrics)
- Optional --resume to pick up training exactly where you left off
"""

from __future__ import annotations

import os
import sys
import json
import time
import argparse
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# wandb
import wandb
try:
    from tqdm import tqdm
except Exception:
    tqdm = None

# Local paths to import vendored modules first
THIS_DIR = Path(__file__).parent
REPO_ROOT = THIS_DIR.parent
CBRAMOD_DIR = REPO_ROOT / "CBraMod"
BRAINLM_DIR = REPO_ROOT / "BrainLM"

# Prefer local copies inside Multi_modal_NTM if present
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(THIS_DIR / "CBraMod"))
sys.path.insert(0, str(THIS_DIR / "BrainLM"))
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

# Models (CBraMod from local vendored path if available)
from models.cbramod import CBraMod  # type: ignore
from brainlm_mae.modeling_brainlm import BrainLMForPretraining  # type: ignore
from brainlm_mae.configuration_brainlm import BrainLMConfig  # type: ignore

# Data (paired + aligned, with optional overlap)
from data_oddball import (
    pad_timepoints_for_brainlm_torch,
    load_a424_coords,
    PairedAlignedDataset,
    collate_paired,
)

# -----------------------------
# Utility: Reproducibility
# -----------------------------
def seed_all(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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

        # Decoders
        self.eeg_decoder = EEGDecodingAdapter(
            channels=eeg_channels,
            patch_num=eeg_patch_num,
            n_layers=eeg_n_layers,
            patch_size=eeg_input_dim,
            d_model=d_model,
        )

        # fMRI factorized head
        self.fmri_proj = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(d_model * 2, d_model), nn.LayerNorm(d_model)
        )
        self.fmri_depthwise = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=3, padding=1, groups=d_model)
        self.fmri_voxel_embed: Optional[nn.Embedding] = None

    def _dbg(self, msg: str) -> None:
        if self.debug:
            print(msg, flush=True)

    def forward(
        self,
        eeg_latents: torch.Tensor,        # (L,B,S,C,D)
        fmri_latents: torch.Tensor,       # (L,B,T,H)
        fmri_target_T: int,
        fmri_target_V: int,
    ):
        self._dbg("\n[Forward] start")
        self._dbg(f"  EEG latents in: {tuple(eeg_latents.shape)} (L,B,S,C,D)")
        self._dbg(f"  fMRI latents in: {tuple(fmri_latents.shape)} (L,B,T,H)")

        eeg_adapted = self.adapter_eeg(eeg_latents)           # (B, N_eeg_tokens, D)
        fmri_adapted = self.adapter_fmri(fmri_latents)        # (B, N_fmri_tokens, D)
        self._dbg(f"  EEG after adapter: {tuple(eeg_adapted.shape)}")
        self._dbg(f"  fMRI after adapter: {tuple(fmri_adapted.shape)}")

        eeg_lower_enc, eeg_higher_enc = self.eeg_encoder(eeg_adapted, eeg_adapted)
        fmri_lower_enc, fmri_higher_enc = self.fmri_encoder(fmri_adapted, fmri_adapted)
        self._dbg(f"  EEG enc higher: {tuple(eeg_higher_enc.shape)} | fMRI enc higher: {tuple(fmri_higher_enc.shape)}")

        eeg_compressed, fmri_compressed, _ = self.compressor(eeg_higher_enc, fmri_higher_enc)
        self._dbg(f"  After compressor EEG: {tuple(eeg_compressed.shape)} fMRI: {tuple(fmri_compressed.shape)}")

        fused_output = self.cross_attn(eeg_compressed, fmri_compressed)  # (B, Tfused, D)
        self._dbg(f"  Fused output: {tuple(fused_output.shape)}")

        # EEG decode
        eeg_reconstructed_layers = self.eeg_decoder(fused_output)        # (L,B,P,C,S)
        eeg_signal = eeg_reconstructed_layers.mean(dim=0).permute(0, 2, 1, 3).contiguous()  # (B,C,P,S)
        self._dbg(f"  EEG signal out: {tuple(eeg_signal.shape)}")

        # fMRI decode
        H = self.fmri_proj(fused_output)          # (B, Tfused, D)
        Hc = H.transpose(1, 2).contiguous()       # (B, D, Tfused)
        Hc = self.fmri_depthwise(Hc)              # (B, D, Tfused)
        if Hc.shape[-1] != fmri_target_T:
            Hc = nn.functional.interpolate(Hc, size=fmri_target_T, mode='linear', align_corners=False)
        H = Hc.transpose(1, 2).contiguous()       # (B, T, D)

        if (self.fmri_voxel_embed is None) or (self.fmri_voxel_embed.num_embeddings != fmri_target_V) or (self.fmri_voxel_embed.embedding_dim != H.shape[-1]):
            self.fmri_voxel_embed = nn.Embedding(fmri_target_V, H.shape[-1]).to(H.device)
        E = self.fmri_voxel_embed.weight          # (V, D)
        fmri_signal = torch.matmul(H, E.t())      # (B, T, V)
        self._dbg(f"  fMRI signal out: {tuple(fmri_signal.shape)}")

        self._dbg("[Forward] end\n")
        return eeg_signal, fmri_signal


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
    stride_sec: Optional[int] = None
    batch_size: int = 2
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
    # EEG tokenization
    eeg_seconds_per_token: int = 40
    # resume
    resume: Optional[Path] = None


def save_checkpoint(path: Path, translator: TranslatorModel, optim: torch.optim.Optimizer,
                    scaler: torch.amp.GradScaler, cfg: TrainConfig, epoch: int,
                    global_step: int, best_val: float) -> None:
    ckpt = {
        "translator_state": translator.state_dict(),
        "optimizer_state": optim.state_dict(),
        "scaler_state": scaler.state_dict(),
        "config": asdict(cfg),
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
    ckpt = torch.load(path, map_location=device)
    translator.load_state_dict(ckpt["translator_state"])
    if "optimizer_state" in ckpt:
        optim.load_state_dict(ckpt["optimizer_state"])
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

    # Dataset / Loader: paired & aligned (optional overlap by stride_sec)
    ds_train = PairedAlignedDataset(
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
    )
    if len(ds_train) == 0:
        print("No paired aligned samples found. Check your paths and filename patterns.")
        return

    pin = device.type == 'cuda'
    dl = DataLoader(ds_train, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers,
                    collate_fn=collate_paired, pin_memory=pin, drop_last=False)

    # Frozen feature extractors
    seq_len_eeg = cfg.window_sec  # EEG tokens per second (1s per token before grouping)
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

    # EEG grouping (e.g., 40s->1 token by default)
    eeg_group = max(1, int(cfg.eeg_seconds_per_token))
    eeg_patch_num_grouped = max(1, int(cfg.window_sec) // eeg_group)

    translator = TranslatorModel(
        eeg_channels=34,
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
        debug=cfg.debug,
    ).to(device)

    optim = torch.optim.AdamW(
        [p for p in translator.parameters() if p.requires_grad],
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    mse_loss = nn.MSELoss()

    # wandb
    run = None
    if not cfg.wandb_off:
        run = wandb.init(
            project=cfg.wandb_project or "oddball_eeg_fmri_paired",
            name=cfg.wandb_run_name,
            config={**asdict(cfg), "model": "Translator+CBraMod(frozen)+BrainLM(frozen)"},
        )
        wandb.watch(translator, log="gradients", log_freq=200)

    scaler = torch.amp.GradScaler('cuda', enabled=(cfg.amp and device.type == 'cuda'))
    start_epoch = 1
    global_step = 0
    best_val = float('inf')

    # Resume?
    if cfg.resume is not None and Path(cfg.resume).exists():
        print(f"Resuming from {cfg.resume}")
        last_epoch, global_step, best_val = load_checkpoint(Path(cfg.resume), translator, optim, scaler, device)
        start_epoch = max(1, int(last_epoch) + 1)

    def group_eeg_latents_seconds(eeg_latents: torch.Tensor, seconds_per_token: int) -> torch.Tensor:
        # eeg_latents: (L,B,P,C,D) where P=window_sec (1s tokens)
        L, B, P, C, D = eeg_latents.shape
        g = max(1, min(seconds_per_token, P))
        P_grp = P // g
        x = eeg_latents[:, :, :P_grp * g].reshape(L, B, P_grp, g, C, D).mean(dim=3)
        return x

    def group_eeg_signal_seconds(x_eeg_sig: torch.Tensor, seconds_per_token: int) -> torch.Tensor:
        # x_eeg_sig: (B,C,P,S)
        B_, C_, P_, S_ = x_eeg_sig.shape
        g = max(1, min(seconds_per_token, P_))
        P_grp = P_ // g
        x = x_eeg_sig[:, :, :P_grp * g, :].reshape(B_, C_, P_grp, g, S_).mean(dim=3)
        return x

    def run_epoch(mode: str, epoch: int):
        nonlocal global_step, best_val
        is_train = mode == 'train'
        translator.train(is_train)

        total_eeg_loss = 0.0
        total_fmri_loss = 0.0
        steps = 0

        total_steps = len(dl)
        iterator = dl
        if tqdm is not None:
            iterator = tqdm(dl, total=total_steps, desc=f"{mode} {epoch}/{cfg.num_epochs}", leave=False)

        for i, batch in enumerate(iterator):
            x_eeg = batch['eeg_window'].to(device, non_blocking=True)   # (B,C,P,S) P=window_sec
            fmri_t = batch['fmri_window'].to(device, non_blocking=True) # (B,T,V) already normalized in dataset

            # Extract latents
            with torch.no_grad():
                eeg_latents = frozen_cbramod.extract_latents(x_eeg)      # (L,B,P,C,D)
                # fMRI prep for BrainLM
                fmri_padded = pad_timepoints_for_brainlm_torch(fmri_t, patch_size=20)  # (B,Tp,V)
                signal_vectors = fmri_padded.permute(0, 2, 1).contiguous()             # (B,V,Tp)

                # Use A424 coords if available; if V!=424, fall back to zeros
                Bf, Tf, Vf = fmri_t.shape
                try:
                    # Prefer local BrainLM resources first, then generic resources, then repo fallback
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
                    if Vf == 424:
                        xyz = torch.from_numpy(coords_np).to(device=device, dtype=torch.float32).unsqueeze(0).repeat(Bf, 1, 1)
                    else:
                        xyz = torch.zeros(Bf, Vf, 3, device=device)
                except Exception:
                    xyz = torch.zeros(Bf, Vf, 3, device=device)

                fmri_latents = frozen_brainlm.extract_latents(signal_vectors, xyz, noise=None)  # (L,B,T_tokens,H)

            # Group EEG tokens (and targets) to match decoder tokenization
            eeg_latents_t = group_eeg_latents_seconds(eeg_latents, cfg.eeg_seconds_per_token)  # (L,B,Pe,C,D)
            x_eeg_grp = group_eeg_signal_seconds(x_eeg, cfg.eeg_seconds_per_token)             # (B,C,Pe,S)

            if cfg.debug:
                print("\n[Batch Debug]")
                print(f"  mode={mode} B={x_eeg.shape[0]}")
                print(f"  EEG latents: {tuple(eeg_latents_t.shape)}  target x_eeg_grp: {tuple(x_eeg_grp.shape)}")
                print(f"  fMRI latents: {tuple(fmri_latents.shape)}  target fmri_t: {tuple(fmri_t.shape)}")

            if is_train:
                optim.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda', enabled=(cfg.amp and device.type == 'cuda')):
                recon_eeg, recon_fmri = translator(
                    eeg_latents_t, fmri_latents,
                    fmri_target_T=int(fmri_t.shape[1]),
                    fmri_target_V=int(fmri_t.shape[2]),
                )
                loss_eeg = mse_loss(recon_eeg, x_eeg_grp)
                loss_fmri = mse_loss(recon_fmri, fmri_t)
                loss = cfg.eeg_loss_w * loss_eeg + cfg.fmri_loss_w * loss_fmri

            if is_train:
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(translator.parameters(), 1.0)
                scaler.step(optim)
                scaler.update()
                global_step += 1

            total_eeg_loss += float(loss_eeg.detach().cpu())
            total_fmri_loss += float(loss_fmri.detach().cpu())
            steps += 1

            # Progress bar: epochs left and steps left
            if tqdm is not None:
                epochs_left = max(0, cfg.num_epochs - epoch)
                steps_left = (total_steps - (i + 1)) + epochs_left * total_steps
                iterator.set_postfix_str(f"left: {epochs_left}e {steps_left}s, loss:{float(loss.detach().cpu()):.3f}")

            if is_train and not cfg.wandb_off:
                wandb.log({
                    "train/step_total_loss": float(loss.detach().cpu()),
                    "train/step_eeg_loss": float(loss_eeg.detach().cpu()),
                    "train/step_fmri_loss": float(loss_fmri.detach().cpu()),
                    "train/eeg_loss_w": cfg.eeg_loss_w,
                    "train/fmri_loss_w": cfg.fmri_loss_w,
                }, step=global_step)

        scalars = {
            f'{mode}_eeg_loss': total_eeg_loss / max(1, steps),
            f'{mode}_fmri_loss': total_fmri_loss / max(1, steps),
        }
        return scalars

    best_path = Path(cfg.output_dir) / 'translator_best.pt'
    last_path = Path(cfg.output_dir) / 'translator_last.pt'

    for epoch in range(start_epoch, cfg.num_epochs + 1):
        t0 = time.time()
        tr = run_epoch('train', epoch)
        # In this paired setup we’re using a single dataset; if you want a held-out val split,
        # make a second PairedAlignedDataset on separate files. For now, treat train loss as val proxy.
        va = {"val_eeg_loss": tr['train_eeg_loss'], "val_fmri_loss": tr['train_fmri_loss']}
        dur = time.time() - t0

        print(f"Epoch {epoch:02d} | {dur:.1f}s | "
              f"train EEG {tr['train_eeg_loss']:.6f} FMRI {tr['train_fmri_loss']:.6f} | "
              f"val EEG {va['val_eeg_loss']:.6f} FMRI {va['val_fmri_loss']:.6f}")

        # Save "last" full checkpoint every epoch
        save_checkpoint(last_path, translator, optim, scaler, cfg, epoch, global_step, best_val)

        # Track best by (val_eeg + val_fmri)
        val_score = va['val_eeg_loss'] + va['val_fmri_loss']
        if val_score < best_val:
            best_val = val_score
            save_checkpoint(best_path, translator, optim, scaler, cfg, epoch, global_step, best_val)
            print(f"  ✅ Saved BEST checkpoint to {best_path}")
            if not cfg.wandb_off:
                wandb.summary["best_val_total_loss"] = best_val
                try:
                    wandb.save(str(best_path))
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

    # Optional: finish wandb
    if not cfg.wandb_off:
        wandb.finish()


def main() -> None:
    parser = argparse.ArgumentParser(description='Train translator-decoder on paired Oddball EEG+fMRI (aligned, optional overlap)')
    parser.add_argument('--eeg_root', type=str, required=True)
    parser.add_argument('--fmri_root', type=str, required=True)
    parser.add_argument('--a424_label_nii', type=str, default=str(THIS_DIR / 'BrainLM' / 'resources' / 'atlases' / 'A424_resampled_to_bold.nii.gz'))
    parser.add_argument('--cbramod_weights', type=str, default=str(THIS_DIR / 'CBraMod' / 'pretrained_weights' / 'pretrained_weights.pth'))
    parser.add_argument('--brainlm_model_dir', type=str, default=str(THIS_DIR / 'BrainLM' / 'pretrained_models' / '2023-06-06-22_15_00-checkpoint-1400'))
    parser.add_argument('--output_dir', type=str, default=str(THIS_DIR / 'translator_runs'))
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--window_sec', type=int, default=30)
    parser.add_argument('--stride_sec', type=int, default=None, help='Stride in seconds between paired windows (controls overlap)')
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

    args = parser.parse_args()

    # Resolve defaults/fallbacks if paths are missing
    # a424_label_nii
    if not Path(args.a424_label_nii).exists():
        candidates = [
            THIS_DIR / 'BrainLM' / 'resources' / 'atlases' / 'A424_resampled_to_bold.nii.gz',
            THIS_DIR / 'BrainLM' / 'resources' / 'atlases' / 'A424.nii.gz',
            THIS_DIR / 'resources' / 'atlases' / 'A424_resampled_to_bold.nii.gz',
            THIS_DIR / 'resources' / 'atlases' / 'A424.nii.gz',
            REPO_ROOT / 'BrainLM' / 'A424_resampled_to_bold.nii.gz',
        ]
        for c in candidates:
            if c.exists():
                args.a424_label_nii = str(c)
                break
        else:
            raise FileNotFoundError(f"A424 atlas not found: {args.a424_label_nii}")

    # CBraMod weights
    if not Path(args.cbramod_weights).exists():
        fallback_cbramod = CBRAMOD_DIR / 'pretrained_weights' / 'pretrained_weights.pth'
        if fallback_cbramod.exists():
            args.cbramod_weights = str(fallback_cbramod)
        else:
            alt = THIS_DIR / 'pretrained_weights' / 'pretrained_weights.pth'
            if alt.exists():
                args.cbramod_weights = str(alt)
            else:
                raise FileNotFoundError(f"CBraMod weights not found: {args.cbramod_weights}")

    # BrainLM checkpoint dir
    if not Path(args.brainlm_model_dir).exists():
        candidates = [
            THIS_DIR / 'BrainLM' / 'pretrained_models' / '2023-06-06-22_15_00-checkpoint-1400',
            REPO_ROOT / 'BrainLM' / 'pretrained_models' / '2023-06-06-22_15_00-checkpoint-1400',
            THIS_DIR / 'pretrained_models' / '2023-06-06-22_15_00-checkpoint-1400',
        ]
        for c in candidates:
            if c.exists():
                args.brainlm_model_dir = str(c)
                break
        else:
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
    )

    train_loop(cfg)


if __name__ == '__main__':
    main()
