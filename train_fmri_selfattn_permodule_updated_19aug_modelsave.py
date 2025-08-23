#!/usr/bin/env python3
"""
Stage-wise training (fMRI-only, self-attention with axial RoPE; no cross-attn, no EEG path),
run one stage at a time with train/test modes and per-module checkpoints.

Changes vs. previous:
- Killed fixed 512 tokens. Adapter targets true S = T×V.
- Removed additive/sinusoidal 1-D PEs. Only axial RoPE inside attention.
- Encoder APIs require (T, V).
- FIXED DECODER: fMRIDecodingAdapter2D predicts (B,T,V) by producing V channels
  first and resizing along time only, removing rank-1-over-voxels pathology.

Everything else (dataset, diagnostics) unchanged, except they now see real S.
"""

from __future__ import annotations
import os, json, math, argparse, time, platform, sys, csv
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Iterable, List, Dict, Any, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# ---- Matplotlib (headless) ----
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from tqdm import tqdm  # progress bar
except Exception:
    tqdm = None

# --- Weights & Biases (optional) ---
try:
    import wandb
except Exception:
    wandb = None

# -----------------------------
# Repo paths
# -----------------------------
THIS_DIR = Path(__file__).parent
REPO_ROOT = THIS_DIR.parent
CBRAMOD_DIR = REPO_ROOT / "CBraMod"
BRAINLM_DIR = REPO_ROOT / "BrainLM"

sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(CBRAMOD_DIR))
sys.path.insert(0, str(THIS_DIR / "CBraMod"))

candidate_blm_roots = [
    BRAINLM_DIR,
    THIS_DIR / "BrainLM",
    Path(os.environ.get("BRAINLM_DIR", "")),
    Path(os.environ.get("BRAINLM_DIR", "")) / "brainlm_mae",
]
for p in candidate_blm_roots:
    if not p:
        continue
    p = Path(p)
    if (p / "brainlm_mae").is_dir():
        sys.path.insert(0, str(p))
        break
    if p.name == "brainlm_mae" and p.is_dir():
        sys.path.insert(0, str(p.parent))
        break

# -----------------------------
# Local modules (UPDATED)
# -----------------------------
from module import (  # type: ignore
    fMRIInputAdapterConv1d,
    HierarchicalEncoder,
    fMRIDecodingAdapter2D,      # <-- use the fixed 2-D decoder
)

# BrainLM imports
from brainlm_mae.modeling_brainlm import BrainLMForPretraining  # type: ignore
from brainlm_mae.configuration_brainlm import BrainLMConfig     # type: ignore

# Data utilities
from data_oddball import (  # type: ignore
    PairedAlignedDataset,
    collate_paired,
    pad_timepoints_for_brainlm_torch,
    load_a424_coords,
    collect_common_sr_keys,
    fixed_subject_keys,
)

# -----------------------------
# Utils / config
# -----------------------------
def seed_all(seed:int):
    np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def fmt_mem() -> str:
    if not torch.cuda.is_available(): return "cuda:N/A"
    a = torch.cuda.memory_allocated() / (1024**2)
    r = torch.cuda.memory_reserved() / (1024**2)
    return f"cuda: alloc={a:.1f}MB, reserv={r:.1f}MB"

def _torch_load_compat(path: Path, map_location, *, allow_weights_only: bool = False):
    try:
        return torch.load(str(path), map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location=map_location)
    except Exception as e1:
        if allow_weights_only:
            try:
                import torch.serialization as ts
                try:
                    from torch.torch_version import TorchVersion  # type: ignore
                    ts.add_safe_globals([TorchVersion])
                except Exception:
                    pass
                return torch.load(str(path), map_location=map_location, weights_only=True)
            except Exception as e2:
                raise RuntimeError(f"Failed to load {path.name} via classic and safe weights_only paths:\n{e1}\n{e2}") from e2
        raise

@dataclass
class TrainCfg:
    # data / models
    eeg_root: Path
    fmri_root: Path
    a424_label_nii: Path
    brainlm_model_dir: Path
    out_dir: Path

    # training + arch
    device: str = "cpu"
    seed: int = 42
    window_sec: int = 30
    tr: float = 2.0
    fmri_voxels: int = 424
    batch_size: int = 8
    num_workers: int = 0
    stride_sec: Optional[int] = None
    fmri_norm: str = "zscore"

    # token dims
    d_model: int = 128
    n_heads: int = 4
    d_ff: int = 512
    dropout: float = 0.1

    # stages
    stage1_epochs: int = 3
    stage2_epochs: int = 5
    stage3_epochs: int = 3
    lr_stage1: float = 1e-4
    lr_stage2: float = 8e-5
    lr_stage3: float = 5e-5
    weight_decay: float = 1e-4
    amp: bool = False
    train_fmri_affine_stage3: bool = False

    # fixed subjects (optional)
    train_subjects: Optional[List[int]] = None
    val_subjects:   Optional[List[int]] = None
    test_subjects:  Optional[List[int]] = None

    # debug controls
    debug: bool = False
    profile_first_n: int = 0
    max_batches_per_epoch: int = 0
    log_every: int = 10

    # resume
    auto_resume: bool = True

    # ----- Loss fields -----
    loss_type: str = "mse"
    recon_loss_w: float = 1.0
    corr_loss_w: float = 0.0
    tv_loss_w: float = 0.0
    huber_delta: float = 1.0
    charbonnier_eps: float = 1e-3

    # ----- Sweep controls -----
    auto_sweep: bool = False
    sweep_epochs: int = 0
    sweep_loss_list: Optional[List[str]] = None
    sweep_corr_w: Optional[List[float]] = None
    sweep_tv_w: Optional[List[float]] = None
    sweep_huber_delta: Optional[List[float]] = None
    sweep_charbonnier_eps: Optional[List[float]] = None
    sweep_lr1: Optional[List[float]] = None
    sweep_lr2: Optional[List[float]] = None
    sweep_lr3: Optional[List[float]] = None
    sweep_max_combos: int = 0
    no_final_full: bool = False

    # --- W&B ---
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None
    wandb_group: Optional[str] = None
    wandb_job_type: Optional[str] = None
    wandb_tags: Optional[List[str]] = None
    wandb_notes: Optional[str] = None
    wandb_mode: str = "online"

# -----------------------------
# Frozen BrainLM
# -----------------------------
class FrozenBrainLM(nn.Module):
    def __init__(self, model_dir: Path, device: torch.device):
        super().__init__()
        cfg_path = model_dir / "config.json"
        w_path   = model_dir / "pytorch_model.bin"
        with open(cfg_path, "r") as f:
            cfg = BrainLMConfig(**json.load(f))
        self.model = BrainLMForPretraining(cfg)
        try:
            ckpt = _torch_load_compat(w_path, map_location=device, allow_weights_only=True)
            self.model.load_state_dict(ckpt, strict=False)
        except Exception as e:
            raise RuntimeError(f"Failed to load BrainLM: {e}")
        self.model.eval()
        for p in self.model.parameters(): p.requires_grad = False
        self.to(device); self.device = device

    @property
    def n_layers_out(self) -> int:
        return int(getattr(self.model.config, "num_hidden_layers", 4)) + 1

    @property
    def hidden_size(self) -> int:
        return int(getattr(self.model.config, "hidden_size", 256))

    @torch.no_grad()
    def extract_latents(self, signal_vectors: torch.Tensor, xyz_vectors: torch.Tensor) -> torch.Tensor:
        embeddings, mask, ids_restore = self.model.vit.embeddings(
            signal_vectors=signal_vectors, xyz_vectors=xyz_vectors, noise=None
        )
        enc = self.model.vit.encoder(hidden_states=embeddings, output_hidden_states=True, return_dict=True)
        return torch.stack(list(enc.hidden_states), dim=0)  # (L,B,Ttok,Dh)

# -----------------------------
# Translator (fMRI-only, axial-RoPE self-attn)
# -----------------------------
class TranslatorFMRISelfAttn(nn.Module):
    """
    fMRI branch:
      adapter_fmri (→ S=T×V) -> fmri_encoder (axial RoPE) -> fmri_decoder_2d -> tanh + affine
    """
    def __init__(self, cfg: TrainCfg, fmri_n_layers:int, fmri_hidden_size:int):
        super().__init__()
        self.cfg = cfg
        T = int(round(cfg.window_sec / cfg.tr))
        V = int(cfg.fmri_voxels)
        S = T * V

        # Adapter: stack BrainLM latents → tokens, retarget to S=T×V
        self.adapter_fmri = fMRIInputAdapterConv1d(
            seq_len=V * T,  # nominal
            n_layers=fmri_n_layers,
            input_dim=fmri_hidden_size,
            output_dim=cfg.d_model,
            target_seq_len=S
        )

        # Self-attention encoder (lower/higher stacks identical here; we use 'higher')
        self.fmri_encoder = HierarchicalEncoder(
            cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout,
            n_layers_per_stack=1, rope_fraction=1.0
        )

        # 2-D decoder (FIXED): predicts (B,T,V)
        self.fmri_decoder = fMRIDecodingAdapter2D(
            target_T=T, target_V=V, d_model=cfg.d_model, rank=32
        )

        # Output affine (per-run learnable)
        self.fmri_out_scale = nn.Parameter(torch.tensor(1.0))
        self.fmri_out_bias  = nn.Parameter(torch.tensor(0.0))

        self.T = T
        self.V = V

    def forward(self, fmri_latents: torch.Tensor, fmri_T:int, fmri_V:int) -> torch.Tensor:
        """
        fmri_latents: (L,B,Ttok,Dh) from BrainLM
        returns: (B,T,V)
        """
        B = fmri_latents.size(1)
        # Project & retarget to S=T×V
        fmri_adapt = self.adapter_fmri(fmri_latents)       # (B,S,D) with S=T×V
        # Axial-RoPE encoder
        _, fmr_hi = self.fmri_encoder(fmri_adapt, fmri_adapt, T=fmri_T, V=fmri_V)  # (B,S,D)
        # 2-D decoder -> (B,T,V)
        fmri_sig = self.fmri_decoder(fmr_hi)
        # squash + affine
        fmri_sig = torch.tanh(fmri_sig)
        fmri_sig = self.fmri_out_scale * fmri_sig + self.fmri_out_bias
        return fmri_sig

# -----------------------------
# Freezing policies
# -----------------------------
def freeze_all(m: nn.Module):
    for p in m.parameters(): p.requires_grad = False

def set_stage(m: TranslatorFMRISelfAttn, stage: int, *, train_affine_stage3: bool = False):
    """
    Stage-wise (forward chain):
      - 1: adapter
      - 2: adapter + encoder
      - 3: adapter + encoder + decoder
    Note: affine params (fmri_out_scale/bias) are ALWAYS trainable.
    """
    freeze_all(m)

    if stage == 1:
        for n, p in m.named_parameters():
            if n.startswith("adapter_fmri."):
                p.requires_grad = True
    elif stage == 2:
        for n, p in m.named_parameters():
            if n.startswith("adapter_fmri.") or n.startswith("fmri_encoder."):
                p.requires_grad = True
    elif stage == 3:
        for n, p in m.named_parameters():
            if (n.startswith("adapter_fmri.") or
                n.startswith("fmri_encoder.") or
                n.startswith("fmri_decoder.")):
                p.requires_grad = True
    else:
        raise ValueError(f"Unknown stage {stage}")

    # Always learn global affine
    m.fmri_out_scale.requires_grad = True
    m.fmri_out_bias.requires_grad  = True

# -----------------------------
# Per-module save/load helpers
# -----------------------------
def _module_paths(out_dir: Path, tag: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    return {
        "adapter": out_dir / f"adapter_fmri_{tag}.pt",
        "encoder": out_dir / f"fmri_encoder_{tag}.pt",
        "decoder": out_dir / f"fmri_decoder_{tag}.pt",
        "affine":  out_dir / f"fmri_affine_{tag}.pt",
    }

def save_modules(out_dir: Path, model: TranslatorFMRISelfAttn, tag: str):
    paths = _module_paths(out_dir, tag)
    torch.save(model.adapter_fmri.state_dict(), paths["adapter"])
    torch.save(model.fmri_encoder.state_dict(), paths["encoder"])
    torch.save(model.fmri_decoder.state_dict(), paths["decoder"])
    torch.save({"scale": model.fmri_out_scale.detach().cpu(),
                "bias":  model.fmri_out_bias.detach().cpu()}, paths["affine"])
    print(f"[save] modules -> {', '.join(p.name for p in paths.values())}")
    return paths

def load_modules_if_exist(out_dir: Path, model: TranslatorFMRISelfAttn, tag: str,
                          which: Optional[Iterable[str]] = None, device: Optional[torch.device] = None):
    paths = _module_paths(out_dir, tag)
    which = set(which or ["adapter","encoder","decoder","affine"])
    loaded = []
    if "adapter" in which and paths["adapter"].exists():
        sd = _torch_load_compat(paths["adapter"], map_location=device or "cpu", allow_weights_only=True)
        model.adapter_fmri.load_state_dict(sd, strict=False); loaded.append("adapter_fmri")
    if "encoder" in which and paths["encoder"].exists():
        sd = _torch_load_compat(paths["encoder"], map_location=device or "cpu", allow_weights_only=True)
        model.fmri_encoder.load_state_dict(sd, strict=False); loaded.append("fmri_encoder")
    if "decoder" in which and paths["decoder"].exists():
        sd = _torch_load_compat(paths["decoder"], map_location=device or "cpu", allow_weights_only=True)
        model.fmri_decoder.load_state_dict(sd, strict=False); loaded.append("fmri_decoder")
    if "affine" in which and paths["affine"].exists():
        sd = _torch_load_compat(paths["affine"], map_location=device or "cpu", allow_weights_only=True)
        if isinstance(sd, dict):
            with torch.no_grad():
                if "scale" in sd: model.fmri_out_scale.copy_(sd["scale"].to(model.fmri_out_scale.device))
                if "bias"  in sd: model.fmri_out_bias.copy_(sd["bias"].to(model.fmri_out_bias.device))
        loaded.append("fmri_affine")
    if loaded:
        print(f"[load] modules restored for tag '{tag}': {', '.join(loaded)}")
    return loaded

# -----------------------------
# Trainstate save/load (optimizer, scaler, epoch, best)
# -----------------------------
def _trainstate_path(out_dir: Path, stage: int, tag: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"trainstate_stage{stage}_{tag}.pt"

def save_trainstate(out_dir: Path, stage: int, tag: str, epoch: int, best_val: float,
                    opt: Optional[torch.optim.Optimizer], scaler: Optional[torch.amp.GradScaler]) -> Path:
    state = {
        "epoch": int(epoch),
        "best_val": float(best_val),
        "optimizer_state": (opt.state_dict() if opt is not None else None),
        "scaler_state": (scaler.state_dict() if scaler is not None else None),
        "timestamp": time.time(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda if torch.cuda.is_available() else None,
    }
    p = _trainstate_path(out_dir, stage, tag)
    torch.save(state, p)
    return p

def load_trainstate_if_exist(out_dir: Path, stage: int, tag: str,
                             opt: Optional[torch.optim.Optimizer],
                             scaler: Optional[torch.amp.GradScaler],
                             device: torch.device) -> Tuple[int, float, bool]:
    if not _trainstate_path(out_dir, stage, tag).exists():
        return 1, float('inf'), False
    p = _trainstate_path(out_dir, stage, tag)
    try:
        sd = _torch_load_compat(p, map_location=device, allow_weights_only=False)
        if opt is not None and sd.get("optimizer_state") is not None:
            try: opt.load_state_dict(sd["optimizer_state"])
            except Exception as e: print(f"[resume] WARN optimizer state: {e}")
        if scaler is not None and sd.get("scaler_state") is not None:
            try: scaler.load_state_dict(sd["scaler_state"])
            except Exception as e: print(f"[resume] WARN scaler state: {e}")
        last_epoch = int(sd.get("epoch", 0))
        best_val = float(sd.get("best_val", float('inf')))
        print(f"[resume] Loaded trainstate from {p.name}: epoch={last_epoch}, best_val={best_val:.6f}")
        return max(1, last_epoch + 1), best_val, True
    except Exception as e:
        print(f"[resume] WARN failed to load trainstate ({p.name}): {e}")
        return 1, float('inf'), False

# -----------------------------
# Data & helpers
# -----------------------------
def make_dataloaders(cfg: TrainCfg, device: torch.device) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    inter_keys = collect_common_sr_keys(cfg.eeg_root, cfg.fmri_root)
    if not inter_keys:
        raise RuntimeError("No (subject,task,run) intersections between EEG and fMRI trees.")

    if cfg.train_subjects or cfg.val_subjects or cfg.test_subjects:
        train_keys, val_keys, test_keys = fixed_subject_keys(
            cfg.eeg_root, cfg.fmri_root,
            cfg.train_subjects or [], cfg.val_subjects or [], cfg.test_subjects or [],
        )
    else:
        subs = sorted({k[0] for k in inter_keys}, key=int)
        n = len(subs); nt = max(1, int(0.8*n)); nv = max(1, int(0.1*n))
        train_sub = set(subs[:nt]); val_sub = set(subs[nt:nt+nv]); test_sub = set(subs[nt+nv:])
        train_keys = tuple(k for k in inter_keys if k[0] in train_sub)
        val_keys   = tuple(k for k in inter_keys if k[0] in val_sub)
        test_keys  = tuple(k for k in inter_keys if k[0] in test_sub)

    def _dl(keys, name):
        ds = PairedAlignedDataset(
            eeg_root=cfg.eeg_root, fmri_root=cfg.fmri_root, a424_label_nii=cfg.a424_label_nii,
            window_sec=cfg.window_sec, original_fs=1000, target_fs=200, tr=cfg.tr,
            channels_limit=34, fmri_norm=cfg.fmri_norm, stride_sec=cfg.stride_sec,
            device='cpu', include_sr_keys=keys
        )
        if cfg.debug:
            print(f"[debug][dataset] {name}: samples={len(ds)} keys={len(keys)}")
        dl = DataLoader(
            ds,
            batch_size=cfg.batch_size,
            shuffle=(name == "train"),
            num_workers=cfg.num_workers,
            pin_memory=(device.type == 'cuda'),
            collate_fn=collate_paired,
            drop_last=False
        )
        if cfg.debug:
            print(f"[debug][dataloader] {name}: batches={len(dl)} (bs={cfg.batch_size}, workers={cfg.num_workers}, pin={device.type=='cuda'})")
        return dl

    return _dl(train_keys, "train"), _dl(val_keys, "val"), (_dl(test_keys, "test") if test_keys else None)

def cache_a424_xyz(device: torch.device) -> Optional[torch.Tensor]:
    candidates = [
        THIS_DIR / 'BrainLM' / 'resources' / 'atlases' / 'A424_Coordinates.dat',
        THIS_DIR / 'resources' / 'atlases' / 'A424_Coordinates.dat',
        REPO_ROOT / 'BrainLM' / 'toolkit' / 'atlases' / 'A424_Coordinates.dat',
    ]
    coords = None
    for p in candidates:
        if p.exists():
            try:
                from data_oddball import load_a424_coords  # type: ignore
                arr = load_a424_coords(p)
                arr = arr / (float(np.max(np.abs(arr))) or 1.0)
                coords = torch.from_numpy(arr.astype(np.float32)).to(device)
                break
            except Exception:
                pass
    return coords

def _grad_norm(model: nn.Module) -> float:
    tot = 0.0
    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            param_norm = p.grad.data.norm(2).item()
            tot += param_norm * param_norm
    return float(math.sqrt(tot)) if tot > 0 else 0.0

# -----------------------------
# Loss & epoch runner
# -----------------------------
def _compute_loss_components(recon: torch.Tensor, target: torch.Tensor, cfg: TrainCfg) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # recon, target: (B, T, V)
    diff = recon - target

    if cfg.loss_type in ("mse", "mse+pearson"):
        recon_loss = F.mse_loss(recon, target)
    elif cfg.loss_type == "mae":
        recon_loss = F.l1_loss(recon, target)
    elif cfg.loss_type == "huber":
        recon_loss = F.huber_loss(recon, target, delta=float(cfg.huber_delta))
    elif cfg.loss_type == "charbonnier":
        eps2 = float(cfg.charbonnier_eps) ** 2
        recon_loss = torch.mean(torch.sqrt(diff * diff + eps2))
    else:
        recon_loss = F.mse_loss(recon, target)

    corr_loss = torch.tensor(0.0, device=recon.device)
    if cfg.loss_type == "mse+pearson" or (cfg.corr_loss_w and cfg.corr_loss_w > 0):
        x = recon - recon.mean(dim=1, keepdim=True)
        y = target - target.mean(dim=1, keepdim=True)
        xs = torch.sqrt(x.pow(2).mean(dim=1) + 1e-6)
        ys = torch.sqrt(y.pow(2).mean(dim=1) + 1e-6)
        r = (x * y).mean(dim=1) / (xs * ys)  # (B, V)
        corr_loss = 1.0 - r.mean()

    tv_loss = torch.tensor(0.0, device=recon.device)
    if cfg.tv_loss_w and cfg.tv_loss_w > 0:
        tv_loss = torch.mean(torch.abs(recon[:, 1:, :] - recon[:, :-1, :]))

    total = (cfg.recon_loss_w * recon_loss
             + cfg.corr_loss_w * corr_loss
             + cfg.tv_loss_w * tv_loss)

    return total, {"recon": recon_loss.detach(), "corr": corr_loss.detach(), "tv": tv_loss.detach()}

def run_epoch(mode:str, dl, model, brainlm, xyz_ref, cfg:TrainCfg, device, scaler, opt=None):
    is_train = (mode == 'train')
    model.train(is_train)
    tot, steps = 0.0, 0
    comp_acc = {"recon": 0.0, "corr": 0.0, "tv": 0.0}

    iterator = tqdm(dl, total=len(dl), leave=False, desc=f"{mode}") if tqdm is not None else dl
    prev_end = time.time()

    for bidx, batch in enumerate(iterator):
        t_fetch = time.time() - prev_end
        t0 = time.time()

        fmri_t = batch['fmri_window'].to(device, non_blocking=True)  # (B,T,V)
        B,T,V = fmri_t.shape

        # BrainLM inputs
        t_pad0 = time.time()
        fmri_pad = pad_timepoints_for_brainlm_torch(fmri_t, patch_size=20)  # (B,Tp,V)
        signal_vectors = fmri_pad.permute(0,2,1).contiguous()               # (B,V,Tp)
        if xyz_ref is not None and V == cfg.fmri_voxels:
            xyz = xyz_ref.unsqueeze(0).repeat(B,1,1)
        else:
            xyz = torch.zeros(B, V, 3, device=device)
        t_pad = time.time() - t_pad0

        # BrainLM latents (frozen)
        t_blm0 = time.time()
        with torch.no_grad():
            fmri_latents = brainlm.extract_latents(signal_vectors, xyz)     # (L,B,Ttok,Dh)
        t_blm = time.time() - t_blm0

        if is_train:
            opt.zero_grad(set_to_none=True)

        # Forward + loss
        t_fwd0 = time.time()
        with torch.amp.autocast('cuda', enabled=(cfg.amp and device.type=='cuda')):
            recon = model(fmri_latents, fmri_T=T, fmri_V=V)                 # (B,T,V)
            total_loss, comps = _compute_loss_components(recon, fmri_t, cfg)
        t_fwd = time.time() - t_fwd0

        t_bwd = t_step = gnorm = 0.0
        if is_train:
            t_bwd0 = time.time()
            scaler.scale(total_loss).backward()
            scaler.unscale_(opt)
            gnorm = _grad_norm(model)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            t_bwd = time.time() - t_bwd0

            t_step0 = time.time()
            scaler.step(opt); scaler.update()
            t_step = time.time() - t_step0

        loss_val = float(total_loss.detach().cpu())
        tot += loss_val; steps += 1
        for k in comp_acc.keys():
            comp_acc[k] += float(comps[k].detach().cpu())

        t_tot = time.time() - t0
        if cfg.debug:
            if bidx < max(1, cfg.profile_first_n):
                mem = fmt_mem()
                print(
                    f"[{mode}] b{bidx:04d} fetch={t_fetch:.3f}s | pad={t_pad:.3f}s | brainlm={t_blm:.3f}s | "
                    f"fwd={t_fwd:.3f}s | bwd={t_bwd:.3f}s | step={t_step:.3f}s | total={t_tot:.3f}s | "
                    f"loss={loss_val:.6f} | gnorm={gnorm:.3f} | {mem} | "
                    f"shapes: fmri={tuple(fmri_t.shape)} latents={tuple(fmri_latents.shape)}"
                )
            elif (bidx+1) % max(1, cfg.log_every) == 0:
                mem = fmt_mem()
                if tqdm is not None:
                    iterator.set_postfix_str(f"loss={loss_val:.4f} fetch={t_fetch:.2f}s fwd={t_fwd:.2f}s {mem}")
                else:
                    print(f"[{mode}] b{bidx+1}/{len(dl)} loss={loss_val:.5f} fetch={t_fetch:.3f}s fwd={t_fwd:.3f}s gnorm={gnorm:.2f} {mem}")

        if cfg.debug and (wandb is not None) and (cfg.wandb_mode != "disabled") and is_train:
            alloc = torch.cuda.memory_allocated()/(1024**2) if torch.cuda.is_available() else 0.0
            reserv = torch.cuda.memory_reserved()/(1024**2) if torch.cuda.is_available() else 0.0
            wandb.log({
                "batch/total_loss": loss_val,
                "batch/recon_loss": float(comps["recon"].cpu()),
                "batch/corr_loss":  float(comps["corr"].cpu()),
                "batch/tv_loss":    float(comps["tv"].cpu()),
                "batch/fetch_s": t_fetch, "batch/fwd_s": t_fwd,
                "batch/bwd_s": t_bwd, "batch/step_s": t_step, "batch/total_s": t_tot,
                "cuda/alloc_mb": alloc, "cuda/reserved_mb": reserv,
            })

        prev_end = time.time()
        if cfg.max_batches_per_epoch and steps >= cfg.max_batches_per_epoch:
            if cfg.debug:
                print(f"[debug] Reached --max_batches_per_epoch={cfg.max_batches_per_epoch}, breaking early.")
            break

    avg = {
        "total": tot / max(1, steps),
        "recon": comp_acc["recon"] / max(1, steps),
        "corr":  comp_acc["corr"] / max(1, steps),
        "tv":    comp_acc["tv"] / max(1, steps),
    }
    return avg

# -----------------------------
# Diagnostics (unchanged)
# -----------------------------
def _ensure_dirs(out_dir: Path):
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)

def _pearsonr_safe(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a).ravel(); b = np.asarray(b).ravel()
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3: return 0.0
    a = a[m]; b = b[m]
    sa = a.std(); sb = b.std()
    if sa == 0.0 or sb == 0.0: return 0.0
    r = np.corrcoef(a, b)[0,1]
    return float(r) if np.isfinite(r) else 0.0

def _lagged_corr(gt: np.ndarray, rc: np.ndarray, max_lag:int) -> Tuple[int, float]:
    lags = range(-max_lag, max_lag+1)
    best_r = -2.0; best_l = 0
    for l in lags:
        if l < 0:
            r = _pearsonr_safe(gt[:l], rc[-l:])
        elif l > 0:
            r = _pearsonr_safe(gt[l:], rc[:-l])
        else:
            r = _pearsonr_safe(gt, rc)
        if np.isfinite(r) and r > best_r:
            best_r, best_l = r, l
    return best_l, best_r

def _lin_calibrate(rc: np.ndarray, gt: np.ndarray) -> Tuple[float, float, np.ndarray, float, float]:
    A = np.vstack([rc, np.ones_like(rc)]).T
    a, b = np.linalg.lstsq(A, gt, rcond=None)[0]
    rc_lin = a*rc + b
    SSE = np.sum((gt - rc_lin)**2)
    SST = np.sum((gt - gt.mean())**2)
    R2  = 1.0 - SSE/(SST + 1e-8)
    NRMSE = np.sqrt(SSE/len(gt)) / (gt.std() + 1e-8)
    return float(a), float(b), rc_lin, float(R2), float(NRMSE)

def _downsample(x: np.ndarray, y: np.ndarray, max_points: int) -> Tuple[np.ndarray, np.ndarray]:
    n = int(x.shape[0])
    if n <= max_points: return x, y
    step = int(math.ceil(n / max_points))
    return x[::step], y[::step]

def _plot_heatmap(arr: np.ndarray, title: str, out_path: Path, xlabel="ROI", ylabel="Time (TR)", cmap="viridis"):
    plt.figure(figsize=(12, 6))
    plt.imshow(arr.T, aspect="auto", interpolation="nearest", origin="upper", cmap=cmap)
    plt.colorbar()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def _plot_topk_and_bars(t: np.ndarray, x_true: np.ndarray, x_rec: np.ndarray,
                        out_dir: Path, TR: float, top_k: int = 12, make_calib: bool = True,
                        plot_max_points: int = 1000):
    T, V = x_true.shape
    rs: List[Tuple[int,float]] = []
    for roi in range(V):
        rs.append((roi, _pearsonr_safe(x_true[:, roi], x_rec[:, roi])))
    rs.sort(key=lambda t_: t_[1], reverse=True)
    top = rs[:min(top_k, len(rs))]

    if len(top) > 0:
        fig, axes = plt.subplots(nrows=len(top), ncols=1, figsize=(12, 2.2*len(top)), sharex=False)
        if len(top) == 1: axes = [axes]
        for ax, (roi, r) in zip(axes, top):
            gt = x_true[:, roi]; rc = x_rec[:, roi]
            tt = np.arange(T) * float(TR)
            tt_d, gtd = _downsample(tt, gt, plot_max_points)
            _,    rcd = _downsample(tt, rc, plot_max_points)
            ax.plot(tt_d, gtd, label="GT")
            ax.plot(tt_d, rcd, label="Recon", alpha=0.9)
            ax.set_title(f"ROI {roi} — r={r:.3f}")
            ax.set_xlabel("Time (s)"); ax.set_ylabel("Z-score")
            ax.legend(loc="upper right")
        fig.tight_layout()
        fig.savefig(out_dir / "plots" / "topK_fmri_corr.png", dpi=150)
        plt.close(fig)

    if make_calib and len(top) > 0:
        fig, axes = plt.subplots(nrows=len(top), ncols=1, figsize=(12, 2.2*len(top)), sharex=False)
        if len(top) == 1: axes = [axes]
        for ax, (roi, r) in zip(axes, top):
            gt = x_true[:, roi]; rc = x_rec[:, roi]
            a, b, rc_lin, R2, nrmse = _lin_calibrate(rc, gt)
            tt = np.arange(T) * float(TR)
            tt_d, gtd = _downsample(tt, gt, plot_max_points)
            _,    rcd = _downsample(tt, rc_lin, plot_max_points)
            ax.plot(tt_d, gtd, label="GT")
            ax.plot(tt_d, rcd, label=f"Recon (calib) a={a:.2f}, b={b:.2f}, R²={R2:.2f}", alpha=0.9)
            ax.set_title(f"ROI {roi} — raw r={r:.3f}")
            ax.set_xlabel("Time (s)"); ax.set_ylabel("Z-score")
            ax.legend(loc="upper right")
        fig.tight_layout()
        fig.savefig(out_dir / "plots" / "topK_fmri_corr_calib.png", dpi=150)
        plt.close(fig)

    all_roi = [roi for roi,_ in rs]
    all_r   = [r for _,r in rs]
    order = np.argsort(all_r)[::-1]
    figw = max(12, 0.03 * len(all_roi) + 10)
    plt.figure(figsize=(figw, 5))
    plt.bar(np.arange(len(all_roi)), np.array(all_r)[order])
    plt.ylim(-1.0, 1.0)
    step = max(1, len(all_roi)//40)
    plt.xticks(np.arange(0,len(all_roi),step), [str(all_roi[i]) for i in order[::step]], rotation=90)
    plt.xlabel("fMRI ROI (sorted by r)")
    plt.ylabel("Pearson r")
    plt.title("fMRI: Pearson correlation per ROI")
    plt.tight_layout()
    plt.savefig(out_dir / "plots" / "corr_bars_fmri.png", dpi=150)
    plt.close()

    return rs

@torch.no_grad()
def generate_fmri_diagnostics(
    cfg: TrainCfg,
    out_dir: Path,
    stage: int,
    model: TranslatorFMRISelfAttn,
    brainlm: FrozenBrainLM,
    dl_eval: DataLoader,
    device: torch.device,
    xyz_ref: Optional[torch.Tensor],
    *,
    top_k: int = 12,
    max_lag_tr: int = 3,
    make_calib_plots: bool = True,
    plot_max_points: int = 1000,
    wandb_run=None,
    tag_for_logging: Optional[str] = None,
):
    _ensure_dirs(out_dir)
    model.eval()

    try:
        batch = next(iter(dl_eval))
    except StopIteration:
        print("[diag] WARNING: empty eval dataloader; skipping diagnostics.")
        return

    fmri_t = batch['fmri_window'].to(device)  # (B,T,V)
    B, T, V = map(int, fmri_t.shape)

    # BrainLM inputs
    fmri_pad = pad_timepoints_for_brainlm_torch(fmri_t, patch_size=20)  # (B,Tp,V)
    signal_vectors = fmri_pad.permute(0,2,1).contiguous()               # (B,V,Tp)
    if xyz_ref is not None and V == cfg.fmri_voxels:
        xyz = xyz_ref.unsqueeze(0).repeat(B,1,1)
    else:
        xyz = torch.zeros(B, V, 3, device=device)

    fmri_latents = brainlm.extract_latents(signal_vectors, xyz)  # (L,B,Ttok,Dh)
    recon = model(fmri_latents, fmri_T=T, fmri_V=V)              # (B,T,V)

    b = 0
    x_true = fmri_t[b].detach().cpu().numpy()
    x_rec  = recon[b].detach().cpu().numpy()

    _plot_heatmap(x_true, "GT (T×V)", out_dir / "plots" / "heatmap_gt.png")
    _plot_heatmap(x_rec,  "Recon (T×V)", out_dir / "plots" / "heatmap_recon.png")
    _plot_heatmap(np.abs(x_true - x_rec), "|GT - Recon| (T×V)", out_dir / "plots" / "heatmap_absdiff.png", cmap="magma")

    rs = _plot_topk_and_bars(
        t=np.arange(T)*float(cfg.tr),
        x_true=x_true, x_rec=x_rec,
        out_dir=out_dir, TR=cfg.tr, top_k=top_k,
        make_calib=make_calib_plots, plot_max_points=plot_max_points
    )

    rows: List[List[float]] = []
    for roi in range(V):
        gt = x_true[:, roi]
        rc = x_rec[:, roi]
        r0 = _pearsonr_safe(gt, rc)
        a, bb, rc_lin, R2, NRMSE = _lin_calibrate(rc, gt)
        best_lag, r_best = _lagged_corr(gt, rc, max_lag_tr)
        rows.append([roi, r0, float(gt.std()), float(rc.std()), a, bb, R2, NRMSE, best_lag, r_best])
    csv_path = out_dir / "tables" / "fmri_roi_diagnostics.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["roi","r0","std_gt","std_rec","slope","intercept","R2","NRMSE","best_lag_TR","r_at_best_lag"])
        for row in rows:
            w.writerow(row)
    print(f"[diag] wrote diagnostics -> {csv_path}")

    if wandb_run is not None:
        try:
            wandb_run.log({"diag/stage": stage, "diag/tag": tag_for_logging or ""})
            for name in ["heatmap_gt.png", "heatmap_recon.png", "heatmap_absdiff.png",
                         "topK_fmri_corr.png", "corr_bars_fmri.png", "topK_fmri_corr_calib.png"]:
                p = out_dir / "plots" / name
                if p.exists():
                    wandb_run.log({f"plots/{name}": wandb.Image(str(p))})
        except Exception:
            pass

# -----------------------------
# Auto-sweep helpers (unchanged)
# -----------------------------
def dataclass_replace(obj, **updates):
    from dataclasses import replace as _replace
    return _replace(obj, **updates)

# (sweep helpers omitted for brevity — keep your existing ones unchanged)
# ... You can paste your existing sweep helpers block here unchanged ...

# -----------------------------
# Config loader (same as before)
# -----------------------------
def _load_config_file(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        if path.suffix.lower() in (".yaml", ".yml"):
            try:
                import yaml  # type: ignore
            except Exception as e:
                raise RuntimeError("PyYAML is required for YAML configs. Install with 'pip install pyyaml'.") from e
            data = yaml.safe_load(f)
        else:
            data = json.load(f)
    if isinstance(data, dict) and "train" in data:
        return data["train"] or {}
    return data or {}

def _apply_config_defaults(parser: argparse.ArgumentParser) -> Dict[str, Any]:
    temp = argparse.ArgumentParser(add_help=False)
    temp.add_argument("--config", type=str, default=None)
    known, _ = temp.parse_known_args()
    cfg_defaults: Dict[str, Any] = {}
    if known.config:
        raw = _load_config_file(Path(known.config))
        if "out_dir" not in raw and "output_dir" in raw:
            raw["out_dir"] = raw["output_dir"]
        if "epochs" in raw and not any(k in raw for k in ("stage1_epochs","stage2_epochs","stage3_epochs")):
            total = int(raw["epochs"])
            s1 = max(1, total // 4)
            s2 = max(1, total // 2)
            s3 = max(1, total - s1 - s2)
            raw["stage1_epochs"], raw["stage2_epochs"], raw["stage3_epochs"] = s1, s2, s3
        if "lr" in raw:
            for k in ("lr1","lr2","lr3"):
                if k not in raw: raw[k] = float(raw["lr"])
        for key in ("amp", "train_affine_stage3", "debug", "auto_resume", "auto_sweep", "no_final_full"):
            if key in raw: raw[key] = bool(raw[key])
        pass_through_keys = [
            "eeg_root","fmri_root","a424_label_nii","brainlm_model_dir","out_dir",
            "device","seed","window_sec","stride_sec","batch_size","num_workers","amp",
            "stage1_epochs","stage2_epochs","stage3_epochs","lr1","lr2","lr3","weight_decay",
            "fmri_norm","train_subjects","val_subjects","test_subjects","train_affine_stage3",
            "debug","profile_first_n","max_batches_per_epoch","log_every",
            "auto_resume",
            "loss_type","recon_loss_w","corr_loss_w","tv_loss_w","huber_delta","charbonnier_eps",
            "auto_sweep","sweep_epochs","sweep_loss_list","sweep_corr_w","sweep_tv_w",
            "sweep_huber_delta","sweep_charbonnier_eps","sweep_lr1","sweep_lr2","sweep_lr3",
            "sweep_max_combos","no_final_full",
            "wandb_project","wandb_entity","wandb_run_name","wandb_group","wandb_job_type",
            "wandb_tags","wandb_notes","wandb_mode",
        ]
        cfg_defaults = {k: raw[k] for k in pass_through_keys if k in raw}
        parser.set_defaults(**cfg_defaults)
    return cfg_defaults

# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Stage-wise fMRI-only training/testing with axial RoPE and 2-D decoder")
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--eeg_root", type=str)
    ap.add_argument("--fmri_root", type=str)
    ap.add_argument("--a424_label_nii", type=str)
    ap.add_argument("--brainlm_model_dir", type=str)
    ap.add_argument("--out_dir", type=str, default=str(THIS_DIR/"runs_fmri_selfattn"))
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--window_sec", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--stride_sec", type=int, default=None)
    ap.add_argument("--fmri_norm", type=str, default="zscore", choices=["zscore","psc","mad","none"])
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--stage1_epochs", type=int, default=3)
    ap.add_argument("--stage2_epochs", type=int, default=5)
    ap.add_argument("--stage3_epochs", type=int, default=3)
    ap.add_argument("--lr1", type=float, default=1e-4)
    ap.add_argument("--lr2", type=float, default=8e-5)
    ap.add_argument("--lr3", type=float, default=5e-5)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--train_affine_stage3", action="store_true")
    ap.add_argument("--train_subjects", type=int, nargs="*", default=None)
    ap.add_argument("--val_subjects",   type=int, nargs="*", default=None)
    ap.add_argument("--test_subjects",  type=int, nargs="*", default=None)

    ap.add_argument("--stage", type=int, required=True, choices=[1,2,3])
    ap.add_argument("--mode", type=str, required=True, choices=["train","test"])
    ap.add_argument("--eval_split", type=str, default="val", choices=["val","test"])
    ap.add_argument("--load_tag", type=str, default=None)

    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--profile_first_n", type=int, default=0)
    ap.add_argument("--max_batches_per_epoch", type=int, default=0)
    ap.add_argument("--log_every", type=int, default=10)

    ap.add_argument("--no_resume", action="store_true")

    ap.add_argument("--loss_type", type=str,
                    choices=["mse","mae","huber","charbonnier","mse+pearson"],
                    default="mse")
    ap.add_argument("--recon_loss_w", type=float, default=1.0)
    ap.add_argument("--corr_loss_w", type=float, default=0.0)
    ap.add_argument("--tv_loss_w", type=float, default=0.0)
    ap.add_argument("--huber_delta", type=float, default=1.0)
    ap.add_argument("--charbonnier_eps", type=float, default=1e-3)

    ap.add_argument("--auto_sweep", action="store_true")
    ap.add_argument("--sweep_epochs", type=int, default=0)
    ap.add_argument("--sweep_loss_list", type=str, nargs="*", default=["mse","mae","huber","charbonnier","mse+pearson"])
    ap.add_argument("--sweep_corr_w", type=float, nargs="*", default=[0.0, 0.2])
    ap.add_argument("--sweep_tv_w", type=float, nargs="*", default=[0.0])
    ap.add_argument("--sweep_huber_delta", type=float, nargs="*", default=[0.5, 1.0])
    ap.add_argument("--sweep_charbonnier_eps", type=float, nargs="*", default=[1e-3, 3e-3])

    ap.add_argument("--sweep_lr1", type=float, nargs="*", default=[1e-4, 5e-5])
    ap.add_argument("--sweep_lr2", type=float, nargs="*", default=[8e-5, 5e-5])
    ap.add_argument("--sweep_lr3", type=float, nargs="*", default=[5e-5, 2e-5])
    ap.add_argument("--sweep_max_combos", type=int, default=0)
    ap.add_argument("--no_final_full", action="store_true")

    ap.add_argument("--wandb_project", type=str, default=None)
    ap.add_argument("--wandb_entity", type=str, default=None)
    ap.add_argument("--wandb_run_name", type=str, default=None)
    ap.add_argument("--wandb_group", type=str, default=None)
    ap.add_argument("--wandb_job_type", type=str, default=None)
    ap.add_argument("--wandb_tags", type=str, nargs="*", default=None)
    ap.add_argument("--wandb_notes", type=str, default=None)
    ap.add_argument("--wandb_mode", type=str, default="online", choices=["online","offline","disabled"])

    _apply_config_defaults(ap)
    args = ap.parse_args()

    cfg = TrainCfg(
        eeg_root=Path(args.eeg_root),
        fmri_root=Path(args.fmri_root),
        a424_label_nii=Path(args.a424_label_nii),
        brainlm_model_dir=Path(args.brainlm_model_dir),
        out_dir=Path(args.out_dir),
        device=args.device,
        seed=args.seed,
        window_sec=args.window_sec,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        stride_sec=args.stride_sec,
        amp=bool(args.amp),
        fmri_norm=args.fmri_norm,
        stage1_epochs=args.stage1_epochs,
        stage2_epochs=args.stage2_epochs,
        stage3_epochs=args.stage3_epochs,
        lr_stage1=args.lr1,
        lr_stage2=args.lr2,
        lr_stage3=args.lr3,
        weight_decay=args.weight_decay,
        train_fmri_affine_stage3=bool(args.train_affine_stage3),
        train_subjects=args.train_subjects,
        val_subjects=args.val_subjects,
        test_subjects=args.test_subjects,
        debug=bool(args.debug),
        profile_first_n=int(args.profile_first_n),
        max_batches_per_epoch=int(args.max_batches_per_epoch),
        log_every=int(args.log_every),
        auto_resume=not bool(args.no_resume),
        loss_type=args.loss_type,
        recon_loss_w=args.recon_loss_w,
        corr_loss_w=args.corr_loss_w,
        tv_loss_w=args.tv_loss_w,
        huber_delta=args.huber_delta,
        charbonnier_eps=args.charbonnier_eps,
        auto_sweep=bool(args.auto_sweep),
        sweep_epochs=int(args.sweep_epochs),
        sweep_loss_list=args.sweep_loss_list,
        sweep_corr_w=args.sweep_corr_w,
        sweep_tv_w=args.sweep_tv_w,
        sweep_huber_delta=args.sweep_huber_delta,
        sweep_charbonnier_eps=args.sweep_charbonnier_eps,
        sweep_lr1=args.sweep_lr1,
        sweep_lr2=args.sweep_lr2,
        sweep_lr3=args.sweep_lr3,
        sweep_max_combos=int(args.sweep_max_combos),
        no_final_full=bool(args.no_final_full),
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=args.wandb_run_name,
        wandb_group=args.wandb_group,
        wandb_job_type=args.wandb_job_type,
        wandb_tags=args.wandb_tags,
        wandb_notes=args.wandb_notes,
        wandb_mode=args.wandb_mode,
    )

    for pth in [cfg.eeg_root, cfg.fmri_root, cfg.a424_label_nii, cfg.brainlm_model_dir]:
        if pth is None:
            raise RuntimeError("Missing required path(s) in config/CLI.")
    if not Path(cfg.a424_label_nii).exists():
        raise FileNotFoundError(f"A424 atlas not found: {cfg.a424_label_nii}")
    if not Path(cfg.brainlm_model_dir).exists():
        raise FileNotFoundError(f"BrainLM checkpoint dir not found: {cfg.brainlm_model_dir}")

    seed_all(cfg.seed)
    device = torch.device(cfg.device)
    os.makedirs(cfg.out_dir, exist_ok=True)
    _ensure_dirs(cfg.out_dir)

    if cfg.debug:
        print(f"[debug] torch {torch.__version__} | cuda available={torch.cuda.is_available()} | device={device}")
        if torch.cuda.is_available():
            print(f"[debug] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[debug] platform={platform.platform()} | python={sys.version.split()[0]}")
        print(f"[debug] cfg={cfg}")
        try:
            import brainlm_mae  # type: ignore
            print(f"[debug] brainlm_mae -> {getattr(brainlm_mae, '__file__', '<pkg>')}")
        except Exception as e:
            print(f"[debug] brainlm_mae import check failed: {e}")
        print(f"[debug] sys.path[:5] = {sys.path[:5]}")

    # Data
    dl_train, dl_val, dl_test = make_dataloaders(cfg, device)

    # Models
    brainlm = FrozenBrainLM(cfg.brainlm_model_dir, device)
    translator = TranslatorFMRISelfAttn(cfg, fmri_n_layers=brainlm.n_layers_out, fmri_hidden_size=brainlm.hidden_size).to(device)
    xyz_ref = cache_a424_xyz(device)
    scaler = torch.amp.GradScaler('cuda', enabled=(cfg.amp and device.type=='cuda'))

    # ---------- Stage + Mode ----------
    stage = int(args.stage)
    mode = str(args.mode)

    # ---------- W&B ----------
    def _wandb_enabled() -> bool:
        return (wandb is not None) and (cfg.wandb_mode != "disabled")

    run = None
    if _wandb_enabled():
        run_name = cfg.wandb_run_name or f"fmri-selfattn-s{stage}-{mode}"
        tags = (cfg.wandb_tags or []) + [f"stage:{stage}", f"mode:{mode}"]
        run = wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            name=run_name,
            group=cfg.wandb_group,
            job_type=cfg.wandb_job_type or f"stage{stage}",
            notes=cfg.wandb_notes,
            tags=tags,
            mode=("disabled" if cfg.wandb_mode == "disabled" else cfg.wandb_mode),
            config={**vars(cfg), "stage": stage, "mode": mode},
        )
        try:
            wandb.watch(translator, log="gradients", log_freq=max(1, cfg.log_every))
        except Exception:
            pass

    # ---------- Warm-start / Resume ----------
    out_dir = Path(cfg.out_dir)

    def _any_last_modules_exist(s: int) -> bool:
        tag = f"stage{s}_last"
        paths = {
            out_dir / f"adapter_fmri_{tag}.pt",
            out_dir / f"fmri_encoder_{tag}.pt",
            out_dir / f"fmri_decoder_{tag}.pt",
            out_dir / f"fmri_affine_{tag}.pt",
        }
        return any(p.exists() for p in paths)

    start_epoch = 1
    best = float('inf')
    resumed = False

    if mode == "train" and cfg.auto_resume and _any_last_modules_exist(stage):
        print(f"[resume] Detected existing stage{stage}_last modules. Resuming training...")
        load_modules_if_exist(out_dir, translator, tag=f"stage{stage}_last",
                              which=["adapter","encoder","decoder","affine"], device=device)
        resumed = True
    else:
        if mode == "train":
            if stage == 2:
                load_modules_if_exist(out_dir, translator, tag="stage1_best", which=["adapter"], device=device)
            elif stage == 3:
                load_modules_if_exist(out_dir, translator, tag="stage2_best", which=["adapter","encoder"], device=device)
        else:
            default_tag = f"stage{stage}_best"
            tag = args.load_tag or default_tag
            need = ["adapter"] if stage == 1 else (["adapter","encoder"] if stage == 2 else ["adapter","encoder","decoder","affine"])
            loaded = load_modules_if_exist(out_dir, translator, tag=tag, which=need, device=device)
            missing = set(need) - set([n.split("_")[0] for n in loaded])
            if missing:
                print(f"[WARN] Some modules for '{tag}' were not found: {sorted(missing)}")

    # ---------- Set trainable ----------
    set_stage(translator, stage, train_affine_stage3=cfg.train_fmri_affine_stage3)
    trainable = [n for n,p in translator.named_parameters() if p.requires_grad]
    print(f"[stage {stage}][{mode}] trainable: {len(trainable)} tensors")
    for n in trainable: print("  -", n)

    # ---------- Run ----------
    if mode == "train":
        if stage == 1:
            epochs, lr = cfg.stage1_epochs, cfg.lr_stage1
        elif stage == 2:
            epochs, lr = cfg.stage2_epochs, cfg.lr_stage2
        else:
            epochs, lr = cfg.stage3_epochs, cfg.lr_stage3

        opt = torch.optim.AdamW([p for p in translator.parameters() if p.requires_grad], lr=lr, weight_decay=cfg.weight_decay)

        if resumed:
            start_epoch, best, _ = load_trainstate_if_exist(out_dir, stage, tag="last", opt=opt, scaler=scaler, device=device)
        if start_epoch > epochs:
            print(f"[resume] start_epoch ({start_epoch}) > total epochs ({epochs}); nothing to do.")
            if run is not None:
                wandb.log({"resume/nothing_to_do": 1, "stage": stage})
                try: run.finish()
                except Exception: pass
            return

        for ep in range(start_epoch, epochs+1):
            t_ep0 = time.time()
            tr = run_epoch("train", dl_train, translator, brainlm, xyz_ref, cfg, device, scaler, opt)
            va = run_epoch("val",   dl_val,   translator, brainlm, xyz_ref, cfg, device, scaler, opt=None)
            t_ep = time.time() - t_ep0
            print(f"[Stage{stage}][{ep}/{epochs}] "
                  f"train_total={tr['total']:.6f} val_total={va['total']:.6f} "
                  f"| train(recon={tr['recon']:.4f}, corr={tr['corr']:.4f}, tv={tr['tv']:.4f}) "
                  f"| val(recon={va['recon']:.4f}, corr={va['corr']:.4f}, tv={va['tv']:.4f}) "
                  f"| epoch_time={t_ep:.1f}s | lr={opt.param_groups[0]['lr']:.2e} | {fmt_mem()}")

            if run is not None:
                alloc = torch.cuda.memory_allocated()/(1024**2) if torch.cuda.is_available() else 0.0
                reserv = torch.cuda.memory_reserved()/(1024**2) if torch.cuda.is_available() else 0.0
                wandb.log({
                    "stage": stage, "epoch": ep,
                    "train/total_loss": tr["total"], "train/recon_loss": tr["recon"],
                    "train/corr_loss": tr["corr"], "train/tv_loss": tr["tv"],
                    "val/total_loss": va["total"], "val/recon_loss": va["recon"],
                    "val/corr_loss": va["corr"], "val/tv_loss": va["tv"],
                    "lr": float(opt.param_groups[0]['lr']),
                    "time/epoch_s": t_ep,
                    "cuda/alloc_mb": alloc,
                    "cuda/reserved_mb": reserv,
                    "resume/start_epoch": start_epoch if ep == start_epoch else None,
                }, step=ep)

            last_paths = save_modules(out_dir, translator, tag=f"stage{stage}_last")
            ts_last = save_trainstate(out_dir, stage, tag="last", epoch=ep, best_val=best, opt=opt, scaler=scaler)
            if run is not None:
                try:
                    art_last = wandb.Artifact(f"fmri_selfattn_stage{stage}_last", type="model")
                    for p in last_paths.values(): art_last.add_file(str(p))
                    art_last.add_file(str(ts_last))
                    run.log_artifact(art_last)
                except Exception:
                    pass

            if va["total"] < best:
                best = va["total"]
                best_paths = save_modules(out_dir, translator, tag=f"stage{stage}_best")
                ts_best = save_trainstate(out_dir, stage, tag="best", epoch=ep, best_val=best, opt=opt, scaler=scaler)
                print(f"[Stage{stage}] ✅ new best {best:.6f}")

                try:
                    generate_fmri_diagnostics(
                        cfg=cfg, out_dir=out_dir, stage=stage,
                        model=translator, brainlm=brainlm, dl_eval=dl_val,
                        device=device, xyz_ref=xyz_ref, top_k=12,
                        make_calib_plots=True, wandb_run=run,
                        tag_for_logging=f"main_best_ep{ep}"
                    )
                except Exception as e:
                    print(f"[diag] WARN diagnostics failed: {e}")

                if run is not None:
                    wandb.summary[f"stage{stage}/best_val_total"] = float(best)
                    try:
                        art_best = wandb.Artifact(f"fmri_selfattn_stage{stage}_best", type="model")
                        for p in best_paths.values(): art_best.add_file(str(p))
                        art_best.add_file(str(ts_best))
                        run.log_artifact(art_best)
                    except Exception:
                        pass

    else:  # test mode
        if args.eval_split == "test" and dl_test is not None:
            res = run_epoch("test", dl_test, translator, brainlm, xyz_ref, cfg, device, scaler, opt=None)
            print(f"[TEST][stage {stage}] test_total_loss={res['total']:.6f}")
            if run is not None:
                wandb.log({"test/total_loss": float(res["total"]), "stage": stage})
            try:
                generate_fmri_diagnostics(
                    cfg=cfg, out_dir=out_dir, stage=stage,
                    model=translator, brainlm=brainlm, dl_eval=dl_test,
                    device=device, xyz_ref=xyz_ref, top_k=12,
                    make_calib_plots=True, wandb_run=run,
                    tag_for_logging="test_eval"
                )
            except Exception as e:
                print(f"[diag] WARN test diagnostics failed: {e}")
        else:
            res = run_epoch("val", dl_val, translator, brainlm, xyz_ref, cfg, device, scaler, opt=None)
            print(f"[TEST][stage {stage}] val_total_loss={res['total']:.6f}")
            if run is not None:
                wandb.log({"val/total_loss": float(res["total"]), "stage": stage})
            try:
                generate_fmri_diagnostics(
                    cfg=cfg, out_dir=out_dir, stage=stage,
                    model=translator, brainlm=brainlm, dl_eval=dl_val,
                    device=device, xyz_ref=xyz_ref, top_k=12,
                    make_calib_plots=True, wandb_run=run,
                    tag_for_logging="val_eval"
                )
            except Exception as e:
                print(f"[diag] WARN val diagnostics failed: {e}")

    if run is not None:
        try:
            run.finish()
        except Exception:
            pass

if __name__ == "__main__":
    main()
