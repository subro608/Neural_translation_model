#!/usr/bin/env python3
"""
Stage-wise training (fMRI-only, self-attention; no cross-attn, no EEG path),
run **one stage at a time** with train/test modes and per-module checkpoints.

Now with DEBUG instrumentation:
  - --debug prints env, dataset sizes, shapes, memory, LR, etc.
  - --profile_first_n N profiles the first N batches with sub-step timers
  - --max_batches_per_epoch caps batches per epoch for quick sanity checks
  - tqdm progress bar with fetch/compute timings and loss
  - gradient norm & CUDA memory (allocated/reserved) every --log_every steps

Usage examples:
  # Stage 1 train, then test on val:
  python train_fmri_selfattn_permodule.py --config <cfg.yaml> --stage 1 --mode train --debug --profile_first_n 2
  python train_fmri_selfattn_permodule.py --config <cfg.yaml> --stage 1 --mode test --eval_split val

  # Stage 2 (warm-starts from Stage 1 best):
  python train_fmri_selfattn_permodule.py --config <cfg.yaml> --stage 2 --mode train
  python train_fmri_selfattn_permodule.py --config <cfg.yaml> --stage 2 --mode test --eval_split test

  # Stage 3 (warm-starts from Stage 2 best):
  python train_fmri_selfattn_permodule.py --config <cfg.yaml> --stage 3 --mode train --train_affine_stage3
  python train_fmri_selfattn_permodule.py --config <cfg.yaml> --stage 3 --mode test --eval_split test
"""

from __future__ import annotations
import os, json, math, argparse, time, platform
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Iterable, List, Dict, Any, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# -----------------------------
# Repo paths
# -----------------------------
THIS_DIR = Path(__file__).parent
REPO_ROOT = THIS_DIR.parent
CBRAMOD_DIR = REPO_ROOT / "CBraMod"
BRAINLM_DIR = REPO_ROOT / "BrainLM"  # may or may not exist depending on your layout

import sys
# Always include the script dir
sys.path.insert(0, str(THIS_DIR))
# Try common locations for CBraMod
sys.path.insert(0, str(CBRAMOD_DIR))
sys.path.insert(0, str(THIS_DIR / "CBraMod"))
# Try common locations for BrainLM (robust across layouts)
candidate_blm_roots = [
    BRAINLM_DIR,                 # <repo-root>/BrainLM
    THIS_DIR / "BrainLM",        # <this-script-dir>/BrainLM  (your current layout)
    Path(os.environ.get("BRAINLM_DIR", "")),
    Path(os.environ.get("BRAINLM_DIR", "")) / "brainlm_mae",
]
for p in candidate_blm_roots:
    if not p:
        continue
    p = Path(p)
    # If we were given the package dir itself, add its parent;
    # if we were given the BrainLM repo root, add that.
    if (p / "brainlm_mae").is_dir():
        sys.path.insert(0, str(p))
        break
    if p.name == "brainlm_mae" and p.is_dir():
        sys.path.insert(0, str(p.parent))
        break

try:
    from tqdm import tqdm  # progress bar
except Exception:
    tqdm = None

# -----------------------------
# Local modules
# -----------------------------
from module import (  # type: ignore
    fMRIInputAdapterConv1d,
    HierarchicalEncoder,
    fMRIDecodingAdapterLite,
)

# BrainLM imports (now robust thanks to path setup above)
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
    train_fmri_affine_stage3: bool = False  # unfreeze fmri_out_scale/bias in stage 3

    # fixed subjects (optional; if set, used for deterministic splits)
    train_subjects: Optional[List[int]] = None
    val_subjects:   Optional[List[int]] = None
    test_subjects:  Optional[List[int]] = None

    # debug controls
    debug: bool = False
    profile_first_n: int = 0     # profile first N batches in each epoch
    max_batches_per_epoch: int = 0  # 0=all
    log_every: int = 10

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
            ckpt = torch.load(str(w_path), map_location=device)
            self.model.load_state_dict(ckpt, strict=False)
        except Exception as e:
            raise RuntimeError(f"Failed to load BrainLM: {e}")
        self.model.eval()
        for p in self.model.parameters(): p.requires_grad = False
        self.to(device); self.device = device

    @property
    def n_layers_out(self) -> int:
        # encoder.hidden_states includes embeddings + layers
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
# Translator (fMRI-only, self-attn)
# -----------------------------
class TranslatorFMRISelfAttn(nn.Module):
    """
    Only fMRI branch:
      adapter_fmri -> fmri_encoder (self-attn) -> fmri_decoder -> tanh + affine
    """
    def __init__(self, cfg: TrainCfg, fmri_n_layers:int, fmri_hidden_size:int):
        super().__init__()
        self.cfg = cfg
        # Adapter from BrainLM stacked latents to d_model tokens (resamples to 512)
        self.adapter_fmri = fMRIInputAdapterConv1d(
            seq_len=cfg.fmri_voxels * int(round(cfg.window_sec/cfg.tr)),
            n_layers=fmri_n_layers, input_dim=fmri_hidden_size,
            output_dim=cfg.d_model, target_seq_len=512
        )
        # Positional enc (buffer)
        self.pos_fmri = self._pos_enc(cfg.d_model, 100_000)

        # Self-attention encoder (lower/higher stacks identical here; use 'higher')
        self.fmri_encoder = HierarchicalEncoder(cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout, n_layers_per_stack=1)

        # Tiny decoder to (B, T*V)
        T = int(round(cfg.window_sec/cfg.tr))
        self.fmri_decoder = fMRIDecodingAdapterLite(target_T=T, target_V=cfg.fmri_voxels, d_model=cfg.d_model, rank=16)

        # Output affine (usually keep frozen until stage 3 if enabled)
        self.fmri_out_scale = nn.Parameter(torch.tensor(1.0))
        self.fmri_out_bias  = nn.Parameter(torch.tensor(0.0))

    @staticmethod
    def _pos_enc(d_model:int, max_len:int):
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32)*(-math.log(10000.0)/d_model))
        pe[:,0::2]=torch.sin(pos*div); pe[:,1::2]=torch.cos(pos*div)
        return nn.Parameter(pe, requires_grad=False)

    def forward(self, fmri_latents: torch.Tensor, fmri_T:int, fmri_V:int) -> torch.Tensor:
        """
        fmri_latents: (L,B,Ttok,Dh) from BrainLM
        returns: (B,T,V)
        """
        fmri_adapt = self.adapter_fmri(fmri_latents)  # (B, 512, D)
        fmri_adapt = fmri_adapt + self.pos_fmri[:fmri_adapt.size(1)].unsqueeze(0).to(fmri_adapt.device)
        _, fmr_hi = self.fmri_encoder(fmri_adapt, fmri_adapt)  # (B, Tf, D)
        fmri_flat = self.fmri_decoder(fmr_hi)                  # (B, T*V)
        fmri_sig  = fmri_flat.view(fmri_adapt.size(0), int(fmri_T), int(fmri_V))
        fmri_sig  = torch.tanh(fmri_sig)
        fmri_sig  = self.fmri_out_scale * fmri_sig + self.fmri_out_bias
        return fmri_sig

# -----------------------------
# Freezing policies
# -----------------------------
def freeze_all(m: nn.Module):
    for p in m.parameters(): p.requires_grad = False

def set_stage(m: TranslatorFMRISelfAttn, stage: int, *, train_affine_stage3: bool = False):
    freeze_all(m)
    if stage == 1:
        for n,p in m.named_parameters():
            if n.startswith("adapter_fmri."): p.requires_grad = True
    elif stage == 2:
        for n,p in m.named_parameters():
            if n.startswith("adapter_fmri.") or n.startswith("fmri_encoder."):
                p.requires_grad = True
    elif stage == 3:
        for n,p in m.named_parameters():
            if (n.startswith("adapter_fmri.") or
                n.startswith("fmri_encoder.") or
                n.startswith("fmri_decoder.")):
                p.requires_grad = True
        m.fmri_out_scale.requires_grad = bool(train_affine_stage3)
        m.fmri_out_bias.requires_grad  = bool(train_affine_stage3)
    else:
        raise ValueError(f"Unknown stage {stage}")

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

def load_modules_if_exist(out_dir: Path, model: TranslatorFMRISelfAttn, tag: str,
                          which: Optional[Iterable[str]] = None, device: Optional[torch.device] = None):
    paths = _module_paths(out_dir, tag)
    which = set(which or ["adapter","encoder","decoder","affine"])
    loaded = []
    if "adapter" in which and paths["adapter"].exists():
        sd = torch.load(paths["adapter"], map_location=device or "cpu")
        model.adapter_fmri.load_state_dict(sd, strict=False); loaded.append("adapter_fmri")
    if "encoder" in which and paths["encoder"].exists():
        sd = torch.load(paths["encoder"], map_location=device or "cpu")
        model.fmri_encoder.load_state_dict(sd, strict=False); loaded.append("fmri_encoder")
    if "decoder" in which and paths["decoder"].exists():
        sd = torch.load(paths["decoder"], map_location=device or "cpu")
        model.fmri_decoder.load_state_dict(sd, strict=False); loaded.append("fmri_decoder")
    if "affine" in which and paths["affine"].exists():
        sd = torch.load(paths["affine"], map_location=device or "cpu")
        if isinstance(sd, dict):
            with torch.no_grad():
                if "scale" in sd: model.fmri_out_scale.copy_(sd["scale"].to(model.fmri_out_scale.device))
                if "bias"  in sd: model.fmri_out_bias.copy_(sd["bias"].to(model.fmri_out_bias.device))
        loaded.append("fmri_affine")
    if loaded:
        print(f"[load] modules restored for tag '{tag}': {', '.join(loaded)}")
    return loaded

# -----------------------------
# Data & helpers
# -----------------------------
def make_dataloaders(cfg: TrainCfg, device: torch.device) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    inter_keys = collect_common_sr_keys(cfg.eeg_root, cfg.fmri_root)
    if not inter_keys:
        raise RuntimeError("No (subject,task,run) intersections between EEG and fMRI trees.")

    # Fixed subject lists (if provided) -> use exactly those
    if cfg.train_subjects or cfg.val_subjects or cfg.test_subjects:
        train_keys, val_keys, test_keys = fixed_subject_keys(
            cfg.eeg_root, cfg.fmri_root,
            cfg.train_subjects or [], cfg.val_subjects or [], cfg.test_subjects or [],
        )
    else:
        # Deterministic 80/10/10 split
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
            ds, batch_size=cfg.batch_size, shuffle=False,
            num_workers=cfg.num_workers, pin_memory=(device.type=='cuda'),
            collate_fn=collate_paired, drop_last=False
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

def run_epoch(mode:str, dl, model, brainlm, xyz_ref, cfg:TrainCfg, device, scaler, opt=None):
    is_train = (mode == 'train')
    model.train(is_train)
    loss_fn = nn.MSELoss()
    tot, steps = 0.0, 0

    iterator = dl
    if tqdm is not None:
        iterator = tqdm(dl, total=len(dl), leave=False, desc=f"{mode}")

    # To measure dataloader fetch time
    prev_end = time.time()

    for bidx, batch in enumerate(iterator):
        t_fetch = time.time() - prev_end  # time spent waiting for this batch
        t0 = time.time()

        fmri_t = batch['fmri_window'].to(device, non_blocking=True)  # (B,T,V)
        B,T,V = fmri_t.shape

        # Prepare BrainLM inputs
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
            loss = loss_fn(recon, fmri_t)
        t_fwd = time.time() - t_fwd0

        # Backward/step
        t_bwd = t_step = gnorm = 0.0
        if is_train:
            t_bwd0 = time.time()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            gnorm = _grad_norm(model)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            t_bwd = time.time() - t_bwd0

            t_step0 = time.time()
            scaler.step(opt); scaler.update()
            t_step = time.time() - t_step0

        # aggregate
        loss_val = float(loss.detach().cpu())
        tot += loss_val; steps += 1

        # logging
        t_tot = time.time() - t0
        if cfg.debug:
            if bidx < max(1, cfg.profile_first_n):
                # detailed first-N profile
                mem = fmt_mem()
                print(
                    f"[{mode}] b{bidx:04d} "
                    f"fetch={t_fetch:.3f}s | pad={t_pad:.3f}s | brainlm={t_blm:.3f}s | "
                    f"fwd={t_fwd:.3f}s | bwd={t_bwd:.3f}s | step={t_step:.3f}s | total={t_tot:.3f}s | "
                    f"loss={loss_val:.6f} | gnorm={gnorm:.3f} | {mem} | "
                    f"shapes: fmri={tuple(fmri_t.shape)} latents={tuple(fmri_latents.shape)}"
                )
            elif (bidx+1) % max(1, cfg.log_every) == 0:
                mem = fmt_mem()
                lr = opt.param_groups[0]['lr'] if is_train else 0.0
                if tqdm is not None:
                    iterator.set_postfix_str(f"loss={loss_val:.4f} fetch={t_fetch:.2f}s fwd={t_fwd:.2f}s {mem}")
                else:
                    print(f"[{mode}] b{bidx+1}/{len(dl)} loss={loss_val:.5f} fetch={t_fetch:.3f}s fwd={t_fwd:.3f}s gnorm={gnorm:.2f} lr={lr:.2e} {mem}")

        prev_end = time.time()

        if cfg.max_batches_per_epoch and steps >= cfg.max_batches_per_epoch:
            if cfg.debug:
                print(f"[debug] Reached --max_batches_per_epoch={cfg.max_batches_per_epoch}, breaking early.")
            break

    return tot / max(1, steps)

# -----------------------------
# Config loader
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
    # Use 'train' section if present
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

        # Map legacy keys to this script
        if "out_dir" not in raw and "output_dir" in raw:
            raw["out_dir"] = raw["output_dir"]

        # Map epochs -> per-stage if not given
        if "epochs" in raw and not any(k in raw for k in ("stage1_epochs","stage2_epochs","stage3_epochs")):
            total = int(raw["epochs"])
            s1 = max(1, total // 4)
            s2 = max(1, total // 2)
            s3 = max(1, total - s1 - s2)
            raw["stage1_epochs"], raw["stage2_epochs"], raw["stage3_epochs"] = s1, s2, s3

        # Map lr -> lr1/2/3 if not given
        if "lr" in raw:
            for k in ("lr1","lr2","lr3"):
                if k not in raw: raw[k] = float(raw["lr"])

        # Normalize booleans/paths
        for key in ("amp", "train_affine_stage3", "debug"):
            if key in raw: raw[key] = bool(raw[key])

        # Pass-through keys we support
        pass_through_keys = [
            "eeg_root","fmri_root","a424_label_nii","brainlm_model_dir","out_dir",
            "device","seed","window_sec","stride_sec","batch_size","num_workers","amp",
            "stage1_epochs","stage2_epochs","stage3_epochs","lr1","lr2","lr3","weight_decay",
            "fmri_norm","train_subjects","val_subjects","test_subjects","train_affine_stage3",
            "debug","profile_first_n","max_batches_per_epoch","log_every",
        ]
        cfg_defaults = {k: raw[k] for k in pass_through_keys if k in raw}
        parser.set_defaults(**cfg_defaults)
    return cfg_defaults

# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Stage-wise fMRI-only (self-attn) training/testing (per-module checkpoints) + YAML/JSON config + DEBUG")
    ap.add_argument("--config", type=str, default=None, help="YAML/JSON config; if YAML has sections, uses 'train'")
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
    ap.add_argument("--train_affine_stage3", action="store_true", help="Unfreeze fmri_out_scale/bias in Stage 3")
    ap.add_argument("--train_subjects", type=int, nargs="*", default=None)
    ap.add_argument("--val_subjects",   type=int, nargs="*", default=None)
    ap.add_argument("--test_subjects",  type=int, nargs="*", default=None)

    # Run control
    ap.add_argument("--stage", type=int, required=True, choices=[1,2,3], help="Which stage to run (1|2|3)")
    ap.add_argument("--mode", type=str, required=True, choices=["train","test"], help="Run mode: train or test")
    ap.add_argument("--eval_split", type=str, default="val", choices=["val","test"], help="Which split to evaluate in test mode")
    ap.add_argument("--load_tag", type=str, default=None,
                    help="Optional override for which checkpoint tag to load in test mode (default: stageX_best). Example: 'stage2_last'.")

    # Debug knobs
    ap.add_argument("--debug", action="store_true", help="Verbose debug output")
    ap.add_argument("--profile_first_n", type=int, default=0, help="Profile/timestamp the first N batches in each epoch")
    ap.add_argument("--max_batches_per_epoch", type=int, default=0, help="Cap number of batches per epoch (0 = all)")
    ap.add_argument("--log_every", type=int, default=10, help="Print mem/grad every N batches when debug")

    # Apply config defaults if provided (CLI still overrides)
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
    )

    # Checks
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

    if cfg.debug:
        print(f"[debug] torch {torch.__version__} | cuda available={torch.cuda.is_available()} | device={device}")
        if torch.cuda.is_available():
            print(f"[debug] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[debug] platform={platform.platform()} | python={sys.version.split()[0]}")
        print(f"[debug] cfg={cfg}")
        # Show where brainlm_mae is being imported from
        try:
            import brainlm_mae  # type: ignore
            print(f"[debug] brainlm_mae -> {getattr(brainlm_mae, '__file__', '<pkg>')}")
        except Exception as e:
            print(f"[debug] brainlm_mae import check failed: {e}")
        # and the front of sys.path (useful when debugging)
        print(f"[debug] sys.path[:5] = {sys.path[:5]}")

    # Data
    dl_train, dl_val, dl_test = make_dataloaders(cfg, device)

    # Models
    brainlm = FrozenBrainLM(cfg.brainlm_model_dir, device)
    translator = TranslatorFMRISelfAttn(cfg, fmri_n_layers=brainlm.n_layers_out, fmri_hidden_size=brainlm.hidden_size).to(device)
    xyz_ref = cache_a424_xyz(device)
    scaler = torch.amp.GradScaler('cuda', enabled=(cfg.amp and device.type=='cuda'))

    # ---------- Warm-start policy per stage ----------
    out_dir = Path(cfg.out_dir)
    stage = int(args.stage)
    mode = str(args.mode)

    if mode == "train":
        if stage == 1:
            pass
        elif stage == 2:
            load_modules_if_exist(out_dir, translator, tag="stage1_best", which=["adapter"], device=device)
        elif stage == 3:
            load_modules_if_exist(out_dir, translator, tag="stage2_best", which=["adapter","encoder"], device=device)
    else:
        # test mode
        default_tag = f"stage{stage}_best"
        tag = args.load_tag or default_tag
        need = ["adapter"] if stage == 1 else (["adapter","encoder"] if stage == 2 else ["adapter","encoder","decoder","affine"])
        loaded = load_modules_if_exist(out_dir, translator, tag=tag, which=need, device=device)
        missing = set(need) - set([n.split("_")[0] for n in loaded])
        if missing:
            print(f"[WARN] Some modules for '{tag}' were not found: {sorted(missing)}")

    # ---------- Set trainable modules ----------
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
        best = float('inf')

        for ep in range(1, epochs+1):
            t_ep0 = time.time()
            tr = run_epoch("train", dl_train, translator, brainlm, xyz_ref, cfg, device, scaler, opt)
            va = run_epoch("val", dl_val, translator, brainlm, xyz_ref, cfg, device, scaler, opt=None)
            t_ep = time.time() - t_ep0
            print(f"[Stage{stage}][{ep}/{epochs}] train={tr:.6f} val={va:.6f} | epoch_time={t_ep:.1f}s | lr={opt.param_groups[0]['lr']:.2e} | {fmt_mem()}")
            save_modules(out_dir, translator, tag=f"stage{stage}_last")
            if va < best:
                best = va
                save_modules(out_dir, translator, tag=f"stage{stage}_best")

    else:  # test mode
        if args.eval_split == "test" and dl_test is not None:
            loss = run_epoch("test", dl_test, translator, brainlm, xyz_ref, cfg, device, scaler, opt=None)
            print(f"[TEST][stage {stage}] test_loss={loss:.6f}")
        else:
            loss = run_epoch("val", dl_val, translator, brainlm, xyz_ref, cfg, device, scaler, opt=None)
            print(f"[TEST][stage {stage}] val_loss={loss:.6f}")

if __name__ == "__main__":
    main()
