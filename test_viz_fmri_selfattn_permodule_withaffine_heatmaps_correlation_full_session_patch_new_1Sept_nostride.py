#!/usr/bin/env python3
"""
Diagnostics for fMRI-only (self-attn) per-module translator — static config (no CLI args)

What's included:
- ALWAYS attempts to load adapter, voxel_pe, encoder, decoder, **and affine** weights for the chosen tag.
- Canonical module-load check (no false "missing" warnings).
- DIAGNOSTICS: per-ROI stats written to CSV:
    roi, r0, std_gt, std_rec, slope, intercept, R2, NRMSE, best_lag_TR, r_at_best_lag
- PRINTS: overall summaries + affine values.
- PLOTS:
    * top-K GT vs Recon
    * heatmaps of target and reconstructed fMRI (T × V)
    * correlation matrices (fMRI-only):
        - Corr(ROI×ROI) for GT
        - Corr(ROI×ROI) for Recon
        - Cross-corr GT vs Recon (ROI×ROI)
    * **Mask heatmap** showing where patch-sparse masking was applied

Updates in this version (kept in sync with training):
- Mirrors training’s **patch-sparse** masking:
  - Split time into patches of size = BrainLM temporal tokenization size (auto-inferred from the checkpoint).
  - Sample a fraction of patches r_patch ~ U[0.5, 0.8].
  - For each selected patch, sample a fraction of ROIs r_v ~ U[0.5, 0.8] and mask those ROIs
    across the full temporal span of that patch.
  - Mask is built once for the full session and applied per-chunk **before** BrainLM.
- Adds **voxel positional encoding** (learned ID + XYZ MLP) exactly as in training.
- Forces BrainLM internal masking OFF and infers its time patch size robustly.
"""

from __future__ import annotations
import os, json, math, platform, sys, csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple, List, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
try:
    import yaml  # type: ignore
except Exception:
    yaml = None  # optional

# =========================
# PATHS & CONFIG
# =========================
EEG_ROOT          = Path(r"D:\Neuroinformatics_research_2025\Oddball\ds116_eeg")
FMRI_ROOT         = Path(r"D:\Neuroinformatics_research_2025\Oddball\ds000116")
A424_LABEL_NII    = Path(r"D:\Neuroinformatics_research_2025\BrainLM\A424_resampled_to_bold.nii.gz")
BRAINLM_MODEL_DIR = Path(r"D:\Neuroinformatics_research_2025\MNI_templates\BrainLM\pretrained_models\2023-06-06-22_15_00-checkpoint-1400")

RUN_DIR           = Path(r"30_aug_reverse_huber5e5_voxelpe_rope_patch_masked_subtrain_123456_fixed")
# Keep outputs tidy but colocated with the run:
SUBSET            = "train"   # "train" | "val" | "test"
OUT_DIR           = RUN_DIR / f"viz_diagnostics_{SUBSET}"

STAGE             = 3       # 1 | 2 | 3 (only used for default tag name)
TAG               = None     # None -> uses f"revstage{STAGE}_best" (matches trainer)
TOP_K             = 34
PLOT_MAX_POINTS   = 1000
MAX_LAG_TR        = 3         # search Pearson r over [-MAX_LAG_TR..+MAX_LAG_TR]
MAKE_CALIB_PLOTS  = False     # disabled: no calibrated plots

# Apply patch-sparse masking before BrainLM, to mirror training
APPLY_TIME_MASK   = True

DEVICE            = "cuda"
SEED              = 42
WINDOW_SEC        = 40
TR                = 2.0
FMRI_NORM         = "zscore"   # "zscore" | "psc" | "mad" | "none"
BATCH_SIZE        = 1
NUM_WORKERS       = 0
# STRIDE_SEC        = 10
# CHANNELS_LIMIT    = 34

# =========================
# Repo paths / sys.path
# =========================
THIS_DIR    = Path(__file__).parent
REPO_ROOT   = THIS_DIR.parent
CBRAMOD_DIR = REPO_ROOT / "CBraMod"
BRAINLM_DIR = REPO_ROOT / "BrainLM"

sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(CBRAMOD_DIR))

# Robust BrainLM import pathing
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

# =========================
# Imports matching trainer
# =========================
from module import (  # type: ignore
    fMRIInputAdapterConv1d,
    HierarchicalEncoder,          # axial RoPE inside; requires (T, V) on forward
    fMRIDecodingAdapter2D,        # 2-D decoder (predict V then time-only resize)
    VoxelPositionalEncoding,      # ← NEW: learned ID + XYZ MLP
)
from brainlm_mae.modeling_brainlm import BrainLMForPretraining  # type: ignore
from brainlm_mae.configuration_brainlm import BrainLMConfig     # type: ignore
from data_oddball import (  # type: ignore
    pad_timepoints_for_brainlm_torch,
    load_a424_coords,
    collect_common_sr_keys,
    fixed_subject_keys,
    normalize_fmri_time_series,
    load_fmri_parcels_cached,
    find_bold_files,
    _parse_key_from_path,
    _canonical_task_token,
)

# =========================
# Utils
# =========================
def seed_all(seed:int):
    np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def fmt_mem() -> str:
    if not torch.cuda.is_available(): return "cuda:N/A"
    a = torch.cuda.memory_allocated() / (1024**2)
    r = torch.cuda.memory_reserved() / (1024**2)
    return f"cuda: alloc={a:.1f}MB, reserv={r:.1f}MB"

def pearsonr_safe(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a).ravel(); b = np.asarray(b).ravel()
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3: return 0.0
    a = a[m]; b = b[m]
    sa = a.std(); sb = b.std()
    if sa == 0.0 or sb == 0.0: return 0.0
    r = np.corrcoef(a, b)[0,1]
    return float(r) if np.isfinite(r) else 0.0

def default_cache_dir() -> Path:
    return Path(os.environ.get("ODDBALL_CACHE_DIR", "~/.cache/oddball_eegfmri")).expanduser()

def downsample(x: np.ndarray, y: np.ndarray, max_points: int) -> Tuple[np.ndarray, np.ndarray]:
    n = int(x.shape[0])
    if n <= max_points: return x, y
    step = int(math.ceil(n / max_points))
    return x[::step], y[::step]

def norm_subj_id(s) -> str:
    ss = str(s)
    if ss.startswith("sub-"): ss = ss[4:]
    ss = ss.lstrip("0")
    return ss if ss else "0"

def lagged_corr(gt: np.ndarray, rc: np.ndarray, max_lag:int) -> Tuple[int, float]:
    """Return (best_lag, best_r) over integer lags in [-max_lag..+max_lag]."""
    lags = range(-max_lag, max_lag+1)
    best_r = -2.0
    best_l = 0
    for l in lags:
        if l < 0:
            r = pearsonr_safe(gt[:l], rc[-l:])
        elif l > 0:
            r = pearsonr_safe(gt[l:], rc[:-l])
        else:
            r = pearsonr_safe(gt, rc)
        if np.isfinite(r) and r > best_r:
            best_r, best_l = r, l
    return best_l, best_r

def lin_calibrate(rc: np.ndarray, gt: np.ndarray) -> Tuple[float, float, np.ndarray, float, float]:
    """Fit gt ≈ a*rc + b; return (a,b,rc_lin,R2,NRMSE)."""
    A = np.vstack([rc, np.ones_like(rc)]).T
    a, b = np.linalg.lstsq(A, gt, rcond=None)[0]
    rc_lin = a*rc + b
    SSE = np.sum((gt - rc_lin)**2)
    SST = np.sum((gt - gt.mean())**2)
    R2  = 1.0 - SSE/(SST + 1e-8)
    NRMSE = np.sqrt(SSE/len(gt)) / (gt.std() + 1e-8)
    return float(a), float(b), rc_lin, float(R2), float(NRMSE)

def save_heatmap(arr_TxV: np.ndarray, out_path: Path, title: str, tr: float,
                 cmap: str = "viridis", vlim: Optional[Tuple[float, float]] = None):
    """
    Save a heatmap for a (T, V) array with Time on X and ROI on Y.
    - arr_TxV: numpy array shaped (T, V)
    - tr: seconds per timepoint (for x-axis scaling)
    - vlim: optional (vmin, vmax) to share a color scale across plots
    """
    T, V = arr_TxV.shape
    plt.figure(figsize=(18, 6))
    im = plt.imshow(
        arr_TxV.T,                # (V, T) so Y=ROI, X=Time
        aspect="auto",
        origin="lower",
        cmap=cmap,
        extent=[0, T * tr, 0, V], # X in seconds, Y in ROI index
        vmin=(vlim[0] if vlim else None),
        vmax=(vlim[1] if vlim else None),
        interpolation="nearest",
    )
    plt.xlabel("Time (s)")
    plt.ylabel("ROI")
    plt.title(title)
    plt.colorbar(im)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()

# ---------- NEW: correlation-matrix helpers (fMRI-only) ----------
def corr_matrix(time_by_feat: np.ndarray) -> np.ndarray:
    """
    time_by_feat: (T, D) -> Pearson correlation matrix (D x D).
    Robust to constant columns (returns 0s instead of NaNs).
    """
    X = np.asarray(time_by_feat, dtype=np.float64)
    X = X - X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True) + 1e-12
    Xn = X / std
    C = np.nan_to_num((Xn.T @ Xn) / max(1, Xn.shape[0] - 1),
                      nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(C, -1.0, 1.0)

def cross_corr_matrix(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    X: (T, D1), Y: (T, D2) -> Corr[X_i, Y_j] (D1 x D2)
    Used here for fMRI GT vs Recon (still fMRI-only).
    """
    X = np.asarray(X, dtype=np.float64); Y = np.asarray(Y, dtype=np.float64)
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    Xs = X.std(axis=0, keepdims=True) + 1e-12
    Ys = Y.std(axis=0, keepdims=True) + 1e-12
    Xn = X / Xs; Yn = Y / Ys
    C = np.nan_to_num((Xn.T @ Yn) / max(1, Xn.shape[0] - 1),
                      nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(C, -1.0, 1.0)

def plot_corrmat(C: np.ndarray, out_path: Path, title: str,
                 xlabel: str, ylabel: str):
    plt.figure(figsize=(10, 8))
    im = plt.imshow(C, aspect="auto", origin="lower",
                    vmin=-1.0, vmax=1.0, cmap="coolwarm")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()

def save_corr_csv(C: np.ndarray, out_path: Path,
                  row_prefix: str, col_prefix: str):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([""] + [f"{col_prefix}{j}" for j in range(C.shape[1])])
        for i in range(C.shape[0]):
            w.writerow([f"{row_prefix}{i}"] +
                       [f"{float(v):.6f}" for v in C[i]])

# === PATCH-SPARSE MASKING (mirror trainer) ===============================
@torch.no_grad()
def _make_patch_sparse_mask(
    B: int, T: int, V: int, device: torch.device,
    *, patch_size: int,
    r_patch_range=(0.5, 0.8),          # fraction of time patches to select (match trainer)
    r_v_in_patch_range=(0.5, 0.8),     # fraction of ROIs masked within each selected patch (match trainer)
) -> torch.Tensor:
    """
    Returns boolean mask M (B, T, V) with True = masked.
    - Split T into patches of length patch_size (last may be shorter).
    - Sample a fraction of patches r_patch in [0.5, 0.8].
    - For EACH selected patch, sample r_v_patch in [0.5, 0.8] and mask that
      fraction of ROIs across the full temporal span of that patch.
    - Same mask for all items in the batch (simple & fast).
    """
    n_patches = int(math.ceil(T / float(patch_size)))
    if n_patches <= 0:
        return torch.zeros((B, T, V), dtype=torch.bool, device=device)

    r_patch = float(torch.rand(1, device=device) * (r_patch_range[1] - r_patch_range[0]) + r_patch_range[0])
    k_patch = max(1, int(round(r_patch * n_patches)))
    sel_patches = torch.randperm(n_patches, device=device)[:k_patch]  # indices in [0, n_patches)

    M = torch.zeros((B, T, V), dtype=torch.bool, device=device)
    for pidx in sel_patches.tolist():
        t_start = pidx * patch_size
        t_end = min((pidx + 1) * patch_size, T)
        r_v = float(torch.rand(1, device=device) * (r_v_in_patch_range[1] - r_v_in_patch_range[0]) + r_v_in_patch_range[0])
        k_v = max(1, int(round(r_v * V)))
        v_idx = torch.randperm(V, device=device)[:k_v]
        M[:, t_start:t_end, v_idx] = True

    return M

@torch.no_grad()
def _apply_zero_mask(x_BTV: torch.Tensor, M_BTV: torch.Tensor) -> torch.Tensor:
    y = x_BTV.clone()
    y[M_BTV] = 0.0
    return y
# =======================================================================

# =========================
# Torch load compat (match trainer)
# =========================
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

# =========================
# Frozen BrainLM (match trainer)
# =========================
class FrozenBrainLM(nn.Module):
    def __init__(self, model_dir: Path, device: torch.device):
        super().__init__()
        cfg_path = Path(model_dir) / "config.json"
        w_path   = Path(model_dir) / "pytorch_model.bin"
        with open(cfg_path, "r") as f:
            cfg = BrainLMConfig(**json.load(f))
        self.model = BrainLMForPretraining(cfg)

        # Force-off internal MAE masking (match training)
        self.model.config.mask_ratio = 0.0
        if hasattr(self.model, "vit") and hasattr(self.model.vit, "embeddings"):
            self.model.vit.embeddings.mask_ratio = 0.0

        ckpt = _torch_load_compat(w_path, map_location=device, allow_weights_only=True)
        self.model.load_state_dict(ckpt, strict=False)
        self.model.eval()
        for p in self.model.parameters(): p.requires_grad = False
        self.to(device); self.device = device

        # Infer time patch size robustly (default 20 if absent)
        def _infer_time_patch_size(model) -> int:
            cand_objs = [getattr(model.vit, "embeddings", None), getattr(model, "config", None)]
            cand_names = ["temporal_patch_size", "time_patch_size", "t_patch_size", "patch_size_time"]
            for obj in cand_objs:
                for name in cand_names:
                    if obj is not None and hasattr(obj, name):
                        try:
                            val = int(getattr(obj, name))
                            if val > 0: return val
                        except Exception:
                            pass
            return 20
        self.time_patch_size = _infer_time_patch_size(self.model)

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
        enc = self.model.vit.encoder(hidden_states=embeddings,
                                     output_hidden_states=True, return_dict=True)
        return torch.stack(list(enc.hidden_states), dim=0)  # (L,B,Ttok,Dh)

# =========================
# Translator head (aligned to trainer; with voxel PE)
# =========================
class TranslatorFMRISelfAttn(nn.Module):
    """
    fMRI-only path:
      adapter_fmri (→ S=T×V)
      + voxel PE (learned ID + XYZ MLP) on reshaped grid (B,T,V,D)
      -> fmri_encoder (axial RoPE) -> fmri_decoder_2d -> tanh + affine
    """
    def __init__(self, fmri_voxels:int, window_sec:int, tr:float,
                 fmri_n_layers:int, fmri_hidden_size:int,
                 d_model:int=128, n_heads:int=4, d_ff:int=512, dropout:float=0.1):
        super().__init__()
        self.fmri_voxels = int(fmri_voxels)
        self.window_sec  = int(window_sec)
        self.tr          = float(tr)

        T = int(self.window_sec / self.tr)  # floor, match dataset windows
        V = int(self.fmri_voxels)
        S = T * V

        # Adapter: stack BrainLM latents → tokens, retarget to S=T×V
        self.adapter_fmri = fMRIInputAdapterConv1d(
            seq_len=S,  # nominal
            n_layers=fmri_n_layers, input_dim=fmri_hidden_size,
            output_dim=d_model, target_seq_len=S
        )

        # NEW: voxel positional encoding (learned ID + XYZ MLP)
        self.voxel_pe = VoxelPositionalEncoding(V=V, D=d_model)

        # Axial-RoPE encoder (two stacks; we use 'higher')
        self.fmri_encoder = HierarchicalEncoder(
            d_model, n_heads, d_ff, dropout,
            n_layers_per_stack=1, rope_fraction=1.0
        )

        # 2-D decoder: predicts (B,T,V) with time-only up/downsample
        self.fmri_decoder = fMRIDecodingAdapter2D(
            target_T=T, target_V=V, d_model=d_model, rank=32
        )

        # Output affine (always learnable)
        self.fmri_out_scale = nn.Parameter(torch.tensor(1.0))
        self.fmri_out_bias  = nn.Parameter(torch.tensor(0.0))

    def forward(self, fmri_latents: torch.Tensor, fmri_T:int, fmri_V:int, xyz: torch.Tensor) -> torch.Tensor:
        # Project & retarget to S=T×V
        fmri_adapt = self.adapter_fmri(fmri_latents)       # (B,S,D)
        # Add voxel PE on (B,T,V,D)
        B = fmri_adapt.shape[0]
        f_grid = fmri_adapt.view(B, fmri_T, fmri_V, -1).contiguous()   # (B,T,V,D)
        pe = self.voxel_pe(B, fmri_T, xyz)                             # (B,T,V,D)
        f_grid = f_grid + pe
        f_tokens = f_grid.view(B, fmri_T * fmri_V, -1).contiguous()    # (B,S,D)

        # Axial-RoPE encoder (requires T,V)
        _, fmr_hi = self.fmri_encoder(f_tokens, f_tokens, T=fmri_T, V=fmri_V)  # (B,S,D)

        # 2-D decoder -> (B,T,V)
        fmri_sig = self.fmri_decoder(fmr_hi)
        # squash + affine
        fmri_sig = torch.tanh(fmri_sig)
        fmri_sig = self.fmri_out_scale * fmri_sig + self.fmri_out_bias
        return fmri_sig

# =========================
# Per-module load helpers (AFFINE- & VOXEL-PE-AWARE)
# =========================
def _module_paths(run_dir: Path, tag: str) -> Dict[str, Path]:
    run_dir = Path(run_dir); run_dir.mkdir(parents=True, exist_ok=True)
    return {
        "adapter": run_dir / f"adapter_fmri_{tag}.pt",
        "voxel_pe": run_dir / f"voxel_pe_{tag}.pt",          # ← NEW
        "encoder": run_dir / f"fmri_encoder_{tag}.pt",
        "decoder": run_dir / f"fmri_decoder_{tag}.pt",
        "affine":  run_dir / f"fmri_affine_{tag}.pt",
    }

def load_modules_if_exist(run_dir: Path, model: TranslatorFMRISelfAttn, tag: str,
                          which: Iterable[str], device: torch.device) -> set:
    """
    Loads any of {adapter, voxel_pe, encoder, decoder, affine} found for `tag`.
    - For 'affine', supports dict {"scale":..., "bias":...} and copies into parameters.
    Returns a set of loaded component names.
    """
    paths = _module_paths(run_dir, tag)
    try:
        print(f"[load] searching module files for tag='{tag}': "
              f"adapter={paths['adapter'].name}, voxel_pe={paths['voxel_pe'].name}, "
              f"encoder={paths['encoder'].name}, decoder={paths['decoder'].name}, affine={paths['affine'].name}")
    except Exception:
        pass
    loaded = set()

    if "adapter" in which and paths["adapter"].exists():
        sd = _torch_load_compat(paths["adapter"], map_location=device, allow_weights_only=True)
        model.adapter_fmri.load_state_dict(sd, strict=False)
        loaded.add("adapter")

    if "voxel_pe" in which and paths["voxel_pe"].exists():
        sd = _torch_load_compat(paths["voxel_pe"], map_location=device, allow_weights_only=True)
        model.voxel_pe.load_state_dict(sd, strict=False)
        loaded.add("voxel_pe")

    if "encoder" in which and paths["encoder"].exists():
        sd = _torch_load_compat(paths["encoder"], map_location=device, allow_weights_only=True)
        model.fmri_encoder.load_state_dict(sd, strict=False)
        loaded.add("encoder")

    if "decoder" in which and paths["decoder"].exists():
        sd = _torch_load_compat(paths["decoder"], map_location=device, allow_weights_only=True)
        model.fmri_decoder.load_state_dict(sd, strict=False)
        loaded.add("decoder")

    if "affine" in which and paths["affine"].exists():
        sd = _torch_load_compat(paths["affine"], map_location=device, allow_weights_only=True)
        if isinstance(sd, dict):
            with torch.no_grad():
                scale_t = sd.get("scale")
                bias_t  = sd.get("bias")
                if scale_t is not None:
                    model.fmri_out_scale.copy_(scale_t.to(model.fmri_out_scale.device))
                if bias_t is not None:
                    model.fmri_out_bias.copy_(bias_t.to(model.fmri_out_bias.device))
            # Print after safely copying, only if keys exist
            if ("scale" in sd) or ("bias" in sd):
                try:
                    s_val = float(scale_t.item()) if hasattr(scale_t, "item") else float(scale_t) if scale_t is not None else None
                except Exception:
                    s_val = None
                try:
                    b_val = float(bias_t.item()) if hasattr(bias_t, "item") else float(bias_t) if bias_t is not None else None
                except Exception:
                    b_val = None
                msg_s = f"scale={s_val:.6f}" if s_val is not None else "scale=N/A"
                msg_b = f"bias={b_val:.6f}"  if b_val is not None else "bias=N/A"
                print(f"[load] loading affine {msg_s}, {msg_b}")
        else:
            # best-effort fallback if someone saved a state_dict
            try:
                model.load_state_dict(sd, strict=False)
            except Exception:
                pass
        loaded.add("affine")

    return loaded

# =========================
# Subject split helpers
# =========================
def load_or_make_split(eeg_root: Path, fmri_root: Path, run_dir: Path):
    p_split = Path(run_dir) / "subject_splits.json"
    inter_keys = collect_common_sr_keys(eeg_root, fmri_root)
    try:
        uniq_subs = sorted({int(k[0]) for k in inter_keys})
        print(f"[split] intersecting (subject,task,run) keys = {len(inter_keys)} | subjects={uniq_subs}")
    except Exception:
        print(f"[split] intersecting keys = {len(inter_keys)}")
    if not inter_keys:
        raise RuntimeError("No (subject,task,run) intersections between EEG and fMRI trees.")
    if p_split.exists():
        with open(p_split, "r") as f:
            js = json.load(f)
        tr = [int(norm_subj_id(s)) for s in js.get("train_subjects", [])]
        va = [int(norm_subj_id(s)) for s in js.get("val_subjects", [])]
        te = [int(norm_subj_id(s)) for s in js.get("test_subjects", [])]
        print(f"[split] loaded subject_splits.json → train={tr} | val={va} | test={te}")
        train_keys, val_keys, test_keys = fixed_subject_keys(eeg_root, fmri_root, tr, va, te)
    else:
        subs = sorted({k[0] for k in inter_keys}, key=int)
        n = len(subs); nt = max(1, int(0.8*n)); nv = max(1, int(0.1*n))
        train_sub = set(subs[:nt]); val_sub = set(subs[nt:nt+nv]); test_sub = set(subs[nt+nv:])
        print(f"[split] creating default split: Nsub={n} → train={len(train_sub)}, val={len(val_sub)}, test={len(test_sub)}")
        train_keys = tuple(k for k in inter_keys if k[0] in train_sub)
        val_keys   = tuple(k for k in inter_keys if k[0] in val_sub)
        test_keys  = tuple(k for k in inter_keys if k[0] in test_sub)
    try:
        tr_sub = sorted({int(k[0]) for k in train_keys}); va_sub = sorted({int(k[0]) for k in val_keys}); te_sub = sorted({int(k[0]) for k in test_keys})
        print(f"[split] train: keys={len(train_keys)} subjects={tr_sub}")
        print(f"[split] val  : keys={len(val_keys)} subjects={va_sub}")
        print(f"[split] test : keys={len(test_keys)} subjects={te_sub}")
    except Exception:
        print(f"[split] train|val|test key counts: {len(train_keys)} | {len(val_keys)} | {len(test_keys)}")
    return train_keys, val_keys, test_keys

# =========================
# Optional: load fixed split from YAML
# =========================
def try_load_fixed_split_from_yaml(cfg_path: Path) -> Optional[Tuple[List[int], List[int], List[int]]]:
    try:
        if not cfg_path.exists():
            print(f"[cfg] YAML not found: {cfg_path}")
            return None
        if yaml is None:
            print(f"[cfg] PyYAML not available; skipping YAML split load.")
            return None
        with open(cfg_path, 'r') as f:
            cfg = yaml.safe_load(f)
        train_list = cfg.get('train', {}).get('train_subjects')
        val_list   = cfg.get('train', {}).get('val_subjects')
        test_list  = cfg.get('train', {}).get('test_subjects')
        if train_list and val_list and test_list:
            tr = [int(s) for s in train_list]
            va = [int(s) for s in val_list]
            te = [int(s) for s in test_list]
            print(f"[cfg] loaded fixed split from YAML: train={tr} | val={va} | test={te}")
            return tr, va, te
        print(f"[cfg] YAML present but missing split keys; ignoring: {cfg_path}")
        return None
    except Exception as e:
        print(f"[cfg] failed to load YAML split: {e}")
        return None

# =========================
# Main run
# =========================
def main():
    # Basic checks
    for p in [EEG_ROOT, FMRI_ROOT, A424_LABEL_NII, BRAINLM_MODEL_DIR, RUN_DIR]:
        if p is None or not Path(p).exists():
            raise FileNotFoundError(f"Missing path: {p}")

    # Seeds / device
    seed_all(SEED)
    device = torch.device(DEVICE)
    out_dir = Path(OUT_DIR)
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)

    print(f"[note] Writing diagnostics under: {out_dir}")
    print(f"[debug] torch {torch.__version__} | cuda={torch.cuda.is_available()} | device={device}")
    if torch.cuda.is_available():
        print(f"[debug] GPU: {torch.cuda.get_device_name(0)}")
    print(f"[debug] platform={platform.platform()} | python={sys.version.split()[0]}")
    try:
        import brainlm_mae  # type: ignore
        print(f"[debug] brainlm_mae -> {getattr(brainlm_mae, '__file__', '<pkg>')}")
    except Exception as e:
        print(f"[debug] brainlm_mae import check failed: {e}")

    # Subject split & dataset
    print(f"[cfg] RUN_DIR={RUN_DIR}")
    print(f"[cfg] OUT_DIR={OUT_DIR}")
    print(f"[cfg] SUBSET={SUBSET} | STAGE={STAGE} | TAG={TAG}")
    print(f"[cfg] EEG_ROOT={EEG_ROOT}")
    print(f"[cfg] FMRI_ROOT={FMRI_ROOT}")
    # Try to load fixed split from YAML; otherwise fallback to on-disk split or default 80/10/10
    yaml_split = try_load_fixed_split_from_yaml(REPO_ROOT / 'Multi_modal_NTM' / 'configs' / 'fmri_selfattn.yaml')
    if yaml_split is not None:
        tr, va, te = yaml_split
        train_keys, val_keys, test_keys = fixed_subject_keys(Path(EEG_ROOT), Path(FMRI_ROOT), tr, va, te)
    else:
        train_keys, val_keys, test_keys = load_or_make_split(Path(EEG_ROOT), Path(FMRI_ROOT), Path(RUN_DIR))
    include_sr = {"train": train_keys, "val": val_keys, "test": test_keys}[SUBSET]
    # Show a few keys
    print(f"[split] using subset='{SUBSET}' with {len(include_sr)} (subject,task,run) keys")
    for i, k in enumerate(include_sr[:min(8, len(include_sr))]):
        print(f"  [split] key[{i}]: sub={k[0]} task={k[1]} run={k[2]}")

    # Select a single full session from the chosen subset
    if len(include_sr) == 0:
        raise RuntimeError(f"Empty subset '{SUBSET}' after split.")
    sel_sub, sel_task, sel_run = include_sr[0]
    print(f"[select] full session → subject={sel_sub} task={sel_task} run={sel_run}")

    # Resolve fMRI file path for selected key
    fmri_path = None
    for p in find_bold_files(Path(FMRI_ROOT)):
        s, t_raw, r = _parse_key_from_path(p)
        t = _canonical_task_token(t_raw, p)
        if (s, t, r) == (sel_sub, sel_task, sel_run):
            fmri_path = p
            break
    if fmri_path is None:
        raise RuntimeError(f"Could not locate fMRI file for key: {(sel_sub, sel_task, sel_run)}")
    print(f"[select] fMRI file: {fmri_path}")

    # Load entire session (T×V), then normalize per-voxel across time
    ts = load_fmri_parcels_cached(Path(fmri_path), str(A424_LABEL_NII), cache_dir=default_cache_dir())  # (T,V) float32
    fmri_full = torch.from_numpy(ts).unsqueeze(0).float()  # (1,T,V)
    fmri_full = normalize_fmri_time_series(fmri_full, mode=FMRI_NORM)
    fmri_t = fmri_full.to(device)
    B, T, V = map(int, fmri_t.shape)
    print(f"[debug] full-session shapes: fmri={tuple(fmri_t.shape)} | {fmt_mem()}")

    # Frozen BrainLM (provides time_patch_size)
    brainlm = FrozenBrainLM(Path(BRAINLM_MODEL_DIR), device)
    print(f"[debug] BrainLM time_patch_size = {brainlm.time_patch_size}")

    # Build a single session-wide PATCH-SPARSE mask (mirrors training)
    if APPLY_TIME_MASK:
        M_full = _make_patch_sparse_mask(
            B=B, T=T, V=V, device=device, patch_size=int(brainlm.time_patch_size)
        )  # True = masked
        mask_full = M_full[0].detach().cpu().numpy()  # (T,V) for visualization
    else:
        M_full = None
        mask_full = np.zeros((T, V), dtype=bool)

    # XYZ coordinates (A424)
    coords = None
    xyz = torch.zeros(B, V, 3, device=device)
    try:
        candidates = [
            THIS_DIR / 'BrainLM' / 'resources' / 'atlases' / 'A424_Coordinates.dat',
            THIS_DIR / 'resources' / 'atlases' / 'A424_Coordinates.dat',
            REPO_ROOT / 'BrainLM' / 'toolkit' / 'atlases' / 'A424_Coordinates.dat',
        ]
        for p in candidates:
            if p.exists():
                arr = load_a424_coords(p)
                arr = arr / (float(np.max(np.abs(arr))) or 1.0)
                coords = torch.from_numpy(arr.astype(np.float32)).to(device)
                print(f"[debug] loaded A424 coords from: {p}")
                break
        if coords is not None and V == 424:
            xyz = coords.unsqueeze(0).repeat(B,1,1)
        else:
            if V != 424:
                print(f"[warn] V={V} != 424; using zero xyz.")
            elif coords is None:
                print(f"[warn] A424_Coordinates.dat not found; using zero xyz.")
    except Exception as e:
        print(f"[warn] failed to load coords: {e}; using zeros.")

    # Translator head (aligned to trainer, with voxel PE)
    translator = TranslatorFMRISelfAttn(
        fmri_voxels=V, window_sec=WINDOW_SEC, tr=TR,
        fmri_n_layers=brainlm.n_layers_out, fmri_hidden_size=brainlm.hidden_size,
        d_model=128, n_heads=4, d_ff=512, dropout=0.1
    ).to(device)
    translator.eval()

    # ---- Load per-module weights (ALWAYS try adapter/voxel_pe/encoder/decoder/AFFINE) ----
    default_tag = f"revstage{STAGE}_best"
    use_tag = TAG or default_tag
    want = ["adapter", "voxel_pe", "encoder", "decoder", "affine"]
    loaded = load_modules_if_exist(Path(RUN_DIR), translator, use_tag, which=want, device=device)

    if not loaded:
        print(f"[WARN] No modules found for tag '{use_tag}' in {RUN_DIR}.")
    else:
        print(f"[load] restored modules for tag '{use_tag}': {', '.join(sorted(loaded))}")

    print(f"[diag] affine scale={translator.fmri_out_scale.item():.6f}, "
          f"bias={translator.fmri_out_bias.item():.6f}")

    # Reconstruct fMRI over the entire session via chunked inference
    # Use floor to match training windows: T_chunk = floor(WINDOW_SEC / TR)
    chunk_T = max(1, int(WINDOW_SEC / float(TR)))
    print(f"[infer] chunked inference: WINDOW_SEC={WINDOW_SEC} → chunk_T={chunk_T} TRs, V={V}")
    recon_full = np.zeros((T, V), dtype=np.float32)

    # Inference
    with torch.inference_mode():
        start = 0
        while start < T:
            end = min(T, start + chunk_T)
            seg_T = end - start
            fmri_seg = fmri_t[:, start:end, :]  # (B, seg_T, V)
            # Apply session-wide PATCH-SPARSE mask slice (mirror training)
            if APPLY_TIME_MASK and M_full is not None:
                fmri_seg = _apply_zero_mask(fmri_seg, M_full[:, start:end, :])

            # pad for BrainLM tokenization (use inferred patch size)
            fmri_pad = pad_timepoints_for_brainlm_torch(fmri_seg, patch_size=int(brainlm.time_patch_size))  # (B,Tp,V)
            signal_vectors = fmri_pad.permute(0,2,1).contiguous()                                           # (B,V,Tp)
            fmri_latents = brainlm.extract_latents(signal_vectors, xyz)                                     # (L,B,Ttok,Dh)
            recon_chunk = translator(fmri_latents, fmri_T=chunk_T, fmri_V=V, xyz=xyz)                       # (B,chunk_T,V)
            r_np = recon_chunk[0, :seg_T, :].detach().cpu().numpy()                                         # (seg_T, V)
            recon_full[start:end, :] = r_np
            start = end
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Correlations & diagnostics on first item
    b = 0
    x_true = fmri_t[b].detach().cpu().numpy()    # (T,V)
    x_rec  = recon_full                          # (T,V)

    # ---------- heatmaps ----------
    SHARED_VLIM: Optional[Tuple[float, float]] = None
    # Example to force identical scales across both:
    # SHARED_VLIM = (min(x_true.min(), x_rec.min()), max(x_true.max(), x_rec.max()))

    save_heatmap(
        x_true,
        out_dir / "plots" / "fmri_target_heatmap.png",
        title="fMRI target (T × V)",
        tr=float(TR),
        vlim=SHARED_VLIM,
    )
    save_heatmap(
        x_rec,
        out_dir / "plots" / "fmri_recon_heatmap.png",
        title="fMRI recon (T × V)",
        tr=float(TR),
        vlim=SHARED_VLIM,
    )
    # Mask heatmap (binary)
    if APPLY_TIME_MASK:
        save_heatmap(
            mask_full.astype(np.float32),
            out_dir / "plots" / "fmri_mask_heatmap.png",
            title="Applied patch-sparse mask (T × V)",
            tr=float(TR),
            cmap="Reds",
            vlim=(0.0, 1.0),
        )

    # ---------- per-ROI diagnostics ----------
    diag_rows: List[List[float]] = []
    rs: List[Tuple[int,float]] = []
    for roi in range(V):
        gt = x_true[:, roi]
        rc = x_rec[:, roi]
        r0 = pearsonr_safe(gt, rc)
        # keep linear diagnostics for CSV compatibility, but don't plot calibrated
        a, bb, rc_lin, R2, NRMSE = lin_calibrate(rc, gt)
        best_lag, r_best = lagged_corr(gt, rc, MAX_LAG_TR)
        diag_rows.append([roi, r0, float(gt.std()), float(rc.std()), a, bb, R2, NRMSE, best_lag, r_best])
        rs.append((roi, r0))
    rs.sort(key=lambda t: t[1], reverse=True)

    # Save diagnostics CSV
    csv_path = out_dir / "tables" / "fmri_roi_diagnostics.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["roi","r0","std_gt","std_rec","slope","intercept","R2","NRMSE","best_lag_TR","r_at_best_lag"])
        for row in diag_rows:
            w.writerow(row)
    print(f"[save] wrote diagnostics -> {csv_path}")

    # Print quick summary
    r_vals = np.array([r for _, r in rs], dtype=np.float32)
    std_gt = np.array([row[2] for row in diag_rows], dtype=np.float32)
    std_rc = np.array([row[3] for row in diag_rows], dtype=np.float32)
    small_var_frac = float(np.mean(std_rc < 0.2))
    print(f"[summary] mean r0={r_vals.mean():.3f} | median r0={np.median(r_vals):.3f} | max r0={r_vals.max():.3f}")
    print(f"[summary] mean std(GT)={std_gt.mean():.3f} | mean std(Rec)={std_rc.mean():.3f} | frac std(Rec)<0.2 = {small_var_frac:.2f}")

    # ---------- fMRI correlation matrices ----------
    C_fmri_gt  = corr_matrix(x_true)              # (V x V)
    C_fmri_rec = corr_matrix(x_rec)               # (V x V)
    C_fmri_x   = cross_corr_matrix(x_true, x_rec) # (V x V) GT vs Recon

    plot_corrmat(
        C_fmri_gt,
        out_dir / "plots" / "corrmat_fmri_gt.png",
        title="Corr — fMRI GT (ROI×ROI)",
        xlabel="ROI (GT)",
        ylabel="ROI (GT)",
    )
    plot_corrmat(
        C_fmri_rec,
        out_dir / "plots" / "corrmat_fmri_recon.png",
        title="Corr — fMRI Recon (ROI×ROI)",
        xlabel="ROI (Recon)",
        ylabel="ROI (Recon)",
    )
    plot_corrmat(
        C_fmri_x,
        out_dir / "plots" / "corrmat_fmri_gt_vs_recon.png",
        title="Cross Corr — fMRI GT vs Recon (ROI×ROI)",
        xlabel="ROI (Recon)",
        ylabel="ROI (GT)",
    )

    save_corr_csv(C_fmri_gt,  out_dir / "tables" / "corrmat_fmri_gt.csv",
                  row_prefix="roi", col_prefix="roi")
    save_corr_csv(C_fmri_rec, out_dir / "tables" / "corrmat_fmri_recon.csv",
                  row_prefix="roi", col_prefix="roi")
    save_corr_csv(C_fmri_x,   out_dir / "tables" / "corrmat_fmri_gt_vs_recon.csv",
                  row_prefix="roi_gt", col_prefix="roi_rec")

    # ---------- Top-K plots (raw) ----------
    top = rs[:min(TOP_K, len(rs))]
    t = np.arange(T) * float(TR)
    if len(top) > 0:
        fig, axes = plt.subplots(nrows=len(top), ncols=1, figsize=(12, 2.2*len(top)), sharex=False)
        if len(top) == 1: axes = [axes]
        for ax, (roi, r) in zip(axes, top):
            gt = x_true[:, roi]; rc = x_rec[:, roi]
            tt, gtd = downsample(t, gt, PLOT_MAX_POINTS)
            _,  rcd = downsample(t, rc, PLOT_MAX_POINTS)
            ax.plot(tt, gtd, label="GT")
            ax.plot(tt, rcd, label="Recon", alpha=0.9)
            # shade masked TRs for this ROI
            if APPLY_TIME_MASK:
                m = mask_full[:, roi].astype(bool)
                if m.any():
                    idx = np.where(m)[0]
                    s = None
                    for i in range(len(idx)):
                        if s is None:
                            s = idx[i]
                        if i == len(idx)-1 or idx[i+1] != idx[i] + 1:
                            e = idx[i]
                            ax.axvspan(s*float(TR), (e+1)*float(TR), color="red", alpha=0.12, lw=0)
                            s = None
            ax.set_title(f"ROI {roi} — r={r:.3f}")
            ax.set_xlabel("Time (s)"); ax.set_ylabel("Z-score")
            ax.legend(loc="upper right")
        fig.tight_layout()
        fig.savefig(out_dir / "plots" / "topK_fmri_corr.png", dpi=150)
        plt.close(fig)

    # Bar chart (all ROIs, raw r)
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

    # Console summary of top-K
    print("\n=== Top ROIs by Pearson r (raw) ===")
    for roi, r in top:
        print(f"  ROI={roi:03d} r={r:.4f}")
    print(f"\nSaved outputs -> {out_dir}")

if __name__ == "__main__":
    main()
