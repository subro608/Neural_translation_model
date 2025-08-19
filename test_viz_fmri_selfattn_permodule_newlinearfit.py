#!/usr/bin/env python3
"""
fMRI translator diagnostics + train+val linear calibration (GPU OLS)

What this does:
- Loads your translator (adapter/encoder/decoder/affine as available).
- Computes recon on TRAIN+VAL to learn per-ROI linear calibration: gt ≈ a*rc + b.
  * Closed-form OLS in PyTorch on GPU (scikit-learn doesn't use GPU).
  * Online accumulators across batches; memory-safe.
- Saves calibration (a,b) to CSV.
- Evaluates on TEST:
  * Writes per-ROI diagnostics with raw r, stds, and calibrated R2/NRMSE.
  * Plots top-K raw and calibrated traces; bar chart of raw r.

Outputs:
  tables/calibration_ab_trainval.csv
  tables/fmri_roi_diagnostics.csv
  plots/topK_fmri_corr.png
  plots/topK_fmri_corr_calibrated.png
  plots/corr_bars_fmri.png
"""

from __future__ import annotations
import os, json, math, platform, sys, csv, time
from pathlib import Path
from typing import Iterable, Tuple, List, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# ========= USER CONFIG =========
EEG_ROOT          = Path(r"D:\Neuroinformatics_research_2025\Oddball\ds116_eeg")
FMRI_ROOT         = Path(r"D:\Neuroinformatics_research_2025\Oddball\ds000116")
A424_LABEL_NII    = Path(r"D:\Neuroinformatics_research_2025\BrainLM\A424_resampled_to_bold.nii.gz")
BRAINLM_MODEL_DIR = Path(r"D:\Neuroinformatics_research_2025\MNI_templates\BrainLM\pretrained_models\2023-06-06-22_15_00-checkpoint-1400")

RUN_DIR           = Path(r"D:\Neuroinformatics_research_2025\Multi_modal_NTM\translator_runs_fmri_selfattn_aug19_stage1")
OUT_DIR           = Path(r"D:\Neuroinformatics_research_2025\Multi_modal_NTM\viz_fmri_selfattn_permodule_stage1_best_19aug_diag_newlinearfit")

CALIB_SPLITS        = ("train", "val")  # fit a,b here
EVAL_SPLIT          = "test"            # evaluate here

STAGE               = 1                 # 1 | 2 | 3
TAG                 = "stage1_best"     # None -> f"stage{STAGE}_best"
TOP_K               = 12
PLOT_MAX_POINTS     = 1000
MAX_LAG_TR          = 3                 # lag search (in TRs) for raw corr report
MAKE_CALIB_PLOTS    = True

DEVICE              = "cuda"            # "cpu" to bypass CUDA
SEED                = 42
WINDOW_SEC          = 30
TR                  = 2.0
FMRI_NORM           = "zscore"          # must match your training
BATCH_SIZE          = 8
NUM_WORKERS         = 0
STRIDE_SEC          = 10
CHANNELS_LIMIT      = 34

# ========= REPO PATHS / IMPORTS =========
THIS_DIR   = Path(__file__).parent
REPO_ROOT  = THIS_DIR.parent
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

from module import (  # type: ignore
    fMRIInputAdapterConv1d,
    HierarchicalEncoder,
    fMRIDecodingAdapterLite,
)
from brainlm_mae.modeling_brainlm import BrainLMForPretraining  # type: ignore
from brainlm_mae.configuration_brainlm import BrainLMConfig     # type: ignore
from data_oddball import (  # type: ignore
    PairedAlignedDataset,
    collate_paired,
    pad_timepoints_for_brainlm_torch,
    load_a424_coords,
    collect_common_sr_keys,
    fixed_subject_keys,
)

# ========= UTILS =========
def seed_all(seed:int):
    np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

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
    lags = range(-max_lag, max_lag+1)
    best_r = -2.0; best_l = 0
    for l in lags:
        if l < 0:  r = pearsonr_safe(gt[:l], rc[-l:])
        elif l>0: r = pearsonr_safe(gt[l:], rc[:-l])
        else:     r = pearsonr_safe(gt, rc)
        if np.isfinite(r) and r > best_r:
            best_r, best_l = r, l
    return best_l, best_r

# ========= MODELS =========
class FrozenBrainLM(nn.Module):
    def __init__(self, model_dir: Path, device: torch.device):
        super().__init__()
        cfg_path = Path(model_dir) / "config.json"
        w_path   = Path(model_dir) / "pytorch_model.bin"
        with open(cfg_path, "r") as f:
            cfg = BrainLMConfig(**json.load(f))
        self.model = BrainLMForPretraining(cfg)
        ckpt = torch.load(str(w_path), map_location=device)
        self.model.load_state_dict(ckpt, strict=False)
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

class TranslatorFMRISelfAttn(nn.Module):
    def __init__(self, fmri_voxels:int, window_sec:int, tr:float,
                 fmri_n_layers:int, fmri_hidden_size:int,
                 d_model:int=128, n_heads:int=4, d_ff:int=512, dropout:float=0.1):
        super().__init__()
        import math as _m
        self.fmri_voxels = int(fmri_voxels)
        self.window_sec  = int(window_sec)
        self.tr          = float(tr)

        self.adapter_fmri = fMRIInputAdapterConv1d(
            seq_len=self.fmri_voxels * int(round(self.window_sec/self.tr)),
            n_layers=fmri_n_layers, input_dim=fmri_hidden_size,
            output_dim=d_model, target_seq_len=512
        )
        pe = torch.zeros(100_000, d_model)
        pos = torch.arange(0, 100_000, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32)*(-_m.log(10000.0)/d_model))
        pe[:,0::2]=torch.sin(pos*div); pe[:,1::2]=torch.cos(pos*div)
        self.pos_fmri = nn.Parameter(pe, requires_grad=False)

        self.fmri_encoder = HierarchicalEncoder(d_model, n_heads, d_ff, dropout, n_layers_per_stack=1)

        T = int(round(self.window_sec/self.tr))
        self.fmri_decoder = fMRIDecodingAdapterLite(target_T=T, target_V=self.fmri_voxels, d_model=d_model, rank=16)

        self.fmri_out_scale = nn.Parameter(torch.tensor(1.0))
        self.fmri_out_bias  = nn.Parameter(torch.tensor(0.0))

    def forward(self, fmri_latents: torch.Tensor, fmri_T:int, fmri_V:int) -> torch.Tensor:
        fmri_adapt = self.adapter_fmri(fmri_latents)  # (B,512,D)
        fmri_adapt = fmri_adapt + self.pos_fmri[:fmri_adapt.size(1)].unsqueeze(0).to(fmri_adapt.device)
        _, fmr_hi = self.fmri_encoder(fmri_adapt, fmri_adapt)  # (B,Tf,D)
        fmri_flat = self.fmri_decoder(fmr_hi)                  # (B, T*V)
        fmri_sig  = fmri_flat.view(fmri_adapt.size(0), int(fmri_T), int(fmri_V))
        fmri_sig  = torch.tanh(fmri_sig)
        fmri_sig  = self.fmri_out_scale * fmri_sig + self.fmri_out_bias
        return fmri_sig

# ========= LOADING HELPERS =========
def _module_paths(run_dir: Path, tag: str) -> Dict[str, Path]:
    run_dir = Path(run_dir); run_dir.mkdir(parents=True, exist_ok=True)
    return {
        "adapter": run_dir / f"adapter_fmri_{tag}.pt",
        "encoder": run_dir / f"fmri_encoder_{tag}.pt",
        "decoder": run_dir / f"fmri_decoder_{tag}.pt",
        "affine":  run_dir / f"fmri_affine_{tag}.pt",
    }

def load_modules_if_exist(run_dir: Path, model: TranslatorFMRISelfAttn, tag: str,
                          which: Iterable[str], device: torch.device) -> set:
    paths = _module_paths(run_dir, tag); loaded = set()
    if "adapter" in which and paths["adapter"].exists():
        sd = torch.load(paths["adapter"], map_location=device)
        model.adapter_fmri.load_state_dict(sd, strict=False); loaded.add("adapter")
    if "encoder" in which and paths["encoder"].exists():
        sd = torch.load(paths["encoder"], map_location=device)
        model.fmri_encoder.load_state_dict(sd, strict=False); loaded.add("encoder")
    if "decoder" in which and paths["decoder"].exists():
        sd = torch.load(paths["decoder"], map_location=device)
        model.fmri_decoder.load_state_dict(sd, strict=False); loaded.add("decoder")
    if "affine" in which and paths["affine"].exists():
        sd = torch.load(paths["affine"], map_location=device)
        if isinstance(sd, dict) and "scale" in sd and "bias" in sd:
            with torch.no_grad():
                model.fmri_out_scale.copy_(sd["scale"].to(model.fmri_out_scale.device))
                model.fmri_out_bias.copy_(sd["bias"].to(model.fmri_out_bias.device))
        else:
            try: model.load_state_dict(sd, strict=False)
            except Exception: pass
        loaded.add("affine")
    return loaded

def load_or_make_split(eeg_root: Path, fmri_root: Path, run_dir: Path):
    p_split = Path(run_dir) / "subject_splits.json"
    from data_oddball import collect_common_sr_keys, fixed_subject_keys
    inter_keys = collect_common_sr_keys(eeg_root, fmri_root)
    if not inter_keys:
        raise RuntimeError("No (subject,task,run) intersections between EEG and fMRI trees.")
    if p_split.exists():
        js = json.loads(Path(p_split).read_text())
        tr = [int(norm_subj_id(s)) for s in js.get("train_subjects", [])]
        va = [int(norm_subj_id(s)) for s in js.get("val_subjects", [])]
        te = [int(norm_subj_id(s)) for s in js.get("test_subjects", [])]
        train_keys, val_keys, test_keys = fixed_subject_keys(eeg_root, fmri_root, tr, va, te)
    else:
        subs = sorted({k[0] for k in inter_keys}, key=int)
        n = len(subs); nt = max(1, int(0.8*n)); nv = max(1, int(0.1*n))
        train_sub = set(subs[:nt]); val_sub = set(subs[nt:nt+nv]); test_sub = set(subs[nt+nv:])
        train_keys = tuple(k for k in inter_keys if k[0] in train_sub)
        val_keys   = tuple(k for k in inter_keys if k[0] in val_sub)
        test_keys  = tuple(k for k in inter_keys if k[0] in test_sub)
    return train_keys, val_keys, test_keys

# ========= ONLINE GPU CALIBRATION =========
class OnlineAB:
    """
    Learns per-ROI a,b for gt ≈ a*rc + b using streaming accumulators:
      mu_x, mu_y, var_x, cov_xy  (computed from sums)
    """
    def __init__(self, V:int, device:torch.device):
        self.V = V
        self.device = device
        dtype = torch.float64
        self.sum_x  = torch.zeros(V, device=device, dtype=dtype)
        self.sum_y  = torch.zeros(V, device=device, dtype=dtype)
        self.sum_xx = torch.zeros(V, device=device, dtype=dtype)
        self.sum_xy = torch.zeros(V, device=device, dtype=dtype)
        self.n      = torch.zeros(V, device=device, dtype=dtype)

    @torch.no_grad()
    def update(self, rc_btV: torch.Tensor, gt_btV: torch.Tensor):
        # rc_btV, gt_btV: (B,T,V) on device; accumulate along B*T
        x = rc_btV.reshape(-1, self.V).to(self.sum_x.dtype)
        y = gt_btV.reshape(-1, self.V).to(self.sum_x.dtype)
        # mask non-finite
        m = torch.isfinite(x) & torch.isfinite(y)
        x = torch.where(m, x, torch.zeros_like(x))
        y = torch.where(m, y, torch.zeros_like(y))
        cnt = m.sum(dim=0, dtype=self.n.dtype)
        self.sum_x  += x.sum(dim=0)
        self.sum_y  += y.sum(dim=0)
        self.sum_xx += (x*x).sum(dim=0)
        self.sum_xy += (x*y).sum(dim=0)
        self.n      += cnt

    @torch.no_grad()
    def finalize(self, eps:float=1e-8) -> Tuple[np.ndarray, np.ndarray]:
        n = torch.clamp(self.n, min=1.0)
        mu_x = self.sum_x / n
        mu_y = self.sum_y / n
        var_x = torch.clamp(self.sum_xx / n - mu_x*mu_x, min=0.0)
        cov_xy = self.sum_xy / n - mu_x*mu_y
        a = cov_xy / (var_x + eps)
        b = mu_y - a*mu_x
        # handle near-zero variance -> a=0, b=mu_y
        zero_var = var_x <= 1e-10
        a = torch.where(zero_var, torch.zeros_like(a), a)
        b = torch.where(zero_var, mu_y, b)
        return a.double().cpu().numpy(), b.double().cpu().numpy()

# ========= MAIN =========
def main():
    # Basics
    for p in [EEG_ROOT, FMRI_ROOT, A424_LABEL_NII, BRAINLM_MODEL_DIR, RUN_DIR]:
        if p is None or not Path(p).exists():
            raise FileNotFoundError(f"Missing path: {p}")
    seed_all(SEED)
    device = torch.device(DEVICE)
    out_dir = Path(OUT_DIR); (out_dir / "plots").mkdir(parents=True, exist_ok=True); (out_dir / "tables").mkdir(parents=True, exist_ok=True)

    print(f"[debug] torch {torch.__version__} | cuda={torch.cuda.is_available()} | device={device}")
    if torch.cuda.is_available(): print(f"[debug] GPU: {torch.cuda.get_device_name(0)}")
    print(f"[debug] platform={platform.platform()} | python={sys.version.split()[0]}")
    try:
        import brainlm_mae  # type: ignore
        print(f"[debug] brainlm_mae -> {getattr(brainlm_mae, '__file__', '<pkg>')}")
    except Exception as e:
        print(f"[debug] brainlm_mae import check failed: {e}")

    # Splits
    train_keys, val_keys, test_keys = load_or_make_split(EEG_ROOT, FMRI_ROOT, RUN_DIR)
    keys_by_split = {"train": train_keys, "val": val_keys, "test": test_keys}

    # Build a tiny helper to make a DataLoader for a split
    def make_loader(split:str) -> Tuple[DataLoader, int, int]:
        include_sr = keys_by_split[split]
        ds = PairedAlignedDataset(
            eeg_root=EEG_ROOT, fmri_root=FMRI_ROOT, a424_label_nii=A424_LABEL_NII,
            window_sec=WINDOW_SEC, original_fs=1000, target_fs=200, tr=TR,
            channels_limit=CHANNELS_LIMIT, fmri_norm=FMRI_NORM, stride_sec=STRIDE_SEC,
            device='cpu', include_sr_keys=include_sr
        )
        if len(ds) == 0: raise RuntimeError(f"Empty dataset for split='{split}'. Check paths/splits.")
        dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
                        pin_memory=(device.type=='cuda'), collate_fn=collate_paired, drop_last=False)
        # Probe one batch to get T,V for constructing the translator once
        batch0 = next(iter(dl))
        T, V = int(batch0['fmri_window'].shape[1]), int(batch0['fmri_window'].shape[2])
        return dl, T, V

    # One loader to get T,V, then reuse translator across splits
    dl_probe, T, V = make_loader(CALIB_SPLITS[0])
    # Models
    brainlm = FrozenBrainLM(BRAINLM_MODEL_DIR, device)
    translator = TranslatorFMRISelfAttn(
        fmri_voxels=V, window_sec=WINDOW_SEC, tr=TR,
        fmri_n_layers=brainlm.n_layers_out, fmri_hidden_size=brainlm.hidden_size,
        d_model=128, n_heads=4, d_ff=512, dropout=0.1
    ).to(device).eval()

    # Load per-module weights
    default_tag = f"stage{STAGE}_best"; use_tag = TAG or default_tag
    need = ["adapter"] if STAGE == 1 else (["adapter","encoder"] if STAGE == 2 else ["adapter","encoder","decoder","affine"])
    loaded = load_modules_if_exist(RUN_DIR, translator, use_tag, which=need, device=device)
    missing = set(need) - loaded
    if missing: print(f"[WARN] Some modules for '{use_tag}' not found in {RUN_DIR}: {sorted(missing)}")
    else:       print(f"[load] restored modules for tag '{use_tag}': {', '.join(sorted(loaded))}")
    print(f"[diag] affine scale={translator.fmri_out_scale.item():.6f}, bias={translator.fmri_out_bias.item():.6f}")

    # === helper: forward pass ===
    @torch.no_grad()
    def forward_batch(fmri_t: torch.Tensor) -> torch.Tensor:
        # fmri_t: (B,T,V) on device
        fmri_pad = pad_timepoints_for_brainlm_torch(fmri_t, patch_size=20)  # (B,Tp,V)
        signal_vectors = fmri_pad.permute(0,2,1).contiguous()               # (B,V,Tp)
        # xyz
        xyz = torch.zeros(signal_vectors.shape[0], V, 3, device=device)
        # try coords if available and V==424
        try:
            candidates = [
                THIS_DIR / 'BrainLM' / 'resources' / 'atlases' / 'A424_Coordinates.dat',
                THIS_DIR / 'resources' / 'atlases' / 'A424_Coordinates.dat',
                REPO_ROOT / 'BrainLM' / 'toolkit' / 'atlases' / 'A424_Coordinates.dat',
            ]
            coords = None
            for p in candidates:
                if p.exists():
                    arr = load_a424_coords(p); arr = arr / (float(np.max(np.abs(arr))) or 1.0)
                    coords = torch.from_numpy(arr.astype(np.float32)).to(device); break
            if coords is not None and V == 424:
                xyz = coords.unsqueeze(0).repeat(signal_vectors.size(0),1,1)
        except Exception:
            pass
        lat = brainlm.extract_latents(signal_vectors, xyz)                  # (L,B,Ttok,Dh)
        recon = translator(lat, fmri_T=T, fmri_V=V)                         # (B,T,V)
        return recon

    # === 1) CALIBRATION on TRAIN+VAL ===
    calib = OnlineAB(V, device)
    for split in CALIB_SPLITS:
        print(f"[calib] accumulating {split} ...")
        dl, _, _ = make_loader(split)
        for batch in dl:
            fmri_t = batch['fmri_window'].to(device, non_blocking=True)
            with torch.no_grad():
                recon = forward_batch(fmri_t)
            calib.update(recon, fmri_t)
    a_np, b_np = calib.finalize()
    # Save calibration
    p_calib = out_dir / "tables" / "calibration_ab_trainval.csv"
    with open(p_calib, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["roi","a","b"])
        for roi in range(V): w.writerow([roi, float(a_np[roi]), float(b_np[roi])])
    print(f"[calib] saved train+val calibration -> {p_calib}")

    # === 2) EVALUATION on TEST ===
    print(f"[eval] evaluating on {EVAL_SPLIT} ...")
    dl_test, _, _ = make_loader(EVAL_SPLIT)

    # Gather one batch to make plots; compute full diagnostics across first batch only (like before)
    batch = next(iter(dl_test))
    fmri_t = batch['fmri_window'].to(device, non_blocking=True)  # (B,T,V)
    with torch.no_grad():
        recon = forward_batch(fmri_t)                            # (B,T,V)

    # use first item for plotting/CSV (consistent with earlier scripts)
    b0 = 0
    x_true = fmri_t[b0].detach().cpu().numpy()   # (T,V)
    x_rec  = recon[b0].detach().cpu().numpy()    # (T,V)

    # Apply calibration to this batch (broadcast a,b: (T,V))
    a = a_np[None, :]     # (1,V)
    b = b_np[None, :]     # (1,V)
    x_rec_cal = (x_rec * a) + b

    # Per-ROI diagnostics (raw r, stds; calibrated R2/NRMSE; raw best-lag r)
    def lin_R2_NRMSE(gt: np.ndarray, pred: np.ndarray) -> Tuple[float,float]:
        m = np.isfinite(gt) & np.isfinite(pred)
        gt = gt[m]; pred = pred[m]
        if len(gt) < 3 or gt.std() == 0: return 0.0, 1.0
        err = gt - pred
        SSE = float(np.sum(err*err))
        R2  = 1.0 - SSE / (float(np.sum((gt - gt.mean())**2)) + 1e-8)
        NRMSE = float(np.sqrt(SSE/len(gt)) / (gt.std() + 1e-8))
        return R2, NRMSE

    diag_rows: List[List[float]] = []
    rs: List[Tuple[int,float]] = []
    for roi in range(V):
        gt = x_true[:, roi]; rc = x_rec[:, roi]; rc_cal = x_rec_cal[:, roi]
        r0 = pearsonr_safe(gt, rc)  # raw corr
        R2c, NRMSEc = lin_R2_NRMSE(gt, rc_cal)
        best_lag, r_best = lagged_corr(gt, rc, MAX_LAG_TR)
        std_gt = float(np.nanstd(gt)); std_rc = float(np.nanstd(rc))
        diag_rows.append([roi, r0, std_gt, std_rc, float(a_np[roi]), float(b_np[roi]), R2c, NRMSEc, best_lag, r_best])
        rs.append((roi, r0))
    rs.sort(key=lambda t: t[1], reverse=True)

    # Save diagnostics CSV
    csv_path = out_dir / "tables" / "fmri_roi_diagnostics.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["roi","r0","std_gt","std_rec","a_fit_trainval","b_fit_trainval","R2_calib","NRMSE_calib","best_lag_TR_raw","r_at_best_lag_raw"])
        for row in diag_rows: w.writerow(row)
    print(f"[save] wrote diagnostics -> {csv_path}")

    # Summary
    r_vals = np.array([r for _, r in rs], dtype=np.float32)
    std_gt = np.array([row[2] for row in diag_rows], dtype=np.float32)
    std_rc = np.array([row[3] for row in diag_rows], dtype=np.float32)
    small_var_frac = float(np.mean(std_rc < 0.2))
    print(f"[summary] mean r0={r_vals.mean():.3f} | median r0={np.median(r_vals):.3f} | max r0={r_vals.max():.3f}")
    print(f"[summary] mean std(GT)={std_gt.mean():.3f} | mean std(Rec)={std_rc.mean():.3f} | frac std(Rec)<0.2 = {small_var_frac:.2f}")

    # === PLOTS (first item) ===
    t = np.arange(T) * float(TR)
    top = rs[:min(TOP_K, len(rs))]

    # Raw
    if len(top) > 0:
        fig, axes = plt.subplots(nrows=len(top), ncols=1, figsize=(12, 2.2*len(top)), sharex=False)
        if len(top) == 1: axes = [axes]
        for ax, (roi, r) in zip(axes, top):
            gt = x_true[:, roi]; rc = x_rec[:, roi]
            tt, gtd = downsample(t, gt, PLOT_MAX_POINTS)
            _,  rcd = downsample(t, rc, PLOT_MAX_POINTS)
            ax.plot(tt, gtd, label="GT")
            ax.plot(tt, rcd, label="Recon", alpha=0.9)
            ax.set_title(f"ROI {roi} — r(raw)={r:.3f}")
            ax.set_xlabel("Time (s)"); ax.set_ylabel("Z-score"); ax.legend(loc="upper right")
        fig.tight_layout()
        fig.savefig(out_dir / "plots" / "topK_fmri_corr.png", dpi=150)
        plt.close(fig)

    # Calibrated
    if MAKE_CALIB_PLOTS and len(top) > 0:
        fig, axes = plt.subplots(nrows=len(top), ncols=1, figsize=(12, 2.2*len(top)), sharex=False)
        if len(top) == 1: axes = [axes]
        for ax, (roi, r_raw) in zip(axes, top):
            gt = x_true[:, roi]; rc_cal = x_rec_cal[:, roi]
            R2c, NRMSEc = diag_rows[roi][6], diag_rows[roi][7]
            tt, gtd = downsample(t, gt, PLOT_MAX_POINTS)
            _,  rcd = downsample(t, rc_cal, PLOT_MAX_POINTS)
            ax.plot(tt, gtd, label="GT")
            ax.plot(tt, rcd, label=f"Recon (calib) a={a_np[roi]:.2f}, b={b_np[roi]:.2f}, R²={R2c:.2f}", alpha=0.9)
            ax.set_title(f"ROI {roi} — r(raw)={r_raw:.3f}")
            ax.set_xlabel("Time (s)"); ax.set_ylabel("Z-score"); ax.legend(loc="upper right")
        fig.tight_layout()
        fig.savefig(out_dir / "plots" / "topK_fmri_corr_calibrated.png", dpi=150)
        plt.close(fig)

    # Bar chart (raw r)
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
    plt.ylabel("Pearson r (raw)")
    plt.title("fMRI: Pearson correlation per ROI")
    plt.tight_layout()
    plt.savefig(out_dir / "plots" / "corr_bars_fmri.png", dpi=150)
    plt.close()

    # Console summary of Top-K
    print("\n=== Top ROIs by Pearson r (raw) ===")
    for roi, r in top: print(f"  ROI={roi:03d} r={r:.4f}")
    print(f"\nSaved outputs -> {out_dir}")

if __name__ == "__main__":
    main()
