#!/usr/bin/env python3
"""
Visualize a FULL fMRI session (A424 parcels) with the SAME patch-sparse masking
as training, but now applied **per 40-second window** (i.e., one patch == one window).

Masking per window (like training when window_sec=40 and TR=2):
- window_len_TR = round(40s / TR) = 20 TRs
- For each non-overlapping window of length window_len_TR:
    - n_patches = 1 (the window itself)
    - r_patch ∈ [0.5, 0.8] -> k_patch = 1 (always select the only patch)
    - r_v ∈ [0.5, 0.8] -> mask that fraction of ROIs within the window
- Mask is applied BEFORE anything else (we just visualize it here).

Outputs (under OUT_DIR/sub<id>_<task>_run<id>/):
- plots/heatmap_gt.png
- plots/heatmap_mask.png
- plots/heatmap_masked.png
- plots/masked_frac_over_time.png
- plots/masked_frac_per_patch.png      (here “patch” == 40s window)
- tables/mask_patch_stats.csv          (per-window stats)
"""

from __future__ import annotations
import os, math, csv, time, glob, re
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import torch

# Matplotlib (no seaborn)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =========================
# PATHS & CONFIG (your values)
# =========================
EEG_ROOT          = Path(r"D:\Neuroinformatics_research_2025\Oddball\ds116_eeg")
FMRI_ROOT         = Path(r"D:\Neuroinformatics_research_2025\Oddball\ds000116")
A424_LABEL_NII    = Path(r"D:\Neuroinformatics_research_2025\BrainLM\A424_resampled_to_bold.nii.gz")

RUN_DIR           = Path(r"D:\Neuroinformatics_research_2025\Multi_modal_NTM\28_aug_allstages_comparison_reverse_mse5e5_voxelpe_masked")
SUBSET            = "test"   # "train" | "val" | "test"
OUT_DIR           = RUN_DIR / f"viz_diagnostics_{SUBSET}"

# (Only for naming consistency; not used functionally)
STAGE             = 1
TAG               = None

# Viz knobs
TOP_K             = 34
PLOT_MAX_POINTS   = 1000
MAX_LAG_TR        = 3
MAKE_CALIB_PLOTS  = False

# Masking control (mirror training)
APPLY_TIME_MASK   = True

# Device & seed (seed to make the mask reproducible)
DEVICE            = "cuda" if torch.cuda.is_available() else "cpu"
SEED              = 42

# Session selection (leave None to auto-pick the first intersecting session)
CHOSEN_SUBJECT: Optional[int] = None
CHOSEN_TASK:    Optional[str] = None
CHOSEN_RUN:     Optional[int] = None

# Dataset metadata
TR = 2.0
WINDOW_SEC = 40                           # <<—— per-window masking (like training)
WINDOW_TR  = int(round(WINDOW_SEC / TR))  # = 20 TRs when TR=2.0
V_EXPECTED = 424

# Masking params (same ranges as training code you’re using now)
R_PATCH_RANGE = (0.5, 0.8)
R_V_IN_PATCH_RANGE = (0.5, 0.8)

# -------------------------
# Imports from your data module
# -------------------------
from data_oddball import (
    collect_common_sr_keys,
    load_fmri_parcels_cached,
)

# -------------------------
# File/key utilities (mirror your canonicalization)
# -------------------------
_RE_SUB_GENERIC = re.compile(r"sub[-_]?0*([0-9]+)", re.IGNORECASE)
_RE_TASK_GENERIC = re.compile(r"task[-_]?([A-Za-z0-9]+)", re.IGNORECASE)
_RE_RUN_GENERIC  = re.compile(r"run[-_]?0*([0-9]+)", re.IGNORECASE)

def _parse_key_from_path(path: Path) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    s = str(path)
    sub  = _RE_SUB_GENERIC.search(s)
    task = _RE_TASK_GENERIC.search(s)
    run  = _RE_RUN_GENERIC.search(s)
    sub_id  = str(int(sub.group(1))) if sub else None
    task_id = (task.group(1).lower() if task else None)
    run_id  = str(int(run.group(1))) if run else None
    return sub_id, task_id, run_id

def _canonical_task_token(raw_task: Optional[str], path: Path) -> Optional[str]:
    s = path.name.lower() if raw_task is None else raw_task.lower()
    whole = str(path).lower()
    if 'auditory' in s or 'auditory' in whole:
        return 'task001'
    if 'visual' in s or 'visual' in whole:
        return 'task002'
    if s.startswith('task') and len(s) > 4 and s[4:].isdigit():
        return f"task{int(s[4:]):03d}"
    if s.isdigit():
        return f"task{int(s):03d}"
    return f"task_{s}" if s else None

def _find_bold_files(root: Path) -> List[Path]:
    pats = [str(root / "**" / "*bold.nii.gz"), str(root / "**" / "*bold.nii")]
    out: List[Path] = []
    for pat in pats:
        out.extend([Path(p) for p in glob.glob(pat, recursive=True)])
    return sorted(set(out), key=lambda p: str(p).lower())

def _resolve_fmri_path_for_key(fmri_root: Path, key: Tuple[str, str, str]) -> Optional[Path]:
    sub_k, task_k, run_k = key
    for p in _find_bold_files(fmri_root):
        s, t_raw, r = _parse_key_from_path(p)
        if s is None or r is None:
            continue
        t = _canonical_task_token(t_raw, p)
        if (s, t, r) == (sub_k, task_k, run_k):
            return p
    return None

# -------------------------
# Masking helpers (same behavior as trainer; per-call on a window)
# -------------------------
@torch.no_grad()
def make_patch_sparse_mask(
    B: int, T: int, V: int, device: torch.device,
    *,
    patch_len_tr: int,
    r_patch_range = R_PATCH_RANGE,
    r_v_in_patch_range = R_V_IN_PATCH_RANGE,
) -> torch.Tensor:
    """
    Returns boolean mask M (B, T, V) with True = masked.
    - Split T into patches of length `patch_len_tr` (last patch may be shorter).
    - Sample fraction of patches r_patch in [r_patch_range].
    - For each selected patch, sample r_v in [r_v_in_patch_range] and mask that
      fraction of ROIs across that patch's time span.
    """
    n_patches = int(math.ceil(T / float(patch_len_tr)))
    if n_patches <= 0:
        return torch.zeros((B, T, V), dtype=torch.bool, device=device)

    r_patch = float(torch.rand(1, device=device) * (r_patch_range[1] - r_patch_range[0]) + r_patch_range[0])
    k_patch = max(1, int(round(r_patch * n_patches)))
    sel_patches = torch.randperm(n_patches, device=device)[:k_patch]

    M = torch.zeros((B, T, V), dtype=torch.bool, device=device)
    for pidx in sel_patches.tolist():
        t_start = pidx * patch_len_tr
        t_end = min((pidx + 1) * patch_len_tr, T)
        r_v = float(torch.rand(1, device=device) * (r_v_in_patch_range[1] - r_v_in_patch_range[0]) + r_v_in_patch_range[0])
        k_v = max(1, int(round(r_v * V)))
        v_idx = torch.randperm(V, device=device)[:k_v]
        M[:, t_start:t_end, v_idx] = True
    return M

@torch.no_grad()
def apply_zero_mask(x_BTV: torch.Tensor, M_BTV: torch.Tensor) -> torch.Tensor:
    y = x_BTV.clone()
    y[M_BTV] = 0.0
    return y

# -------------------------
# Plot helpers
# -------------------------
def _ensure_dirs(out_dir: Path):
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)

def plot_heatmap(arr_TV: np.ndarray, title: str, out_path: Path, xlabel="ROI", ylabel="Time (TR)", cmap="viridis"):
    plt.figure(figsize=(12, 6))
    plt.imshow(arr_TV.T, aspect="auto", interpolation="nearest", origin="upper", cmap=cmap)
    plt.colorbar()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def plot_series(x: np.ndarray, y: np.ndarray, title: str, out_path: Path, xlabel="TR", ylabel="Masked fraction"):
    plt.figure(figsize=(12, 4))
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

# -------------------------
# Main
# -------------------------
def main():
    t0 = time.time()
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    device = torch.device(DEVICE)

    # Path checks
    for p in (EEG_ROOT, FMRI_ROOT, A424_LABEL_NII):
        if not Path(p).exists():
            raise FileNotFoundError(f"Path does not exist: {p}")

    # Find intersecting (sub, task, run)
    keys = collect_common_sr_keys(EEG_ROOT, FMRI_ROOT)
    if not keys:
        raise RuntimeError("No intersecting (subject, task, run) found between EEG and fMRI roots.")

    # Choose session
    def _matches(k):
        s_ok = (CHOSEN_SUBJECT is None or int(k[0]) == int(CHOSEN_SUBJECT))
        t_ok = (CHOSEN_TASK is None or k[1] == CHOSEN_TASK)
        r_ok = (CHOSEN_RUN is None or int(k[2]) == int(CHOSEN_RUN))
        return s_ok and t_ok and r_ok

    key = next((k for k in keys if _matches(k)), keys[0])
    sub, task, run = key
    print(f"[select] Using session: sub={sub} task={task} run={run}")

    # Resolve fMRI path & load A424 parcels (cached)
    fmri_path = _resolve_fmri_path_for_key(FMRI_ROOT, key)
    if fmri_path is None:
        raise RuntimeError(f"Could not resolve fMRI path for key={key}")

    cache_dir = Path(os.environ.get("ODDBALL_CACHE_DIR", "~/.cache/oddball_eegfmri")).expanduser()
    cache_dir.mkdir(parents=True, exist_ok=True)

    fmri_TV = load_fmri_parcels_cached(fmri_path, str(A424_LABEL_NII), cache_dir=cache_dir).astype(np.float32, copy=False)  # (T,V)
    T, V = fmri_TV.shape
    assert V == V_EXPECTED, f"Expected V={V_EXPECTED}, got V={V}"
    print(f"[load] fMRI parcels: T={T}, V={V}, WINDOW_TR={WINDOW_TR} (from {WINDOW_SEC}s @ TR={TR}s)")

    # Build per-window mask (like training: each 40s window gets its own mask; patch_len == window_len)
    x = torch.from_numpy(fmri_TV).unsqueeze(0).to(device)          # (1,T,V)
    M_full = torch.zeros((1, T, V), dtype=torch.bool, device=device)

    if APPLY_TIME_MASK:
        start = 0
        while start < T:
            end = min(T, start + WINDOW_TR)
            seg_T = end - start
            # patch_len_tr == WINDOW_TR (even for last shorter chunk; helper handles t_end=min(...))
            M_win = make_patch_sparse_mask(
                B=1, T=seg_T, V=V, device=device,
                patch_len_tr=WINDOW_TR,
                r_patch_range=R_PATCH_RANGE,
                r_v_in_patch_range=R_V_IN_PATCH_RANGE,
            )
            M_full[:, start:end, :] = M_win
            start = end

    # Apply mask
    x_masked = apply_zero_mask(x, M_full) if APPLY_TIME_MASK else x

    x_np        = x.squeeze(0).detach().cpu().numpy()
    x_masked_np = x_masked.squeeze(0).detach().cpu().numpy()
    M_np_bool   = M_full.squeeze(0).detach().cpu().numpy().astype(bool)
    M_np        = M_np_bool.astype(np.float32)

    # Prepare output dirs
    out_dir = OUT_DIR / f"sub{sub}_{task}_run{run}"
    _ensure_dirs(out_dir)

    # Heatmaps
    plot_heatmap(x_np,        "GT (T×V)",                         out_dir / "plots" / "heatmap_gt.png",     cmap="viridis")
    plot_heatmap(M_np,        "Mask (1=masked) (T×V)",            out_dir / "plots" / "heatmap_mask.png",   cmap="gray_r")
    plot_heatmap(x_masked_np, "Masked Input (zeros on masked)",   out_dir / "plots" / "heatmap_masked.png", cmap="magma")

    # Timewise masked fraction
    frac_time = M_np.mean(axis=1)  # per-TR fraction masked
    plot_series(np.arange(T), frac_time, "Masked fraction over time (per TR)", out_dir / "plots" / "masked_frac_over_time.png")

    # Per-window (“per-patch”) stats
    n_windows = int(math.ceil(T / float(WINDOW_TR)))
    patch_rows: List[List[float]] = []
    sel_idx, sel_frac = [], []
    for widx in range(n_windows):
        t_start = widx * WINDOW_TR
        t_end   = min((widx + 1) * WINDOW_TR, T)
        M_win   = M_np_bool[t_start:t_end, :]      # (Tw, V)
        # Any masking in the window?
        sel_flag = int(M_win.any())
        # fraction of ROIs that were masked at least once in this window
        roi_mask_any = (M_win.sum(axis=0) > 0)
        k_v = int(roi_mask_any.sum())
        frac_v = float(k_v) / float(V)
        patch_rows.append([widx, t_start, t_end, k_v, frac_v, sel_flag])
        if sel_flag:
            sel_idx.append(widx)
            sel_frac.append(frac_v)

    # Bar plot across windows (here “patch” == 40s window)
    if len(sel_idx) > 0:
        plt.figure(figsize=(max(8, 0.3*len(sel_idx)+5), 4))
        plt.bar(sel_idx, sel_frac)
        plt.ylim(0, 1.0)
        plt.xlabel("Window index (40s / 20 TRs)")
        plt.ylabel("Masked ROI fraction")
        plt.title("Masked ROI fraction per window")
        plt.tight_layout()
        plt.savefig(out_dir / "plots" / "masked_frac_per_patch.png", dpi=150)
        plt.close()

    # CSV
    csv_path = out_dir / "tables" / "mask_patch_stats.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["window_idx", "t_start_TR", "t_end_TR", "k_v_masked_at_least_once", "masked_fraction", "selected_flag"])
        for row in patch_rows:
            w.writerow(row)

    # Save raw arrays for reuse
    np.save(out_dir / "tables" / "fmri_gt_TV.npy", x_np)
    np.save(out_dir / "tables" / "fmri_mask_bool_TV.npy", M_np_bool)
    np.save(out_dir / "tables" / "fmri_masked_TV.npy", x_masked_np)

    # Console summary
    overall_frac = float(M_np.mean())
    touched_time_frac = float((M_np.sum(axis=1) > 0).mean())
    print(f"[summary] sub={sub} task={task} run={run} | T={T} V={V} TR={TR}s | windows={n_windows} (size={WINDOW_TR} TR)")
    print(f"[summary] overall_mask_fraction={overall_frac:.3f} | time_touched_fraction={touched_time_frac:.3f}")
    print(f"[write] plots -> {out_dir / 'plots'}")
    print(f"[write] tables -> {out_dir / 'tables'}")
    print(f"[done] elapsed={time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
