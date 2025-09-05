#!/usr/bin/env python3
"""
Variance per ROI with heatmaps (no z-scoring, no normalization).

Outputs (default ./debug_out/roi_variance):
  - var_sub<sub>_task<task>_run<run>.png                      (bar plot)
  - var_heatmap_full_sub<sub>_task<task>_run<run>.png         (1×V heatmap)
  - var_heatmap_patches_abs_sub<sub>_task<task>_run<run>.png  (P×V heatmap, absolute variance)

Usage:
  python plot_fmri_roi_variance.py
  python plot_fmri_roi_variance.py --sub 1 --task task001 --run 1 --patch 20
"""

from __future__ import annotations
import argparse, re, glob, sys, os, math
from pathlib import Path
from typing import Optional, List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------- Paths (edit as needed) -----------------------
FMRI_ROOT       = Path(r"D:\Neuroinformatics_research_2025\Oddball\ds000116")
A424_LABEL_NII  = Path(r"D:\Neuroinformatics_research_2025\BrainLM\A424_resampled_to_bold.nii.gz")
OUT_DIR         = Path("./debug_out/roi_variance")

# ----------------------- Repo helpers -----------------------
THIS_DIR    = Path(__file__).parent
REPO_ROOT   = THIS_DIR.parent
CBRAMOD_DIR = REPO_ROOT / "CBraMod"
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(CBRAMOD_DIR))
sys.path.insert(0, str(THIS_DIR / "CBraMod"))

# Data helper
from data_oddball import (  # type: ignore
    load_fmri_parcels_cached,
)

# File discovery for ds000116
_RE_SUB  = re.compile(r"sub[-_]?0*([0-9]+)", re.I)
_RE_TASK = re.compile(r"task[-_]?([A-Za-z0-9]+)", re.I)
_RE_RUN  = re.compile(r"run[-_]?0*([0-9]+)", re.I)

def _parse_key_from_path(path: Path):
    s = str(path)
    sub  = _RE_SUB.search(s)
    task = _RE_TASK.search(s)
    run  = _RE_RUN.search(s)
    sub_id  = str(int(sub.group(1))) if sub else None
    task_id = (task.group(1).lower() if task else None)
    run_id  = str(int(run.group(1))) if run else None
    return sub_id, task_id, run_id

def _canonical_task_token(raw_task: Optional[str], path: Path) -> Optional[str]:
    s = path.name.lower() if raw_task is None else raw_task.lower()
    whole = str(path).lower()
    if 'auditory' in s or 'auditory' in whole: return 'task001'
    if 'visual'   in s or 'visual'   in whole: return 'task002'
    if s.startswith('task') and len(s) > 4 and s[4:].isdigit():
        return f"task{int(s[4:]):03d}"
    if s.isdigit(): return f"task{int(s):03d}"
    return f"task_{s}" if s else None

def _find_bold_files(root: Path) -> List[Path]:
    pats = [str(root / "**" / "*bold.nii.gz"), str(root / "**" / "*bold.nii")]
    out: List[Path] = []
    for pat in pats: out.extend([Path(p) for p in glob.glob(pat, recursive=True)])
    return sorted(set(out), key=lambda p: str(p).lower())

def _resolve_fmri_path_for_key(fmri_root: Path, key: tuple[str, str, str]) -> Optional[Path]:
    sub_k, task_k, run_k = key
    for p in _find_bold_files(fmri_root):
        s, t_raw, r = _parse_key_from_path(p)
        if s is None or r is None: continue
        t = _canonical_task_token(t_raw, p)
        if (s, t, r) == (sub_k, task_k, run_k): return p
    return None

def _ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)

# ----------------------- Plot helpers -----------------------
def _plot_variance_bar(roi_var: np.ndarray, out_png: Path, title: str):
    V = roi_var.shape[0]
    x = np.arange(V)
    fig_h = 4.5 if V <= 600 else 6.5
    plt.figure(figsize=(14, fig_h))
    plt.bar(x, roi_var, width=0.8)
    plt.xlabel("ROI index"); plt.ylabel("Variance (full session)")
    plt.title(title)
    plt.tight_layout(); plt.savefig(out_png, dpi=130)
    plt.close(); print(f"[plot] saved {out_png}")

def _plot_heatmap(mat: np.ndarray, out_png: Path, title: str,
                  xlabel: str, ylabel: str):
    # single-row heatmap gets a bit more height to be visible
    H, W = mat.shape
    fig_h = 2.8 if H == 1 else max(4.0, min(0.012*H + 3.0, 10.0))
    fig_w = max(10.0, min(0.02*W + 6.0, 18.0))
    plt.figure(figsize=(fig_w, fig_h))
    im = plt.imshow(mat, aspect="auto", interpolation="nearest")  # no vmin/vmax clipping
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.title(title)
    plt.xlabel(xlabel); plt.ylabel(ylabel)
    plt.tight_layout(); plt.savefig(out_png, dpi=130)
    plt.close(); print(f"[plot] saved {out_png}")

# ----------------------- CLI -----------------------
def main():
    ap = argparse.ArgumentParser(description="Variance per ROI: bar + heatmaps (no z-scoring).")
    ap.add_argument("--sub", type=int, default=1)
    ap.add_argument("--task", type=str, default="task001")
    ap.add_argument("--run", type=int, default=1)
    ap.add_argument("--patch", type=int, default=20, help="Patch size in TR for patch-wise variance heatmap.")
    ap.add_argument("--out_dir", type=str, default=str(OUT_DIR))
    args = ap.parse_args()

    key = (str(args.sub), args.task, str(args.run))
    fmri_path = _resolve_fmri_path_for_key(FMRI_ROOT, key)
    if fmri_path is None:
        raise RuntimeError(f"Could not resolve fMRI path for key={key} under {FMRI_ROOT}")

    cache_dir = Path(os.environ.get("ODDBALL_CACHE_DIR", "~/.cache/oddball_eegfmri")).expanduser()
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Load entire session: X_TV (T_full, V) — used as-is (no extra normalization)
    X_TV = load_fmri_parcels_cached(fmri_path, str(A424_LABEL_NII), cache_dir=cache_dir).astype(np.float32, copy=False)
    T_full, V = X_TV.shape
    print(f"[session] sub={args.sub} task={args.task} run={args.run} | fMRI (T={T_full}, V={V})")

    out_dir = Path(args.out_dir)
    _ensure_dir(out_dir)

    # ---------- Full-session variance ----------
    roi_var = np.var(X_TV, axis=0)  # (V,)
    bar_png  = out_dir / f"var_sub{args.sub}_task{args.task}_run{args.run}.png"
    full_png = out_dir / f"var_heatmap_full_sub{args.sub}_task{args.task}_run{args.run}.png"

    _plot_variance_bar(roi_var, bar_png,
        title=f"Per-ROI variance — sub={args.sub}, {args.task}, run={args.run}")

    _plot_heatmap(roi_var[None, :], full_png,
        title="Full-session ROI variance (1×V heatmap)",
        xlabel="ROI index", ylabel="")

    # ---------- Patch-wise absolute variance (P×V) ----------
    patch = max(1, int(args.patch))
    n_patches = int(math.ceil(T_full / float(patch)))
    var_patches = np.zeros((n_patches, V), dtype=np.float64)
    for pidx in range(n_patches):
        t0, t1 = pidx * patch, min(T_full, (pidx + 1) * patch)
        var_patches[pidx, :] = np.var(X_TV[t0:t1, :], axis=0)

    patches_abs_png = out_dir / f"var_heatmap_patches_abs_sub{args.sub}_task{args.task}_run{args.run}.png"
    _plot_heatmap(
        var_patches, patches_abs_png,
        title=f"Patch-wise ROI variance (absolute) — patch={patch} TR",
        xlabel="ROI index", ylabel="Patch index (0 = first)"
    )

    print("[done] variance plots written to:", out_dir.resolve())

if __name__ == "__main__":
    main()
