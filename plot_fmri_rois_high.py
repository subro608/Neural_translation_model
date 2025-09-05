#!/usr/bin/env python3
"""
Find "stripe" ROIs from patch-wise ROI×ROI correlations, then plot their full-session time series.

Heuristic:
- Split time into 20-TR patches.
- For each patch:
    * compute ROI×ROI corr over time (TR)
    * score each ROI by mean(abs(corr_i,:)) and corr with global signal
    * flag low-variance ROIs within the patch
    * collect candidates = top-N by mean|r| ∪ top-N by corr_to_global ∪ low-variance
- Aggregate candidates across patches; rank ROIs by occurrence count, break ties by avg mean|r|.
- Plot full-session time series for the top-K ROIs (stacked), with 20-TR gridlines.

Outputs (default ./debug_out/stripe_rois_timeseries):
  - ts_sub<sub>_task<task>_run<run>_top<top_k>.png
  - stripe_roi_summary.csv  (roi, count, avg_mean_abs_corr, avg_corr_global)

Usage:
  python find_stripe_rois_and_plot_timeseries.py
  python find_stripe_rois_and_plot_timeseries.py --sub 1 --task task001 --run 1 --top_k 12 --per_patch_top 10
"""

from __future__ import annotations
import argparse, re, glob, csv, sys, math, os
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------- Paths (edit as needed) -----------------------
FMRI_ROOT       = Path(r"D:\Neuroinformatics_research_2025\Oddball\ds000116")
A424_LABEL_NII  = Path(r"D:\Neuroinformatics_research_2025\BrainLM\A424_resampled_to_bold.nii.gz")
OUT_DIR         = Path("./debug_out/stripe_rois_timeseries")
PATCH_SIZE_TR   = 20
TR_SECONDS      = 2.0

# ----------------------- Repo helpers -----------------------
THIS_DIR   = Path(__file__).parent
REPO_ROOT  = THIS_DIR.parent
CBRAMOD_DIR = REPO_ROOT / "CBraMod"
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(CBRAMOD_DIR))
sys.path.insert(0, str(THIS_DIR / "CBraMod"))

# We only need data helpers
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

def _resolve_fmri_path_for_key(fmri_root: Path, key: Tuple[str, str, str]) -> Optional[Path]:
    sub_k, task_k, run_k = key
    for p in _find_bold_files(fmri_root):
        s, t_raw, r = _parse_key_from_path(p)
        if s is None or r is None: continue
        t = _canonical_task_token(t_raw, p)
        if (s, t, r) == (sub_k, task_k, run_k): return p
    return None

def _ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)

def _add_patch_vlines(ax, T_full:int, patch:int=20):
    for t in range(0, T_full+1, patch):
        ax.axvline(t, color="k", alpha=0.25, lw=0.8)

# ----------------------- Core logic -----------------------
def _per_patch_metrics(Y: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Y: (T_patch, V). Returns dict with:
      - 'mean_abs_corr': (V,)
      - 'corr_to_global': (V,)
      - 'std': (V,)
    """
    T, V = Y.shape
    # Corr matrix over ROIs
    C = np.corrcoef(Y.T)                       # (V,V)
    # guard NaNs from zero-variance ROIs
    C = np.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(C, 0.0)
    mean_abs_corr = np.mean(np.abs(C), axis=1)

    # Correlation to global signal
    g = Y.mean(axis=1)                         # (T,)
    g = (g - g.mean()) / (g.std() + 1e-8)
    Yz = (Y - Y.mean(axis=0)) / (Y.std(axis=0) + 1e-8)
    corr_to_global = (Yz * g[:, None]).mean(axis=0)  # cosine w.r.t. zscored g; ~ Pearson r

    # Per-ROI std in this patch
    std = Y.std(axis=0)

    return {"mean_abs_corr": mean_abs_corr, "corr_to_global": corr_to_global, "std": std}

def _select_candidates_for_patch(metrics: Dict[str, np.ndarray],
                                 per_patch_top:int,
                                 std_thr:float) -> List[int]:
    V = metrics["mean_abs_corr"].shape[0]
    # top by mean|r|
    idx_mean = np.argsort(metrics["mean_abs_corr"])[::-1][:min(per_patch_top, V)]
    # top by corr_to_global
    idx_g = np.argsort(metrics["corr_to_global"])[::-1][:min(per_patch_top, V)]
    # low variance
    idx_lowvar = np.where(metrics["std"] < std_thr)[0]
    return sorted(set(idx_mean.tolist() + idx_g.tolist() + idx_lowvar.tolist()))

def _rank_stripe_rois_over_patches(X_TV: np.ndarray,
                                   patch:int,
                                   per_patch_top:int,
                                   std_thr:float,
                                   top_k:int):
    """
    Returns list of (roi, count, avg_mean_abs_corr, avg_corr_global), sorted for plotting.
    """
    T_full, V = X_TV.shape
    n_patches = int(math.ceil(T_full / float(patch)))
    counts = np.zeros(V, dtype=int)
    sum_mean_abs_corr = np.zeros(V, dtype=float)
    sum_corr_global   = np.zeros(V, dtype=float)

    for pidx in range(n_patches):
        t0 = pidx * patch
        t1 = min(T_full, (pidx+1) * patch)
        Y = X_TV[t0:t1, :]  # (T_p, V)

        m = _per_patch_metrics(Y)
        cand = _select_candidates_for_patch(m, per_patch_top=per_patch_top, std_thr=std_thr)

        # accumulate stats only for candidates to emphasize striped behavior
        counts[cand] += 1
        sum_mean_abs_corr[cand] += m["mean_abs_corr"][cand]
        sum_corr_global[cand]   += m["corr_to_global"][cand]

    # Build table and sort
    rows = []
    for roi in range(V):
        if counts[roi] == 0: continue
        rows.append((
            roi,
            int(counts[roi]),
            float(sum_mean_abs_corr[roi] / max(1, counts[roi])),
            float(sum_corr_global[roi] / max(1, counts[roi]))
        ))
    rows.sort(key=lambda r: (r[1], r[2]), reverse=True)
    return rows[:top_k], n_patches

def _plot_timeseries(X_TV: np.ndarray, rois: List[int], out_png: Path, title:str):
    T_full, V = X_TV.shape
    n = len(rois)
    if n == 0:
        print("[warn] no ROIs selected; nothing to plot.")
        return
    fig_h = max(2.0 * n, 3.0)
    fig, axes = plt.subplots(nrows=n, ncols=1, figsize=(12, fig_h), sharex=True)
    if n == 1: axes = [axes]
    t = np.arange(T_full)
    for ax, r in zip(axes, rois):
        y = X_TV[:, r]
        ax.plot(t, y, label=f"ROI {r}")
        _add_patch_vlines(ax, T_full, patch=PATCH_SIZE_TR)
        ax.set_ylabel("z-score")
        ax.legend(loc="upper right", fontsize=9)
    axes[-1].set_xlabel("TR")
    fig.suptitle(title, y=0.995)
    fig.tight_layout()
    plt.savefig(out_png, dpi=130)
    plt.close(fig)
    print(f"[plot] saved {out_png}")

# ----------------------- CLI -----------------------
def main():
    ap = argparse.ArgumentParser(description="Find stripe ROIs and plot their full-session time series.")
    ap.add_argument("--sub", type=int, default=1)
    ap.add_argument("--task", type=str, default="task001")
    ap.add_argument("--run", type=int, default=1)
    ap.add_argument("--per_patch_top", type=int, default=10, help="Top-N per patch by mean|r| and by corr-to-global.")
    ap.add_argument("--std_thr", type=float, default=1e-6, help="Std threshold to flag low-variance ROIs in a patch.")
    ap.add_argument("--top_k", type=int, default=12, help="How many ROIs to plot (ranked by occurrence).")
    ap.add_argument("--out_dir", type=str, default=str(OUT_DIR))
    args = ap.parse_args()

    key = (str(args.sub), args.task, str(args.run))
    fmri_path = _resolve_fmri_path_for_key(FMRI_ROOT, key)
    if fmri_path is None:
        raise RuntimeError(f"Could not resolve fMRI path for key={key} under {FMRI_ROOT}")

    cache_dir = Path(os.environ.get("ODDBALL_CACHE_DIR", "~/.cache/oddball_eegfmri")).expanduser()
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Load entire session: X_TV (T_full, V)
    X_TV = load_fmri_parcels_cached(fmri_path, str(A424_LABEL_NII), cache_dir=cache_dir).astype(np.float32, copy=False)
    T_full, V = X_TV.shape
    print(f"[session] sub={args.sub} task={args.task} run={args.run} | fMRI (T={T_full}, V={V})")

    # Rank stripe ROIs
    top_rows, n_patches = _rank_stripe_rois_over_patches(
        X_TV=X_TV,
        patch=PATCH_SIZE_TR,
        per_patch_top=args.per_patch_top,
        std_thr=float(args.std_thr),
        top_k=args.top_k
    )
    rois = [roi for roi, _, _, _ in top_rows]
    print(f"[select] top {len(rois)} stripe-like ROIs (by occurrence across {n_patches} patches): {rois}")

    out_dir = Path(args.out_dir)
    _ensure_dir(out_dir)

    # Write summary CSV
    csv_path = out_dir / "stripe_roi_summary.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["roi", "count_patches", "avg_mean_abs_corr", "avg_corr_global"])
        for roi, cnt, mabs, cg in top_rows:
            w.writerow([roi, cnt, f"{mabs:.6f}", f"{cg:.6f}"])
    print(f"[write] {csv_path}")

    # Plot full-session time series for the selected ROIs
    out_png = out_dir / f"ts_sub{args.sub}_task{args.task}_run{args.run}_top{len(rois)}.png"
    _plot_timeseries(
        X_TV=X_TV,
        rois=rois,
        out_png=out_png,
        title=f"Stripe-ROI time series — sub={args.sub}, {args.task}, run={args.run}"
    )

if __name__ == "__main__":
    main()
