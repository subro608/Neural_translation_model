#!/usr/bin/env python3
"""
Plot input temporal fMRI to the encoder (padded to patch=20 TR) for a full session.

Session fixed to: sub=1, task=task001, run=1
- Loads ds000116 A424 parcels (T,V)
- Pads time to multiple of 20 TR -> (Tp,V)
- Plots time series for specified ROI indices over the *entire padded* session
- Shades padded region (Tp > T_full), draws vertical lines every 20 TR

Usage examples:
  python plot_input_timeseries_for_encoder.py --rois 0 10 42
  python plot_input_timeseries_for_encoder.py --first_n 6

Outputs:
  ./debug_out/roi_timeseries_input_encoder/ts_sub1_task001_run1_rois-<...>.png
"""

from __future__ import annotations
import argparse, re, glob, math, sys, platform
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------- Paths (edit to your setup) -----------------------
EEG_ROOT          = Path(r"D:\Neuroinformatics_research_2025\Oddball\ds116_eeg")  # not used here but kept for parity
FMRI_ROOT         = Path(r"D:\Neuroinformatics_research_2025\Oddball\ds000116")
A424_LABEL_NII    = Path(r"D:\Neuroinformatics_research_2025\BrainLM\A424_resampled_to_bold.nii.gz")

# Fixed session as requested
CHOSEN_SUBJECT = 1
CHOSEN_TASK    = "task001"
CHOSEN_RUN     = 1

PATCH_SIZE_TR = 20
OUT_DIR = Path("./debug_out/roi_timeseries_input_encoder")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ----------------------- Repo pathing (match your project) -----------------------
THIS_DIR    = Path(__file__).parent
REPO_ROOT   = THIS_DIR.parent
CBRAMOD_DIR = REPO_ROOT / "CBraMod"
BRAINLM_DIR = REPO_ROOT / "BrainLM"
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(CBRAMOD_DIR))
sys.path.insert(0, str(THIS_DIR / "CBraMod"))

# Make brainlm_mae importable (for pad helper)
for p in [BRAINLM_DIR, THIS_DIR / "BrainLM", Path(os.environ.get("BRAINLM_DIR", "")),
          Path(os.environ.get("BRAINLM_DIR", "")) / "brainlm_mae"]:
    if not p: continue
    p = Path(p)
    if (p / "brainlm_mae").is_dir():
        sys.path.insert(0, str(p)); break
    if p.name == "brainlm_mae" and p.is_dir():
        sys.path.insert(0, str(p.parent)); break

# Data helpers from your repo
from data_oddball import (  # type: ignore
    load_fmri_parcels_cached,
    pad_timepoints_for_brainlm_torch,
)

# ----------------------- Helpers -----------------------
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
    for pat in pats:
        out.extend([Path(p) for p in glob.glob(pat, recursive=True)])
    return sorted(set(out), key=lambda p: str(p).lower())

def _resolve_fmri_path_for_key(fmri_root: Path, key: Tuple[str, str, str]) -> Optional[Path]:
    sub_k, task_k, run_k = key
    for p in _find_bold_files(fmri_root):
        s, t_raw, r = _parse_key_from_path(p)
        if s is None or r is None: continue
        t = _canonical_task_token(t_raw, p)
        if (s, t, r) == (sub_k, task_k, run_k): return p
    return None

def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def _add_patch_vlines(ax, Tp: int, patch: int = 20, **kwargs):
    """Draw vertical lines at every patch boundary (0, 20, 40, ...)."""
    for t in range(0, Tp + 1, patch):
        ax.axvline(t, alpha=0.25, linewidth=0.8, **kwargs)

# ----------------------- Main -----------------------
def main():
    ap = argparse.ArgumentParser(description="Plot encoder input time series (padded) for chosen ROIs.")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--rois", type=int, nargs="*", default=None, help="Explicit ROI indices (e.g., 0 10 42)")
    g.add_argument("--first_n", type=int, default=6, help="If --rois not provided, plot first N ROIs (default 6)")
    args = ap.parse_args()

    # Resolve this specific session
    key = (str(CHOSEN_SUBJECT), CHOSEN_TASK, str(CHOSEN_RUN))
    fmri_path = _resolve_fmri_path_for_key(FMRI_ROOT, key)
    if fmri_path is None:
        raise RuntimeError(f"Could not resolve fMRI path for key={key} under {FMRI_ROOT}")

    cache_dir = Path(os.environ.get("ODDBALL_CACHE_DIR", "~/.cache/oddball_eegfmri")).expanduser()
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Load full session parcels: (T_full, V)
    fmri_TV = load_fmri_parcels_cached(fmri_path, str(A424_LABEL_NII), cache_dir=cache_dir).astype(np.float32, copy=False)
    T_full, V = fmri_TV.shape
    print(f"[env] torch={torch.__version__} | device={DEVICE} | python={sys.version.split()[0]} | {platform.platform()}")
    print(f"[session] sub={CHOSEN_SUBJECT} task={CHOSEN_TASK} run={CHOSEN_RUN} | fMRI (T={T_full}, V={V})")

    # Build encoder input by padding time to multiple of 20
    x_BTV = torch.from_numpy(fmri_TV).unsqueeze(0).to(DEVICE)                  # (1, T_full, V)
    x_pad = pad_timepoints_for_brainlm_torch(x_BTV, patch_size=PATCH_SIZE_TR)   # (1, Tp, V)
    Tp = int(x_pad.shape[1])
    print(f"[prep] padded timepoints Tp={Tp} (was T_full={T_full}); patch={PATCH_SIZE_TR} TR")

    # Select ROIs
    if args.rois is not None and len(args.rois) > 0:
        rois = [int(r) for r in args.rois if 0 <= int(r) < V]
        if not rois:
            raise ValueError("No valid ROI indices given (0-based, must be < V).")
    else:
        n = max(1, int(args.first_n))
        rois = list(range(min(n, V)))
    print(f"[plot] ROIs: {rois}")

    # Prepare output
    _ensure_dir(OUT_DIR)
    roitag = "-".join(map(str, rois)) if len(rois) <= 16 else f"{len(rois)}rois"
    out_png = OUT_DIR / f"ts_sub{CHOSEN_SUBJECT}_task{CHOSEN_TASK}_run{CHOSEN_RUN}_rois-{roitag}.png"

    # Plot
    nrows = len(rois)
    fig_h = max(2.0 * nrows, 3.0)
    fig, axes = plt.subplots(nrows=nrows, ncols=1, figsize=(12, fig_h), sharex=True)
    if nrows == 1:
        axes = [axes]

    t_axis = np.arange(Tp)  # padded axis
    for ax, roi in zip(axes, rois):
        y = x_pad[0, :, roi].detach().cpu().numpy()  # (Tp,)
        ax.plot(t_axis, y, label=f"ROI {roi}")
        # Shade padded region beyond T_full
        if Tp > T_full:
            ax.axvspan(T_full-0.5, Tp-0.5, color="gray", alpha=0.15, label="padded tail")
        _add_patch_vlines(ax, Tp, patch=PATCH_SIZE_TR, color="k")
        ax.set_ylabel("z-score")
        ax.legend(loc="upper right", fontsize=9)

    axes[-1].set_xlabel("TR (padded)")
    fig.suptitle(f"Encoder Input (padded) — sub={CHOSEN_SUBJECT}, {CHOSEN_TASK}, run={CHOSEN_RUN}", y=0.995)
    fig.tight_layout()
    plt.savefig(out_png, dpi=130)
    plt.close(fig)
    print(f"[done] saved {out_png.resolve()}")

if __name__ == "__main__":
    main()
