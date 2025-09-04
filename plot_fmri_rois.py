import argparse
import os
from math import ceil
from typing import Optional, Tuple, Sequence

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Use your project helpers for exact fMRI format (T, 424) A424 parcels
try:
    from data_oddball import load_fmri_parcels_cached  # (bold_path, atlas_path, cache_dir) -> np.ndarray (T,424)
except Exception:
    load_fmri_parcels_cached = None  # Fallback handled below


def _default_cache_dir() -> Path:
    return Path(os.environ.get("ODDBALL_CACHE_DIR", "~/.cache/oddball_eegfmri")).expanduser()


def load_fmri_from_npz(npz_path: str) -> np.ndarray:
    with np.load(npz_path) as npz:
        if "arr" in npz:
            arr = npz["arr"]
        else:
            # Best effort: take the first array
            keys = [k for k in npz.files]
            if not keys:
                raise ValueError(f"No arrays found in npz: {npz_path}")
            arr = npz[keys[0]]
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D (T,V) in npz, got shape {arr.shape}")
    return arr.astype(np.float32, copy=False)


def load_fmri_from_bold(bold_path: str, a424_label_nii: str, cache_dir: Optional[str]) -> np.ndarray:
    if load_fmri_parcels_cached is None:
        raise ImportError("Could not import data_oddball.load_fmri_parcels_cached")
    cache_p = Path(cache_dir) if cache_dir is not None else _default_cache_dir()
    arr = load_fmri_parcels_cached(Path(bold_path), str(a424_label_nii), cache_dir=cache_p)
    if arr.ndim != 2 or arr.shape[1] != 424:
        raise ValueError(f"Expected (T,424) after extraction, got {arr.shape}")
    return arr.astype(np.float32, copy=False)


def ensure_time_by_roi(arr: np.ndarray) -> np.ndarray:
    # Keep (T, 424). If first dim is 424 and second is not, transpose.
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {arr.shape}")
    T, V = arr.shape
    if V == 424 and T != 424:
        return arr
    if T == 424 and V != 424:
        return arr.T
    # Ambiguous or already fine; return as-is
    return arr


def normalize_fmri(arr_TV: np.ndarray, mode: str = "none", eps: float = 1e-8) -> np.ndarray:
    if mode == "none":
        return arr_TV
    if mode == "zscore":
        mean = arr_TV.mean(axis=0, keepdims=True)
        std = arr_TV.std(axis=0, keepdims=True)
        std = np.where(std < eps, 1.0, std)
        return (arr_TV - mean) / std
    if mode == "psc":
        mean = arr_TV.mean(axis=0, keepdims=True)
        return 100.0 * (arr_TV - mean) / (mean + eps)
    if mode == "mad":
        median = np.median(arr_TV, axis=0, keepdims=True)
        mad = np.median(np.abs(arr_TV - median), axis=0, keepdims=True)
        scaled = mad * 1.4826
        return (arr_TV - median) / (scaled + eps)
    raise ValueError(f"Unsupported normalization mode: {mode}")


def plot_rois(
    time_by_roi: np.ndarray,
    rois: Optional[Sequence[int]] = None,
    num_rois: int = 30,
    start: Optional[int] = None,
    end: Optional[int] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 10),
    ncols: int = 5,
    line_width: float = 0.9,
) -> plt.Figure:
    """Create a grid plot of selected ROI time series (default: first N)."""
    t_by_r = time_by_roi
    t_start = 0 if start is None else max(0, start)
    t_end = t_by_r.shape[0] if end is None else min(t_by_r.shape[0], end)
    t_by_r = t_by_r[t_start:t_end]

    if rois is None:
        n = min(num_rois, t_by_r.shape[1])
        rois = list(range(n))
    else:
        rois = [int(r) for r in rois if 0 <= int(r) < t_by_r.shape[1]]
        if not rois:
            raise ValueError("No valid ROI indices to plot.")
        n = len(rois)

    nrows = ceil(n / ncols)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, sharex=True)
    axes = np.atleast_1d(axes).reshape(nrows, ncols)

    time_axis = np.arange(t_by_r.shape[0])

    for plot_idx, roi_idx in enumerate(rois):
        r = plot_idx // ncols
        c = plot_idx % ncols
        ax = axes[r, c]
        ax.plot(time_axis, t_by_r[:, roi_idx], linewidth=line_width)
        ax.set_title(f"ROI {roi_idx}", fontsize=9)
        ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)

    # Hide any unused subplots
    for idx in range(n, nrows * ncols):
        r = idx // ncols
        c = idx % ncols
        axes[r, c].axis("off")

    for r in range(nrows):
        axes[r, 0].set_ylabel("Signal")
    for c in range(ncols):
        axes[-1, c].set_xlabel("Time (samples)")

    if title:
        fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Plot fMRI A424 parcel time series (T,424) using your exact format. "
            "Input can be a cached fmri_*.npz (arr=(T,424)) or a BOLD NIfTI with the A424 atlas."
        )
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--npz", type=str, help="Path to cached fmri_*.npz with arr=(T,424)")
    src.add_argument("--bold", type=str, help="Path to BOLD NIfTI (.nii or .nii.gz)")
    parser.add_argument("--a424_label_nii", type=str, default=None,
                        help="Required if using --bold: path to A424 label NIfTI aligned to BOLD space")
    parser.add_argument("--cache_dir", type=str, default=None,
                        help="Optional cache dir for A424 parcel extraction (defaults to ODDball cache path)")
    parser.add_argument(
        "--num-rois",
        type=int,
        default=30,
        help="Number of ROIs to plot (default: 30)",
    )
    parser.add_argument(
        "--rois",
        type=str,
        default=None,
        help="Comma-separated ROI indices to plot (overrides --num-rois)",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=None,
        help="Start time index (inclusive). Default: 0",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="End time index (exclusive). Default: full length",
    )
    parser.add_argument(
        "--norm",
        type=str,
        default="none",
        choices=["none", "zscore", "psc", "mad"],
        help="Normalization across time per ROI (matches pipeline options)",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Optional figure title",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Optional output image path (e.g., plot.png). If omitted, shows the plot",
    )
    args = parser.parse_args()

    if args.npz is not None:
        arr = load_fmri_from_npz(args.npz)
    else:
        if not args.a424_label_nii:
            raise SystemExit("--a424_label_nii is required when using --bold")
        arr = load_fmri_from_bold(args.bold, args.a424_label_nii, args.cache_dir)
    arr = ensure_time_by_roi(arr)
    arr = normalize_fmri(arr, mode=args.norm)

    roi_list = None
    if args.rois:
        roi_list = [int(s) for s in args.rois.replace(',', ' ').split() if s.strip()]

    fig = plot_rois(
        arr,
        rois=roi_list,
        num_rois=args.num_rois,
        start=args.start,
        end=args.end,
        title=args.title,
    )

    if args.save is not None:
        fig.savefig(args.save, dpi=200)
        print(f"Saved figure to: {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()



