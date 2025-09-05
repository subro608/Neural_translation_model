#!/usr/bin/env python3
"""
Print A424 ROI xyz coordinates.

- Searches common locations for A424_Coordinates.dat
- Optionally normalizes coords to [-1, 1] (same as in your model code)
- Prints shape, per-axis min/max, and the first N rows

Usage:
  python print_a424_xyz.py --normalize --head 10
  python print_a424_xyz.py --path "C:/.../A424_Coordinates.dat" --no-normalize
"""

from __future__ import annotations
import argparse
from pathlib import Path
import sys
import numpy as np

# Optional: use your project loader if available, else fall back to a simple np.loadtxt
def _load_coords(path: Path) -> np.ndarray:
    try:
        # Try your helper if it's importable
        from data_oddball import load_a424_coords  # type: ignore
        return load_a424_coords(path).astype(np.float32)
    except Exception:
        # Assume whitespace-delimited Nx3 text
        arr = np.loadtxt(str(path), dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise RuntimeError(f"Expected (N,3) coords, got {arr.shape}")
        return arr

def _find_default_path(script_dir: Path) -> Path | None:
    repo_root = script_dir.parent
    candidates = [
        script_dir / "BrainLM" / "resources" / "atlases" / "A424_Coordinates.dat",
        script_dir / "resources" / "atlases" / "A424_Coordinates.dat",
        repo_root / "BrainLM" / "toolkit" / "atlases" / "A424_Coordinates.dat",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None

def main():
    ap = argparse.ArgumentParser(description="Print A424 ROI xyz coordinates.")
    ap.add_argument("--path", type=str, default=None, help="Path to A424_Coordinates.dat")
    ap.add_argument("--normalize", dest="normalize", action="store_true", help="Normalize to [-1,1] via max |val|")
    ap.add_argument("--no-normalize", dest="normalize", action="store_false", help="Use raw coordinates")
    ap.add_argument("--head", type=int, default=10, help="How many rows to print")
    ap.set_defaults(normalize=True)
    args = ap.parse_args()

    script_dir = Path(__file__).parent
    path = Path(args.path) if args.path else _find_default_path(script_dir)
    if not path or not path.exists():
        print("ERROR: A424_Coordinates.dat not found. Provide --path to the file.", file=sys.stderr)
        sys.exit(1)

    coords = _load_coords(path)  # (N,3)
    if args.normalize:
        denom = float(np.max(np.abs(coords))) or 1.0
        coords = coords / denom

    N = coords.shape[0]
    print(f"[file] {path}")
    print(f"[shape] (N,3) = {coords.shape}  (expected N≈424)")
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    means = coords.mean(axis=0)
    print(f"[stats] x: min={mins[0]:.4f} max={maxs[0]:.4f} mean={means[0]:.4f}")
    print(f"        y: min={mins[1]:.4f} max={maxs[1]:.4f} mean={means[1]:.4f}")
    print(f"        z: min={mins[2]:.4f} max={maxs[2]:.4f} mean={means[2]:.4f}")
    print(f"[normalize] {'ON' if args.normalize else 'OFF'}")

    head = max(1, min(args.head, N))
    print(f"\n[first {head} rows (roi_idx, x, y, z)]:")
    for i in range(head):
        x, y, z = coords[i]
        print(f"{i:03d}\t{x:+.6f}\t{y:+.6f}\t{z:+.6f}")

if __name__ == "__main__":
    main()
