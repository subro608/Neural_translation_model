#!/usr/bin/env python3
r"""
print_zero_rois_run1.py

For each subject, print ROI indices that are exactly zero across ALL windows of:
  - run 1, auditory task
  - run 1, visual task

Assumptions:
- Task is inferred from the fMRI file path (case-insensitive substring match):
    'auditory' in path -> auditory
    'visual'   in path -> visual
- Subject and run are parsed from the path; run must be "1".
- Uses PairedAlignedDataset to window the data; we intersect zero-ROIs across windows
  so only ROIs that are zero for the entire run are reported.

Edit the CONFIG block if paths differ.
"""

from __future__ import annotations
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple, Set
import re

import torch
import numpy as np

# ---------------- CONFIG ----------------
EEG_ROOT       = r"D:\Neuroinformatics_research_2025\Oddball\ds116_eeg"
FMRI_ROOT      = r"D:\Neuroinformatics_research_2025\Oddball\ds000116"
A424_LABEL_NII = r"D:\Neuroinformatics_research_2025\BrainLM\A424_resampled_to_bold.nii.gz"

WINDOW_SEC     = 40
STRIDE_SEC     = None
FMRI_NORM      = "zscore"   # 'zscore' | 'psc' | 'mad' | 'none'
CHANNELS_LIMIT = 34
DEVICE_TENSORS = "cpu"

# ------------- repo imports / paths -------------
THIS_DIR  = Path(__file__).parent
REPO_ROOT = THIS_DIR.parent
CBRAMOD_DIR = REPO_ROOT / "CBraMod"
BRAINLM_DIR = REPO_ROOT / "BrainLM"
sys.path.insert(0, str(THIS_DIR))
sys.path.append(str(CBRAMOD_DIR))
sys.path.append(str(BRAINLM_DIR))

from data_oddball import PairedAlignedDataset, find_bold_files, extract_a424_from_native  # type: ignore

def infer_task_from_path(p: Path) -> Optional[str]:
    s = str(p).lower()
    if "auditory" in s:
        return "auditory"
    if "visual" in s:
        return "visual"
    return None


def main():
    print("[audit] scanning BOLD files under:", FMRI_ROOT)
    bold_files = find_bold_files(Path(FMRI_ROOT))
    if len(bold_files) == 0:
        print("[audit] no BOLD files found. Check FMRI_ROOT path.")
        return

    re_sub = re.compile(r"sub[-_]?0*([0-9]+)", re.IGNORECASE)

    # per-file report and intersections
    subj_to_zero: Dict[str, Set[int]] = {}  # across ALL files for a subject
    srt_to_zero: Dict[Tuple[str, str, str], Set[int]] = {}  # (sub, run, task) across files in that group
    per_file_rows: list[Tuple[str, str, str, str, int, list[int]]] = []  # (sub, run, task, path, count, indices)
    V_global: Optional[int] = None

    for i, bpath in enumerate(sorted(bold_files, key=lambda p: str(p).lower())):
        s = str(bpath)
        msub = re_sub.search(s)
        subj = msub.group(1) if msub else None
        if subj is None:
            continue

        try:
            parcel_ts, _, _, _ = extract_a424_from_native(str(bpath), str(A424_LABEL_NII))  # (T,V)
        except Exception:
            continue
        if parcel_ts.size == 0:
            continue
        V = int(parcel_ts.shape[1])
        if V_global is None:
            V_global = V

        zero_mask = (parcel_ts == 0).all(axis=0)  # (V,)
        zero_list = np.nonzero(zero_mask)[0].tolist()
        zero_idx = set(zero_list)

        # record per-file
        per_file_rows.append((subj, '-', '-', str(bpath), len(zero_list), zero_list))

        if subj not in subj_to_zero:
            subj_to_zero[subj] = set(zero_idx)
        else:
            subj_to_zero[subj] &= zero_idx

        # by (subject, run, task)
        # try to parse run and task from filename
        run_match = re.search(r"run[-_]?0*([0-9]+)", s, flags=re.IGNORECASE)
        run_id = run_match.group(1) if run_match else "?"
        task = infer_task_from_path(bpath) or "?"
        key_srt = (subj, run_id, task)
        if key_srt not in srt_to_zero:
            srt_to_zero[key_srt] = set(zero_idx)
        else:
            srt_to_zero[key_srt] &= zero_idx

        if (i + 1) % 10 == 0:
            print(f"[audit] processed {i+1}/{len(bold_files)} files... current subject={subj}")

    if not subj_to_zero:
        print("[audit] no subjects aggregated; nothing to report.")
        return

    # Per-file report (compact)
    print("\n=== Per-file zero-ROIs (count) ===")
    for subj, _, _, path, cnt, _idx in sorted(per_file_rows, key=lambda r: (int(r[0]) if r[0].isdigit() else 9999, r[3])):
        print(f"Subject {subj}: {cnt} zero-ROIs | {path}")

    # Per (subject, run, task) intersection
    print("\n=== ROIs exactly zero across ALL files in (subject, run, task) ===")
    for (subj, run_id, task), zs in sorted(srt_to_zero.items(), key=lambda k: (int(k[0][0]) if k[0][0].isdigit() else 9999, k[0][1], k[0][2])):
        zsorted = sorted(zs)
        print(f"Subject {subj} | run {run_id} | task {task}: {len(zsorted)} zero-ROIs")
        if zsorted:
            print(f"  {zsorted}")

    # Per-subject across all files
    print("\n=== ROIs exactly zero across ALL runs/tasks per subject ===")
    for subj in sorted(subj_to_zero.keys(), key=lambda x: int(x)):
        zs = sorted(subj_to_zero[subj])
        print(f"Subject {subj}: {len(zs)} zero-ROIs")
        if zs:
            print(f"  {zs}")
    print("\n[audit] done.")

if __name__ == "__main__":
    main()
