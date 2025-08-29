#!/usr/bin/env python3
# compute_oddball_hours_noargs.py
# Self-contained: set your paths and options in the CONSTANTS below (no CLI args).

from __future__ import annotations

import csv
import glob
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

# ========= EDIT THESE CONSTANTS =========
EEG_ROOT      = r"D:\Neuroinformatics_research_2025\Oddball\ds116_eeg"
FMRI_ROOT     = r"D:\Neuroinformatics_research_2025\Oddball\ds000116"
ORIGINAL_FS   = 1000      # EEG sampling rate (Hz)
TR_SECONDS    = 2.0       # fMRI TR in seconds
SUBJECTS      = None      # e.g., ["7","11","12"] or None for all paired subjects
CSV_OUT_PATH  = r"D:\tmp\oddball_stats.csv"  # set to None to skip CSV
# =======================================

# Dependencies:
#   pip install scipy nibabel

try:
    import scipy.io as sio
except Exception as e:
    print("[error] scipy is required. Install with: pip install scipy")
    raise

try:
    import nibabel as nib
except Exception as e:
    print("[error] nibabel is required. Install with: pip install nibabel")
    raise


# ------------ Path parsing helpers (mirrors your pipeline) ------------
_RE_SUB_GENERIC  = re.compile(r"sub[-_]?0*([0-9]+)", re.IGNORECASE)
_RE_TASK_GENERIC = re.compile(r"task[-_]?([A-Za-z0-9]+)", re.IGNORECASE)
_RE_RUN_GENERIC  = re.compile(r"run[-_]?0*([0-9]+)", re.IGNORECASE)

def _parse_key_from_path(path: Path) -> Tuple[str | None, str | None, str | None]:
    s = str(path)
    sub  = _RE_SUB_GENERIC.search(s)
    task = _RE_TASK_GENERIC.search(s)
    run  = _RE_RUN_GENERIC.search(s)
    sub_id  = str(int(sub.group(1))) if sub else None
    task_id = (task.group(1).lower() if task else None)
    run_id  = str(int(run.group(1))) if run else None
    return sub_id, task_id, run_id

def _canonical_task_token(raw_task: str | None, path: Path) -> str | None:
    s = (raw_task.lower() if raw_task else path.name.lower())
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

def find_eeg_files(root: Path) -> List[Path]:
    patterns = [
        str(root / "**" / "EEG_rereferenced.mat"),
        str(root / "**" / "*.mat"),
    ]
    files: List[Path] = []
    for pat in patterns:
        files.extend([Path(p) for p in glob.glob(pat, recursive=True)])
    # prefer exact filename if present
    files = sorted(set(files), key=lambda p: ("EEG_rereferenced.mat" not in p.name, str(p).lower()))
    return files

def find_bold_files(root: Path) -> List[Path]:
    patterns = [
        str(root / "**" / "*bold.nii.gz"),
        str(root / "**" / "*bold.nii"),
    ]
    files: List[Path] = []
    for pat in patterns:
        files.extend([Path(p) for p in glob.glob(pat, recursive=True)])
    return sorted(set(files), key=lambda p: str(p).lower())


# ------------ Duration helpers ------------
def duration_eeg_seconds(mat_path: Path, original_fs: int) -> float:
    """
    Reads EEG .mat and returns duration in seconds.
    Searches keys: data_reref, eeg, data, EEG, eegdata.
    Transposes if needed (expects C x T).
    """
    mat = sio.loadmat(str(mat_path))
    arr = None
    for key in ['data_reref', 'eeg', 'data', 'EEG', 'eegdata']:
        if key in mat:
            arr = mat[key]
            break
    if arr is None:
        return 0.0
    if arr.shape[0] > arr.shape[1]:
        arr = arr.T
    return float(arr.shape[1]) / float(original_fs)

def duration_fmri_seconds(bold_path: Path, tr: float) -> float:
    """Reads NIfTI and returns duration in seconds as T * TR."""
    img = nib.load(str(bold_path))
    shape = img.shape
    T = shape[-1] if len(shape) >= 4 else 0
    return float(T) * float(tr)


# ------------ Core computation ------------
def normalize_subject_list(x) -> Iterable[str] | None:
    if x is None:
        return None
    return [str(int(s)) for s in x]

def build_file_maps(eeg_root: Path, fmri_root: Path):
    """Return dicts mapping (sub, task, run) -> Path for EEG and fMRI files."""
    eeg_map: Dict[Tuple[str, str, str], Path] = {}
    fmri_map: Dict[Tuple[str, str, str], Path] = {}

    for p in find_eeg_files(eeg_root):
        sub, task_raw, run = _parse_key_from_path(p)
        if sub and run:
            task = _canonical_task_token(task_raw, p)
            if task:
                eeg_map[(sub, task, run)] = p

    for p in find_bold_files(fmri_root):
        sub, task_raw, run = _parse_key_from_path(p)
        if sub and run:
            task = _canonical_task_token(task_raw, p)
            if task:
                fmri_map[(sub, task, run)] = p

    return eeg_map, fmri_map

def human_hours(seconds: float) -> str:
    return f"{seconds/3600.0:.2f} h"


def main():
    eeg_root = Path(EEG_ROOT)
    fmri_root = Path(FMRI_ROOT)

    eeg_map, fmri_map = build_file_maps(eeg_root, fmri_root)

    # Intersection keys (paired sessions)
    keys_all = sorted(set(eeg_map.keys()) & set(fmri_map.keys()), key=lambda t: (int(t[0]), t[1], int(t[2])))

    subj_filter = normalize_subject_list(SUBJECTS)
    if subj_filter:
        keys = [k for k in keys_all if k[0] in set(subj_filter)]
    else:
        keys = keys_all

    if not keys:
        print("[info] No paired (subject, task, run) sessions found with the given roots/filters.")
        sys.exit(0)

    total_sessions = 0
    total_eeg_sec = 0.0
    total_fmri_sec = 0.0
    total_paired_sec = 0.0

    by_subject: Dict[str, Dict[str, float]] = {}
    by_task: Dict[str, Dict[str, float]] = {}
    per_session_rows: List[Dict[str, str]] = []

    for (sub, task, run) in keys:
        eeg_path = eeg_map[(sub, task, run)]
        fmri_path = fmri_map[(sub, task, run)]

        eeg_sec = duration_eeg_seconds(eeg_path, ORIGINAL_FS)
        fmri_sec = duration_fmri_seconds(fmri_path, TR_SECONDS)
        paired_sec = min(eeg_sec, fmri_sec)

        total_sessions += 1
        total_eeg_sec += eeg_sec
        total_fmri_sec += fmri_sec
        total_paired_sec += paired_sec

        sb = by_subject.setdefault(sub, {"sessions": 0.0, "eeg_sec": 0.0, "fmri_sec": 0.0, "paired_sec": 0.0})
        sb["sessions"] += 1
        sb["eeg_sec"] += eeg_sec
        sb["fmri_sec"] += fmri_sec
        sb["paired_sec"] += paired_sec

        tb = by_task.setdefault(task, {"sessions": 0.0, "eeg_sec": 0.0, "fmri_sec": 0.0, "paired_sec": 0.0})
        tb["sessions"] += 1
        tb["eeg_sec"] += eeg_sec
        tb["fmri_sec"] += fmri_sec
        tb["paired_sec"] += paired_sec

        per_session_rows.append({
            "subject": sub,
            "task": task,
            "run": run,
            "eeg_seconds": f"{eeg_sec:.1f}",
            "fmri_seconds": f"{fmri_sec:.1f}",
            "paired_overlap_seconds": f"{paired_sec:.1f}",
            "eeg_hours": f"{eeg_sec/3600.0:.3f}",
            "fmri_hours": f"{fmri_sec/3600.0:.3f}",
            "paired_overlap_hours": f"{paired_sec/3600.0:.3f}",
            "eeg_path": str(eeg_path),
            "fmri_path": str(fmri_path),
        })

    # ----- Print summary -----
    print("========== Oddball EEG/fMRI — Paired Sessions Summary ==========")
    print(f"Total paired sessions: {total_sessions}")
    print(f"Total EEG:             {human_hours(total_eeg_sec)}")
    print(f"Total fMRI:            {human_hours(total_fmri_sec)}")
    print(f"Total paired-overlap:  {human_hours(total_paired_sec)}  [usable for aligned windows]")
    print(f"Unique subjects:       {len(by_subject)}")
    print(f"Tasks present:         {', '.join(sorted(by_task.keys()))}")
    print()

    print("---- Per-subject ----")
    for sub in sorted(by_subject.keys(), key=lambda s: int(s)):
        m = by_subject[sub]
        print(f"sub-{int(sub):02d}: sessions={int(m['sessions'])}  "
              f"EEG={human_hours(m['eeg_sec'])}  fMRI={human_hours(m['fmri_sec'])}  "
              f"paired={human_hours(m['paired_sec'])}")

    print()
    print("---- Per-task ----")
    for task in sorted(by_task.keys()):
        m = by_task[task]
        print(f"{task}: sessions={int(m['sessions'])}  "
              f"EEG={human_hours(m['eeg_sec'])}  fMRI={human_hours(m['fmri_sec'])}  "
              f"paired={human_hours(m['paired_sec'])}")

    # ----- Optional CSV output -----
    if CSV_OUT_PATH and per_session_rows:
        out_path = Path(CSV_OUT_PATH).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Per-session CSV
        with open(out_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(per_session_rows[0].keys()))
            w.writeheader()
            w.writerows(per_session_rows)

        # Per-subject rollup
        per_subject_csv = out_path.with_name(out_path.stem + "_per_subject.csv")
        with open(per_subject_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["subject", "sessions", "eeg_hours", "fmri_hours", "paired_hours"])
            for sub in sorted(by_subject.keys(), key=lambda s: int(s)):
                m = by_subject[sub]
                w.writerow([
                    sub,
                    int(m["sessions"]),
                    f"{m['eeg_sec']/3600.0:.3f}",
                    f"{m['fmri_sec']/3600.0:.3f}",
                    f"{m['paired_sec']/3600.0:.3f}",
                ])

        print(f"\n[write] Per-session CSV  → {out_path}")
        print(f"[write] Per-subject CSV  → {per_subject_csv}")


if __name__ == "__main__":
    main()
