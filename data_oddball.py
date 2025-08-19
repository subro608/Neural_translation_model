#!/usr/bin/env python3
"""
Oddball EEG/fMRI preprocessing and datasets — with automatic caching
and paired-window materialization (NatView-style) for fast training.

What’s cached automatically:
- EEG downsampled arrays (C, Tds) per .mat  → ~/.cache/oddball_eegfmri/eeg_*.npz
- fMRI A424 parcel time series (T, 424) per BOLD → ~/.cache/oddball_eegfmri/fmri_*.npz
- Fully shaped, normalized paired windows (EEG (C,P,S), fMRI (T,V))
  for a given dataset configuration & split of (subject, task, run)
  → ~/.cache/oddball_eegfmri/paired/paired_<signature>.pkl

Public API preserved:
- collect_common_sr_keys
- fixed_subject_keys
- PairedAlignedDataset (now auto-uses paired window caches)
- collate_paired

Diagnostics:
- diagnose_one_pair(...) prints BEFORE/AFTER & first window stats
- prewarm_cache(...) fills all caches for a given split
"""

from __future__ import annotations

import os
import json
import glob
import hashlib
import pickle
from dataclasses import dataclass
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Iterable, Set, Any

import numpy as np
import torch
from torch.utils.data import Dataset

import scipy.io as sio
import scipy.signal
import nibabel as nib
from nilearn.image import resample_to_img
import argparse

# =============================
# Cache utilities (automatic)
# =============================
CACHE_VERSION = "oddball_cache_2025-08-19a"

def _default_cache_dir() -> Path:
    return Path(os.environ.get("ODDBALL_CACHE_DIR", "~/.cache/oddball_eegfmri")).expanduser()

def _safe_mkdirs(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _file_sig(p: Path) -> str:
    try:
        st = p.stat()
        return f"{str(p.resolve())}|{int(st.st_mtime)}|{st.st_size}"
    except Exception:
        return f"{str(p)}|0|0"

def _hash_key(*parts: str) -> str:
    h = hashlib.sha1()
    for s in parts:
        h.update(s.encode("utf-8"))
    return h.hexdigest()

# Small in-memory LRU to avoid repeat disk reads within a run
class _LRU:
    def __init__(self, k: int = 4):
        from collections import OrderedDict
        self.k = int(k)
        self._od: "OrderedDict[str, Any]" = OrderedDict()

    def get(self, key: str):
        if key in self._od:
            self._od.move_to_end(key)
            return self._od[key]
        return None

    def put(self, key: str, val: Any):
        self._od[key] = val
        self._od.move_to_end(key)
        while len(self._od) > self.k:
            self._od.popitem(last=False)

_MEM = _LRU(6)

# =============================
# EEG helpers
# =============================
def downsample_eeg(eeg_data: np.ndarray, original_fs: int = 1000, target_fs: int = 200) -> np.ndarray:
    factor = original_fs // target_fs
    if factor < 1 or original_fs % target_fs != 0:
        raise ValueError(f"original_fs ({original_fs}) must be an integer multiple of target_fs ({target_fs})")
    x = scipy.signal.decimate(eeg_data, factor, axis=1, ftype='iir', zero_phase=True)
    return x.copy()

def patchify_eeg(eeg: np.ndarray, patch_size: int = 200) -> Tuple[np.ndarray, int]:
    channels, total_time = eeg.shape
    num_patches = total_time // patch_size
    if num_patches == 0:
        raise ValueError(f"EEG too short: {total_time} points < {patch_size}")
    usable = num_patches * patch_size
    eeg_trim = eeg[:, :usable]
    patches = eeg_trim.reshape(channels, num_patches, patch_size)
    return patches, num_patches

def zscore_eeg_channelwise(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Z-score per channel across the S dimension (kept identical to your pipeline).
    Accepts (C,P,S) or (B,C,P,S). Returns same dimensionality as input.
    """
    orig_dim = x.dim()
    if orig_dim == 3:
        x = x.unsqueeze(0)  # (1,C,P,S)
    assert x.dim() == 4, f"Expected 4D (B,C,P,S), got {tuple(x.shape)}"
    mean = x.mean(dim=3, keepdim=True)
    std = x.std(dim=3, keepdim=True, unbiased=False).clamp(min=eps)
    x = (x - mean) / std
    return x.squeeze(0) if orig_dim == 3 else x

# =============================
# fMRI helpers
# =============================
def extract_a424_from_native(bold_path: str, atlas_path: str) -> Tuple[np.ndarray, nib.Nifti1Image, np.ndarray, np.ndarray]:
    """
    Returns:
      parcel_ts: (T, 424) float64
      bold_img: nib image
      atlas_resampled_data: (X,Y,Z) labels in BOLD native grid
      parcel_ids: (424,) 1..424
    """
    bold_img = nib.load(bold_path)
    bold_data = bold_img.get_fdata()  # (X,Y,Z,T)
    atlas_img = nib.load(atlas_path)
    try:
        atlas_resampled = resample_to_img(atlas_img, bold_img, interpolation='nearest', force_resample=True, copy_header=True)
    except TypeError:
        atlas_resampled = resample_to_img(atlas_img, bold_img, interpolation='nearest')

    atlas_data = atlas_resampled.get_fdata()  # (X,Y,Z)
    T = bold_data.shape[-1]
    bold_2d = bold_data.reshape(-1, T)
    atlas_flat = atlas_data.reshape(-1)

    parcel_ids = np.arange(1, 425, dtype=int)
    parcel_ts = np.zeros((T, parcel_ids.size), dtype=np.float64)
    for i, pid in enumerate(parcel_ids):
        idx = np.where(atlas_flat == pid)[0]
        if idx.size > 0:
            parcel_ts[:, i] = np.nanmean(bold_2d[idx, :], axis=0)
    parcel_ts = np.nan_to_num(parcel_ts, nan=0.0, posinf=0.0, neginf=0.0)
    return parcel_ts, bold_img, atlas_data, parcel_ids

def pad_timepoints_for_brainlm_torch(signal_vectors_btv: torch.Tensor, patch_size: int = 20) -> torch.Tensor:
    """
    Pads along time dimension to a multiple of patch_size.
    Expects (B, T, V). Returns (B, T_pad, V).
    """
    B, T, V = signal_vectors_btv.shape
    remainder = T % patch_size
    if remainder == 0:
        return signal_vectors_btv
    padding_needed = patch_size - remainder
    padded = torch.zeros((B, T + padding_needed, V), dtype=signal_vectors_btv.dtype, device=signal_vectors_btv.device)
    padded[:, :T, :] = signal_vectors_btv
    return padded

def normalize_fmri_time_series(fmri_btv: torch.Tensor, mode: str = 'zscore', eps: float = 1e-8) -> torch.Tensor:
    if mode == 'none':
        return fmri_btv.to(dtype=torch.float32)
    x = fmri_btv.to(dtype=torch.float32)
    if mode == 'zscore':
        mean_t = x.mean(dim=1, keepdim=True)
        std_t = x.std(dim=1, keepdim=True, unbiased=False).clamp(min=eps)
        return (x - mean_t) / std_t
    elif mode == 'psc':
        mean_t = x.mean(dim=1, keepdim=True)
        return 100.0 * (x - mean_t) / (mean_t + eps)
    elif mode == 'mad':
        median_t = x.median(dim=1, keepdim=True).values
        mad = (x - median_t).abs().median(dim=1, keepdim=True).values
        scaled_mad = mad * 1.4826
        return (x - median_t) / (scaled_mad + eps)
    else:
        raise ValueError(f"Unsupported fmri normalization mode: {mode}")

# =============================
# A424 coordinate helpers
# =============================
def load_a424_coords(dat_path: Path) -> np.ndarray:
    try:
        arr = np.loadtxt(str(dat_path)).astype(np.float32)
    except Exception as e:
        raise RuntimeError(f"Failed to load A424 coords from {dat_path}: {e}")
    arr = arr[np.argsort(arr[:, 0])]
    coords = np.zeros((424, 3), dtype=np.float32)
    for roi in range(1, 425):
        idx = roi - 1
        if idx < arr.shape[0] and int(arr[idx, 0]) == roi:
            coords[idx] = arr[idx, 1:4]
        else:
            coords[idx] = 0.0
    return coords

# =============================
# Disk caches (EEG/FMRI arrays)
# =============================
def _eeg_cache_path(cache_dir: Path, mat_path: Path, original_fs: int, target_fs: int, channels_limit: int) -> Path:
    key = _hash_key("EEG", CACHE_VERSION, _file_sig(mat_path), str(original_fs), str(target_fs), str(channels_limit))[:16]
    return cache_dir / f"eeg_{mat_path.stem}_{key}.npz"

def _fmri_cache_path(cache_dir: Path, bold_path: Path, atlas_path: Path) -> Path:
    key = _hash_key("FMRI_A424", CACHE_VERSION, _file_sig(bold_path), _file_sig(Path(atlas_path)))[:16]
    return cache_dir / f"fmri_{Path(bold_path).stem}_{key}.npz"

def _eeg_mem_key(mat_path: Path, original_fs: int, target_fs: int, channels_limit: int) -> str:
    return f"EEG|{str(mat_path)}|{original_fs}|{target_fs}|{channels_limit}|{CACHE_VERSION}"

def _fmri_mem_key(bold_path: Path, atlas_path: Path) -> str:
    return f"FMRI|{str(bold_path)}|{str(atlas_path)}|{CACHE_VERSION}"

def load_eeg_ds_cached(mat_path: Path, *, original_fs: int, target_fs: int, channels_limit: int, cache_dir: Path) -> np.ndarray:
    _safe_mkdirs(cache_dir)
    mkey = _eeg_mem_key(mat_path, original_fs, target_fs, channels_limit)
    arr = _MEM.get(mkey)
    if arr is not None:
        return arr

    cpath = _eeg_cache_path(cache_dir, mat_path, original_fs, target_fs, channels_limit)
    if cpath.exists():
        with np.load(cpath) as npz:
            ds = npz["arr"].astype(np.float32, copy=False)
        _MEM.put(mkey, ds)
        return ds

    # Compute, save
    mat = sio.loadmat(str(mat_path))
    eeg_key = None
    for k in ['data_reref', 'eeg', 'data', 'EEG', 'eegdata']:
        if k in mat:
            eeg_key = k
            break
    if eeg_key is None:
        raise KeyError(f"No EEG array in {mat_path}")
    data = mat[eeg_key]
    if data.shape[0] > data.shape[1]:
        data = data.T
    # Limit channels BEFORE downsample to match pipeline
    # (you did this order in dataset previously)
    # If you ever want to change, bump CACHE_VERSION
    if data.shape[0] > channels_limit:
        data = data[:channels_limit]
    ds = downsample_eeg(data, original_fs, target_fs).astype(np.float32, copy=False)
    np.savez_compressed(cpath, arr=ds)
    _MEM.put(mkey, ds)
    return ds

def load_fmri_parcels_cached(bold_path: Path, atlas_path: Path, *, cache_dir: Path) -> np.ndarray:
    _safe_mkdirs(cache_dir)
    mkey = _fmri_mem_key(bold_path, atlas_path)
    arr = _MEM.get(mkey)
    if arr is not None:
        return arr

    cpath = _fmri_cache_path(cache_dir, bold_path, atlas_path)
    if cpath.exists():
        with np.load(cpath) as npz:
            ts = npz["arr"].astype(np.float32, copy=False)
        _MEM.put(mkey, ts)
        return ts

    parcel_ts, _, _, _ = extract_a424_from_native(str(bold_path), str(atlas_path))
    ts = parcel_ts.astype(np.float32, copy=False)
    np.savez_compressed(cpath, arr=ts)
    _MEM.put(mkey, ts)
    return ts

# =============================
# File discovery + parsing
# =============================
def find_eeg_files(root: Path) -> List[Path]:
    patterns = [
        str(root / "**" / "EEG_rereferenced.mat"),
        str(root / "**" / "*.mat"),
    ]
    files: List[Path] = []
    for pat in patterns:
        files.extend([Path(p) for p in glob.glob(pat, recursive=True)])
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
    if raw_task is None:
        s = path.name.lower()
    else:
        s = raw_task.lower()

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

# =============================
# Intersections & splits
# =============================
def collect_common_sr_keys(eeg_root: Path, fmri_root: Path) -> List[Tuple[str, str, str]]:
    eeg_files = find_eeg_files(eeg_root)
    fmri_files = find_bold_files(fmri_root)

    key_to_eeg: Dict[Tuple[str, str, str], Path] = {}
    key_to_fmri: Dict[Tuple[str, str, str], Path] = {}
    for p in eeg_files:
        sub, task_raw, run = _parse_key_from_path(p)
        if sub and run:
            task = _canonical_task_token(task_raw, p)
            if task:
                key_to_eeg[(sub, task, run)] = p
    for p in fmri_files:
        sub, task_raw, run = _parse_key_from_path(p)
        if sub and run:
            task = _canonical_task_token(task_raw, p)
            if task:
                key_to_fmri[(sub, task, run)] = p

    inter = set(key_to_eeg.keys()) & set(key_to_fmri.keys())
    return sorted(inter, key=lambda t: (int(t[0]), str(t[1]), int(t[2])))

def fixed_subject_keys(
    eeg_root: Path,
    fmri_root: Path,
    train_subjects: Iterable[int],
    val_subjects: Iterable[int],
    test_subjects: Iterable[int],
) -> Tuple[Tuple[Tuple[str, str, str], ...], Tuple[Tuple[str, str, str], ...], Tuple[Tuple[str, str, str], ...]]:
    inter_keys = collect_common_sr_keys(eeg_root, fmri_root)
    tr = {str(int(s)) for s in train_subjects}
    va = {str(int(s)) for s in val_subjects}
    te = {str(int(s)) for s in test_subjects}

    def _assert_disjoint(a: Set[str], b: Set[str], aname: str, bname: str):
        inter = a & b
        if inter:
            raise ValueError(f"Subject leakage: {aname} ∩ {bname} = {sorted(inter)}")

    _assert_disjoint(tr, va, "train_subjects", "val_subjects")
    _assert_disjoint(tr, te, "train_subjects", "test_subjects")
    _assert_disjoint(va, te, "val_subjects", "test_subjects")

    train_keys = tuple(k for k in inter_keys if k[0] in tr)
    val_keys   = tuple(k for k in inter_keys if k[0] in va)
    test_keys  = tuple(k for k in inter_keys if k[0] in te)

    if not train_keys or not val_keys:
        raise RuntimeError(f"Empty split: train={len(train_keys)} val={len(val_keys)} test={len(test_keys)}")

    return train_keys, val_keys, test_keys

# =============================
# Window-level paired cache
# =============================
def _paired_cache_dir(base: Path) -> Path:
    p = base / "paired"
    _safe_mkdirs(p)
    return p

def _paired_signature(
    *,
    eeg_root: Path,
    fmri_root: Path,
    a424_label_nii: Path,
    include_sr_keys: Tuple[Tuple[str, str, str], ...],
    window_sec: int,
    stride_sec: int,
    original_fs: int,
    target_fs: int,
    tr: float,
    channels_limit: int,
    fmri_norm: str,
) -> str:
    parts = [
        CACHE_VERSION,
        str(eeg_root.resolve()),
        str(fmri_root.resolve()),
        _file_sig(a424_label_nii),
        f"w{window_sec}",
        f"s{stride_sec}",
        f"fs{original_fs}->{target_fs}",
        f"tr{tr}",
        f"ch{channels_limit}",
        f"norm{fmri_norm}",
    ]
    # include file signatures for all included keys (robust invalidation)
    for sub, task, run in include_sr_keys:
        # attempt typical paths for deterministic signature
        # (we’ll re-derive actual paths below anyway)
        parts.append(f"{sub}-{task}-{run}")
    return _hash_key(*parts)[:20]

def _paired_cache_path(base: Path, sig: str) -> Path:
    return _paired_cache_dir(base) / f"paired_{sig}.pkl"

def _find_paths_for_keys(eeg_root: Path, fmri_root: Path, keys: Tuple[Tuple[str, str, str], ...]) -> List[Tuple[str, str, Path, Path]]:
    eeg_files = find_eeg_files(eeg_root)
    fmri_files = find_bold_files(fmri_root)
    key_to_eeg: Dict[Tuple[str, str, str], Path] = {}
    key_to_fmri: Dict[Tuple[str, str, str], Path] = {}

    for p in eeg_files:
        sub, task_raw, run = _parse_key_from_path(p)
        if sub and run:
            task = _canonical_task_token(task_raw, p)
            if task:
                key_to_eeg[(sub, task, run)] = p
    for p in fmri_files:
        sub, task_raw, run = _parse_key_from_path(p)
        if sub and run:
            task = _canonical_task_token(task_raw, p)
            if task:
                key_to_fmri[(sub, task, run)] = p

    out: List[Tuple[str, str, Path, Path]] = []
    for k in keys:
        if k in key_to_eeg and k in key_to_fmri:
            out.append((k[0], k[1], key_to_eeg[k], key_to_fmri[k]))
    return out

def _make_windows_for_pair(
    eeg_ds: np.ndarray,
    fmri_ts: np.ndarray,
    *,
    window_sec: int,
    stride_sec: int,
    target_fs: int,
    tr: float,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    eeg_duration = eeg_ds.shape[1] / float(target_fs)
    fmri_duration = fmri_ts.shape[0] * float(tr)
    max_duration = min(eeg_duration, fmri_duration)
    max_start = max(0, int(max_duration) - int(window_sec))
    windows: List[Tuple[np.ndarray, np.ndarray]] = []
    for start_sec in range(0, max_start + 1, int(stride_sec)):
        # EEG window
        start_sample = int(start_sec * target_fs)
        end_sample = start_sample + int(window_sec * target_fs)
        eeg_window = eeg_ds[:, start_sample:end_sample]  # (C, W*fs)
        if eeg_window.shape[1] != window_sec * target_fs:
            continue
        eeg_patches = eeg_window.reshape(eeg_window.shape[0], window_sec, target_fs)  # (C,P,S)

        # fMRI window
        start_vol = int(start_sec / tr)
        end_vol = start_vol + int(window_sec / tr)
        if end_vol > fmri_ts.shape[0]:
            continue
        fmri_window = fmri_ts[start_vol:end_vol, :]  # (T,V)

        windows.append((eeg_patches, fmri_window))
    return windows

def build_paired_window_cache(
    *,
    eeg_root: Path,
    fmri_root: Path,
    a424_label_nii: Path,
    include_sr_keys: Tuple[Tuple[str, str, str], ...],
    window_sec: int,
    stride_sec: Optional[int],
    original_fs: int,
    target_fs: int,
    tr: float,
    channels_limit: int,
    fmri_norm: str,
    cache_dir: Path,
) -> Path:
    stride = int(stride_sec) if stride_sec is not None else int(window_sec)
    sig = _paired_signature(
        eeg_root=eeg_root,
        fmri_root=fmri_root,
        a424_label_nii=a424_label_nii,
        include_sr_keys=include_sr_keys,
        window_sec=window_sec,
        stride_sec=stride,
        original_fs=original_fs,
        target_fs=target_fs,
        tr=tr,
        channels_limit=channels_limit,
        fmri_norm=fmri_norm,
    )
    out_path = _paired_cache_path(cache_dir, sig)
    if out_path.exists():
        return out_path

    pairs = _find_paths_for_keys(eeg_root, fmri_root, include_sr_keys)
    cached_list: List[Dict[str, Any]] = []
    for sub, task, eeg_p, fmri_p in pairs:
        # derive run from filenames
        _, _, run = _parse_key_from_path(eeg_p)
        # Load cached per-file arrays
        eeg_ds = load_eeg_ds_cached(eeg_p, original_fs=original_fs, target_fs=target_fs, channels_limit=channels_limit, cache_dir=cache_dir)
        fmri_ts = load_fmri_parcels_cached(fmri_p, str(a424_label_nii), cache_dir=cache_dir)
        # Enumerate windows
        windows = _make_windows_for_pair(
            eeg_ds, fmri_ts,
            window_sec=window_sec, stride_sec=stride, target_fs=target_fs, tr=tr,
        )
        for idx_win, (eeg_patch, fmri_win) in enumerate(windows):
            # Normalize as in your pipeline
            eeg_t = torch.from_numpy(eeg_patch).unsqueeze(0).float()
            eeg_t = zscore_eeg_channelwise(eeg_t).squeeze(0)          # (C,P,S)
            fmri_t = torch.from_numpy(fmri_win).unsqueeze(0).float()
            fmri_t = normalize_fmri_time_series(fmri_t, mode=fmri_norm).squeeze(0)  # (T,V)
            cached_list.append({
                "eeg_window": eeg_t.numpy(),
                "fmri_window": fmri_t.numpy(),
                "meta": {
                    "subject": sub, "task": task, "run": run or "",
                    "eeg_path": str(eeg_p), "fmri_path": str(fmri_p),
                    "window_idx": idx_win,
                    "window_sec": window_sec, "stride_sec": stride,
                }
            })
    _safe_mkdirs(_paired_cache_dir(cache_dir))
    with open(out_path, "wb") as f:
        pickle.dump(cached_list, f)
    return out_path

def load_paired_window_cache(path: Path) -> List[Dict[str, Any]]:
    with open(path, "rb") as f:
        return pickle.load(f)

# =============================
# Datasets
# =============================
class PairedAlignedDataset(Dataset):
    """
    Builds aligned EEG–fMRI windows for the SAME subject, task, run and time interval,
    now with AUTOMATIC paired-window caching (NatView-style).

    If a cache exists for the exact configuration + included keys, it is used.
    Otherwise, the cache is built once and then reused on subsequent runs.

    Returns items:
      - eeg_window: (C, P=window_sec, S=target_fs)  [z-scored per channel]
      - fmri_window: (T=round(window_sec/TR), V)    [normalized per voxel by fmri_norm]
      - meta dict: subject, task, run, window_idx, paths
    """
    def __init__(
        self,
        eeg_root: Path,
        fmri_root: Path,
        a424_label_nii: Path,
        window_sec: int,
        original_fs: int = 1000,
        target_fs: int = 200,
        tr: float = 2.0,
        channels_limit: int = 34,
        fmri_norm: str = 'zscore',
        stride_sec: Optional[int] = None,
        device: str = 'cpu',
        include_sr_keys: Optional[Tuple[Tuple[str, str, str], ...]] = None,
        include_subjects: Optional[Iterable[int]] = None,
    ) -> None:
        self.eeg_root = Path(eeg_root)
        self.fmri_root = Path(fmri_root)
        self.a424_label_nii = Path(a424_label_nii)
        self.window_sec = int(window_sec)
        self.original_fs = int(original_fs)
        self.target_fs = int(target_fs)
        self.tr = float(tr)
        self.channels_limit = int(channels_limit)
        self.fmri_norm = str(fmri_norm)
        self.stride_sec = int(stride_sec) if stride_sec is not None else int(window_sec)
        self.device = torch.device(device)
        self.cache_dir = _default_cache_dir()
        _safe_mkdirs(self.cache_dir)

        # Determine included keys deterministically
        all_keys = collect_common_sr_keys(self.eeg_root, self.fmri_root)  # [(sub,task,run), ...]
        if include_sr_keys is not None:
            inc = set(include_sr_keys)
            keys = tuple(sorted([k for k in all_keys if k in inc], key=lambda t: (int(t[0]), t[1], int(t[2]))))
        elif include_subjects is not None:
            subj_whitelist = {str(int(s)) for s in include_subjects}
            keys = tuple(sorted([k for k in all_keys if k[0] in subj_whitelist], key=lambda t: (int(t[0]), t[1], int(t[2]))))
        else:
            keys = tuple(all_keys)

        # Build or load paired-window cache for this exact configuration
        sig = _paired_signature(
            eeg_root=self.eeg_root,
            fmri_root=self.fmri_root,
            a424_label_nii=self.a424_label_nii,
            include_sr_keys=keys,
            window_sec=self.window_sec,
            stride_sec=self.stride_sec,
            original_fs=self.original_fs,
            target_fs=self.target_fs,
            tr=self.tr,
            channels_limit=self.channels_limit,
            fmri_norm=self.fmri_norm,
        )
        cache_path = _paired_cache_path(self.cache_dir, sig)
        if not cache_path.exists():
            cache_path = build_paired_window_cache(
                eeg_root=self.eeg_root,
                fmri_root=self.fmri_root,
                a424_label_nii=self.a424_label_nii,
                include_sr_keys=keys,
                window_sec=self.window_sec,
                stride_sec=self.stride_sec,
                original_fs=self.original_fs,
                target_fs=self.target_fs,
                tr=self.tr,
                channels_limit=self.channels_limit,
                fmri_norm=self.fmri_norm,
                cache_dir=self.cache_dir,
            )
        self._cached_list = load_paired_window_cache(cache_path)

    def __len__(self) -> int:
        return len(self._cached_list)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self._cached_list[idx]
        eeg = torch.from_numpy(item["eeg_window"]).to(self.device)   # already z-scored (C,P,S)
        fmri = torch.from_numpy(item["fmri_window"]).to(self.device) # already normalized (T,V)
        return {
            "eeg_window": eeg,
            "fmri_window": fmri,
            "meta": item["meta"],
        }

def collate_paired(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    eeg = torch.stack([b['eeg_window'] for b in batch], dim=0)
    fmri = torch.stack([b['fmri_window'] for b in batch], dim=0)
    metas = [b['meta'] for b in batch]
    return {'eeg_window': eeg, 'fmri_window': fmri, 'meta': metas}

# =============================
# CLI sanity check & diagnostics
# =============================
def _duration_eeg_seconds(mat_path: Path, original_fs: int) -> float:
    mat = sio.loadmat(str(mat_path))
    for key in ['data_reref', 'eeg', 'data', 'EEG', 'eegdata']:
        if key in mat:
            arr = mat[key]
            break
    else:
        return 0.0
    if arr.shape[0] > arr.shape[1]:
        arr = arr.T
    return float(arr.shape[1]) / float(original_fs)

def _duration_fmri_seconds(bold_path: Path, tr: float) -> float:
    img = nib.load(str(bold_path))
    shape = img.shape
    T = shape[-1] if len(shape) >= 4 else 0
    return float(T) * float(tr)

def _summ(name: str, arr: np.ndarray):
    try:
        mn, mx = np.nanmin(arr), np.nanmax(arr)
        mu, sd = np.nanmean(arr), np.nanstd(arr)
        print(f"{name:20s} shape={arr.shape} min/max={mn:.4f}/{mx:.4f} mean/std={mu:.4f}/{sd:.4f}")
    except Exception as e:
        print(f"{name:20s} shape={arr.shape} (stats err: {e})")

def diagnose_one_pair(
    eeg_root: Path,
    fmri_root: Path,
    a424_label_nii: Path,
    original_fs: int = 1000,
    target_fs: int = 200,
    tr: float = 2.0,
    channels_limit: int = 34,
    window_sec: int = 40,
):
    keys = collect_common_sr_keys(eeg_root, fmri_root)
    if not keys:
        print("No intersecting (subject, task, run) found.")
        return
    # find paths for first key
    (sub, task, run) = keys[0]
    eeg_p = None
    fmri_p = None
    for p in find_eeg_files(eeg_root):
        s, t, r = _parse_key_from_path(p)
        if (s, _canonical_task_token(t, p), r) == (sub, task, run):
            eeg_p = p; break
    for p in find_bold_files(fmri_root):
        s, t, r = _parse_key_from_path(p)
        if (s, _canonical_task_token(t, p), r) == (sub, task, run):
            fmri_p = p; break
    if eeg_p is None or fmri_p is None:
        print("Could not resolve file paths for first key.")
        return
    print(f"[diagnostics] sub={sub} task={task} run={run}")
    cache_dir = _default_cache_dir()

    eeg_ds = load_eeg_ds_cached(eeg_p, original_fs=original_fs, target_fs=target_fs, channels_limit=channels_limit, cache_dir=cache_dir)
    fmri_ts = load_fmri_parcels_cached(fmri_p, str(a424_label_nii), cache_dir=cache_dir)
    _summ("EEG downsampled", eeg_ds)
    _summ("fMRI parcels", fmri_ts)

    # one window
    start_sec = 0
    start_samp = start_sec * target_fs
    eeg_win = eeg_ds[:, start_samp:start_samp + window_sec * target_fs]
    eeg_patch = eeg_win.reshape(eeg_ds.shape[0], window_sec, target_fs)
    eeg_t = torch.from_numpy(eeg_patch).unsqueeze(0).float()
    eeg_t = zscore_eeg_channelwise(eeg_t).squeeze(0).numpy()
    _summ("EEG window (C,P,S)", eeg_t)

    start_vol = int(start_sec / tr)
    fmri_win = fmri_ts[start_vol:start_vol + int(window_sec / tr), :]
    fmri_t = torch.from_numpy(fmri_win).unsqueeze(0).float()
    fmri_t = normalize_fmri_time_series(fmri_t, mode='zscore').squeeze(0).numpy()
    _summ("fMRI window (T,V)", fmri_t)

def prewarm_cache(
    *,
    eeg_root: Path,
    fmri_root: Path,
    a424_label_nii: Path,
    include_sr_keys: Tuple[Tuple[str, str, str], ...],
    window_sec: int,
    stride_sec: Optional[int],
    original_fs: int = 1000,
    target_fs: int = 200,
    tr: float = 2.0,
    channels_limit: int = 34,
    fmri_norm: str = 'zscore',
) -> Path:
    """
    Precompute paired-window cache for a given split (call once; then dataset loads instantly).
    """
    path = build_paired_window_cache(
        eeg_root=Path(eeg_root),
        fmri_root=Path(fmri_root),
        a424_label_nii=Path(a424_label_nii),
        include_sr_keys=include_sr_keys,
        window_sec=int(window_sec),
        stride_sec=int(stride_sec) if stride_sec is not None else None,
        original_fs=int(original_fs),
        target_fs=int(target_fs),
        tr=float(tr),
        channels_limit=int(channels_limit),
        fmri_norm=str(fmri_norm),
        cache_dir=_default_cache_dir(),
    )
    print(f"[prewarm_cache] built → {path}")
    return path

# =============================
# Minimal CLI (unchanged args)
# =============================
def _normalize_subject_list(x) -> Optional[List[int]]:
    if x is None:
        return None
    if isinstance(x, str):
        xs = [s for s in x.replace(',', ' ').split() if s]
        return [int(s) for s in xs]
    if isinstance(x, Iterable):
        return [int(s) for s in x]
    return None

def main():
    parser = argparse.ArgumentParser(
        description="Sanity check + instant paired-window caching (automatic)."
    )
    parser.add_argument('--eeg_root', type=str, required=True)
    parser.add_argument('--fmri_root', type=str, required=True)
    parser.add_argument('--a424_label_nii', type=str, required=True)
    parser.add_argument('--window_sec', type=int, default=40)
    parser.add_argument('--original_fs', type=int, default=1000)
    parser.add_argument('--target_fs', type=int, default=200)
    parser.add_argument('--tr', type=float, default=2.0)
    parser.add_argument('--fmri_norm', type=str, default='zscore', choices=['zscore', 'psc', 'mad', 'none'])
    parser.add_argument('--channels_limit', type=int, default=34)
    parser.add_argument('--stride_sec', type=int, default=None)
    parser.add_argument('--subjects', type=str, default=None,
                        help='Comma/space-separated subject IDs for quick demo split (optional).')
    args = parser.parse_args()

    # Diagnostics on first pair (quick)
    diagnose_one_pair(Path(args.eeg_root), Path(args.fmri_root), Path(args.a424_label_nii),
                      original_fs=args.original_fs, target_fs=args.target_fs, tr=args.tr,
                      channels_limit=args.channels_limit, window_sec=args.window_sec)

    # Build a tiny demo dataset (auto-cache)
    subj_list = _normalize_subject_list(args.subjects)
    if subj_list:
        # build include keys from those subjects
        all_keys = collect_common_sr_keys(Path(args.eeg_root), Path(args.fmri_root))
        include = tuple([k for k in all_keys if int(k[0]) in set(subj_list)])
    else:
        include = tuple(collect_common_sr_keys(Path(args.eeg_root), Path(args.fmri_root))[:6])  # small demo

    # Prewarm (optional; dataset would also build on first use)
    prewarm_cache(
        eeg_root=Path(args.eeg_root),
        fmri_root=Path(args.fmri_root),
        a424_label_nii=Path(args.a424_label_nii),
        include_sr_keys=include,
        window_sec=args.window_sec,
        stride_sec=args.stride_sec if args.stride_sec is not None else args.window_sec,
        original_fs=args.original_fs,
        target_fs=args.target_fs,
        tr=args.tr,
        channels_limit=args.channels_limit,
        fmri_norm=args.fmri_norm,
    )

    ds = PairedAlignedDataset(
        eeg_root=Path(args.eeg_root),
        fmri_root=Path(args.fmri_root),
        a424_label_nii=Path(args.a424_label_nii),
        window_sec=args.window_sec,
        original_fs=args.original_fs,
        target_fs=args.target_fs,
        tr=args.tr,
        channels_limit=args.channels_limit,
        fmri_norm=args.fmri_norm,
        stride_sec=args.stride_sec,
        device='cpu',
        include_sr_keys=include,
    )
    print(f"[dataset] cached windows: {len(ds)}")

if __name__ == "__main__":
    main()

# Example:
# python data_oddball.py \
#   --eeg_root D:\Neuroinformatics_research_2025\Oddball\ds116_eeg \
#   --fmri_root D:\Neuroinformatics_research_2025\Oddball\ds000116 \
#   --a424_label_nii D:\Neuroinformatics_research_2025\BrainLM\A424_resampled_to_bold.nii.gz \
#   --window_sec 40 --original_fs 1000 --target_fs 200 --tr 2.0 \
#   --fmri_norm zscore --channels_limit 34 --stride_sec 10 \
#   --subjects "7 11 12"
