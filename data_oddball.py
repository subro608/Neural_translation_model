#!/usr/bin/env python3
"""
Oddball EEG/fMRI preprocessing and datasets.

Provides:
- EEG helpers: downsample, patchify, channel-wise z-score normalization
- fMRI helpers: A424 extraction in native space, padding, per-parcel normalization
- A424 coordinate loader (xyz centroids)
- File discovery and splitting utilities
- Datasets and collate fns for EEG and fMRI
"""

from __future__ import annotations

import json
import glob
from dataclasses import dataclass
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

import scipy.io as sio
import scipy.signal
import nibabel as nib
from nilearn.image import resample_to_img
import matplotlib.pyplot as plt
import argparse


# -----------------------------
# EEG helpers
# -----------------------------
def downsample_eeg(eeg_data: np.ndarray, original_fs: int = 1000, target_fs: int = 200) -> np.ndarray:
    factor = original_fs // target_fs
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
    Z-score per channel across the entire window (P*S).
    Accepts (C,P,S) or (B,C,P,S). Returns same dimensionality as input.
    """
    orig_dim = x.dim()
    if orig_dim == 3:  # (C,P,S) -> (1,C,P,S)
        x = x.unsqueeze(0)
    assert x.dim() == 4, f"Expected 4D tensor (B,C,P,S), got {tuple(x.shape)}"
    # mean/std per-channel over (P,S)
    mean = x.mean(dim=3, keepdim=True)
    std = x.std(dim=3, keepdim=True, unbiased=False)
    x = (x - mean) / std
    return x.squeeze(0) if orig_dim == 3 else x


# -----------------------------
# fMRI helpers
# -----------------------------
def extract_a424_from_native(bold_path: str, atlas_path: str) -> Tuple[np.ndarray, nib.Nifti1Image, np.ndarray, np.ndarray]:
    bold_img = nib.load(bold_path)
    bold_data = bold_img.get_fdata()  # (X,Y,Z,T)
    atlas_img = nib.load(atlas_path)
    try:
        atlas_resampled = resample_to_img(
            atlas_img, bold_img, interpolation='nearest', force_resample=True, copy_header=True
        )
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


def pad_timepoints_for_brainlm(parcel_ts: np.ndarray, patch_size: int = 20) -> Tuple[np.ndarray, int]:
    num_timepoints, _ = parcel_ts.shape
    remainder = num_timepoints % patch_size
    if remainder == 0:
        return parcel_ts, num_timepoints
    pad = patch_size - remainder
    padded = np.zeros((num_timepoints + pad, parcel_ts.shape[1]))
    padded[:num_timepoints] = parcel_ts
    return padded, num_timepoints + pad


def pad_timepoints_for_brainlm_torch(signal_vectors_btv: torch.Tensor, patch_size: int = 20) -> torch.Tensor:
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
        # z-score per voxel over time (B,T,V) -> mean/std over dim=1
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


# -----------------------------
# A424 coordinate helpers
# -----------------------------
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


# -----------------------------
# File discovery and splits
# -----------------------------
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
    files = sorted(set(files), key=lambda p: str(p).lower())
    return files


def split_files(files: List[Path], ratios: Tuple[float, float, float], seed: int) -> Tuple[List[Path], List[Path], List[Path]]:
    n = len(files)
    if n == 0:
        return [], [], []
    rng = np.random.default_rng(seed)
    indices = np.arange(n)
    rng.shuffle(indices)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]
    idx_to_file = {i: files[i] for i in range(n)}
    return [idx_to_file[i] for i in train_idx], [idx_to_file[i] for i in val_idx], [idx_to_file[i] for i in test_idx]


# -----------------------------
# Datasets
# -----------------------------
class EEGWindowsDataset(Dataset):
    def __init__(
        self,
        eeg_mat_files: List[Path],
        original_fs: int,
        target_fs: int,
        window_sec: int,
        channels_limit: int = 34,
        device: str = 'cpu',
    ) -> None:
        self.eeg_mat_files = eeg_mat_files
        self.original_fs = original_fs
        self.target_fs = target_fs
        self.window_sec = window_sec
        self.channels_limit = channels_limit
        self.device = torch.device(device)
        self.index: List[Tuple[int, int, int]] = []

        for i, fpath in enumerate(self.eeg_mat_files):
            try:
                mat = sio.loadmat(str(fpath))
                eeg_key = None
                for key in ['data_reref', 'eeg', 'data', 'EEG', 'eegdata']:
                    if key in mat:
                        eeg_key = key
                        break
                if eeg_key is None:
                    continue
                data = mat[eeg_key]
                if data.shape[0] > data.shape[1]:
                    data = data.T
                ds = downsample_eeg(data, self.original_fs, self.target_fs)
                if ds.shape[0] > channels_limit:
                    ds = ds[:channels_limit]
                patches, num_patches = patchify_eeg(ds, patch_size=self.target_fs)
                window_patches = window_sec
                for start_patch in range(0, max(1, num_patches - window_patches + 1)):
                    self.index.append((i, start_patch, window_patches))
            except Exception:
                continue

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        file_idx, start_patch, window_patches = self.index[idx]
        fpath = self.eeg_mat_files[file_idx]
        mat = sio.loadmat(str(fpath))
        eeg_key = None
        for key in ['data_reref', 'eeg', 'data', 'EEG', 'eegdata']:
            if key in mat:
                eeg_key = key
                break
        data = mat[eeg_key]
        if data.shape[0] > data.shape[1]:
            data = data.T
        ds = downsample_eeg(data, self.original_fs, self.target_fs)
        if ds.shape[0] > self.channels_limit:
            ds = ds[:self.channels_limit]
        patches, _ = patchify_eeg(ds, patch_size=self.target_fs)
        window = patches[:, start_patch:start_patch + window_patches, :].copy()
        x = torch.from_numpy(window).unsqueeze(0).float().to(self.device)  # (1,C,P,S)
        x = zscore_eeg_channelwise(x).squeeze(0).cpu()
        return {
            'eeg_window': x,  # (C,P,S)
            'meta_file': str(fpath),
            'meta_start_patch': start_patch,
        }


class FMRIDataset(Dataset):
    def __init__(
        self,
        bold_files: List[Path],
        a424_label_nii: Path,
        window_sec: int,
        tr: float = 2.0,
        fmri_norm: str = 'zscore',
    ) -> None:
        self.bold_files = bold_files
        self.a424_label_nii = a424_label_nii
        self.window_sec = window_sec
        self.tr = tr
        self.fmri_norm = fmri_norm
        self.index: List[Tuple[int, int, int]] = []
        for i, bpath in enumerate(self.bold_files):
            try:
                parcel_ts, _, _, _ = extract_a424_from_native(str(bpath), str(a424_label_nii))
                T = parcel_ts.shape[0]
                window_T = int(round(window_sec / tr))
                if T < window_T:
                    continue
                for start_t in range(0, T - window_T + 1, window_T):
                    self.index.append((i, start_t, window_T))
            except Exception:
                continue

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        file_idx, start_t, window_T = self.index[idx]
        bpath = self.bold_files[file_idx]
        parcel_ts, _, _, _ = extract_a424_from_native(str(bpath), str(self.a424_label_nii))
        window = parcel_ts[start_t:start_t + window_T, :].astype(np.float32)  # (T,V)
        fmri_t = torch.from_numpy(window)  # (T,V)
        fmri_t = normalize_fmri_time_series(fmri_t.unsqueeze(0), mode=self.fmri_norm).squeeze(0)
        return {
            'fmri_window': fmri_t,
            'meta_file': str(bpath),
            'meta_start_t': start_t,
        }


# -----------------------------
# Collate functions
# -----------------------------
def collate_eeg(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    eeg_windows = torch.stack([b['eeg_window'] for b in batch], dim=0)
    return {'eeg_window': eeg_windows}


def collate_fmri(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    fmri_windows = torch.stack([b['fmri_window'] for b in batch], dim=0)
    return {'fmri_window': fmri_windows}


# -----------------------------
# Paired EEG–fMRI alignment (same subject/task/run/time)
# -----------------------------
_RE_SUB_GENERIC = re.compile(r"sub[-_]?0*([0-9]+)", re.IGNORECASE)
_RE_TASK_GENERIC = re.compile(r"task[-_]?([A-Za-z0-9]+)", re.IGNORECASE)
_RE_RUN_GENERIC = re.compile(r"run[-_]?0*([0-9]+)", re.IGNORECASE)


def _parse_key_from_path(path: Path) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    s = str(path)
    sub = _RE_SUB_GENERIC.search(s)
    task = _RE_TASK_GENERIC.search(s)
    run = _RE_RUN_GENERIC.search(s)
    sub_id = str(int(sub.group(1))) if sub else None
    task_id = (task.group(1).lower() if task else None)
    run_id = str(int(run.group(1))) if run else None
    return sub_id, task_id, run_id


def _parse_sr_from_path(path: Path) -> Tuple[Optional[str], Optional[str]]:
    s = str(path)
    sub = _RE_SUB_GENERIC.search(s)
    run = _RE_RUN_GENERIC.search(s)
    sub_id = str(int(sub.group(1))) if sub else None
    run_id = str(int(run.group(1))) if run else None
    return sub_id, run_id


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


class PairedAlignedDataset(Dataset):
    """
    Builds aligned EEG–fMRI windows for the same subject/task/run and time interval.

    Each item returns:
      - eeg_window: (C, P=window_sec, S=target_fs)  [z-scored per channel]
      - fmri_window: (T=round(window_sec/TR), V)    [z-scored per voxel]
      - meta dict
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
    ) -> None:
        self.eeg_root = eeg_root
        self.fmri_root = fmri_root
        self.a424_label_nii = a424_label_nii
        self.window_sec = int(window_sec)
        self.original_fs = original_fs
        self.target_fs = target_fs
        self.tr = tr
        self.channels_limit = channels_limit
        self.fmri_norm = fmri_norm
        self.stride_sec = int(stride_sec) if stride_sec is not None else int(window_sec)
        self.device = torch.device(device)

        eeg_files = find_eeg_files(eeg_root)
        fmri_files = find_bold_files(fmri_root)

        # Build maps RELAXED by (sub,run) to tolerate task naming mismatch
        key_to_eeg: Dict[Tuple[str, str], Path] = {}
        key_to_fmri: Dict[Tuple[str, str], Path] = {}
        for p in eeg_files:
            k = _parse_sr_from_path(p)
            if all(k):
                key_to_eeg[(k[0], k[1])] = p
        for p in fmri_files:
            k = _parse_sr_from_path(p)
            if all(k):
                key_to_fmri[(k[0], k[1])] = p

        # Intersect keys and create aligned window index
        self.index: List[Tuple[Tuple[str, str], Path, Path, int]] = []
        inter_keys = sorted(set(key_to_eeg.keys()) & set(key_to_fmri.keys()))
        for key in inter_keys:
            eeg_p = key_to_eeg[key]
            fmri_p = key_to_fmri[key]
            dur_eeg = _duration_eeg_seconds(eeg_p, original_fs)
            dur_fmri = _duration_fmri_seconds(fmri_p, tr)
            max_start = int(max(0, min(dur_eeg, dur_fmri) - window_sec))
            for start_sec in range(0, max_start + 1, self.stride_sec):
                self.index.append((key, eeg_p, fmri_p, start_sec))

        if len(self.index) == 0:
            # Debug: show a few parsed keys to help adapt regex if needed
            print("[PairedAlignedDataset] No intersections. Example EEG keys (sub,run):",
                  list(key_to_eeg.keys())[:5])
            print("[PairedAlignedDataset] Example fMRI keys (sub,run):",
                  list(key_to_fmri.keys())[:5])

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        (sub, run), eeg_path, fmri_path, start_sec = self.index[idx]

        # EEG load -> downsample -> patchify -> slice aligned window
        mat = sio.loadmat(str(eeg_path))
        for key in ['data_reref', 'eeg', 'data', 'EEG', 'eegdata']:
            if key in mat:
                eeg_arr = mat[key]
                break
        else:
            raise KeyError(f"No EEG array in {eeg_path}")
        if eeg_arr.shape[0] > eeg_arr.shape[1]:
            eeg_arr = eeg_arr.T
        eeg_ds = downsample_eeg(eeg_arr, self.original_fs, self.target_fs)
        if eeg_ds.shape[0] > self.channels_limit:
            eeg_ds = eeg_ds[:self.channels_limit]
        start_patch = int(round(start_sec))
        end_patch = start_patch + self.window_sec
        total_patches = eeg_ds.shape[1] // self.target_fs
        if end_patch > total_patches:
            start_patch = max(0, total_patches - self.window_sec)
            end_patch = start_patch + self.window_sec
        eeg_window = eeg_ds[:, start_patch * self.target_fs:end_patch * self.target_fs]
        eeg_patches = eeg_window.reshape(eeg_window.shape[0], self.window_sec, self.target_fs)
        x_eeg = torch.from_numpy(eeg_patches).unsqueeze(0).float().to(self.device)
        x_eeg = zscore_eeg_channelwise(x_eeg).squeeze(0).cpu()  # (C,P,S)

        # fMRI load -> A424 time series -> slice aligned window -> normalize
        parcel_ts, _, _, _ = extract_a424_from_native(str(fmri_path), str(self.a424_label_nii))
        start_vol = int(round(start_sec / self.tr))
        win_T = int(round(self.window_sec / self.tr))
        if start_vol + win_T > parcel_ts.shape[0]:
            start_vol = max(0, parcel_ts.shape[0] - win_T)
        fmri_window = parcel_ts[start_vol:start_vol + win_T, :].astype(np.float32)
        x_fmri = torch.from_numpy(fmri_window)
        x_fmri = normalize_fmri_time_series(x_fmri.unsqueeze(0), mode=self.fmri_norm).squeeze(0).cpu()

        return {
            'eeg_window': x_eeg,           # (C,P,S) z-scored per channel
            'fmri_window': x_fmri,         # (T,V)   z-scored per voxel
            'meta': {
                'subject': sub, 'run': run,
                'start_sec': start_sec,
                'eeg_path': str(eeg_path), 'fmri_path': str(fmri_path),
            }
        }


def collate_paired(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    eeg = torch.stack([b['eeg_window'] for b in batch], dim=0)
    fmri = torch.stack([b['fmri_window'] for b in batch], dim=0)
    metas = [b['meta'] for b in batch]
    return {'eeg_window': eeg, 'fmri_window': fmri, 'meta': metas}


def main() -> None:
    parser = argparse.ArgumentParser(description='Quick sanity check: load aligned EEG/fMRI and plot small sample')
    parser.add_argument('--eeg_root', type=str, required=True, help='Path to Oddball EEG root (e.g., Oddball/ds116_eeg)')
    parser.add_argument('--fmri_root', type=str, required=True, help='Path to Oddball fMRI root (e.g., Oddball/ds000116)')
    parser.add_argument('--a424_label_nii', type=str, required=True, help='Path to A424 atlas NIfTI in BOLD native space')
    parser.add_argument('--window_sec', type=int, default=40)
    parser.add_argument('--original_fs', type=int, default=1000)
    parser.add_argument('--target_fs', type=int, default=200)
    parser.add_argument('--tr', type=float, default=2.0)
    parser.add_argument('--fmri_norm', type=str, default='zscore', choices=['zscore','psc','mad','none'])
    parser.add_argument('--channels_limit', type=int, default=34)
    parser.add_argument('--stride_sec', type=int, default=None)
    args = parser.parse_args()

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
    )

    if len(ds) == 0:
        print('No aligned samples found.')
        return

    sample = ds[0]
    eeg = sample['eeg_window']  # normalized (C,P,S)
    fmri = sample['fmri_window']  # normalized (T,V)
    meta = sample['meta']

    # Recompute RAW windows for before/after comparison
    eeg_path = Path(meta['eeg_path'])
    fmri_path = Path(meta['fmri_path'])
    start_sec = int(meta['start_sec'])

    # RAW EEG
    mat = sio.loadmat(str(eeg_path))
    for key in ['data_reref', 'eeg', 'data', 'EEG', 'eegdata']:
        if key in mat:
            eeg_arr = mat[key]
            break
    else:
        raise KeyError(f"No EEG array in {eeg_path}")
    if eeg_arr.shape[0] > eeg_arr.shape[1]:
        eeg_arr = eeg_arr.T
    eeg_ds_raw = downsample_eeg(eeg_arr, args.original_fs, args.target_fs)
    if eeg_ds_raw.shape[0] > args.channels_limit:
        eeg_ds_raw = eeg_ds_raw[:args.channels_limit]
    start_patch = int(round(start_sec))
    end_patch = start_patch + args.window_sec
    total_patches = eeg_ds_raw.shape[1] // args.target_fs
    if end_patch > total_patches:
        start_patch = max(0, total_patches - args.window_sec)
        end_patch = start_patch + args.window_sec
    eeg_window_raw = eeg_ds_raw[:, start_patch * args.target_fs:end_patch * args.target_fs]
    eeg_raw = torch.from_numpy(eeg_window_raw.reshape(eeg_window_raw.shape[0], args.window_sec, args.target_fs)).float()  # (C,P,S)
    eeg_norm = zscore_eeg_channelwise(eeg_raw.unsqueeze(0)).squeeze(0)

    # RAW fMRI
    parcel_ts, _, _, _ = extract_a424_from_native(str(fmri_path), str(args.a424_label_nii))
    start_vol = int(round(start_sec / args.tr))
    win_T = int(round(args.window_sec / args.tr))
    if start_vol + win_T > parcel_ts.shape[0]:
        start_vol = max(0, parcel_ts.shape[0] - win_T)
    fmri_window_raw = parcel_ts[start_vol:start_vol + win_T, :].astype(np.float32)
    fmri_raw = torch.from_numpy(fmri_window_raw)  # (T,V)
    fmri_normd = normalize_fmri_time_series(fmri_raw.unsqueeze(0), mode=args.fmri_norm).squeeze(0)

    # Report shapes and before/after stats
    print(f"EEG shape (C,P,S): raw={tuple(eeg_raw.shape)} norm={tuple(eeg_norm.shape)}")
    print(f"EEG stats raw:   min={float(eeg_raw.min()):.4f} max={float(eeg_raw.max()):.4f} "
          f"mean={float(eeg_raw.mean()):.4f} std={float(eeg_raw.std(unbiased=False)):.4f}")
    print(f"EEG stats zsc:   min={float(eeg_norm.min()):.4f} max={float(eeg_norm.max()):.4f} "
          f"mean={float(eeg_norm.mean()):.4f} std={float(eeg_norm.std(unbiased=False)):.4f}")
    print(f"EEG raw sample [ch0, first 10 pts]: {eeg_raw[0].reshape(-1)[:10]}")
    print(f"EEG zsc sample [ch0, first 10 pts]: {eeg_norm[0].reshape(-1)[:10]}")

    print(f"fMRI shape (T,V): raw={tuple(fmri_raw.shape)} norm={tuple(fmri_normd.shape)}")
    print(f"fMRI stats raw:   min={float(fmri_raw.min()):.4f} max={float(fmri_raw.max()):.4f} "
          f"mean={float(fmri_raw.mean()):.4f} std={float(fmri_raw.std(unbiased=False)):.4f}")
    print(f"fMRI stats zsc:   min={float(fmri_normd.min()):.4f} max={float(fmri_normd.max()):.4f} "
          f"mean={float(fmri_normd.mean()):.4f} std={float(fmri_normd.std(unbiased=False)):.4f}")
    print(f"fMRI raw sample [t0..t1, ROI0]: {fmri_raw[:2, 0]}")
    print(f"fMRI zsc sample [t0..t1, ROI0]: {fmri_normd[:2, 0]}")

    # Prepare small plot: 10 EEG channels and 10 fMRI ROIs
    num_eeg = min(10, eeg.shape[0])
    num_fmri = min(10, fmri.shape[1])

    # EEG time axis in seconds (show first 10 seconds)
    eeg_total_pts = eeg.shape[1] * eeg.shape[2]
    eeg_pts_10s = min(int(10 * args.target_fs), eeg_total_pts)
    eeg_time = torch.arange(eeg_pts_10s).float() / float(args.target_fs)
    eeg_series = eeg[:num_eeg].reshape(num_eeg, -1)[:, :eeg_pts_10s]

    # fMRI time axis in seconds (show first 10 seconds)
    fmri_pts_10s = min(int(round(10.0 / float(args.tr))), fmri.shape[0])
    fmri_time = (torch.arange(fmri_pts_10s).float() * float(args.tr))
    fmri_series = fmri[:fmri_pts_10s, :num_fmri].transpose(0, 1)  # (num_fmri, T10)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

    # EEG plot (z-scored)
    for i in range(num_eeg):
        axes[0].plot(eeg_time.numpy(), eeg_series[i].numpy(), label=f'ch{i+1}', linewidth=0.8)
    axes[0].set_title(f'EEG (first {num_eeg} channels, z-score per channel)')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Z-scored amplitude')
    axes[0].grid(alpha=0.3)
    if num_eeg <= 10:
        axes[0].legend(loc='upper right', fontsize=8)

    # fMRI plot (z-scored)
    for i in range(num_fmri):
        axes[1].plot(fmri_time.numpy(), fmri_series[i].numpy(), label=f'ROI{i+1}', linewidth=0.8)
    axes[1].set_title(f'fMRI (first {num_fmri} ROIs, z-score per voxel)')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Z-scored signal')
    axes[1].grid(alpha=0.3)
    if num_fmri <= 10:
        axes[1].legend(loc='upper right', fontsize=8)

    plt.show()


if __name__ == '__main__':
    main()
