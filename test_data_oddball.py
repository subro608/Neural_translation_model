#!/usr/bin/env python3
"""
Fixed-configuration test harness for data_oddball.py

- Uses your exact train/test paths and fixed subject splits (no argparse).
- Confirms task-aware pairing (subject, task, run).
- Summarizes coverage: expect up to 3 runs for each of 2 tasks per subject.
- Probes dataset items for shape & normalization and runs a DataLoader batch test.

Place this file next to data_oddball.py and run:
    python test_data_oddball_fixed.py
"""

from __future__ import annotations
from pathlib import Path
from collections import defaultdict
from typing import Iterable, List, Optional, Tuple

import numpy as np
import torch

import data_oddball as D  # your module


# =========================
# CONFIG (hardcoded)
# =========================
TRAIN_CFG = {
    "eeg_root": "D:/Neuroinformatics_research_2025/Oddball/ds116_eeg",
    "fmri_root": "D:/Neuroinformatics_research_2025/Oddball/ds000116",
    "a424_label_nii": "D:/Neuroinformatics_research_2025/BrainLM/A424_resampled_to_bold.nii.gz",
    "device": "cuda",
    "seed": 42,
    "window_sec": 40,
    "stride_sec": 10,
    "original_fs": 1000,
    "target_fs": 200,
    "tr": 2.0,
    "channels_limit": 34,
    "fmri_norm": "zscore",
    "batch_size": 16,
    # fixed subject splits
    "train_subjects": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    "val_subjects":   [16],
    "test_subjects":  [17],
}

# You can switch to TEST_CFG if you want, but for dataset checks these are identical fields.
TEST_CFG = {
    "eeg_root": "D:/Neuroinformatics_research_2025/Oddball/ds116_eeg",
    "fmri_root": "D:/Neuroinformatics_research_2025/Oddball/ds000116",
    "a424_label_nii": "D:/Neuroinformatics_research_2025/BrainLM/A424_resampled_to_bold.nii.gz",
    "device": "cuda",
    "seed": 42,
    "window_sec": 40,
    "stride_sec": 10,
    "original_fs": 1000,
    "target_fs": 200,
    "tr": 2.0,
    "channels_limit": 34,
    "fmri_norm": "zscore",
    "batch_size": 8,
    # reusing train/val/test subject ids
    "train_subjects": TRAIN_CFG["train_subjects"],
    "val_subjects": TRAIN_CFG["val_subjects"],
    "test_subjects": TRAIN_CFG["test_subjects"],
}


# =========================
# Helpers
# =========================
def _pretty_counts(keys: List[Tuple[str, str, str]]) -> None:
    """
    keys are (sub, task, run) strings.
    """
    by_sub = defaultdict(list)
    for sub, task, run in keys:
        by_sub[sub].append((task, run))

    print("\n=== Intersection summary (subject → tasks → runs) ===")
    for sub in sorted(by_sub, key=lambda s: int(s)):
        tasks = defaultdict(list)
        for task, run in by_sub[sub]:
            tasks[task].append(run)
        print(f"subject {sub}:")
        for task in sorted(tasks):
            runs_sorted = sorted(tasks[task], key=lambda r: int(r))
            print(f"  {task}: runs {runs_sorted} (count={len(runs_sorted)})")
    print("=====================================================\n")


def _check_expected_2x3(keys: List[Tuple[str, str, str]], expect_task_tokens=("task001", "task002")) -> None:
    """
    Warn (but don't fail) if a subject doesn't have up to 3 runs for both tasks.
    """
    print("Checking presence of up to 3 runs × 2 tasks per subject...")
    by_subtask = defaultdict(set)
    for sub, task, run in keys:
        by_subtask[(sub, task)].add(run)

    by_sub = defaultdict(set)
    for sub, task, run in keys:
        by_sub[sub].add(task)

    for sub in sorted(by_sub, key=lambda s: int(s)):
        for task in expect_task_tokens:
            runs = sorted(by_subtask.get((sub, task), set()), key=lambda r: int(r))
            if len(runs) == 0:
                print(f"  [WARN] subject {sub}: missing {task}")
            elif len(runs) < 3:
                print(f"  [WARN] subject {sub}: {task} has only runs {runs} (expected up to ['1','2','3'])")
    print("Done.\n")


def _approx_zero_mean_unit_std(x: torch.Tensor, dim: int, tol_mean=0.15, tol_std_low=0.80, tol_std_high=1.20):
    """
    Check mean≈0, std≈1 along a dimension. Returns (ok, max_abs_mean, max_dev_from_1).
    """
    mean = x.mean(dim=dim)
    std = x.std(dim=dim, unbiased=False)
    max_abs_mean = float(mean.abs().max().cpu())
    max_dev_from_1 = float((std - 1.0).abs().max().cpu())
    ok = (max_abs_mean <= tol_mean) and (tol_std_low <= float(std.min()) <= float(std.max()) <= tol_std_high)
    return ok, max_abs_mean, max_dev_from_1


def _build_dataset_for_keys(cfg, include_sr_keys: Tuple[Tuple[str, str, str], ...]) -> D.PairedAlignedDataset:
    ds = D.PairedAlignedDataset(
        eeg_root=Path(cfg["eeg_root"]),
        fmri_root=Path(cfg["fmri_root"]),
        a424_label_nii=Path(cfg["a424_label_nii"]),
        window_sec=cfg["window_sec"],
        original_fs=cfg["original_fs"],
        target_fs=cfg["target_fs"],
        tr=cfg["tr"],
        channels_limit=cfg["channels_limit"],
        fmri_norm=cfg["fmri_norm"],
        stride_sec=cfg["stride_sec"],
        device="cpu",
        include_sr_keys=include_sr_keys,   # filter by exact (sub, task, run) triples
    )
    return ds


# =========================
# Tests
# =========================
def test_intersection(cfg) -> List[Tuple[str, str, str]]:
    keys = D.collect_common_sr_keys(Path(cfg["eeg_root"]), Path(cfg["fmri_root"]))
    print(f"Found {len(keys)} intersecting (subject, task, run) keys.")
    if len(keys) == 0:
        print("[ERROR] No intersections. Check your folder paths and name patterns.")
        return []
    _pretty_counts(keys)
    _check_expected_2x3(keys)
    return keys


def test_split(cfg, split_name: str, subjects: Iterable[int]) -> None:
    print(f"\n=== Split: {split_name} (subjects={list(subjects)}) ===")
    # Use fixed_subject_keys to get the exact (sub, task, run) triples for this split
    train_keys, val_keys, test_keys = D.fixed_subject_keys(
        eeg_root=Path(cfg["eeg_root"]),
        fmri_root=Path(cfg["fmri_root"]),
        train_subjects=TRAIN_CFG["train_subjects"],
        val_subjects=TRAIN_CFG["val_subjects"],
        test_subjects=TRAIN_CFG["test_subjects"],
    )
    if split_name == "train":
        keys = train_keys
    elif split_name == "val":
        keys = val_keys
    else:
        keys = test_keys

    print(f"  Keys in split: {len(keys)}")
    if len(keys) == 0:
        print(f"  [WARN] No keys for {split_name} — skipping dataset test.")
        return

    # Summarize coverage for this split
    _pretty_counts(list(keys))
    _check_expected_2x3(list(keys))

    # Build dataset constrained to these keys
    ds = _build_dataset_for_keys(cfg, include_sr_keys=keys)
    print(f"  PairedAlignedDataset size: {len(ds)} windows (window_sec={cfg['window_sec']}, stride_sec={cfg['stride_sec']}).")
    if len(ds) == 0:
        print("  [WARN] Dataset is empty for this split. Check atlas path and naming.")
        return

    # Probe a couple of samples
    probe_n = min(2, len(ds))
    for i in range(probe_n):
        sample = ds[i]
        x_eeg = sample["eeg_window"]   # (C,P,S)
        x_fmri = sample["fmri_window"] # (T,V)
        meta = sample["meta"]
        C, P, S = x_eeg.shape
        T, V = x_fmri.shape
        expected_T = int(round(cfg["window_sec"] / cfg["tr"]))
        print(f"  [{i}] meta={meta}")
        print(f"      EEG shape (C,P,S)=({C},{P},{S})  expected=({cfg['channels_limit']},{cfg['window_sec']},{cfg['target_fs']})")
        print(f"      fMRI shape (T,V)=({T},{V})       expected=({expected_T}, 424)")

        # shape checks
        assert P == cfg["window_sec"], "EEG P != window_sec"
        assert S == cfg["target_fs"], "EEG S != target_fs"
        assert T == expected_T, "fMRI T != window_sec/TR"

        # normalization checks
        eeg_ok, eeg_m, eeg_s = _approx_zero_mean_unit_std(x_eeg, dim=2)
        fmri_ok, fmri_m, fmri_s = _approx_zero_mean_unit_std(x_fmri, dim=0)
        print(f"      EEG z-score ok={eeg_ok} (max|mean|={eeg_m:.3f}, max|std-1|={eeg_s:.3f})")
        print(f"      fMRI z-score ok={fmri_ok} (max|mean|={fmri_m:.3f}, max|std-1|={fmri_s:.3f})")

    # DataLoader + collate check
    from torch.utils.data import DataLoader
    dl = DataLoader(
        ds, batch_size=min(cfg.get("batch_size", 8), max(1, len(ds))),
        shuffle=False, num_workers=0, collate_fn=D.collate_paired, pin_memory=False
    )
    batch = next(iter(dl))
    eeg = batch["eeg_window"]   # (B,C,P,S)
    fmri = batch["fmri_window"] # (B,T,V)
    print("  DataLoader batch shapes:", tuple(eeg.shape), "(EEG)", tuple(fmri.shape), "(fMRI)")
    assert eeg.shape[0] == fmri.shape[0], "Batch size mismatch between EEG and fMRI"
    print("  OK: collate_paired returns aligned (B,C,P,S) and (B,T,V).")


def quick_padding_probe() -> None:
    # minimal probe for pad_timepoints_for_brainlm_torch
    x = torch.zeros(2, 53, 424)  # (B,T=53,V)
    xp = D.pad_timepoints_for_brainlm_torch(x, patch_size=20)
    print("\nPadding probe: input T=53 -> padded T =", xp.shape[1], "(should be 60)")


def main():
    torch.set_num_threads(1)
    np.set_printoptions(precision=3, suppress=True)

    cfg = TRAIN_CFG  # use TRAIN paths/settings for dataset checks

    print("=== Fixed config ===")
    print("EEG root      :", cfg["eeg_root"])
    print("fMRI root     :", cfg["fmri_root"])
    print("A424 atlas NII:", cfg["a424_label_nii"])
    print("window_sec / stride_sec:", cfg["window_sec"], "/", cfg["stride_sec"])
    print("target_fs / TR:", cfg["target_fs"], "/", cfg["tr"])
    print("channels_limit:", cfg["channels_limit"])
    print("splits: train/val/test =", len(cfg["train_subjects"]), len(cfg["val_subjects"]), len(cfg["test_subjects"]))

    # 1) global intersection check
    keys_all = test_intersection(cfg)
    if not keys_all:
        return

    # 2) split-wise dataset checks
    test_split(cfg, "train", cfg["train_subjects"])
    test_split(cfg, "val",   cfg["val_subjects"])
    test_split(cfg, "test",  cfg["test_subjects"])

    # 3) small padding probe
    quick_padding_probe()

    print("\nAll tests completed. If you saw WARNs about missing tasks/runs,")
    print("double-check file naming or the auditory/visual ↔ task001/task002 mapping in data_oddball._canonical_task_token().")


if __name__ == "__main__":
    main()
