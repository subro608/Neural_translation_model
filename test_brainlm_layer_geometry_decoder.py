#!/usr/bin/env python3
"""
Decoder-based ROI×ROI (time-dependent) correlation from BrainLM.

What it does
------------
- Loads ONE full fMRI session (A424).
- Pads time to a multiple of 20 TR (BrainLM’s patch).
- Runs BrainLM **decoder** to reconstruct time series (no masking).
- Computes ROI×ROI Pearson correlation over time from the decoder outputs.
- Saves PNGs for:
    * Full-session correlation per ablation
    * Per-patch (20 TR) correlation per ablation

Ablations
---------
- sig_xyz  : signal + xyz
- sig_only : signal + zeros(xyz)
- xyz_only : zeros(signal) + xyz

Notes
-----
- We call the *full* model forward (not just encoder) and take `out.logits`.
- Handles both cases where `out.logits` is a tensor or a tuple (logits, latent).
- CLS is irrelevant here (decoder output is time-by-ROI).
- Uses only matplotlib; one plot per figure; default colors.

Outputs
-------
./debug_out/decoder_temporal_roi_corr/
  sig_xyz/
    full_session_corr.png
    patch_000_T0-19_corr.png
    patch_001_T20-39_corr.png
    ...
  sig_only/...
  xyz_only/...

"""

from __future__ import annotations
import os, sys, re, glob, math, platform, json
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -----------------------
# CONFIG — edit as needed
# -----------------------
EEG_ROOT          = Path(r"D:\Neuroinformatics_research_2025\Oddball\ds116_eeg")
FMRI_ROOT         = Path(r"D:\Neuroinformatics_research_2025\Oddball\ds000116")
A424_LABEL_NII    = Path(r"D:\Neuroinformatics_research_2025\BrainLM\A424_resampled_to_bold.nii.gz")
BRAINLM_MODEL_DIR = Path(r"D:\Neuroinformatics_research_2025\MNI_templates\BrainLM\pretrained_models\2023-06-06-22_15_00-checkpoint-1400")

CHOSEN_SUBJECT: Optional[int] = None
CHOSEN_TASK:    Optional[str] = None
CHOSEN_RUN:     Optional[int] = None

TR = 2.0
PATCH_SIZE_TR = 20
FIG_DPI = 115

OUT_DIR = Path("./debug_out/decoder_temporal_roi_corr")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED   = 42

# -----------------------
# Repo pathing (match your project)
# -----------------------
THIS_DIR    = Path(__file__).parent
REPO_ROOT   = THIS_DIR.parent
CBRAMOD_DIR = REPO_ROOT / "CBraMod"
BRAINLM_DIR = REPO_ROOT / "BrainLM"

sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(CBRAMOD_DIR))
sys.path.insert(0, str(THIS_DIR / "CBraMod"))

# Make brainlm_mae importable
for p in [BRAINLM_DIR, THIS_DIR / "BrainLM", Path(os.environ.get("BRAINLM_DIR", "")),
          Path(os.environ.get("BRAINLM_DIR", "")) / "brainlm_mae"]:
    if not p: continue
    p = Path(p)
    if (p / "brainlm_mae").is_dir():
        sys.path.insert(0, str(p)); break
    if p.name == "brainlm_mae" and p.is_dir():
        sys.path.insert(0, str(p.parent)); break

# Exact class you already use (ensures consistent embeddings/encoder behavior)
from train_fmri_selfattn_permodule_updated_29aug_reverse_modelsave_patchmasked import FrozenBrainLM  # type: ignore

# Data helpers
from data_oddball import (  # type: ignore
    collect_common_sr_keys,
    load_fmri_parcels_cached,
    pad_timepoints_for_brainlm_torch,
    load_a424_coords,
)

# -----------------------
# Small utils
# -----------------------
def seed_all(seed:int):
    np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def _parse_key_from_path(path: Path):
    _RE_SUB  = re.compile(r"sub[-_]?0*([0-9]+)", re.I)
    _RE_TASK = re.compile(r"task[-_]?([A-Za-z0-9]+)", re.I)
    _RE_RUN  = re.compile(r"run[-_]?0*([0-9]+)", re.I)
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

def _save_heatmap(mat: np.ndarray, title: str, out_path: Path):
    plt.figure(figsize=(6.2, 5.2))
    im = plt.imshow(mat, aspect='auto')
    plt.colorbar(im)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=FIG_DPI)
    plt.close()

# -----------------------
# Decoder helper
# -----------------------
@torch.no_grad()
def decode_time_series_via_brainlm(
    brainlm: FrozenBrainLM,
    signal_vectors: torch.Tensor,   # (B,V,Tp)
    xyz_vectors: torch.Tensor,      # (B,V,3)
    patch_size_tr: int = 20,
) -> torch.Tensor:
    """
    Returns reconstructed time series from the BrainLM decoder:
      (B, Tp, V) where Tp = #padded TR (multiple of patch_size_tr)
    Cropping to original T should be done by the caller.
    """
    # Call the full model forward so we hit the decoder
    out = brainlm.model(
        signal_vectors=signal_vectors,
        xyz_vectors=xyz_vectors,
        output_hidden_states=False,
        return_dict=True,
    )
    logits = out.logits
    # Some BrainLM variants pack (logits, latent) in logits
    if isinstance(logits, (tuple, list)) and len(logits) >= 1:
        logits = logits[0]

    # Expecting (B, V, Ttok, patch_size)
    if logits.dim() != 4:
        raise RuntimeError(f"Unexpected decoder logits shape: {tuple(logits.shape)} (want B×V×Ttok×{patch_size_tr})")

    B, V, Ttok, P = logits.shape
    if P != patch_size_tr:
        # If a different patch size is configured, we still flatten time
        pass

    # Flatten time and permute to (B, Tp, V)
    recon = logits.reshape(B, V, Ttok * P).permute(0, 2, 1).contiguous()
    return recon  # (B, Tp, V)

# -----------------------
# Main
# -----------------------
def main():
    print(f"[env] torch={torch.__version__} | device={DEVICE} | python={sys.version.split()[0]} | {platform.platform()}")
    seed_all(SEED)
    ensure_dir(OUT_DIR)

    # 1) Pick a session present in both trees
    keys = collect_common_sr_keys(EEG_ROOT, FMRI_ROOT)
    if not keys: raise RuntimeError("No intersecting (subject, task, run).")
    def _matches(k):
        return ((CHOSEN_SUBJECT is None or int(k[0]) == int(CHOSEN_SUBJECT)) and
                (CHOSEN_TASK    is None or k[1] == CHOSEN_TASK) and
                (CHOSEN_RUN     is None or int(k[2]) == int(CHOSEN_RUN)))
    key = next((k for k in keys if _matches(k)), keys[0])
    sub, task, run = key
    print(f"[select] Using session: sub={sub} task={task} run={run}")

    # 2) Load FULL fMRI session (A424 parcels)
    fmri_path = _resolve_fmri_path_for_key(FMRI_ROOT, key)
    if fmri_path is None:
        raise RuntimeError(f"Could not resolve fMRI path for key={key}")
    cache_dir = Path(os.environ.get("ODDBALL_CACHE_DIR", "~/.cache/oddball_eegfmri")).expanduser()
    cache_dir.mkdir(parents=True, exist_ok=True)

    fmri_TV = load_fmri_parcels_cached(fmri_path, str(A424_LABEL_NII), cache_dir=cache_dir).astype(np.float32, copy=False)  # (T_full,V)
    T_full, V = fmri_TV.shape
    print(f"[session] full fMRI: (T={T_full}, V={V})")

    # 3) Prepare BrainLM inputs: pad time to multiple of PATCH_SIZE_TR
    x_BTV = torch.from_numpy(fmri_TV).unsqueeze(0).to(DEVICE)                       # (1, T_full, V)
    x_pad = pad_timepoints_for_brainlm_torch(x_BTV, patch_size=PATCH_SIZE_TR)       # (1, Tp, V) -> Tp multiple of 20
    signal_vectors = x_pad.permute(0, 2, 1).contiguous()                            # (1, V, Tp)
    Tp = int(x_pad.shape[1])
    print(f"[prep] padded timepoints Tp={Tp} (was T_full={T_full})")

    # 4) xyz coords (normalized)
    xyz = torch.zeros(1, V, 3, device=DEVICE)
    try:
        for cand in [
            THIS_DIR / 'BrainLM' / 'resources' / 'atlases' / 'A424_Coordinates.dat',
            THIS_DIR / 'resources' / 'atlases' / 'A424_Coordinates.dat',
            REPO_ROOT / 'BrainLM' / 'toolkit' / 'atlases' / 'A424_Coordinates.dat',
        ]:
            if cand.exists():
                arr = load_a424_coords(cand)
                arr = arr / (float(np.max(np.abs(arr))) or 1.0)
                xyz = torch.from_numpy(arr.astype(np.float32)).to(DEVICE).unsqueeze(0)
                print(f"[coords] loaded A424 coords from: {cand}")
                break
    except Exception as e:
        print(f"[coords] WARN {e} (using zeros)")

    # 5) Init frozen BrainLM (same class you use in training)
    brainlm = FrozenBrainLM(BRAINLM_MODEL_DIR, torch.device(DEVICE), verbose=False)

    # 6) Ablations
    conditions = {
        "sig_xyz":  {"use_signal": True,  "use_xyz": True},
        "sig_only": {"use_signal": True,  "use_xyz": False},
        "xyz_only": {"use_signal": False, "use_xyz": True},
    }

    # 7) Decode → recon time series (B,Tp,V), crop to T_full, compute ROI×ROI corr
    n_patches = int(math.ceil(T_full / float(PATCH_SIZE_TR)))
    print(f"[run] PATCH_SIZE_TR={PATCH_SIZE_TR} -> n_patches={n_patches}")

    for tag, cfg in conditions.items():
        out_tag_dir = OUT_DIR / tag
        ensure_dir(out_tag_dir)

        sig_in = signal_vectors.clone() if cfg["use_signal"] else torch.zeros_like(signal_vectors)
        xyz_in = xyz.clone()             if cfg["use_xyz"]    else torch.zeros_like(xyz)

        with torch.no_grad():
            recon_BTV = decode_time_series_via_brainlm(
                brainlm, signal_vectors=sig_in, xyz_vectors=xyz_in, patch_size_tr=PATCH_SIZE_TR
            )  # (1, Tp, V)

        # Crop back to original length
        recon_TV = recon_BTV[0, :T_full, :].detach().cpu().numpy()  # (T_full, V)

        # ---- Full-session ROI×ROI correlation over time
        S_full = np.corrcoef(recon_TV.T)  # (V,V)
        _save_heatmap(S_full, f"Decoder ROI×ROI corr — full session — {tag}",
                      out_tag_dir / "full_session_corr.png")

        # ---- Per-patch (20 TR) ROI×ROI correlation over time
        for pidx in range(n_patches):
            t0 = pidx * PATCH_SIZE_TR
            t1 = min((pidx + 1) * PATCH_SIZE_TR, T_full)
            if (t1 - t0) < 2:
                continue
            patch_tag = f"patch_{pidx:03d}_T{t0}-{t1-1}"

            seg = recon_TV[t0:t1, :]  # (seg_len, V)
            S = np.corrcoef(seg.T)    # (V,V)

            patch_dir = out_tag_dir   # keep flat under tag; change here if you want subfolders
            ensure_dir(patch_dir)
            _save_heatmap(S, f"Decoder ROI×ROI corr — {tag} — {patch_tag}",
                          patch_dir / f"{patch_tag}_corr.png")

        print(f"[save] {tag}: full_session_corr + {n_patches} patch PNGs in {out_tag_dir}")

    print(f"[done] outputs in: {OUT_DIR.resolve()}")

if __name__ == "__main__":
    main()
