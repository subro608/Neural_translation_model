#!/usr/bin/env python3
"""
Full-session patch-wise analysis:

For each 20-TR patch across the entire fMRI session:
  A) Pre-BrainLM temporal ROI×ROI (Pearson over TRs)  -> PNG
  B) BrainLM latent ablations:
       - sig_xyz  (signal+xyz)
       - sig_only (signal, xyz=0)
       - xyz_only (signal=0, xyz)
     For each layer: token×token (ROI×ROI) Pearson over embedding dim D -> PNG

No masking. No .npy/.npz artifacts.

Directory structure:
  ./debug_out/fullsession_patches/
    pre_brainlm_temporal/
      patch_000_T0-19_corr.png
      patch_001_T20-39_corr.png
      ...
    latents_ablation/
      sig_xyz/patch_000_T0-19/layer_00_corr.png
      sig_xyz/patch_000_T0-19/layer_01_corr.png
      ...
      sig_only/patch_000_T0-19/...
      xyz_only/patch_000_T0-19/...
"""

from __future__ import annotations
import os, sys, re, glob, math, platform
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

OUT_DIR = Path("./debug_out/fullsession_patches")
FIG_DPI = 115

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

# Import the exact extractor from your training code (keeps behavior identical)
from train_fmri_selfattn_permodule_updated_29aug_reverse_modelsave_patchmasked import FrozenBrainLM  # type: ignore

# Data helpers
from data_oddball import (  # type: ignore
    collect_common_sr_keys,
    load_fmri_parcels_cached,
    pad_timepoints_for_brainlm_torch,
    load_a424_coords,
)

# -----------------------
# Utils
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

    # 3) xyz coords (normalized to [-1,1])
    xyz = torch.zeros(1, V, 3, device=DEVICE)
    try:
        for cand in [
            THIS_DIR / 'BrainLM' / 'resources' / 'atlases' / 'A424_Coordinates.dat',
            THIS_DIR / 'resources' / 'atlases' / 'A424_Coordinates.dat',
            REPO_ROOT / 'BrainLM' / 'toolkit' / 'atlases' / 'A424_Coordinates.dat',
        ]:
            if cand.exists():
                from data_oddball import load_a424_coords  # type: ignore
                arr = load_a424_coords(cand)
                arr = arr / (float(np.max(np.abs(arr))) or 1.0)
                xyz = torch.from_numpy(arr.astype(np.float32)).to(DEVICE).unsqueeze(0)
                print(f"[coords] loaded A424 coords from: {cand}")
                break
    except Exception as e:
        print(f"[coords] WARN {e} (using zeros)")

    # 4) Init frozen BrainLM (same extractor as training)
    brainlm = FrozenBrainLM(BRAINLM_MODEL_DIR, torch.device(DEVICE), verbose=False)

    # 5) Iterate patches across ENTIRE session
    pre_dir = OUT_DIR / "pre_brainlm_temporal"
    ensure_dir(pre_dir)

    ablate_root = OUT_DIR / "latents_ablation"
    ensure_dir(ablate_root)

    conditions = {
        "sig_xyz":  {"use_signal": True,  "use_xyz": True},
        "sig_only": {"use_signal": True,  "use_xyz": False},
        "xyz_only": {"use_signal": False, "use_xyz": True},
    }

    n_patches = int(math.ceil(T_full / float(PATCH_SIZE_TR)))
    print(f"[run] PATCH_SIZE_TR={PATCH_SIZE_TR} -> n_patches={n_patches}")

    for pidx in range(n_patches):
        t0 = pidx * PATCH_SIZE_TR
        t1 = min((pidx + 1) * PATCH_SIZE_TR, T_full)
        seg_len = t1 - t0
        if seg_len < 2:
            print(f"[skip] patch {pidx:03d} too short (len={seg_len})")
            continue

        patch_tag = f"patch_{pidx:03d}_T{t0}-{t1-1}"
        print(f"[patch] {patch_tag} (len={seg_len})")

        # ----- A) Pre-BrainLM temporal ROI×ROI (Pearson over TRs) -----
        win = fmri_TV[t0:t1, :]  # (seg_len, V)
        S_corr = np.corrcoef(win.T)  # (V,V)
        _save_heatmap(S_corr, f"Pre-BrainLM ROI×ROI corr — {patch_tag}",
                      pre_dir / f"{patch_tag}_corr.png")

        # If you ever want cosine across TRs, uncomment:
        # Yn = win.astype(np.float64) / (np.linalg.norm(win, axis=0, keepdims=True) + 1e-8)
        # S_cos = Yn.T @ Yn
        # _save_heatmap(S_cos, f"Pre-BrainLM ROI×ROI cosine — {patch_tag}",
        #               pre_dir / f"{patch_tag}_cosine.png")

        # ----- B) BrainLM latent ablations (per-layer ROI×ROI over D) -----
        # Prepare BrainLM input for THIS patch (pad to 20 if needed)
        x_BTV = torch.from_numpy(win).unsqueeze(0).to(DEVICE)                       # (1, seg_len, V)
        x_pad = pad_timepoints_for_brainlm_torch(x_BTV, patch_size=PATCH_SIZE_TR)   # (1, Tp, V) -> Tp=20
        signal_vectors = x_pad.permute(0, 2, 1).contiguous()                        # (1, V, Tp)

        for tag, cfg in conditions.items():
            cond_dir = ablate_root / tag / patch_tag
            ensure_dir(cond_dir)

            sig_in = signal_vectors if cfg["use_signal"] else torch.zeros_like(signal_vectors)
            xyz_in = xyz            if cfg["use_xyz"]    else torch.zeros_like(xyz)

            with torch.no_grad():
                hs_LBND = brainlm.extract_latents(signal_vectors=sig_in, xyz_vectors=xyz_in)  # (L,1,N,D)

            # Drop CLS if present → (L,1,T,D), where T = V*Ttok (here Ttok=1 for 20 TR)
            L, Bsz, N, D = hs_LBND.shape
            has_cls = (N % V) != 0
            hs_LBTD = hs_LBND[:, :, 1:, :] if has_cls else hs_LBND
            L, Bsz, T_tokens, D = hs_LBTD.shape

            if T_tokens != V:
                print(f"[warn][{patch_tag}/{tag}] tokens={T_tokens} != V={V} (has_cls={has_cls})")

            # Per-layer token×token (ROI×ROI) Pearson over embedding dim D
            for ell in range(L):
                X = hs_LBTD[ell, 0].detach().cpu().numpy()  # (T_tokens, D) ~ (V,D)
                S_tok = np.corrcoef(X, rowvar=True)         # (V,V)
                out_png = cond_dir / f"layer_{ell:02d}_corr.png"
                _save_heatmap(S_tok, f"{tag} — ttok×ttok corr — {patch_tag} — layer {ell}", out_png)

    print(f"[done] outputs in: {OUT_DIR.resolve()}")

if __name__ == "__main__":
    main()
