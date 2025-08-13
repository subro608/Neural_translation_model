#!/usr/bin/env python3
r"""
viz_brainlm_latents.py  (axis-stable visuals, no internal masking, CLS handling, ROI debug)

Consistent conventions across ALL plots:
  • ROI on X (columns)
  • Time on Y (rows) whenever time is present
  • Colorbar on all heatmaps

Saves:
  - input_time_by_roi.png
  - input_padded_time_by_roi.png
  - input_roi_temporal_std.png
  - input_time_mean_over_rois.png
  - input_value_hist.png
  - xyz_coords_3d.png
  - roi_time_layer{L}.png
  - roi_by_hidden_layer{L}.png
  - debug_zero_voxel_rois.txt
  - debug_lowvar_rois.txt
  - roi_voxel_counts.csv
  - roi_temporal_std.csv
  - paths_batch{idx}.json
  - (optional via DUMP_NPY) fmri_t_sample0.npy, fmri_pad_sample0.npy, latent_layer{L}_sample0.npy
"""

from __future__ import annotations
import json, sys, csv
from types import SimpleNamespace
from pathlib import Path
from typing import Optional, Tuple, Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# ---------- Optional: atlas voxel debug ----------
try:
    import nibabel as nib  # type: ignore
    _HAVE_NIB = True
except Exception:
    _HAVE_NIB = False

# ---------- Repo paths ----------
THIS_DIR = Path(__file__).parent
REPO_ROOT = THIS_DIR.parent
CBRAMOD_DIR = REPO_ROOT / "CBraMod"
BRAINLM_DIR = REPO_ROOT / "BrainLM"
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(BRAINLM_DIR))
sys.path.append(str(CBRAMOD_DIR))
sys.path.append(str(BRAINLM_DIR))

from data_oddball import (  # type: ignore
    PairedAlignedDataset,
    collate_paired,
    pad_timepoints_for_brainlm_torch,
    load_a424_coords,
)
from brainlm_mae.modeling_brainlm import BrainLMForPretraining  # type: ignore
from brainlm_mae.configuration_brainlm import BrainLMConfig     # type: ignore

# ---------- Hardcoded configuration ----------
EEG_ROOT = r"D:\Neuroinformatics_research_2025\Oddball\ds116_eeg"
FMRI_ROOT = r"D:\Neuroinformatics_research_2025\Oddball\ds000116"
A424_LABEL_NII = r"D:\Neuroinformatics_research_2025\BrainLM\A424_resampled_to_bold.nii.gz"
BRAINLM_MODEL_DIR = r"D:\Neuroinformatics_research_2025\MNI_templates\BrainLM\pretrained_models\2023-06-06-22_15_00-checkpoint-1400"
OUTPUT_DIR = r"D:\Neuroinformatics_research_2025\Multi_modal_NTM\translator_runs\viz_latents"

DEVICE = "cuda"
WINDOW_SEC = 40          # use 40s to get 1 time-chunk per ROI set
STRIDE_SEC = None
FMRI_NORM = "zscore"
BATCH_SIZE = 4
NUM_WORKERS = 0
LAYERS = [0, 2, 4]       # valid for this BrainLM (L=5)
BATCH_IDX = 0

# Advanced toggles
SWAP_ORDER   = False      # only set True if your BrainLM flattens [roi,chunk] not [chunk,roi]
DUMP_NPY     = False      # dump NumPy arrays for offline checks
KEEP_MASK    = False      # keep internal BrainLM masking (False disables masking)
CLS_AT_END   = False      # set True if the special token is appended (not prepended)
LOWVAR_THRESH = 1e-6      # temporal std threshold to flag low-variance ROIs
MARK_SUSPECT = True       # draw thin lines over suspect ROI columns (zero-voxel ∪ low-variance)

TR = 2.0                  # seconds per fMRI TR

# ---------- BrainLM wrapper (mask disabled by default) ----------
class FrozenBrainLM(nn.Module):
    def __init__(self, model_dir: Path, device: torch.device, keep_mask: bool = False) -> None:
        super().__init__()
        with open(model_dir / "config.json", "r") as f:
            cfg = json.load(f)
        self.model = BrainLMForPretraining(BrainLMConfig(**cfg))
        weights_path = model_dir / "pytorch_model.bin"
        ckpt = torch.load(str(weights_path), map_location=device)
        self.model.load_state_dict(ckpt, strict=False)

        if not keep_mask:
            for obj in [self.model,
                        getattr(self.model, "config", None),
                        getattr(self.model, "vit", None),
                        getattr(getattr(self.model, "vit", None), "embeddings", None)]:
                if obj is not None and hasattr(obj, "mask_ratio"):
                    try:
                        setattr(obj, "mask_ratio", 0.0)
                    except Exception:
                        pass

        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.to(device)
        self.device = device

    @torch.no_grad()
    def extract_latents(self, signal_vectors: torch.Tensor, xyz_vectors: torch.Tensor):
        """
        Inputs:
          signal_vectors: (B, V, Tp)
          xyz_vectors:    (B, V, 3)
        Returns:
          h_vis: (L, B, N_vis, H)   (may include a CLS/special token among N_vis)
        """
        vit = self.model.vit
        emb, mask, ids_restore = vit.embeddings(signal_vectors=signal_vectors, xyz_vectors=xyz_vectors, noise=None)
        enc = vit.encoder(hidden_states=emb, output_hidden_states=True, return_dict=True)
        hidden_layers = enc.hidden_states  # tuple of (B, N_vis, H)
        h_vis = torch.stack(list(hidden_layers), dim=0)  # (L,B,N_vis,H)
        return h_vis

# ---------- token utilities ----------
def maybe_strip_special_token(h_tokens: torch.Tensor, V: int, cls_at_end: bool = False):
    """
    h_tokens: (L,B,N,H)
    If N % V != 0 but (N-1) % V == 0, assume 1 special token (CLS) and strip it.
    """
    L, B, N, H = h_tokens.shape
    if N % V == 0:
        return h_tokens, N, False
    if (N - 1) % V == 0:
        print(f"[fix] Detected 1 special token (N={N}, V={V}). Stripping {'END' if cls_at_end else 'FRONT'} token.")
        if cls_at_end:
            h_tokens = h_tokens[:, :, :-1, :]
        else:
            h_tokens = h_tokens[:, :, 1:, :]
        return h_tokens, N - 1, True
    raise AssertionError(f"N={N} not divisible by V={V}, and (N-1) not divisible either — cannot reshape safely.")

def reshape_tokens(hL_full: torch.Tensor, V: int, swap_order: bool = False) -> torch.Tensor:
    """
    (B, N_full, H) -> (B, time_chunks, V, H), assuming default flatten [chunk, roi].
    If your model flattens [roi, chunk], set swap_order=True.
    """
    B, N_full, H = hL_full.shape
    assert V > 0 and N_full % V == 0, f"N_full={N_full} not divisible by V={V}"
    time_chunks = N_full // V
    if not swap_order:
        return hL_full.view(B, time_chunks, V, H).contiguous()
    else:
        return hL_full.view(B, V, time_chunks, H).permute(0, 2, 1, 3).contiguous()

# ---------- plotting helpers (axis-stable) ----------
def _add_colorbar(ax):
    mappable = ax.images[-1] if ax.images else None
    if mappable is not None:
        plt.colorbar(mappable, ax=ax, fraction=0.046, pad=0.04)

def save_fmri_input_heatmap(X: torch.Tensor, path: Path, title: str):
    """
    X: (T, V). We plot time (rows) × ROI (cols). ROI on X, time on Y.
    """
    A = X.detach().cpu().numpy()  # (T,V)
    fig, ax = plt.subplots(figsize=(12, 4))
    im = ax.imshow(A, aspect='auto', origin='upper')  # rows=time, cols=ROI
    ax.set_xlabel("ROI (V)")
    ax.set_ylabel("time (TR index)")
    ax.set_title(title)
    _add_colorbar(ax)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)

def overlay_vertical_markers(ax, cols: Optional[List[int]]):
    if not cols: return
    ymin, ymax = ax.get_ylim()
    for c in cols:
        ax.axvline(x=c, linewidth=0.6)

def save_fmri_input_heatmap_marked(X: torch.Tensor, path: Path, title: str, mark_cols: Optional[List[int]]):
    A = X.detach().cpu().numpy()  # (T,V)
    fig, ax = plt.subplots(figsize=(12, 4))
    im = ax.imshow(A, aspect='auto', origin='upper')
    overlay_vertical_markers(ax, mark_cols)
    ax.set_xlabel("ROI (V)")
    ax.set_ylabel("time (TR index)")
    ax.set_title(title)
    _add_colorbar(ax)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)

def save_roi_time_heatmap(x_btvh: torch.Tensor, out_path: Path, mark_cols: Optional[List[int]] = None):
    """
    x_btvh: (B, time_chunks, V, H)
    Plot time_chunks (rows) × ROI (cols). Value = ||latent|| over H, mean over batch.
    """
    with torch.no_grad():
        val = torch.linalg.vector_norm(x_btvh, ord=2, dim=-1).mean(dim=0)  # (time_chunks, V)
        A = val.detach().cpu().numpy()  # (time_chunks, V)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.imshow(A, aspect='auto', origin='upper')
    overlay_vertical_markers(ax, mark_cols)
    ax.set_xlabel("ROI (V)")
    ax.set_ylabel("time chunk")
    ax.set_title("Time-chunks × ROI (||latent|| over H, batch-avg)")
    _add_colorbar(ax)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)

def save_roi_by_hidden_heatmap(x_btvh: torch.Tensor, out_path: Path):
    """
    x_btvh: (B, time_chunks, V, H)
    Aggregate over batch+time → (V,H). Plot hidden (rows) × ROI (cols).
    """
    X = x_btvh.mean(dim=0).mean(dim=0)   # (V,H)
    A = X.detach().cpu().numpy().T       # (H,V), ROI on X
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(A, aspect='auto', origin='upper')
    ax.set_xlabel("ROI (V)")
    ax.set_ylabel("hidden dim (H=256)")
    ax.set_title("Hidden × ROI (batch+time aggregated)")
    _add_colorbar(ax)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)

def save_debug_text_list(path: Path, rows: List[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(f"{r}\n")

def save_csv_two_cols(path: Path, header_a: str, header_b: str, rows: List[tuple]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([header_a, header_b])
        for a, b in rows:
            w.writerow([a, b])

# ---------- main ----------
@torch.no_grad()
def main():
    args = SimpleNamespace(
        eeg_root=EEG_ROOT,
        fmri_root=FMRI_ROOT,
        a424_label_nii=A424_LABEL_NII,
        brainlm_model_dir=BRAINLM_MODEL_DIR,
        output_dir=OUTPUT_DIR,
        device=DEVICE,
        window_sec=WINDOW_SEC,
        stride_sec=STRIDE_SEC,
        fmri_norm=FMRI_NORM,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        layers=LAYERS,
        batch_idx=BATCH_IDX,
        swap_order=SWAP_ORDER,
        dump_npy=DUMP_NPY,
        keep_mask=KEEP_MASK,
        cls_at_end=CLS_AT_END,
        lowvar_thresh=LOWVAR_THRESH,
        mark_suspect=MARK_SUSPECT,
    )

    dev = torch.device(args.device)
    outdir = Path(args.output_dir); outdir.mkdir(parents=True, exist_ok=True)

    # Dataset & batch
    ds = PairedAlignedDataset(
        eeg_root=Path(args.eeg_root),
        fmri_root=Path(args.fmri_root),
        a424_label_nii=Path(args.a424_label_nii),
        window_sec=int(args.window_sec),
        original_fs=1000, target_fs=200, tr=TR,
        channels_limit=34, fmri_norm=args.fmri_norm,
        stride_sec=args.stride_sec, device='cpu', include_sr_keys=None,
    )
    if len(ds) == 0:
        raise RuntimeError("Dataset is empty. Check paths.")

    dl = DataLoader(ds, batch_size=int(args.batch_size), shuffle=False,
                    num_workers=int(args.num_workers), collate_fn=collate_paired,
                    pin_memory=(dev.type == 'cuda'), drop_last=False)

    it = iter(dl)
    for _ in range(int(args.batch_idx)):
        _ = next(it)
    batch = next(it)

    fmri_t = batch['fmri_window'].to(dev, non_blocking=True)  # (B,T,V)
    B, T, V = fmri_t.shape
    print(f"[data] batch shape: B={B}, T={T}, V={V}")

    # Save minimal metadata for this batch (best-effort)
    metas = batch.get('meta', [])
    try:
        meta_slim = []
        for m in metas:
            meta_slim.append({
                "subject": m.get("subject"),
                "run": m.get("run"),
                "start_sec": m.get("start_sec"),
                "eeg_path": m.get("eeg_path"),
                "fmri_path": m.get("fmri_path"),
            })
        with open(outdir / f"paths_batch{int(args.batch_idx)}.json", "w", encoding="utf-8") as f:
            json.dump(meta_slim, f, indent=2)
        print(f"[data] wrote paths metadata to {outdir / ('paths_batch'+str(int(args.batch_idx))+'.json')}")
    except Exception as e:
        print(f"[warn] failed to write paths metadata: {e}")

    # Padding to BrainLM patch size (20)
    fmri_pad = pad_timepoints_for_brainlm_torch(fmri_t, patch_size=20).to(dev)  # (B,Tp,V)
    Tp = fmri_pad.shape[1]
    print(f"[data] padded timepoints: Tp={Tp}, padded_by={Tp - T}")
    if Tp == T:
        max_diff = (fmri_pad - fmri_t).abs().max().item()
        print(f"[data] Tp==T → max |pad - orig| = {max_diff:.3e}")

    # Atlas coords (optional)
    xyz = None
    for pth in [
        THIS_DIR / 'BrainLM' / 'resources' / 'atlases' / 'A424_Coordinates.dat',
        THIS_DIR / 'resources' / 'atlases' / 'A424_Coordinates.dat',
        REPO_ROOT / 'BrainLM' / 'toolkit' / 'atlases' / 'A424_Coordinates.dat',
    ]:
        if Path(pth).exists() and V == 424:
            coords_np = load_a424_coords(pth)
            coords_np = coords_np / (np.max(np.abs(coords_np)) or 1.0)
            xyz = torch.from_numpy(coords_np).float().to(dev)[None, ...].repeat(B, 1, 1)
            print(f"[coords] Using atlas coords at: {pth}")
            break
    if xyz is None:
        xyz = torch.zeros(B, V, 3, device=dev)
        print("[coords] Using zero coords (no atlas or V!=424).")

    # ---- ROI debug: zero-voxel and low-variance (sample 0) ----
    zero_vox: List[int] = []
    if _HAVE_NIB and Path(args.a424_label_nii).exists():
        try:
            lab = nib.load(str(args.a424_label_nii)).get_fdata().astype(int)
            vc = np.bincount(lab.ravel(), minlength=V + 1)[1:1 + V]  # labels 1..V → idx 0..V-1
            zero_vox = np.where(vc == 0)[0].tolist()
            print(f"[debug] zero-voxel ROIs: count={len(zero_vox)} (first 20): {zero_vox[:20]}")
            save_csv_two_cols(outdir / "roi_voxel_counts.csv", "roi", "voxels",
                              [(i, int(vc[i])) for i in range(V)])
            save_debug_text_list(outdir / "debug_zero_voxel_rois.txt", [str(i) for i in zero_vox])
        except Exception as e:
            print(f"[warn] Failed to read label NIfTI for voxel counts: {e}")
    else:
        if not _HAVE_NIB:
            print("[warn] nibabel not installed; skipping zero-voxel check.")
        else:
            print("[warn] label NIfTI missing; skipping zero-voxel check.")

    roi_std0 = fmri_t[0].std(dim=0)  # (V,)
    low_var_idx = (roi_std0 < args.lowvar_thresh).nonzero().flatten().tolist()
    print(f"[debug] low-variance ROIs (std<{args.lowvar_thresh:g}): count={len(low_var_idx)} (first 20): {low_var_idx[:20]}")
    save_csv_two_cols(outdir / "roi_temporal_std.csv", "roi", "std",
                      [(i, float(roi_std0[i].item())) for i in range(V)])
    save_debug_text_list(outdir / "debug_lowvar_rois.txt",
                         [f"{i}\t{roi_std0[i].item():.6g}" for i in low_var_idx])

    suspects = sorted(set(zero_vox).union(low_var_idx)) if args.mark_suspect else None

    # ---- Input visuals (stable axes) ----
    save_fmri_input_heatmap_marked(fmri_t[0],  outdir / "input_time_by_roi.png",
                                   title="Input fMRI: time (rows) × ROI (cols)", mark_cols=suspects)
    save_fmri_input_heatmap_marked(fmri_pad[0], outdir / "input_padded_time_by_roi.png",
                                   title="Input fMRI (padded): time (rows) × ROI (cols)", mark_cols=suspects)

    # Temporal std across ROIs (bar)
    std_np = roi_std0.detach().cpu().numpy()
    fig, ax = plt.subplots(figsize=(12, 2.6))
    ax.bar(np.arange(std_np.size), std_np, width=1.0)
    if suspects:
        for c in suspects: ax.axvline(c, linewidth=0.6)
    ax.set_title("Per-ROI temporal std (sample 0)")
    ax.set_xlabel("ROI index"); ax.set_ylabel("std")
    plt.tight_layout(); fig.savefig(outdir / "input_roi_temporal_std.png", dpi=150); plt.close(fig)

    # Per-time mean across ROIs
    time_mean = fmri_t[0].mean(dim=1).detach().cpu().numpy()
    fig, ax = plt.subplots(figsize=(8, 2.6))
    ax.plot(time_mean)
    ax.set_title("Per-time mean across ROIs (sample 0)")
    ax.set_xlabel("time (TR index)"); ax.set_ylabel("mean signal")
    plt.tight_layout(); fig.savefig(outdir / "input_time_mean_over_rois.png", dpi=150); plt.close(fig)

    # Histogram of input values
    vals = fmri_t[0].detach().cpu().numpy().ravel()
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.hist(vals, bins=50)
    ax.set_title("Input value distribution (sample 0)")
    plt.tight_layout(); fig.savefig(outdir / "input_value_hist.png", dpi=150); plt.close(fig)

    # Optional: XYZ scatter of ROI coords
    arr_xyz = xyz[0].detach().cpu().numpy()
    if not np.allclose(arr_xyz, 0):
        from mpl_toolkits.mplot3d import Axes3D  # noqa
        fig = plt.figure(figsize=(6, 5)); ax = fig.add_subplot(111, projection='3d')
        ax.scatter(arr_xyz[:, 0], arr_xyz[:, 1], arr_xyz[:, 2], s=6)
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
        ax.set_title("ROI coordinates (sample 0)")
        plt.tight_layout(); fig.savefig(outdir / "xyz_coords_3d.png", dpi=150); plt.close(fig)

    if args.dump_npy:
        np.save(outdir / "fmri_t_sample0.npy", fmri_t[0].detach().cpu().numpy())
        np.save(outdir / "fmri_pad_sample0.npy", fmri_pad[0].detach().cpu().numpy())

    # ---- BrainLM latents ----
    blm = FrozenBrainLM(Path(args.brainlm_model_dir), device=dev, keep_mask=args.keep_mask)
    sig = fmri_pad.permute(0, 2, 1).contiguous()  # (B,V,Tp)
    h_vis = blm.extract_latents(sig, xyz)         # (L,B,N_vis,H)
    L, _, N_vis, H = h_vis.shape
    print(f"[brainlm] L={L}, H={H}, N_vis={N_vis}")

    # CLS/special token handling
    h_full, N_full, stripped = maybe_strip_special_token(h_vis, V=V, cls_at_end=args.cls_at_end)
    if not stripped and (N_full % V != 0):
        raise AssertionError(f"After CLS check: N={N_full} still not divisible by V={V}")
    time_chunks = N_full // V
    print(f"[brainlm] time_chunks={time_chunks}, N_full(after strip)={N_full}")

    # Layers to plot (validate range)
    layers = args.layers or [-1]
    for l in layers:
        l_eff = l if l >= 0 else (L + l)
        if not (0 <= l_eff < L):
            print(f"[warn] layer {l} out of range [0,{L-1}]; skipping.")
            continue
        hL = h_full[l_eff]                                   # (B,N_full,H)
        x = reshape_tokens(hL, V=V, swap_order=args.swap_order)  # (B,time_chunks,V,H)

        save_roi_time_heatmap(x, outdir / f"roi_time_layer{l_eff}.png", mark_cols=suspects)
        save_roi_by_hidden_heatmap(x, outdir / f"roi_by_hidden_layer{l_eff}.png")
        if args.dump_npy:
            np.save(outdir / f"latent_layer{l_eff}_sample0.npy", x[0].detach().cpu().numpy())

    print(f"[done] Outputs saved to: {outdir.resolve()}")

if __name__ == "__main__":
    main()
