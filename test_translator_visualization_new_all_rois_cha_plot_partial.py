#!/usr/bin/env python3
"""
Top-k correlation visualization for EEG & fMRI (A424) reconstructions + leakage checks.

Updates in this version
-----------------------
- Supports MODE: 'both' | 'eeg2fmri' | 'fmri2eeg' | 'partial_eeg' | 'partial_fmri'
- Random contiguous partial masks with user-settable fraction (VISIBLE_FRAC)
- Uses pathlib.Path for all paths (compatible with data_oddball utilities)
- Keeps strict subject split (reads subject_splits.json next to checkpoint)
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List, Iterable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# ---------- Repo-local imports ----------
THIS_DIR = Path(__file__).parent
REPO_ROOT = THIS_DIR.parent
CBRAMOD_DIR = REPO_ROOT / "CBraMod"
BRAINLM_DIR = REPO_ROOT / "BrainLM"

import sys
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(THIS_DIR / "CBraMod"))
sys.path(insert_index := 0, string := str(THIS_DIR / "BrainLM"))  # small trick to keep both early in sys.path
sys.path.append(str(CBRAMOD_DIR))
sys.path.append(str(BRAINLM_DIR))

from module import (  # type: ignore
    BidirectionalAdaptiveCompressor,
    ConvEEGInputAdapter,
    fMRIInputAdapterConv1d,
    HierarchicalEncoder,
    CrossAttentionLayer,
    EEGDecodingAdapterLite,
    fMRIDecodingAdapterLite,
)

from models.cbramod import CBraMod  # type: ignore
from brainlm_mae.modeling_brainlm import BrainLMForPretraining  # type: ignore
from brainlm_mae.configuration_brainlm import BrainLMConfig     # type: ignore

from data_oddball import (  # type: ignore
    pad_timepoints_for_brainlm_torch,
    load_a424_coords,
    PairedAlignedDataset,
    collate_paired,
    collect_common_sr_keys,
    fixed_subject_keys,
)

# ---------- Hardcoded configuration (edit as needed) ----------
EEG_ROOT = Path(r"D:\Neuroinformatics_research_2025\Oddball\ds116_eeg")
FMRI_ROOT = Path(r"D:\Neuroinformatics_research_2025\Oddball\ds000116")
A424_LABEL_NII = Path(r"D:\Neuroinformatics_research_2025\BrainLM\A424_resampled_to_bold.nii.gz")
CBRAMOD_WEIGHTS = Path(r"D:\Neuroinformatics_research_2025\MNI_templates\CBraMod\pretrained_weights\pretrained_weights.pth")
BRAINLM_MODEL_DIR = Path(r"D:\Neuroinformatics_research_2025\MNI_templates\BrainLM\pretrained_models\2023-06-06-22_15_00-checkpoint-1400")
CHECKPOINT = Path(r"D:\Neuroinformatics_research_2025\Multi_modal_NTM\translator_run_2_aug17_updated_fmri_6_eeg_0_patchfmri_translator_fixed\translator_best.pt")
OUT_DIR = Path(r"D:\Neuroinformatics_research_2025\Multi_modal_NTM\viz_output_aug17_fixed_train_patchfmri_eeg0_fmri6")

DEVICE = "cuda"
SEED = 42
FMRI_NORM = "zscore"
WINDOW_SEC = 40
ORIGINAL_FS = 1000
TARGET_FS = 200
STRIDE_SEC = 10
CHANNELS_LIMIT = 34
BATCH_SIZE = 8
NUM_WORKERS = 0
EEG_SECONDS_PER_TOKEN = 1
TR = 2.0

SUBSET = "train"  # "train" | "val" | "test" | "all"
MODE = "partial_fmri"  # "both" | "eeg2fmri" | "fmri2eeg" | "partial_eeg" | "partial_fmri"
VISIBLE_FRAC = 0.5     # fraction of time kept visible in partial modes

# Plotting downsample caps
PLOT_MAX_POINTS_EEG = 200
PLOT_MAX_POINTS_FMRI = 1000

TOP_K = 34

# -----------------------------
# Positional Encoding (sin/cos)
# -----------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 100_000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        return x + self.pe[: x.size(1), :].unsqueeze(0)

# -----------------------------
# Frozen feature extractors
# -----------------------------
class FrozenCBraMod(nn.Module):
    def __init__(self, in_dim: int, d_model: int, seq_len: int, n_layer: int, nhead: int, dim_feedforward: int,
                 weights_path: Path, device: torch.device) -> None:
        super().__init__()
        self.model = CBraMod(
            in_dim=in_dim, out_dim=in_dim, d_model=d_model,
            seq_len=seq_len, n_layer=n_layer, nhead=nhead, dim_feedforward=dim_feedforward,
        )
        try:
            try:
                ckpt = torch.load(str(weights_path), map_location=device, weights_only=False)
            except TypeError:
                ckpt = torch.load(str(weights_path), map_location=device)
            if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
                self.model.load_state_dict(ckpt['model_state_dict'], strict=False)
            elif isinstance(ckpt, dict) and 'state_dict' in ckpt:
                self.model.load_state_dict(ckpt['state_dict'], strict=False)
            else:
                self.model.load_state_dict(ckpt, strict=False)
        except Exception as e:
            raise RuntimeError(f"Failed to load CBraMod weights from {weights_path}: {e}")
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.to(device)

    @torch.no_grad()
    def extract_latents(self, x: torch.Tensor) -> torch.Tensor:
        patch_emb = self.model.patch_embedding(x)  # (B,P,C,D) -> internal layout
        cur = patch_emb
        outs = []
        for layer in self.model.encoder.layers:
            cur = layer(cur)
            outs.append(cur)
        return torch.stack(outs, dim=0)  # (L,B,P,C,D)

class FrozenBrainLM(nn.Module):
    def __init__(self, model_dir: Path, device: torch.device) -> None:
        super().__init__()
        with open(model_dir / "config.json", "r") as f:
            cfg = json.load(f)
        config = BrainLMConfig(**cfg)
        self.model = BrainLMForPretraining(config)
        try:
            try:
                ckpt = torch.load(str(model_dir / "pytorch_model.bin"), map_location=device, weights_only=True)
            except TypeError:
                ckpt = torch.load(str(model_dir / "pytorch_model.bin"), map_location=device)
            self.model.load_state_dict(ckpt, strict=False)
        except Exception as e:
            raise RuntimeError(f"Failed to load BrainLM weights from {model_dir}: {e}")
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.to(device)

    @torch.no_grad()
    def extract_latents(self, signal_vectors: torch.Tensor, xyz: torch.Tensor) -> torch.Tensor:
        embeddings, _, _ = self.model.vit.embeddings(
            signal_vectors=signal_vectors, xyz_vectors=xyz, noise=None
        )
        enc = self.model.vit.encoder(hidden_states=embeddings, output_hidden_states=True, return_dict=True)
        return torch.stack(list(enc.hidden_states), dim=0)  # (L,B,Ttok,H)

# -----------------------------
# Translator (Lite + small + PEs)
# -----------------------------
class TranslatorModel(nn.Module):
    def __init__(
        self,
        eeg_channels: int,
        eeg_patch_num: int,
        eeg_n_layers: int,
        eeg_input_dim: int,
        fmri_n_layers: int,
        fmri_hidden_size: int,
        fmri_tokens_target: int,
        fmri_target_T: int,
        fmri_target_V: int,
        d_model: int = 128,
        n_heads: int = 4,
        d_ff: int = 512,
        dropout: float = 0.1,
        voxel_count: int = 424,
        debug: bool = False,
    ) -> None:
        super().__init__()
        self.debug = debug
        self._voxel_count = int(voxel_count)
        self._fmri_n_layers = int(fmri_n_layers)
        self._fmri_hidden_size = int(fmri_hidden_size)

        self.adapter_eeg = ConvEEGInputAdapter(
            seq_len=eeg_patch_num, n_layers=eeg_n_layers,
            channels=eeg_channels, input_dim=eeg_input_dim, output_dim=d_model,
        )
        self.adapter_fmri = fMRIInputAdapterConv1d(
            seq_len=fmri_tokens_target, n_layers=fmri_n_layers,
            input_dim=fmri_hidden_size, output_dim=d_model, target_seq_len=512,
        )

        self.pos_eeg  = PositionalEncoding(d_model)
        self.pos_fmri = PositionalEncoding(d_model)

        self.eeg_encoder  = HierarchicalEncoder(d_model, n_heads, d_ff, dropout, n_layers_per_stack=1)
        self.fmri_encoder = HierarchicalEncoder(d_model, n_heads, d_ff, dropout, n_layers_per_stack=1)
        self.cross_attn   = CrossAttentionLayer(d_model, n_heads, dropout)
        self.compressor   = BidirectionalAdaptiveCompressor()

        self.eeg_decoder = EEGDecodingAdapterLite(
            channels=eeg_channels, patch_num=eeg_patch_num,
            patch_size=eeg_input_dim, d_model=d_model, rank=32,
        )
        self.fmri_decoder = fMRIDecodingAdapterLite(
            target_T=fmri_target_T, target_V=fmri_target_V,
            d_model=d_model, rank=16,
        )

        self.fmri_out_scale = nn.Parameter(torch.tensor(1.0))
        self.fmri_out_bias  = nn.Parameter(torch.tensor(0.0))

    def forward(self, eeg_latents, fmri_latents, fmri_target_T: int, fmri_target_V: int):
        if int(fmri_target_V) != self._voxel_count:
            raise ValueError(f"TranslatorModel expects V={self._voxel_count}, got V={fmri_target_V}")

        eeg_adapt  = self.pos_eeg( self.adapter_eeg(eeg_latents) )
        fmri_adapt = self.pos_fmri(self.adapter_fmri(fmri_latents))

        _, eeg_hi = self.eeg_encoder(eeg_adapt, eeg_adapt)
        _, fmr_hi = self.fmri_encoder(fmri_adapt, fmri_adapt)
        eeg_c, fmr_c, _ = self.compressor(eeg_hi, fmr_hi)
        fused = self.cross_attn(eeg_c, fmr_c)

        eeg_signal = self.eeg_decoder(fused)                 # (B,C,Pe,S)
        fmri_flat  = self.fmri_decoder(fused)                # (B, T*V)
        fmri_signal = fmri_flat.view(-1, int(fmri_target_T), int(fmri_target_V))
        fmri_signal = torch.tanh(fmri_signal)
        fmri_signal = self.fmri_out_scale * fmri_signal + self.fmri_out_bias
        return eeg_signal, fmri_signal

# -----------------------------
# Helpers
# -----------------------------
def group_eeg_latents_seconds(eeg_latents: torch.Tensor, seconds_per_token: int) -> torch.Tensor:
    L, B, P, C, D = eeg_latents.shape
    g = max(1, min(seconds_per_token, P))
    P_grp = max(1, P // g)
    return eeg_latents[:, :, :P_grp*g].reshape(L, B, P_grp, g, C, D).mean(dim=3)

def group_eeg_signal_seconds(x_eeg_sig: torch.Tensor, seconds_per_token: int) -> torch.Tensor:
    B, C, P, S = x_eeg_sig.shape
    g = max(1, min(seconds_per_token, P))
    P_grp = max(1, P // g)
    return x_eeg_sig[:, :, :P_grp*g, :].reshape(B, C, P_grp, g, S).mean(dim=3)

def _rand_block_mask_time(B: int, L: int, vis_frac: float, device) -> torch.Tensor:
    """
    Return (B,L) mask where True=MASKED, constructed by keeping a single
    contiguous visible block of length ~vis_frac*L, masking the rest.
    """
    M = torch.ones(B, L, dtype=torch.bool, device=device)
    vis = max(1, int(round(vis_frac * L))); vis = min(vis, L)
    starts = torch.randint(0, max(1, L - vis + 1), (B,), device=device) if L > vis else torch.zeros(B, dtype=torch.long, device=device)
    for i in range(B):
        M[i, starts[i]:starts[i]+vis] = False
    return M

# -----------------------------
# Correlation utilities
# -----------------------------
def pearsonr_safe(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return 0.0
    a = a[m]; b = b[m]
    sa = np.std(a); sb = np.std(b)
    if sa == 0.0 or sb == 0.0:
        return 0.0
    r = np.corrcoef(a, b)[0, 1]
    return float(r) if np.isfinite(r) else 0.0

def downsample(x: np.ndarray, y: np.ndarray, max_points: int) -> Tuple[np.ndarray, np.ndarray]:
    n = x.shape[0]
    if n <= max_points:
        return x, y
    step = int(math.ceil(n / max_points))
    return x[::step], y[::step]

# -----------------------------
# Subject ID helpers & leakage checks
# -----------------------------
def norm_subj_id(s) -> str:
    ss = str(s)
    if ss.startswith("sub-"):
        ss = ss[4:]
    ss = ss.lstrip("0")
    return ss if ss else "0"

def subjects_from_ds_index(ds_index: Iterable, limit: Optional[int] = None) -> List[str]:
    out: List[str] = []
    if not ds_index:
        return out
    it = ds_index if limit is None else ds_index[:limit]
    for entry in it:
        try:
            cand = entry[0]
            if isinstance(cand, (tuple, list)) and len(cand) >= 1:
                subj = cand[0]
            else:
                subj = cand
            out.append(norm_subj_id(subj))
        except Exception:
            continue
    return out

# -----------------------------
# Build frozen models & translator
# -----------------------------
@dataclass
class VizConfig:
    eeg_root: Path
    fmri_root: Path
    a424_label_nii: Path
    cbramod_weights: Path
    brainlm_model_dir: Path
    checkpoint: Path
    out_dir: Path
    device: str = "cuda"
    seed: int = 42
    fmri_norm: str = "zscore"
    window_sec: int = 40
    original_fs: int = 1000
    target_fs: int = 200
    stride_sec: Optional[int] = 10
    channels_limit: int = 34
    batch_size: int = 8
    num_workers: int = 0
    eeg_seconds_per_token: int = 1
    tr: float = 2.0

def build_frozen_models(cfg: VizConfig, device: torch.device):
    seq_len_eeg = cfg.window_sec
    frozen_eeg = FrozenCBraMod(
        in_dim=cfg.target_fs, d_model=cfg.target_fs, seq_len=seq_len_eeg,
        n_layer=12, nhead=8, dim_feedforward=800,
        weights_path=cfg.cbramod_weights, device=device
    )
    frozen_fmri = FrozenBrainLM(cfg.brainlm_model_dir, device=device)
    return frozen_eeg, frozen_fmri

def build_translator(
    cfg: VizConfig,
    device: torch.device,
    fmri_tokens_target: int,
    fmri_target_T: int,
    fmri_target_V: int,
    fmri_n_layers: int,
    fmri_hidden_size: int
) -> TranslatorModel:
    eeg_group = max(1, int(cfg.eeg_seconds_per_token))
    eeg_patch_num_grouped = max(1, int(cfg.window_sec) // eeg_group)
    translator = TranslatorModel(
        eeg_channels=cfg.channels_limit,
        eeg_patch_num=eeg_patch_num_grouped,
        eeg_n_layers=12,
        eeg_input_dim=cfg.target_fs,
        fmri_n_layers=fmri_n_layers,
        fmri_hidden_size=fmri_hidden_size,
        fmri_tokens_target=fmri_tokens_target,  # MUST be T * V
        fmri_target_T=fmri_target_T,
        fmri_target_V=fmri_target_V,
        d_model=128, n_heads=4, d_ff=512, dropout=0.1,
        voxel_count=fmri_target_V,
        debug=False,
    ).to(device)

    try:
        ckpt = torch.load(str(cfg.checkpoint), map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(str(cfg.checkpoint), map_location=device)
    state = ckpt.get("translator_state", ckpt)
    current = translator.state_dict()
    filtered = {k: v for k, v in state.items()
                if k in current and getattr(current[k], 'shape', None) == getattr(v, 'shape', None)}
    translator.load_state_dict(filtered, strict=False)
    translator.eval()
    return translator

# -----------------------------
# Main eval & plotting
# -----------------------------
def run_once(cfg: VizConfig, subset: str = "test", mode: str = "both", visible_frac: float = 0.5):
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

    if torch.cuda.is_available() and cfg.device.startswith("cuda"):
        try:
            torch.set_float32_matmul_precision('high')
        except Exception:
            pass

    device = torch.device(cfg.device)
    out_dir = cfg.out_dir
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)

    # Strict split from checkpoint folder
    inter_keys = collect_common_sr_keys(cfg.eeg_root, cfg.fmri_root)
    if len(inter_keys) == 0:
        raise RuntimeError("No aligned (subject, run) pairs found; check paths.")

    run_dir = Path(cfg.checkpoint).parent
    p_split = run_dir / "subject_splits.json"
    if not p_split.exists():
        raise FileNotFoundError(f"subject_splits.json not found at {p_split}")
    with open(p_split, "r") as f:
        j = json.load(f)
    train_subj = [norm_subj_id(s) for s in j["train_subjects"]]
    val_subj   = [norm_subj_id(s) for s in j["val_subjects"]]
    test_subj  = [norm_subj_id(s) for s in j["test_subjects"]]
    train_keys, val_keys, test_keys = fixed_subject_keys(cfg.eeg_root, cfg.fmri_root,
                                                         [int(s) for s in train_subj],
                                                         [int(s) for s in val_subj],
                                                         [int(s) for s in test_subj])

    include_sr = None
    allowed_subjects = None
    if subset == "train":
        include_sr = train_keys
        allowed_subjects = set(train_subj)
    elif subset == "val":
        include_sr = val_keys
        allowed_subjects = set(val_subj)
    elif subset == "test":
        include_sr = test_keys
        allowed_subjects = set(test_subj)
    elif subset == "all":
        include_sr = None
        allowed_subjects = None
    else:
        raise ValueError(f"Unknown subset: {subset}")

    if include_sr is not None:
        include_subjects_list = sorted({int(norm_subj_id(s)) for (s, _, _) in include_sr})
        print(f"[INFO] Running subset='{subset}' with subjects: {include_subjects_list}")
    else:
        print(f"[INFO] Running subset='{subset}' with all intersecting subjects")

    ds = PairedAlignedDataset(
        eeg_root=cfg.eeg_root,
        fmri_root=cfg.fmri_root,
        a424_label_nii=cfg.a424_label_nii,
        window_sec=cfg.window_sec,
        original_fs=cfg.original_fs,
        target_fs=cfg.target_fs,
        tr=cfg.tr,
        channels_limit=cfg.channels_limit,
        fmri_norm=cfg.fmri_norm,
        stride_sec=cfg.stride_sec,
        device='cpu',
        include_sr_keys=include_sr,
    )
    if len(ds) == 0:
        raise RuntimeError("Dataset is empty for chosen subset.")

    ds_index = getattr(ds, 'index', [])
    ds_subjects = set(subjects_from_ds_index(ds_index))
    if allowed_subjects is not None and ds_subjects:
        extras = sorted({int(s) for s in ds_subjects if s not in allowed_subjects})
        if extras:
            raise AssertionError(
                f"[LEAKAGE] Subset='{subset}' contains unexpected subjects: {extras}. "
                f"Allowed: {sorted(int(s) for s in allowed_subjects)} ; Present: {sorted(int(s) for s in ds_subjects)}"
            )
        print(f"[CHECK] Leakage check passed for subset='{subset}'. "
              f"Subjects present: {sorted(int(s) for s in ds_subjects)}")

    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False,
                    num_workers=cfg.num_workers, collate_paired=collate_paired,
                    pin_memory=(device.type == 'cuda'))

    frozen_eeg, frozen_fmri = build_frozen_models(cfg, device)

    batch = next(iter(dl))

    first_batch_subjects = subjects_from_ds_index(ds_index, limit=min(cfg.batch_size, len(ds_index)))
    if first_batch_subjects:
        print(f"[BATCH] First batch subjects (best-effort): {first_batch_subjects}")

    x_eeg = batch['eeg_window'].to(device)   # (B,C,P,S)
    fmri_t = batch['fmri_window'].to(device) # (B,T,V)
    B, C, P, S = x_eeg.shape
    _, T, V = fmri_t.shape

    # Observed inputs incl. partial modes
    x_eeg_obs = x_eeg.clone()
    fmri_obs  = fmri_t.clone()
    if mode == "both":
        pass
    elif mode == "eeg2fmri":
        fmri_obs[:] = 0.0
    elif mode == "fmri2eeg":
        x_eeg_obs[:] = 0.0
    elif mode == "partial_eeg":
        M_eeg = _rand_block_mask_time(B, P, visible_frac, device)              # (B,P) True=mask
        x_eeg_obs = x_eeg_obs.masked_fill(M_eeg[:, None, :, None], 0.0)        # broadcast (B,1,P,1)
    elif mode == "partial_fmri":
        M_t = _rand_block_mask_time(B, T, visible_frac, device)                # (B,T) True=mask
        fmri_obs = fmri_obs.masked_fill(M_t[:, :, None], 0.0)                  # broadcast (B,T,1)
    else:
        raise ValueError("Supported modes: 'both' | 'eeg2fmri' | 'fmri2eeg' | 'partial_eeg' | 'partial_fmri'")

    # BrainLM inputs
    patch = int(getattr(frozen_fmri.model.config, "timepoint_patching_size", 20))
    fmri_padded = pad_timepoints_for_brainlm_torch(fmri_obs, patch_size=patch)   # (B,Tp,V)
    signal_vectors = fmri_padded.permute(0, 2, 1).contiguous()                    # (B,V,Tp)

    # Sync voxel count with current V
    frozen_fmri.model.config.num_brain_voxels = int(V)
    if hasattr(frozen_fmri.model, "vit") and hasattr(frozen_fmri.model.vit, "embeddings"):
        frozen_fmri.model.vit.embeddings.num_brain_voxels = int(V)
    if hasattr(frozen_fmri.model, "decoder"):
        frozen_fmri.model.decoder.num_brain_voxels = int(V)

    # XYZ coords
    try:
        cand = [
            THIS_DIR / 'BrainLM' / 'resources' / 'atlases' / 'A424_Coordinates.dat',
            THIS_DIR / 'resources' / 'atlases' / 'A424_Coordinates.dat',
            REPO_ROOT / 'BrainLM' / 'toolkit' / 'atlases' / 'A424_Coordinates.dat',
        ]
        for pth in cand:
            if pth.exists():
                a424_dat = pth
                break
        else:
            a424_dat = cand[-1]
        coords_np = load_a424_coords(a424_dat)
        coords_np = coords_np / (np.max(np.abs(coords_np)) or 1.0)
        if V == 424:
            xyz = torch.from_numpy(coords_np).to(device=device, dtype=torch.float32)[None, ...].repeat(B, 1, 1)
        else:
            xyz = torch.zeros(B, V, 3, device=device)
    except Exception:
        xyz = torch.zeros(B, V, 3, device=device)

    # Frozen latents
    with torch.no_grad():
        eeg_latents = frozen_eeg.extract_latents(x_eeg_obs)     # (L,B,P,C,D)
        fmri_latents = frozen_fmri.extract_latents(signal_vectors, xyz)  # (L,B,Ttok,H)

    fmri_n_layers = int(getattr(frozen_fmri.model.config, "num_hidden_layers", 4)) + 1
    fmri_hidden_size = int(getattr(frozen_fmri.model.config, "hidden_size", 256))

    translator = build_translator(
        cfg, device,
        fmri_tokens_target=int(fmri_t.shape[1]) * int(fmri_t.shape[2]),  # T * V
        fmri_target_T=int(fmri_t.shape[1]),
        fmri_target_V=int(fmri_t.shape[2]),
        fmri_n_layers=fmri_n_layers,
        fmri_hidden_size=fmri_hidden_size
    )

    # Group EEG for decoder
    eeg_latents_t = group_eeg_latents_seconds(eeg_latents, cfg.eeg_seconds_per_token)
    x_eeg_grp     = group_eeg_signal_seconds(x_eeg, cfg.eeg_seconds_per_token)  # (B,C,Pe,S)

    with torch.no_grad():
        recon_eeg, recon_fmri = translator(
            eeg_latents_t, fmri_latents,
            fmri_target_T=int(fmri_t.shape[1]),
            fmri_target_V=int(fmri_t.shape[2]),
        )

    # Compute correlations on first item of batch (b=0)
    b = 0

    # EEG correlations (per channel)
    x_true_eeg = x_eeg_grp[b].detach().cpu().numpy()       # (C, Pe, S)
    x_rec_eeg  = recon_eeg[b].detach().cpu().numpy()       # (C, Pe, S)
    C_, Pe, S_ = x_true_eeg.shape
    eeg_rs = []
    for c in range(C_):
        gt = x_true_eeg[c].reshape(Pe * S_)
        rc = x_rec_eeg[c].reshape(Pe * S_)
        r = pearsonr_safe(gt, rc)
        eeg_rs.append((c, r))
    eeg_rs.sort(key=lambda t: t[1], reverse=True)
    top_eeg = eeg_rs[:min(TOP_K, len(eeg_rs))]

    # fMRI correlations (per ROI)
    x_true_fmri = fmri_t[b].detach().cpu().numpy()         # (T, V)
    x_rec_fmri  = recon_fmri[b].detach().cpu().numpy()     # (T, V)
    T_, V_ = x_true_fmri.shape
    fmri_rs = []
    for v in range(V_):
        gt = x_true_fmri[:, v]
        rc = x_rec_fmri[:, v]
        r = pearsonr_safe(gt, rc)
        fmri_rs.append((v, r))
    fmri_rs.sort(key=lambda t: t[1], reverse=True)
    top_fmri = fmri_rs[:min(TOP_K, len(fmri_rs))]

    # Save CSV tables
    import csv
    with open(out_dir / "tables" / "eeg_channel_pearson_r.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["channel", "pearson_r"])
        for ch, r in eeg_rs:
            w.writerow([ch, r])
    with open(out_dir / "tables" / "fmri_roi_pearson_r.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["roi", "pearson_r"])
        for roi, r in fmri_rs:
            w.writerow([roi, r])

    # ---------- Plotting: Top-10 EEG ----------
    if len(top_eeg) > 0:
        fig_eeg, axes_eeg = plt.subplots(nrows=len(top_eeg), ncols=1,
                                         figsize=(12, 2.2*len(top_eeg)), sharex=False)
        if len(top_eeg) == 1:
            axes_eeg = [axes_eeg]
        t_eeg = np.arange(Pe * S_) / float(TARGET_FS)  # seconds
        for ax, (ch, r) in zip(axes_eeg, top_eeg):
            gt = x_true_eeg[ch].reshape(Pe*S_)
            rc = x_rec_eeg[ch].reshape(Pe*S_)
            te, gt_d = downsample(t_eeg, gt, PLOT_MAX_POINTS_EEG)
            _,  rc_d = downsample(t_eeg, rc, PLOT_MAX_POINTS_EEG)
            ax.plot(te, gt_d, label="EEG GT")
            ax.plot(te, rc_d, label="EEG Recon", alpha=0.85)
            ax.set_title(f"EEG Channel {ch} — r={r:.3f}")
            ax.set_xlabel("Time (s)"); ax.set_ylabel("Amplitude (z-sc)")
            ax.legend(loc="upper right")
        fig_eeg.tight_layout()
        fig_eeg.savefig(out_dir / "plots" / "top10_eeg_corr.png", dpi=150)
        plt.close(fig_eeg)

    # ---------- Plotting: Top-10 fMRI ----------
    if len(top_fmri) > 0:
        fig_fmri, axes_fmri = plt.subplots(nrows=len(top_fmri), ncols=1,
                                           figsize=(12, 2.2*len(top_fmri)), sharex=False)
        if len(top_fmri) == 1:
            axes_fmri = [axes_fmri]
        t_fmri = np.arange(T_) * float(cfg.tr)
        for ax, (roi, r) in zip(axes_fmri, top_fmri):
            gt = x_true_fmri[:, roi]
            rc = x_rec_fmri[:, roi]
            tf, gt_d = downsample(t_fmri, gt, PLOT_MAX_POINTS_FMRI)
            _,  rc_d = downsample(t_fmri, rc, PLOT_MAX_POINTS_FMRI)
            ax.plot(tf, gt_d, label="fMRI GT")
            ax.plot(tf, rc_d, label="fMRI Recon", alpha=0.85)
            ax.set_title(f"fMRI ROI {roi} — r={r:.3f}")
            ax.set_xlabel("Time (s)"); ax.set_ylabel("Z-scored signal")
            ax.legend(loc="upper right")
        fig_fmri.tight_layout()
        fig_fmri.savefig(out_dir / "plots" / "top10_fmri_corr.png", dpi=150)
        plt.close(fig_fmri)

    # ---------- Bar charts across ALL channels/ROIs ----------
    # EEG bars
    all_ch = [ch for ch, _ in eeg_rs]
    all_r  = [r for _, r in eeg_rs]
    order = np.argsort(all_r)[::-1]
    figw = max(12, 0.3 * len(all_ch) + 6)
    plt.figure(figsize=(figw, 5))
    plt.bar(np.arange(len(all_ch)), np.array(all_r)[order])
    plt.ylim(-1.0, 1.0)
    idxs = np.arange(len(all_ch))
    step = max(1, len(all_ch) // 30)
    tick_positions = idxs[::step]
    tick_labels = [str(all_ch[i]) for i in order[::step]]
    plt.xticks(tick_positions, tick_labels, rotation=90)
    plt.xlabel("EEG Channel (sorted by r)")
    plt.ylabel("Pearson r")
    plt.title("EEG: Pearson correlation per channel (all)")
    plt.tight_layout()
    plt.savefig(out_dir / "plots" / "corr_bars_eeg.png", dpi=150)
    plt.close()

    # fMRI bars
    all_roi = [roi for roi, _ in fmri_rs]
    all_rr  = [r for _, r in fmri_rs]
    order_f = np.argsort(all_rr)[::-1]
    figw = max(12, 0.03 * len(all_roi) + 10)
    plt.figure(figsize=(figw, 5))
    plt.bar(np.arange(len(all_roi)), np.array(all_rr)[order_f])
    plt.ylim(-1.0, 1.0)
    idxs = np.arange(len(all_roi))
    step = max(1, len(all_roi) // 40)
    tick_positions = idxs[::step]
    tick_labels = [str(all_roi[i]) for i in order_f[::step]]
    plt.xticks(tick_positions, tick_labels, rotation=90)
    plt.xlabel("fMRI ROI (sorted by r)")
    plt.ylabel("Pearson r")
    plt.title("fMRI: Pearson correlation per ROI (all)")
    plt.tight_layout()
    plt.savefig(out_dir / "plots" / "corr_bars_fmri.png", dpi=150)
    plt.close()

    # Console summary
    print("\n=== Top EEG channels by Pearson r ===")
    for ch, r in top_eeg:
        print(f"  ch={ch:02d}  r={r:.4f}")
    print("\n=== Top fMRI ROIs by Pearson r ===")
    for roi, r in top_fmri:
        print(f"  ROI={roi:03d} r={r:.4f}")

def main():
    cfg = VizConfig(
        eeg_root=EEG_ROOT,
        fmri_root=FMRI_ROOT,
        a424_label_nii=A424_LABEL_NII,
        cbramod_weights=CBRAMOD_WEIGHTS,
        brainlm_model_dir=BRAINLM_MODEL_DIR,
        checkpoint=CHECKPOINT,
        out_dir=OUT_DIR,
        device=DEVICE,
        seed=SEED,
        fmri_norm=FMRI_NORM,
        window_sec=WINDOW_SEC,
        original_fs=ORIGINAL_FS,
        target_fs=TARGET_FS,
        stride_sec=STRIDE_SEC,
        channels_limit=CHANNELS_LIMIT,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        eeg_seconds_per_token=EEG_SECONDS_PER_TOKEN,
        tr=TR,
    )
    # Quick path sanity
    for p in [cfg.eeg_root, cfg.fmri_root, cfg.a424_label_nii, cfg.cbramod_weights, cfg.brainlm_model_dir, cfg.checkpoint]:
        if p is None or not Path(p).exists():
            raise FileNotFoundError(f"Missing path: {p}")
    run_once(cfg, subset=SUBSET, mode=MODE, visible_frac=VISIBLE_FRAC)

if __name__ == "__main__":
    main()
