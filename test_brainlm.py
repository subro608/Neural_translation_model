#!/usr/bin/env python3
"""
BrainLM shape-debugger:
- Loads one real fMRI session (A424 parcels)
- Prepares BrainLM inputs (pad-to-20TR, (B,V,Tp) + xyz)
- Runs embeddings + encoder with output_hidden_states=True
- Prints shapes/stats for:
    * raw fMRI window
    * padded window (for BrainLM)
    * signal_vectors (B,V,Tp)
    * xyz_vectors (B,V,3)
    * embeddings (B,N,D)
    * mask / ids_restore (if provided)
    * every encoder hidden state h[i]
    * encoder.last_hidden_state (final)
No CLI args; everything is configured below.
"""

from __future__ import annotations
import os, time, math, glob, re, json, platform, sys
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn
from transformers.models.nystromformer.modeling_nystromformer import NystromformerLayer  # type: ignore

# -----------------------
# CONFIG — EDIT AS NEEDED
# -----------------------
EEG_ROOT          = Path(r"D:\Neuroinformatics_research_2025\Oddball\ds116_eeg")
FMRI_ROOT         = Path(r"D:\Neuroinformatics_research_2025\Oddball\ds000116")
A424_LABEL_NII    = Path(r"D:\Neuroinformatics_research_2025\BrainLM\A424_resampled_to_bold.nii.gz")
BRAINLM_MODEL_DIR = Path(r"D:\Neuroinformatics_research_2025\MNI_templates\BrainLM\pretrained_models\2023-06-06-22_15_00-checkpoint-1400")

# Choose a specific session (or leave None to auto-pick first intersection)
CHOSEN_SUBJECT: Optional[int] = None    # e.g., 7
CHOSEN_TASK:    Optional[str] = None    # 'task001' / 'task002' / etc.
CHOSEN_RUN:     Optional[int] = None    # e.g., 1

# fMRI/BrainLM tokenization
TR = 2.0
WINDOW_SEC = 40                       # like training; 40s window
WINDOW_TR  = int(round(WINDOW_SEC / TR))  # expected 20 when TR=2
PATCH_SIZE_TR = 20                    # BrainLM temporal patch size

# Device/seed
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED   = 42

# Full-model hook tracing
HOOK_FULL_MODEL: bool = True
# Set to None to trace all modules; otherwise only these types
HOOK_TYPES = (
    NystromformerLayer,
    nn.Linear,
    nn.LayerNorm,
    nn.Dropout,
    nn.GELU,
    nn.ReLU,
)

# -----------------------
# Robust repo pathing for BrainLM
# -----------------------
THIS_DIR    = Path(__file__).parent
REPO_ROOT   = THIS_DIR.parent
CBRAMOD_DIR = REPO_ROOT / "CBraMod"
BRAINLM_DIR = REPO_ROOT / "BrainLM"

sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(CBRAMOD_DIR))

candidate_blm_roots = [
    BRAINLM_DIR,
    THIS_DIR / "BrainLM",
    Path(os.environ.get("BRAINLM_DIR", "")),
    Path(os.environ.get("BRAINLM_DIR", "")) / "brainlm_mae",
]
for p in candidate_blm_roots:
    if not p:
        continue
    p = Path(p)
    if (p / "brainlm_mae").is_dir():
        sys.path.insert(0, str(p)); break
    if p.name == "brainlm_mae" and p.is_dir():
        sys.path.insert(0, str(p.parent)); break

# -----------------------
# Imports from your repo
# -----------------------
from brainlm_mae.modeling_brainlm import BrainLMForPretraining  # type: ignore
from brainlm_mae.configuration_brainlm import BrainLMConfig     # type: ignore
from data_oddball import (                                      # type: ignore
    collect_common_sr_keys,
    load_fmri_parcels_cached,
    pad_timepoints_for_brainlm_torch,
    load_a424_coords,
)

# -----------------------
# Utilities
# -----------------------
def seed_all(seed: int):
    np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def fmt_mem() -> str:
    if not torch.cuda.is_available(): return "cuda:N/A"
    a = torch.cuda.memory_allocated() / (1024**2)
    r = torch.cuda.memory_reserved() / (1024**2)
    return f"cuda: alloc={a:.1f}MB, reserv={r:.1f}MB"

def debug_tensor(name: str, t: torch.Tensor, *, stats: bool = True):
    shape = tuple(t.shape)
    dtype = str(t.dtype)
    dev   = str(t.device)
    msg = f"[shape] {name:<32} = {shape} | {dtype} | {dev}"
    if stats and t.numel() > 0:
        with torch.no_grad():
            try:
                tmin = float(torch.nanmin(t).item())
                tmax = float(torch.nanmax(t).item())
                tmean = float(torch.nanmean(t).item())
                msg += f" | min={tmin:.4f} max={tmax:.4f} mean={tmean:.4f}"
            except Exception:
                pass
    print(msg)

# -----------------------
# Hook helpers for model-wide I/O shapes
# -----------------------
def _shape_of(obj):
    try:
        import torch
    except Exception:
        pass
    if isinstance(obj, torch.Tensor):
        return tuple(obj.shape)
    if isinstance(obj, (list, tuple)):
        return [
            _shape_of(x)
            for x in obj
            if isinstance(x, (torch.Tensor, list, tuple, dict))
        ]
    if isinstance(obj, dict):
        return {k: _shape_of(v) for k, v in obj.items() if isinstance(v, (torch.Tensor, list, tuple, dict))}
    return type(obj).__name__

def register_shape_hooks(root: nn.Module, prefix: str, *, types_filter=None):
    handles = []
    counter = {"i": 0}

    for sub_name, sub_mod in root.named_modules():
        if sub_name == "":
            # skip root itself to avoid duplicate top-level prints
            continue
        if (types_filter is not None) and (not isinstance(sub_mod, types_filter)):
            continue
        full_name = f"{prefix}.{sub_name}" if sub_name else prefix

        def _pre(mod, inputs, __fn=full_name):
            idx = counter["i"]; counter["i"] += 1
            try:
                shapes = _shape_of(inputs if len(inputs) != 1 else inputs[0])
            except Exception:
                shapes = "<unavailable>"
            print(f"[hook-pre] {idx:04d}:{__fn} IN  {shapes}")

        def _post(mod, inputs, output, __fn=full_name):
            idx = counter["i"]; counter["i"] += 1
            try:
                shapes = _shape_of(output)
            except Exception:
                shapes = "<unavailable>"
            print(f"[hook-post]{idx:04d}:{__fn} OUT {shapes}")

        handles.append(sub_mod.register_forward_pre_hook(_pre))
        handles.append(sub_mod.register_forward_hook(_post))

    return handles

# Regex helpers to find files/keys (loosely matching your other scripts)
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
    for pat in pats:
        out.extend([Path(p) for p in glob.glob(pat, recursive=True)])
    return sorted(set(out), key=lambda p: str(p).lower())

def _resolve_fmri_path_for_key(fmri_root: Path, key: Tuple[str, str, str]) -> Optional[Path]:
    sub_k, task_k, run_k = key
    for p in _find_bold_files(fmri_root):
        s, t_raw, r = _parse_key_from_path(p)
        if s is None or r is None:
            continue
        t = _canonical_task_token(t_raw, p)
        if (s, t, r) == (sub_k, task_k, run_k):
            return p
    return None

def _torch_load_compat(path: Path, map_location, *, allow_weights_only: bool = False):
    try:
        return torch.load(str(path), map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location=map_location)
    except Exception as e1:
        if allow_weights_only:
            try:
                import torch.serialization as ts
                try:
                    from torch.torch_version import TorchVersion  # type: ignore
                    ts.add_safe_globals([TorchVersion])
                except Exception:
                    pass
                return torch.load(str(path), map_location=map_location, weights_only=True)
            except Exception as e2:
                raise RuntimeError(f"Failed to load {path.name} via classic and safe weights_only paths:\n{e1}\n{e2}") from e2
        raise

# -----------------------
# Main
# -----------------------
def main():
    t0 = time.time()
    print(f"[env] torch={torch.__version__} | cuda_avail={torch.cuda.is_available()} | device={DEVICE}")
    if torch.cuda.is_available():
        print(f"[env] GPU: {torch.cuda.get_device_name(0)}")
    print(f"[env] python={sys.version.split()[0]} | platform={platform.platform()}")

    # Basic path checks
    for p in (EEG_ROOT, FMRI_ROOT, A424_LABEL_NII, BRAINLM_MODEL_DIR):
        if not Path(p).exists():
            raise FileNotFoundError(f"Path not found: {p}")

    seed_all(SEED)
    device = torch.device(DEVICE)

    # 1) Pick a session key from intersection
    keys = collect_common_sr_keys(EEG_ROOT, FMRI_ROOT)
    if not keys:
        raise RuntimeError("No intersecting (subject, task, run) between EEG_ROOT and FMRI_ROOT.")

    def _matches(k):
        s_ok = (CHOSEN_SUBJECT is None or int(k[0]) == int(CHOSEN_SUBJECT))
        t_ok = (CHOSEN_TASK is None or k[1] == CHOSEN_TASK)
        r_ok = (CHOSEN_RUN is None or int(k[2]) == int(CHOSEN_RUN))
        return s_ok and t_ok and r_ok

    key = next((k for k in keys if _matches(k)), keys[0])
    sub, task, run = key
    print(f"[select] Using session: sub={sub} task={task} run={run}")

    # 2) Load the fMRI session (T,V=424)
    fmri_path = _resolve_fmri_path_for_key(FMRI_ROOT, key)
    if fmri_path is None:
        raise RuntimeError(f"Could not resolve fMRI path for key={key}")
    cache_dir = Path(os.environ.get("ODDBALL_CACHE_DIR", "~/.cache/oddball_eegfmri")).expanduser()
    cache_dir.mkdir(parents=True, exist_ok=True)

    fmri_TV = load_fmri_parcels_cached(fmri_path, str(A424_LABEL_NII), cache_dir=cache_dir).astype(np.float32, copy=False)  # (T,V)
    T_full, V = fmri_TV.shape
    print(f"[load] fMRI parcels loaded: (T={T_full}, V={V}) | {fmt_mem()}")

    # 3) Choose a 40s window (20 TRs when TR=2s)
    if T_full < WINDOW_TR:
        raise RuntimeError(f"Session too short: T={T_full} < WINDOW_TR={WINDOW_TR}")
    start_tr = (T_full - WINDOW_TR) // 2
    end_tr   = start_tr + WINDOW_TR
    fmri_win = fmri_TV[start_tr:end_tr, :]              # (WINDOW_TR, V)
    x = torch.from_numpy(fmri_win).unsqueeze(0).to(device)  # (B=1, T=WINDOW_TR, V)
    debug_tensor("fmri_window (B,T,V)", x)

    # 4) Prepare BrainLM inputs
    #    pad to multiple of PATCH_SIZE_TR==20 along time, then permute to (B,V,Tp)
    x_pad = pad_timepoints_for_brainlm_torch(x, patch_size=PATCH_SIZE_TR)  # (B, Tp, V)
    debug_tensor("pad_timepoints (B,Tp,V)", x_pad)
    Tp = int(x_pad.shape[1])
    Ttok = int(math.ceil(Tp / PATCH_SIZE_TR))  # number of temporal tokens per ROI (expected Tp/20)
    print(f"[tokenization] TR={TR}s | WINDOW_TR={WINDOW_TR} | Tp(after pad)={Tp} | PATCH={PATCH_SIZE_TR} -> Ttok={Ttok} per ROI")

    signal_vectors = x_pad.permute(0, 2, 1).contiguous()  # (B,V,Tp)
    debug_tensor("signal_vectors (B,V,Tp)", signal_vectors)

    # xyz vectors (A424 coords in [-1,1], replicated across batch)
    coords = None
    xyz = torch.zeros(1, V, 3, device=device)
    try:
        candidates = [
            THIS_DIR / 'BrainLM' / 'resources' / 'atlases' / 'A424_Coordinates.dat',
            THIS_DIR / 'resources' / 'atlases' / 'A424_Coordinates.dat',
            REPO_ROOT / 'BrainLM' / 'toolkit' / 'atlases' / 'A424_Coordinates.dat',
        ]
        for p in candidates:
            if p.exists():
                arr = load_a424_coords(p)
                arr = arr / (float(np.max(np.abs(arr))) or 1.0)
                coords = torch.from_numpy(arr.astype(np.float32)).to(device)
                print(f"[coords] loaded A424 coords from: {p}")
                break
        if coords is not None and V == 424:
            xyz = coords.unsqueeze(0).repeat(1, 1, 1)      # (B,V,3)
        else:
            print(f"[coords] using zeros (coords missing or V!=424)")
    except Exception as e:
        print(f"[coords] WARN failed to load coords: {e} (using zeros)")
    debug_tensor("xyz_vectors (B,V,3)", xyz, stats=False)

    # 5) Load BrainLM (frozen)
    cfg_path = Path(BRAINLM_MODEL_DIR) / "config.json"
    w_path   = Path(BRAINLM_MODEL_DIR) / "pytorch_model.bin"
    with open(cfg_path, "r") as f:
        blm_cfg = BrainLMConfig(**json.load(f))
    model = BrainLMForPretraining(blm_cfg).to(device)
    sd = _torch_load_compat(w_path, map_location=device, allow_weights_only=True)
    model.load_state_dict(sd, strict=False)
    model.eval()
    for p in model.parameters(): p.requires_grad = False
    print(f"[BrainLM] loaded. hidden_size={getattr(model.config, 'hidden_size', 'NA')}, "
          f"num_hidden_layers={getattr(model.config, 'num_hidden_layers', 'NA')} | {fmt_mem()}")
    model.config.mask_ratio = 0.0
    if hasattr(model.vit, "embeddings"):
        model.vit.embeddings.mask_ratio = 0.0
    # (Optional safety for some variants)
    if hasattr(model.vit, "mask_ratio"):
        model.vit.mask_ratio = 0.0
    # 6) Embeddings (pre-encoder)
    t1 = time.time()
    embeddings, mask, ids_restore = model.vit.embeddings(
        signal_vectors=signal_vectors, xyz_vectors=xyz, noise=None
    )  # embeddings: (B, N_tokens, D)
    t2 = time.time()
    debug_tensor("embeddings (B,N,D)", embeddings)
    if mask is not None:
        debug_tensor("mask", mask, stats=False)
    if ids_restore is not None:
        debug_tensor("ids_restore", ids_restore, stats=False)
    print(f"[timing] embeddings: {t2 - t1:.3f}s | {fmt_mem()}")

    # Quick expectation: N_tokens ≈ V * Ttok (no CLS), but model specifics may differ.
    B, N_tokens, D = embeddings.shape
    print(f"[tokens] V={V}, Ttok={Ttok} => expected ~{V*Ttok} tokens; got N={N_tokens}")

    # 7) Encoder with hidden states
    t3 = time.time()
    enc_out = model.vit.encoder(
        hidden_states=embeddings,
        output_hidden_states=True,
        return_dict=True
    )
    t4 = time.time()

    # Print all hidden states (layer-by-layer)
    hs = enc_out.hidden_states  # tuple(len = num_layers+1), hs[0] is input to layer1
    print(f"[encoder] layers (including input) = {len(hs)}")
    for i, h in enumerate(hs):
        debug_tensor(f"encoder.hidden_states[{i}]", h)

    # Final encoder output
    last = enc_out.last_hidden_state
    debug_tensor("encoder.last_hidden_state (final)", last)
    print(f"[timing] encoder: {t4 - t3:.3f}s | total elapsed: {time.time() - t0:.3f}s | {fmt_mem()}")

    print("\n[done] BrainLM shape-debug complete.")

    # 8) Full-model forward with hooks (encoder+decoder) to show I/O at each layer
    if HOOK_FULL_MODEL:
        print("\n[trace] Registering hooks for full model forward (encoder + decoder)...")
        handles = []
        try:
            handles += register_shape_hooks(model.vit, prefix="vit", types_filter=HOOK_TYPES)
            handles += register_shape_hooks(model.decoder, prefix="decoder", types_filter=HOOK_TYPES)

            t5 = time.time()
            with torch.no_grad():
                out = model(
                    signal_vectors=signal_vectors,
                    xyz_vectors=xyz,
                    output_hidden_states=True,
                    return_dict=True,
                )
            t6 = time.time()

            # Summaries
            logits, latent = out.logits
            debug_tensor("pretraining.logits (B,V,Ttok_patch)", logits)
            debug_tensor("pretraining.latent (B,N,D)", latent)
            if hasattr(out, "mask") and isinstance(out.mask, torch.Tensor):
                debug_tensor("pretraining.mask (B,seq)", out.mask, stats=False)
            if out.hidden_states is not None:
                print(f"[trace] encoder hidden_states len = {len(out.hidden_states)}")
            print(f"[timing] full-model forward: {t6 - t5:.3f}s | {fmt_mem()}")
        finally:
            for h in handles:
                try:
                    h.remove()
                except Exception:
                    pass
            print("[trace] Hooks removed.")

if __name__ == "__main__":
    main()
