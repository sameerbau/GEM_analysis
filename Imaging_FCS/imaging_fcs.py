#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
imaging_fcs.py

Imaging FCS analysis of GEM particle spinning-disk movies.

Three complementary outputs:
  1. Global iMSD  — ensemble diffusion coefficient D via STICS over the full
                    field; independent cross-validation of tracking results.
  2. Tiled D map  — D computed in non-overlapping spatial tiles, producing a
                    heat map that can be overlaid on the ER/organelle mask.
  3. N&B maps     — Number (concentration) and Brightness (per-particle photon
                    yield) computed per pixel; does not give D but maps where
                    GEMs accumulate and whether they remain intact 40-mers.

Input:
  Folder of raw TIFF movie stacks (T, Y, X), uint16 from spinning-disk
  confocal.  Optional co-registered ER/membrane binary mask TIFs.

Output (written to <tiff_dir>/imfcs_results/):
  imfcs_summary.csv       — per-movie D table (iMSD vs tracking)
  <movie>_imsd.png        — iMSD curve + linear fit
  <movie>_dmap.png        — D heat map ± ER contour overlay
  <movie>_nb.png          — N map and B map ± ER contour overlay

Usage:
  python imaging_fcs.py /path/to/tiff/folder [options]

  --tracking-csv PATH   CSV with columns "movie","median_D" for comparison
  --pixel-um  FLOAT     µm per pixel           (default 0.094)
  --dt        FLOAT     seconds per frame       (default 0.1)
  --tile-px   INT       tile side in pixels for D map  (default 64)
  --max-lag   INT       max lag frames for STICS       (default 30)
  --fit-lags  INT INT   lag range used for iMSD fit    (default 1 20)
  --camera    sCMOS|EMCCD   camera type for N&B noise correction

NOTE ON PIXEL SIZE:
  The existing analysis scripts use CONVERSION = 0.094 µm/pixel,
  consistent with a ~60–100x Nikon spinning-disk objective.  If your
  acquisition pixel size differs, pass --pixel-um explicitly.
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit

try:
    import tifffile
except ImportError:
    raise SystemExit("tifffile not found — run:  pip install tifffile")

# ── Default parameters ────────────────────────────────────────────────────────

PIXEL_UM   = 0.094   # µm per pixel  (Nikon spinning-disk, 60–100x)
DT         = 0.1     # seconds per frame
TILE_PX    = 64      # tile size (pixels) for D heat map  →  ~6 µm at 0.094 µm/px
MAX_LAG    = 30      # maximum lag (frames) for STICS
LAG_MIN    = 1       # first lag used in iMSD linear fit  (skip τ=0, shot noise)
LAG_MAX    = 20      # last  lag used in iMSD linear fit
FIT_RADIUS = 10      # pixels around STICS centre used for Gaussian fit
# PSF w0 ≈ 2.3 px (τ=0), grows to ~7-10 px at τ=20 for D=0.05-0.1 µm²/s.
# r=10 gives good signal (~15-35 % of peak) at large lags; old r=15 included
# pure-noise pixels that biased the Gaussian fit and inflated recovered w0.

# Camera noise correction for N&B
# sCMOS: excess noise factor ≈ 1  |  EMCCD: ≈ 2
CAMERA_NOISE = {"sCMOS": 1.0, "EMCCD": 2.0}

# ── GPU detection ─────────────────────────────────────────────────────────────

def _detect_gpu() -> bool:
    """Return True if CuPy is installed and at least one CUDA GPU is present."""
    try:
        import cupy as cp
        cp.zeros(1)          # triggers device init; raises if no usable GPU
        return True
    except Exception:
        return False

HAS_GPU = _detect_gpu()


def _get_xp(use_gpu: bool):
    """Return cupy if GPU is requested and available, otherwise numpy."""
    if use_gpu and HAS_GPU:
        import cupy
        return cupy
    return np

# ── I/O helpers ───────────────────────────────────────────────────────────────

def load_tiff_stack(path: Path) -> np.ndarray:
    """Return (T, Y, X) float32 array from a TIFF stack."""
    raw = tifffile.imread(str(path))
    raw = raw.astype(np.float32)
    if raw.ndim == 2:                     # single frame → add T axis
        raw = raw[np.newaxis]
    elif raw.ndim == 4:                   # (T, 1, Y, X) → squeeze channel
        raw = raw[:, 0]
    if raw.ndim != 3:
        raise ValueError(f"Expected (T,Y,X) after loading, got shape {raw.shape}")
    return raw


def find_er_mask(tiff_path: Path) -> Path | None:
    """Look for a binary ER/membrane mask alongside the movie."""
    parent = tiff_path.parent
    # Patterns used in this repo: *membrane*-1.tif  or  ER_*.tif
    candidates = (
        list(parent.glob("*membrane*-1.tif")) +
        list(parent.glob("*membrane*mask*.tif")) +
        list(parent.glob("ER_*.tif"))
    )
    return candidates[0] if candidates else None


def load_er_mask(path: Path) -> np.ndarray:
    """Return (Y, X) binary float32 mask."""
    raw = tifffile.imread(str(path)).astype(np.float32)
    if raw.ndim > 2:
        raw = raw[0]                      # take first frame/channel
    return (raw > 0).astype(np.float32)


def collect_tiff_movies(folder: Path) -> list[Path]:
    """Return TIFF stacks, skipping membrane/ER/mask files."""
    skip = {"membrane", "er_", "mask", "_cp_", "er_max", "gem_max"}
    tiffs = sorted(folder.glob("*.tif")) + sorted(folder.glob("*.tiff"))
    return [t for t in tiffs
            if not any(s in t.name.lower() for s in skip)]


# ── Preprocessing ─────────────────────────────────────────────────────────────

def correct_bleaching(movie: np.ndarray) -> np.ndarray:
    """
    Divide each frame by its spatial mean, normalised to frame 0.

    Using the per-frame mean (rather than median) is intentional: after
    subtracting the per-pixel background the frame median is close to zero
    (most pixels are dark), so fitting a polynomial to medians produces a
    near-flat trend that fails to correct bleaching. The mean tracks the
    total GEM signal and correctly compensates the slow exponential decay.
    """
    frame_means = movie.mean(axis=(1, 2)).clip(min=1e-6)  # (T,)
    norm = frame_means[0] / frame_means
    return movie * norm[:, np.newaxis, np.newaxis]


def subtract_background(movie: np.ndarray, percentile: float = 5) -> np.ndarray:
    """Subtract the per-pixel temporal low-percentile (rolling background)."""
    bg = np.percentile(movie, percentile, axis=0)
    return np.maximum(movie - bg[np.newaxis], 0.0)


def preprocess(movie: np.ndarray) -> np.ndarray:
    movie = correct_bleaching(movie)
    movie = subtract_background(movie)
    return movie


def _fft_crop(movie: np.ndarray) -> np.ndarray:
    """Crop to the largest power-of-2 square centred on the image.

    rfft2 on 998×998 (= 2×499, prime) is ~10× slower than on 512×512.
    Cropping is only done for the global iMSD; tiled analysis uses the
    full movie and is unaffected (tiles are always 64×64 = 2⁶).
    """
    T, Y, X = movie.shape
    side = 1 << int(np.floor(np.log2(min(Y, X))))  # largest power of 2 ≤ min dim
    y0   = (Y - side) // 2
    x0   = (X - side) // 2
    return movie[:, y0: y0 + side, x0: x0 + side]


def _align_mask(mask: np.ndarray, target_shape: tuple) -> np.ndarray:
    """Match ER mask to (Y, X) by centre-crop then zero-pad if needed.

    Handles the common case where the mask and the movie were acquired at
    different binning or saved at different sizes (e.g. 998 vs 512).
    """
    H, W   = target_shape
    mH, mW = mask.shape
    y0 = max(0, (mH - H) // 2)
    x0 = max(0, (mW - W) // 2)
    cropped = mask[y0: y0 + min(mH, H), x0: x0 + min(mW, W)]
    if cropped.shape == (H, W):
        return cropped
    padded = np.zeros((H, W), dtype=mask.dtype)
    padded[: cropped.shape[0], : cropped.shape[1]] = cropped
    return padded


# ── STICS core ────────────────────────────────────────────────────────────────

def compute_stics(movie: np.ndarray, max_lag: int = MAX_LAG,
                  use_gpu: bool = False,
                  intensity_mask_frac: float = 0.10) -> np.ndarray:
    """
    Spatio-temporal image correlation spectroscopy (STICS).

    G(ξ, η, τ) = <δI(r,t)·δI(r+ρ, t+τ)>_{r,t} / <I>²

    Implemented via FFT cross-correlation (O(N² log N) per lag).
    Runs on GPU (CuPy) when use_gpu=True and a CUDA device is available.

    Parameters
    ----------
    movie              : (T, Y, X) float32 array, background-subtracted
    max_lag            : maximum lag in frames
    use_gpu            : use CuPy/CUDA if available (ignored if HAS_GPU is False)
    intensity_mask_frac: zero out pixels whose temporal mean < this fraction of
                         the brightest 1% mean; removes dark background from the
                         global ACF so cytoplasm/nuclei don't dilute the signal.
                         Set to 0 to disable.

    Returns
    -------
    acf : (max_lag+1, Y, X) float32 numpy array — fftshifted, centre = (Y//2, X//2)
    """
    xp = _get_xp(use_gpu)

    T, Y, X = movie.shape
    arr    = xp.asarray(movie, dtype=xp.float32)
    mean_t = arr.mean(axis=0)                              # (Y, X)

    # Spatial intensity mask: exclude dark background pixels
    if intensity_mask_frac > 0:
        mn = mean_t.get() if use_gpu and HAS_GPU else np.asarray(mean_t)
        bright_ref = float(np.percentile(mn, 99))
        thresh     = intensity_mask_frac * bright_ref
        imask      = xp.asarray((mn >= thresh).astype(np.float32))
        arr       *= imask[xp.newaxis]
        mean_t    *= imask

    df     = arr - mean_t[xp.newaxis]; del arr
    g_mean = float((mean_t.get() if use_gpu and HAS_GPU else np.asarray(mean_t)).mean())
    if g_mean < 1e-6:
        raise ValueError("Near-zero mean after preprocessing — check input data.")

    # Precompute rfft2 for all frames; complex64 halves memory vs complex128
    F_all = xp.fft.rfft2(df).astype(xp.complex64); del df  # (T, Y, X//2+1)

    acf = xp.zeros((max_lag + 1, Y, X), dtype=xp.float32)
    for tau in range(max_lag + 1):
        n     = T - tau
        cross = (xp.conj(F_all[:n]) * F_all[tau: tau + n]).mean(axis=0)
        cc    = xp.real(xp.fft.irfft2(cross, s=(Y, X)))
        acf[tau] = xp.fft.fftshift(cc / (Y * X * g_mean ** 2))

    # Return as numpy regardless of backend
    return acf.get() if use_gpu and HAS_GPU else np.asarray(acf)


# ── Gaussian fitting on STICS frames ─────────────────────────────────────────

def _gaussian2d(xy, amplitude, w2, offset):
    """Isotropic 2-D Gaussian: A·exp(-(x²+y²)/w²) + C.

    w is the PSF 1/e² radius (beam waist) because h⊗h for a Gaussian PSF
    h=exp(-2r²/w₀²) gives exp(-|ξ|²/w₀²), i.e. no factor of 2 in exponent.
    This means slope of w²(τ) = 4D (standard iMSD result).
    """
    x, y = xy
    return (offset + amplitude * np.exp(-(x ** 2 + y ** 2) / w2)).ravel()


def fit_stics_waist(g_frame: np.ndarray,
                    pixel_um: float = PIXEL_UM,
                    r: int = FIT_RADIUS,
                    exclude_centre: bool = False) -> float:
    """
    Fit an isotropic 2-D Gaussian to the central region of one STICS lag frame.

    exclude_centre=True masks the (0,0) pixel to avoid the τ=0 shot-noise spike.

    Returns beam waist w (1/e² radius) in µm, or NaN on failure.
    """
    Y, X   = g_frame.shape
    cy, cx = Y // 2, X // 2
    r      = min(r, cy - 1, cx - 1)

    patch  = g_frame[cy - r: cy + r + 1, cx - r: cx + r + 1]
    xi     = np.arange(-r, r + 1, dtype=np.float32)
    yi     = np.arange(-r, r + 1, dtype=np.float32)
    xx, yy = np.meshgrid(xi, yi)
    rr     = np.hypot(xx, yy)

    mask = (rr <= r)
    if exclude_centre:
        mask &= (rr >= 1.0)

    if mask.sum() < 5:
        return np.nan

    x_data = (xx[mask], yy[mask])
    y_data = patch[mask].ravel()

    amp0   = float(y_data.max() - y_data.min())
    # Initial guess: PSF waist ≈ 2.5 px (≈ 0.235 µm at 0.094 µm/px)
    w2_0   = max(6.0, 1.0)
    off0   = float(y_data.min())

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            popt, _ = curve_fit(
                _gaussian2d, x_data, y_data,
                p0=[amp0, w2_0, off0],
                bounds=([-np.inf, 1.0, -np.inf],   # w ≥ 1 px (≥ pixel_um)
                        [np.inf, (r * 0.95) ** 2, np.inf]),
                maxfev=4000,
            )
        w_px = float(np.sqrt(abs(popt[1])))
        return w_px * pixel_um                             # convert px → µm
    except RuntimeError:
        return np.nan


# ── iMSD extraction ───────────────────────────────────────────────────────────

def compute_imsd(acf: np.ndarray,
                 pixel_um: float = PIXEL_UM,
                 dt: float = DT,
                 lag_min: int = LAG_MIN,
                 lag_max: int = LAG_MAX) -> dict:
    """
    Extract imaging-MSD from a STICS ACF stack.

    Model:  w²(τ) = w₀² + 4·D·τ   (pure Brownian diffusion)

    D (µm²/s) is the diffusion coefficient; w₀ (µm) is the PSF waist
    recovered as the τ→0 intercept and should match the optical PSF.

    Returns a dict with keys: D, w0, slope, intercept, taus, w2, waists,
    valid_mask, r2.
    """
    max_lag = acf.shape[0] - 1
    lag_max = min(lag_max, max_lag)

    waists = np.full(max_lag + 1, np.nan)
    for tau_idx in range(max_lag + 1):
        # Exclude centre pixel at τ=0 to avoid shot-noise spike
        waists[tau_idx] = fit_stics_waist(
            acf[tau_idx], pixel_um=pixel_um,
            exclude_centre=(tau_idx == 0),
        )

    taus  = np.arange(max_lag + 1, dtype=float) * dt
    w2    = waists ** 2
    valid = (np.isfinite(w2)
             & (np.arange(max_lag + 1) >= lag_min)
             & (np.arange(max_lag + 1) <= lag_max))

    result = dict(taus=taus, w2=w2, waists=waists, valid_mask=valid,
                  D=np.nan, w0=np.nan, slope=np.nan, intercept=np.nan, r2=np.nan)

    if valid.sum() < 3:
        warnings.warn("Fewer than 3 valid lag points — iMSD fit skipped.")
        return result

    slope, intercept = np.polyfit(taus[valid], w2[valid], 1)
    D   = slope / 4.0
    w0  = float(np.sqrt(max(intercept, 0.0)))

    # R²
    y_hat = slope * taus[valid] + intercept
    ss_res = float(np.sum((w2[valid] - y_hat) ** 2))
    ss_tot = float(np.sum((w2[valid] - w2[valid].mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    result.update(D=D, w0=w0, slope=slope, intercept=intercept, r2=r2)

    if r2 < 0.5:
        warnings.warn(
            f"Global iMSD R²={r2:.3f} < 0.5 — the iMSD fit is unreliable for this movie. "
            f"Use dmap_median_D (tiled D map) as the primary diffusion estimate instead. "
            f"Possible causes: mixed cell compartments, sample drift, or heavy photobleaching.",
            UserWarning, stacklevel=2,
        )
    return result


# ── Tiled D map ───────────────────────────────────────────────────────────────

def compute_d_map(movie: np.ndarray,
                  tile_px: int  = TILE_PX,
                  pixel_um: float = PIXEL_UM,
                  dt: float = DT,
                  max_lag: int = MAX_LAG,
                  lag_min: int = LAG_MIN,
                  lag_max: int = LAG_MAX,
                  use_gpu: bool = False) -> np.ndarray:
    """
    Divide movie into non-overlapping square tiles; compute iMSD D in each.

    Dispatches to the GPU-accelerated path when use_gpu=True (all tiles are
    processed in a single batched FFT — much faster than the per-tile CPU loop).
    Falls back to the sequential CPU loop otherwise.

    Returns D_map of shape (ny, nx) in µm²/s.
    """
    if use_gpu and HAS_GPU:
        return _compute_d_map_gpu(movie, tile_px=tile_px, pixel_um=pixel_um,
                                  dt=dt, max_lag=max_lag,
                                  lag_min=lag_min, lag_max=lag_max)

    T, Y, X = movie.shape
    ny, nx  = Y // tile_px, X // tile_px
    D_map   = np.full((ny, nx), np.nan, dtype=np.float32)
    global_mean = float(movie.mean())

    print(f"   Tiled D map: {ny}×{nx} tiles of {tile_px}×{tile_px} px "
          f"({tile_px * pixel_um:.1f} µm)  [CPU]")

    for iy in range(ny):
        for ix in range(nx):
            y0, y1 = iy * tile_px, (iy + 1) * tile_px
            x0, x1 = ix * tile_px, (ix + 1) * tile_px
            tile   = movie[:, y0:y1, x0:x1]

            if tile.mean() < 0.01 * global_mean:
                continue

            try:
                acf = compute_stics(tile, max_lag=max_lag, intensity_mask_frac=0)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    res = compute_imsd(acf, pixel_um=pixel_um, dt=dt,
                                       lag_min=lag_min, lag_max=lag_max)
                d  = res["D"]
                r2 = res["r2"]
                if np.isfinite(d) and 1e-5 < d < 10.0:
                    D_map[iy, ix] = d
            except Exception:
                pass

        print(f"     row {iy + 1}/{ny}", end="\r", flush=True)
    print()
    return D_map


def _compute_d_map_gpu(movie: np.ndarray,
                       tile_px: int = TILE_PX,
                       pixel_um: float = PIXEL_UM,
                       dt: float = DT,
                       max_lag: int = MAX_LAG,
                       lag_min: int = LAG_MIN,
                       lag_max: int = LAG_MAX) -> np.ndarray:
    """
    GPU-accelerated tiled D map.

    All tiles are batched into a single rfft2 call, so the FFT runs once for
    the entire field rather than tile-by-tile.  Typical speedup vs CPU: 20–100×
    depending on GPU and image size.

    Memory estimate (worst case, 998×998 → 15×15 tiles, 300 frames):
      tiles on GPU  ≈ 300 × 225 × 64² × 4 B  ≈ 1.1 GB
      F_all         ≈ 300 × 225 × 64 × 33 × 8 B  ≈ 1.1 GB
    Peak usage ~ 2.2 GB; a 4 GB GPU handles this comfortably.

    Gaussian fitting uses log-linear OLS (vectorised NumPy on CPU) after
    transferring the small (~30 MB) ACF patch array back from the GPU.
    """
    import cupy as cp

    T, Y, X = movie.shape
    ny, nx  = Y // tile_px, X // tile_px
    n_tiles = ny * nx

    # ── Reshape movie into tile batch ─────────────────────────────────────────
    # (T, ny*tp, nx*tp) → (T, ny, nx, tp, tp) → (T, n_tiles, tp, tp)
    mc    = movie[:, : ny * tile_px, : nx * tile_px]
    tiles = (mc.reshape(T, ny, tile_px, nx, tile_px)
               .transpose(0, 1, 3, 2, 4)
               .reshape(T, n_tiles, tile_px, tile_px))

    global_mean = float(movie.mean())

    # ── GPU: batch FFT + STICS ───────────────────────────────────────────────
    g      = cp.asarray(tiles, dtype=cp.float32)           # (T, n_tiles, tp, tp)
    mean_t = g.mean(axis=0)                                 # (n_tiles, tp, tp)
    g_mean = mean_t.mean(axis=(-2, -1))                    # (n_tiles,)
    empty  = cp.asnumpy(g_mean < global_mean * 0.01)       # (n_tiles,) bool
    g     -= mean_t; del mean_t                             # in-place δI; free mem

    # Batch rfft2 across all tiles and frames simultaneously
    F = cp.fft.rfft2(g, axes=(-2, -1)).astype(cp.complex64)
    del g

    # For each lag: cross-spectrum → irfft2 → fftshift; store central patch only
    cy = cx = tile_px // 2
    r       = min(FIT_RADIUS, cy - 1)
    psz     = 2 * r + 1
    norm    = (tile_px ** 2 * cp.maximum(g_mean, 1e-12) ** 2)[:, None, None]

    patches_gpu = cp.zeros((max_lag + 1, n_tiles, psz, psz), dtype=cp.float32)

    for tau in range(max_lag + 1):
        n     = T - tau
        cross = (cp.conj(F[:n]) * F[tau: tau + n]).mean(axis=0)
        cc    = cp.fft.fftshift(
            cp.real(cp.fft.irfft2(cross, s=(tile_px, tile_px), axes=(-2, -1))),
            axes=(-2, -1),
        ) / norm
        patches_gpu[tau] = cc[:, cy - r: cy + r + 1, cx - r: cx + r + 1]

    del F, norm

    # Transfer patches to CPU (~30 MB); GPU work is done
    patches = patches_gpu.get(); del patches_gpu

    # ── CPU: vectorised log-linear Gaussian fit ───────────────────────────────
    # G(r) = A · exp(-r²/w²)  →  log G = log A - r²/w²
    # OLS per tile per lag → slope a = -1/w²  → w² = pixel_um²/(-a)
    xi       = np.arange(-r, r + 1, dtype=np.float32)
    xx, yy   = np.meshgrid(xi, xi)
    rr_flat  = (xx ** 2 + yy ** 2).ravel()               # (psz²,)

    w2_um2 = np.full((max_lag + 1, n_tiles), np.nan, dtype=np.float32)

    for tau_idx in range(max_lag + 1):
        patch   = patches[tau_idx].reshape(n_tiles, -1)   # (n_tiles, psz²)
        px_ok   = (rr_flat >= (1.0 if tau_idx == 0 else 0.0)) & (rr_flat <= r ** 2)
        g_px    = patch[:, px_ok]                          # (n_tiles, n_px)
        r2_v    = rr_flat[px_ok]                           # (n_px,)
        r2_b    = np.broadcast_to(r2_v, g_px.shape)

        ok      = g_px > 0
        log_g   = np.where(ok, np.log(np.maximum(g_px, 1e-30)), np.nan)
        # Binary-weighted OLS (weight=0 for non-positive pixels)
        w       = ok.astype(np.float32)
        log_g_w = np.where(ok, log_g, 0.0)
        r2_w    = np.where(ok, r2_b, 0.0)

        Sw   = w.sum(axis=-1)                              # (n_tiles,)
        Swx2 = (w * r2_b ** 2).sum(axis=-1)
        Swx  = (w * r2_b).sum(axis=-1)
        Swxy = (r2_w * log_g_w).sum(axis=-1)
        Swy  = log_g_w.sum(axis=-1)

        det   = Swx2 * Sw - Swx ** 2
        valid = (np.abs(det) > 1e-12) & (Sw >= 3)
        a     = np.where(valid, (Sw * Swxy - Swx * Swy) / np.where(valid, det, 1), np.nan)
        # a < 0  →  w² = pixel_um² / (-a)
        w2_px = np.where((a < -1e-8) & valid, -1.0 / a, np.nan)
        w2    = w2_px * pixel_um ** 2
        lo, hi = (0.5 * pixel_um) ** 2, (r * pixel_um) ** 2
        w2_um2[tau_idx] = np.where((w2 > lo) & (w2 < hi), w2, np.nan)

    # ── CPU: vectorised iMSD line fit per tile ────────────────────────────────
    # w²(τ) = w₀² + 4·D·τ  →  slope = 4D
    taus = np.arange(max_lag + 1, dtype=np.float32) * dt
    vm   = (np.arange(max_lag + 1) >= lag_min) & (np.arange(max_lag + 1) <= lag_max)
    tv   = taus[vm, np.newaxis]                            # (n_v, 1) for broadcast
    w2v  = w2_um2[vm]                                      # (n_v, n_tiles)

    fin    = np.isfinite(w2v)
    tv_b   = np.broadcast_to(tv, w2v.shape)
    n_v    = fin.sum(axis=0).astype(np.float32)
    St     = np.where(fin, tv_b, 0.0).sum(axis=0)
    St2    = np.where(fin, tv_b ** 2, 0.0).sum(axis=0)
    Sw_    = np.where(fin, w2v, 0.0).sum(axis=0)
    Stw    = np.where(fin, tv_b * w2v, 0.0).sum(axis=0)

    det    = n_v * St2 - St ** 2
    good   = (np.abs(det) > 1e-14) & (n_v >= 3)
    slope  = np.where(good, (n_v * Stw - St * Sw_) / np.where(good, det, 1.0), np.nan)
    intercept = np.where(good, (Sw_ * St2 - St * Stw) / np.where(good, det, 1.0), np.nan)
    D_vals = slope / 4.0

    D_vals = np.where(
        good & ~empty & (D_vals > 1e-5) & (D_vals < 10.0),
        D_vals, np.nan,
    )
    n_valid = int(np.isfinite(D_vals).sum())
    print(f"   Tiled D map: {ny}×{nx} tiles of {tile_px}×{tile_px} px "
          f"({tile_px * pixel_um:.1f} µm)  [{n_valid}/{n_tiles} valid tiles, GPU]")
    return D_vals.reshape(ny, nx).astype(np.float32)


# ── N&B analysis ─────────────────────────────────────────────────────────────

def number_and_brightness(movie: np.ndarray,
                           camera: str = "sCMOS",
                           smooth_sigma: float = 1.5) -> tuple:
    """
    Digman–Gratton Number and Brightness analysis.

    B_apparent = (Var(I) − s·<I>) / <I>
        where s = camera excess noise factor (1 for sCMOS, 2 for EMCCD)
    N_apparent = <I>² / (Var(I) − s·<I>)

    B maps particle brightness per pixel transit — a proxy for GEM integrity:
    intact 40-mers produce uniform B; aggregates appear brighter, fragments
    dimmer.  N maps local particle concentration.

    Neither B nor N gives D.

    Returns (N_map, B_map, mean_map, var_map), all (Y, X) float32.
    """
    s      = CAMERA_NOISE.get(camera, 1.0)
    mean_I = movie.mean(axis=0)
    var_I  = movie.var(axis=0, ddof=1)

    # Camera-corrected signal variance
    var_sig = np.maximum(var_I - s * mean_I, 1e-6)
    mean_I  = np.maximum(mean_I, 1e-6)

    B_map = var_sig / mean_I
    N_map = mean_I ** 2 / var_sig

    if smooth_sigma > 0:
        B_map = gaussian_filter(B_map, sigma=smooth_sigma)
        N_map = gaussian_filter(N_map, sigma=smooth_sigma)

    return N_map.astype(np.float32), B_map.astype(np.float32), mean_I, var_I


# ── Plotting ──────────────────────────────────────────────────────────────────

def _er_contour(ax, er_mask, D_or_N_map, tile_px, pixel_um, use_tile_coords=False):
    """Overlay ER mask contour on an image axes."""
    if er_mask is None:
        return
    H, W = D_or_N_map.shape
    if use_tile_coords:
        # D map: align mask to the pixel footprint of the tile grid, then downsample
        er_full = _align_mask(er_mask, (H * tile_px, W * tile_px))
        er_ds   = er_full.reshape(H, tile_px, W, tile_px).mean(axis=(1, 3))
        xs = np.linspace(0, W * tile_px * pixel_um, W)
        ys = np.linspace(0, H * tile_px * pixel_um, H)
    else:
        # N/B map: align mask to full movie frame size
        er_ds = _align_mask(er_mask, (H, W))
        xs = np.linspace(0, W * pixel_um, W)
        ys = np.linspace(0, H * pixel_um, H)

    ax.contour(xs, ys, er_ds, levels=[0.5], colors="cyan",
               linewidths=0.8, alpha=0.8)


def plot_imsd(result: dict, movie_name: str = "",
              tracking_D: float | None = None,
              out_path: Path | None = None):
    fig, ax = plt.subplots(figsize=(5.5, 4.5))

    taus  = result["taus"]
    w2    = result["w2"]
    valid = result["valid_mask"]

    ax.scatter(taus, w2, s=25, c="steelblue", zorder=3, label="w²(τ) from STICS")

    if np.isfinite(result["D"]):
        tau_line = np.linspace(taus[valid].min(), taus[valid].max(), 100)
        w2_line  = result["slope"] * tau_line + result["intercept"]
        ax.plot(tau_line, w2_line, "r-", lw=1.8,
                label=f"iMSD fit  D = {result['D']:.4f} µm²/s  "
                      f"(R² = {result['r2']:.3f})")
        ax.axhline(result["intercept"], ls=":", lw=0.8, c="gray",
                   label=f"w₀ = {result['w0']:.3f} µm")

    if tracking_D is not None:
        # Draw tracking prediction using same w₀ intercept
        tau_line = np.linspace(taus[valid].min(), taus[valid].max(), 100)
        w2_track = 4 * tracking_D * tau_line + result.get("intercept", 0)
        ax.plot(tau_line, w2_track, "g--", lw=1.5,
                label=f"Tracking  D = {tracking_D:.4f} µm²/s")

    ax.set_xlabel("τ  (s)")
    ax.set_ylabel("w²(τ)  (µm²)")
    ax.set_title(f"iMSD — {movie_name}")
    ax.legend(fontsize=8)
    fig.tight_layout()

    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_d_map(D_map: np.ndarray,
               er_mask: np.ndarray | None = None,
               tile_px: int = TILE_PX,
               pixel_um: float = PIXEL_UM,
               movie_name: str = "",
               tracking_D: float | None = None,
               out_path: Path | None = None):
    ncols = 2 if er_mask is not None else 1
    fig, axes = plt.subplots(1, ncols, figsize=(5.5 * ncols, 4.5))
    if ncols == 1:
        axes = [axes]

    ny, nx   = D_map.shape
    extent   = [0, nx * tile_px * pixel_um,
                ny * tile_px * pixel_um, 0]          # µm
    vmin = np.nanpercentile(D_map, 5)
    vmax = np.nanpercentile(D_map, 95)
    cmap = "inferno"

    for i, ax in enumerate(axes):
        im = ax.imshow(D_map, cmap=cmap, vmin=vmin, vmax=vmax,
                       extent=extent, aspect="equal")
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("D  (µm²/s)")

        if i == 1 and er_mask is not None:
            _er_contour(ax, er_mask, D_map, tile_px, pixel_um,
                        use_tile_coords=True)
            ax.set_title(f"D map + ER  — {movie_name}")
        else:
            ax.set_title(f"D map  — {movie_name}")

        ax.set_xlabel("x  (µm)")
        ax.set_ylabel("y  (µm)")

    if tracking_D is not None:
        fig.suptitle(f"Tracking median D = {tracking_D:.4f} µm²/s", fontsize=9,
                     color="grey")

    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_nb(N_map: np.ndarray, B_map: np.ndarray,
            er_mask: np.ndarray | None = None,
            pixel_um: float = PIXEL_UM,
            movie_name: str = "",
            out_path: Path | None = None):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    for ax, data, cmap, label, title in [
        (axes[0], N_map, "viridis",
         "Apparent N  (particles / pixel)", f"N map  — {movie_name}"),
        (axes[1], B_map, "plasma",
         "Apparent B  (photons / particle)", f"B map  — {movie_name}"),
    ]:
        H, W = data.shape
        extent = [0, W * pixel_um, H * pixel_um, 0]
        vmin = np.nanpercentile(data, 2)
        vmax = np.nanpercentile(data, 98)
        im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax,
                       extent=extent, aspect="equal")
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label(label)
        _er_contour(ax, er_mask, data, tile_px=None, pixel_um=pixel_um,
                    use_tile_coords=False)
        ax.set_title(title)
        ax.set_xlabel("x  (µm)")
        ax.set_ylabel("y  (µm)")

    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def _print_er_comparison(D_map, er_mask, tile_px):
    """Print median D inside vs outside ER tiles."""
    ny, nx  = D_map.shape
    er_full = _align_mask(er_mask, (ny * tile_px, nx * tile_px))
    er_ds   = er_full.reshape(ny, tile_px, nx, tile_px).mean(axis=(1, 3)) > 0.5
    D_in    = D_map[er_ds & np.isfinite(D_map)]
    D_out   = D_map[~er_ds & np.isfinite(D_map)]
    print(f"   D inside  ER: median={np.nanmedian(D_in):.4f}  n={len(D_in)}")
    print(f"   D outside ER: median={np.nanmedian(D_out):.4f}  n={len(D_out)}")
    return float(np.nanmedian(D_in)), float(np.nanmedian(D_out)), len(D_in), len(D_out)


# ── Per-movie analysis ────────────────────────────────────────────────────────

def analyse_movie(tiff_path: Path,
                  tracking_D: float | None = None,
                  out_dir: Path | None = None,
                  pixel_um: float = PIXEL_UM,
                  dt: float = DT,
                  tile_px: int = TILE_PX,
                  max_lag: int = MAX_LAG,
                  lag_min: int = LAG_MIN,
                  lag_max: int = LAG_MAX,
                  camera: str = "sCMOS",
                  use_gpu: bool = False) -> dict:
    name  = tiff_path.stem
    out   = out_dir or (tiff_path.parent / "imfcs_results")
    out.mkdir(parents=True, exist_ok=True)

    gpu_tag = "GPU ✓" if (use_gpu and HAS_GPU) else ("GPU requested but unavailable — CPU" if use_gpu else "CPU")
    print(f"\n── {name} {'─' * max(0, 60 - len(name))}  [{gpu_tag}]")

    # Load & preprocess
    movie = load_tiff_stack(tiff_path)
    T, Y, X = movie.shape
    print(f"   {T} frames × {Y}×{X} px  "
          f"({Y * pixel_um:.1f}×{X * pixel_um:.1f} µm  |  {T * dt:.1f} s total)")

    movie = preprocess(movie)

    # Warn if PSF likely sub-pixel (common misconfiguration)
    psf_expected_um = 0.25                                 # typical for NA≥1
    if psf_expected_um / pixel_um < 1.5:
        warnings.warn(
            f"PSF (~{psf_expected_um} µm) spans only "
            f"{psf_expected_um / pixel_um:.1f} pixels at {pixel_um} µm/px. "
            "STICS Gaussian fitting may be unreliable — verify pixel-um.",
            stacklevel=2,
        )

    summary: dict = dict(movie=name, pixel_um=pixel_um, dt=dt,
                         n_frames=T, tracking_D=tracking_D)

    # ── 1. Global iMSD ────────────────────────────────────────────────────────
    print("   [1/3] Global STICS → iMSD …")
    movie_fft = _fft_crop(movie)
    if movie_fft.shape[1] != Y or movie_fft.shape[2] != X:
        print(f"   FFT crop: {Y}×{X} → {movie_fft.shape[1]}×{movie_fft.shape[2]} px "
              f"(centre crop to power-of-2 for speed)")
    acf    = compute_stics(movie_fft, max_lag=max_lag, use_gpu=use_gpu)
    result = compute_imsd(acf, pixel_um=pixel_um, dt=dt,
                          lag_min=lag_min, lag_max=lag_max)

    print(f"   iMSD  D = {result['D']:.4f} µm²/s  |  "
          f"w₀ = {result['w0']:.3f} µm  |  R² = {result['r2']:.3f}")
    if tracking_D is not None and np.isfinite(result["D"]):
        ratio = result["D"] / tracking_D
        print(f"   Ratio iMSD/tracking = {ratio:.2f}")

    summary.update(imsd_D=result["D"], imsd_w0=result["w0"], imsd_r2=result["r2"])

    plot_imsd(result, movie_name=name, tracking_D=tracking_D,
              out_path=out / f"{name}_imsd.png")

    # ── 2. Tiled D map ────────────────────────────────────────────────────────
    print("   [2/3] Tiled D map …")
    D_map = compute_d_map(movie, tile_px=tile_px, pixel_um=pixel_um, dt=dt,
                          max_lag=max_lag, lag_min=lag_min, lag_max=lag_max,
                          use_gpu=use_gpu)

    valid_tiles = np.isfinite(D_map).sum()
    print(f"   {valid_tiles} valid tiles  |  "
          f"median D = {np.nanmedian(D_map):.4f} µm²/s")
    summary["dmap_median_D"] = float(np.nanmedian(D_map))
    summary["n_valid_tiles"] = int(valid_tiles)

    er_mask = None
    er_path = find_er_mask(tiff_path)
    if er_path:
        er_mask = load_er_mask(er_path)
        print(f"   ER mask: {er_path.name}")
        d_in, d_out, n_in, n_out = _print_er_comparison(D_map, er_mask, tile_px)
        summary.update(D_inside_ER=d_in, D_outside_ER=d_out,
                       n_tiles_inside_ER=n_in, n_tiles_outside_ER=n_out)

    plot_d_map(D_map, er_mask=er_mask, tile_px=tile_px, pixel_um=pixel_um,
               movie_name=name, tracking_D=tracking_D,
               out_path=out / f"{name}_dmap.png")

    # ── 3. N&B ────────────────────────────────────────────────────────────────
    print("   [3/3] N&B …")
    N_map, B_map, _, _ = number_and_brightness(movie, camera=camera)
    print(f"   B median = {float(np.median(B_map)):.2f}  |  "
          f"N median = {float(np.median(N_map)):.3f}")
    summary.update(nb_median_B=float(np.median(B_map)),
                   nb_median_N=float(np.median(N_map)))

    plot_nb(N_map, B_map, er_mask=er_mask, pixel_um=pixel_um,
            movie_name=name, out_path=out / f"{name}_nb.png")

    return summary


# ── Batch loop ────────────────────────────────────────────────────────────────

def batch_analyse(tiff_dir: Path,
                  tracking_csv: Path | None = None,
                  out_dir: Path | None = None,
                  pixel_um: float = PIXEL_UM,
                  dt: float = DT,
                  tile_px: int = TILE_PX,
                  max_lag: int = MAX_LAG,
                  lag_min: int = LAG_MIN,
                  lag_max: int = LAG_MAX,
                  camera: str = "sCMOS",
                  use_gpu: bool = False) -> pd.DataFrame | None:
    tiffs = collect_tiff_movies(tiff_dir)
    if not tiffs:
        print(f"No TIFF movie stacks found in {tiff_dir}")
        return None
    gpu_status = f"GPU ✓ (CuPy)" if (use_gpu and HAS_GPU) else ("no GPU — CPU only" if not HAS_GPU else "CPU (--gpu to enable)")
    print(f"Processing {len(tiffs)} movie(s)  |  {gpu_status}")

    # Optional tracking D lookup
    tracking: dict = {}
    if tracking_csv and tracking_csv.exists():
        df_t = pd.read_csv(tracking_csv)
        for _, row in df_t.iterrows():
            tracking[str(row["movie"])] = float(row["median_D"])
        print(f"Loaded tracking results for {len(tracking)} movies.")

    out = out_dir or (tiff_dir / "imfcs_results")
    summaries = []

    for tiff in tiffs:
        tr_D = tracking.get(tiff.stem) or tracking.get(tiff.name)
        try:
            s = analyse_movie(tiff, tracking_D=tr_D, out_dir=out,
                              pixel_um=pixel_um, dt=dt, tile_px=tile_px,
                              max_lag=max_lag, lag_min=lag_min, lag_max=lag_max,
                              camera=camera, use_gpu=use_gpu)
            summaries.append(s)
        except Exception as exc:
            print(f"  ERROR on {tiff.name}: {exc}")

    if not summaries:
        return None

    df = pd.DataFrame(summaries)
    csv_path = out / "imfcs_summary.csv"
    df.to_csv(csv_path, index=False, float_format="%.6f")
    print(f"\nSummary → {csv_path}")
    print(df.to_string(index=False))
    return df


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(
        description="Imaging FCS (STICS iMSD + N&B) for GEM spinning-disk movies",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("tiff_dir", type=Path,
                   help="Directory containing raw TIFF movie stacks")
    p.add_argument("--tracking-csv", type=Path, default=None,
                   help='CSV with columns "movie","median_D" for iMSD comparison')
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Output directory (default: <tiff_dir>/imfcs_results)")
    p.add_argument("--pixel-um", type=float, default=PIXEL_UM,
                   help="µm per pixel")
    p.add_argument("--dt", type=float, default=DT,
                   help="Seconds per frame")
    p.add_argument("--tile-px", type=int, default=TILE_PX,
                   help="Tile side in pixels for D map")
    p.add_argument("--max-lag", type=int, default=MAX_LAG,
                   help="Maximum lag (frames) for STICS")
    p.add_argument("--fit-lags", type=int, nargs=2,
                   default=[LAG_MIN, LAG_MAX], metavar=("MIN", "MAX"),
                   help="Lag range (frames) used for iMSD linear fit")
    p.add_argument("--camera", choices=["sCMOS", "EMCCD"], default="sCMOS",
                   help="Camera type (sets N&B noise correction factor)")
    p.add_argument("--gpu", action="store_true", default=False,
                   help="Use GPU acceleration via CuPy (requires CUDA + cupy install)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.gpu and not HAS_GPU:
        print("WARNING: --gpu requested but CuPy/CUDA not available — running on CPU.")
    batch_analyse(
        tiff_dir    = args.tiff_dir,
        tracking_csv= args.tracking_csv,
        out_dir     = args.out_dir,
        pixel_um    = args.pixel_um,
        dt          = args.dt,
        tile_px     = args.tile_px,
        max_lag     = args.max_lag,
        lag_min     = args.fit_lags[0],
        lag_max     = args.fit_lags[1],
        camera      = args.camera,
        use_gpu     = args.gpu,
    )
