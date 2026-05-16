"""
spots_stics.py — Spots-reconstruction STICS pipeline for GEM diffusion.

Pipeline
--------
  preprocess
    → detect_spots_per_frame   (DoG, no linking — avoids all tracking problems)
    → reconstruct_spots_movie  (stamp Gaussian PSF at each detection)
    → compute_stics / compute_imsd / compute_d_map

Why no linking
--------------
Tracking requires a linking step (search radius, gap fill, min-track-length)
whose parameters are hard to optimise, fail at high density, and introduce
trajectory-swap errors that bias D upward.  Detection without linking uses only
a single scale / threshold choice; false positives add noise but do NOT bias D
because they are uncorrelated across frames.

The reconstructed movie contains only PSF-scale objects above the brightness
threshold — fast diffusers and autofluorescence are suppressed because they are
spread over many pixels (low DoG response per pixel) or are below the threshold.
"""

import warnings
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage
from scipy.optimize import curve_fit

# ── constants (match imaging_fcs.py defaults) ────────────────────────────────
PIXEL_UM   = 0.094   # µm / pixel
DT         = 0.1     # s / frame
MAX_LAG    = 30      # frames
LAG_MIN    = 1
LAG_MAX    = 20
TILE_PX    = 64
FIT_RADIUS = 10

# PSF sigma in pixels: w₀ ≈ 0.28 µm (1/e² radius) → σ = w₀/2 ≈ 1.5 px
# Used as DoG inner scale and reconstruction stamp width.
DEFAULT_SIGMA_PX      = 1.5
DEFAULT_THRESH_SIGMA  = 5.0   # DoG response must exceed mean + k·std per frame
DEFAULT_MIN_DIST_PX   = 3     # minimum centre-to-centre separation (px)


# ═══════════════════════════════════════════════════════════════════════════════
# I/O
# ═══════════════════════════════════════════════════════════════════════════════

def load_tiff_stack(path: Path) -> np.ndarray:
    """Return (T, Y, X) float32 array from a multi-frame TIFF."""
    raw = tifffile.imread(str(path)).astype(np.float32)
    if raw.ndim == 2:
        raw = raw[np.newaxis]
    elif raw.ndim == 4:
        raw = raw[:, 0]
    if raw.ndim != 3:
        raise ValueError(f"Expected (T,Y,X) after loading, got {raw.shape}")
    return raw


# ═══════════════════════════════════════════════════════════════════════════════
# Preprocessing  (verbatim from imaging_fcs.py)
# ═══════════════════════════════════════════════════════════════════════════════

def correct_bleaching(movie: np.ndarray) -> np.ndarray:
    """Normalise each frame to frame-0 mean to remove photobleaching."""
    frame_means = movie.mean(axis=(1, 2)).clip(min=1e-6)
    norm = frame_means[0] / frame_means
    return movie * norm[:, np.newaxis, np.newaxis]


def subtract_background(movie: np.ndarray, percentile: float = 5) -> np.ndarray:
    """Subtract per-pixel temporal low-percentile (rolling background)."""
    bg = np.percentile(movie, percentile, axis=0)
    return np.maximum(movie - bg[np.newaxis], 0.0)


def correct_drift(movie: np.ndarray, max_shift_px: int = 20) -> tuple:
    """Rigid drift correction via phase cross-correlation (integer-pixel)."""
    T, Y, X = movie.shape
    shifts = np.zeros((T, 2), dtype=np.float32)
    out    = np.empty_like(movie)
    out[0] = movie[0]
    cum_dy, cum_dx = 0.0, 0.0
    for t in range(1, T):
        prev_fft = np.fft.fft2(movie[t - 1])
        cur_fft  = np.fft.fft2(movie[t])
        cc       = np.real(np.fft.ifft2(prev_fft * np.conj(cur_fft)))
        cc       = np.fft.fftshift(cc)
        peak     = np.unravel_index(cc.argmax(), cc.shape)
        dy, dx   = peak[0] - Y // 2, peak[1] - X // 2
        if abs(dy) > max_shift_px or abs(dx) > max_shift_px:
            dy, dx = 0, 0
        cum_dy += dy; cum_dx += dx
        shifts[t] = [cum_dy, cum_dx]
        out[t] = np.roll(np.roll(movie[t], int(round(cum_dy)), axis=0),
                         int(round(cum_dx)), axis=1)
    return out, shifts


def preprocess(movie: np.ndarray,
               correct_drift_flag: bool = True) -> tuple:
    """Bleach correction → background subtraction → optional drift correction."""
    movie = correct_bleaching(movie)
    movie = subtract_background(movie)
    if correct_drift_flag:
        movie, shifts = correct_drift(movie)
    else:
        shifts = np.zeros((movie.shape[0], 2), dtype=np.float32)
    return movie, shifts


def _fft_crop(movie: np.ndarray) -> np.ndarray:
    """Centre-crop to largest power-of-2 square (speeds up rfft2)."""
    T, Y, X = movie.shape
    side = 1 << int(np.floor(np.log2(min(Y, X))))
    y0, x0 = (Y - side) // 2, (X - side) // 2
    return movie[:, y0: y0 + side, x0: x0 + side]


# ═══════════════════════════════════════════════════════════════════════════════
# STICS core  (CPU-only, stripped of GPU / detrend branches)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_stics(movie: np.ndarray,
                  max_lag: int = MAX_LAG,
                  intensity_mask_frac: float = 0.0) -> np.ndarray:
    """
    STICS via FFT cross-correlation.

    G(ξ,η,τ) = <δI(r,t)·δI(r+ρ,t+τ)> / <I>²

    intensity_mask_frac: zero out pixels whose mean < this fraction of the
      brightest 1% mean (useful for raw movies; set to 0 for spots movies
      where background = 0 is already meaningful).
    """
    T, Y, X = movie.shape
    arr     = movie.astype(np.float32)
    mean_t  = arr.mean(axis=0)                         # (Y, X)

    if intensity_mask_frac > 0:
        bright_ref = float(np.percentile(mean_t, 99))
        imask      = (mean_t >= intensity_mask_frac * bright_ref).astype(np.float32)
        arr       *= imask[np.newaxis]
        mean_t    *= imask

    g_mean = float(mean_t.mean())
    if g_mean < 1e-9:
        raise ValueError("Near-zero mean — check input movie.")

    df    = arr - mean_t[np.newaxis]
    F_all = np.fft.rfft2(df).astype(np.complex64)

    norm  = Y * X * g_mean ** 2
    acf   = np.zeros((max_lag + 1, Y, X), dtype=np.float32)
    for tau in range(max_lag + 1):
        n     = T - tau
        cross = (np.conj(F_all[:n]) * F_all[tau: tau + n]).mean(axis=0)
        cc    = np.real(np.fft.irfft2(cross, s=(Y, X)))
        acf[tau] = np.fft.fftshift(cc / norm)

    return acf


# ═══════════════════════════════════════════════════════════════════════════════
# Gaussian fitting + iMSD  (verbatim from imaging_fcs.py)
# ═══════════════════════════════════════════════════════════════════════════════

def _gaussian2d(xy, amplitude, w2, offset):
    x, y = xy
    return (offset + amplitude * np.exp(-(x ** 2 + y ** 2) / w2)).ravel()


def fit_stics_waist(g_frame: np.ndarray,
                    pixel_um: float = PIXEL_UM,
                    r: int = FIT_RADIUS,
                    exclude_centre: bool = False) -> float:
    """Fit isotropic 2-D Gaussian; return beam waist w (µm) or NaN."""
    Y, X   = g_frame.shape
    cy, cx = Y // 2, X // 2
    r      = min(r, cy - 1, cx - 1)
    patch  = g_frame[cy - r: cy + r + 1, cx - r: cx + r + 1]
    xi     = np.arange(-r, r + 1, dtype=np.float32)
    yi     = np.arange(-r, r + 1, dtype=np.float32)
    xx, yy = np.meshgrid(xi, yi)
    rr     = np.hypot(xx, yy)
    mask   = rr <= r
    if exclude_centre:
        mask &= (rr >= 1.0)
    if mask.sum() < 5:
        return np.nan
    x_data = (xx[mask], yy[mask])
    y_data = patch[mask].ravel()
    amp0   = float(y_data.max() - y_data.min())
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            popt, _ = curve_fit(
                _gaussian2d, x_data, y_data,
                p0=[amp0, max(6.0, 1.0), float(y_data.min())],
                bounds=([-np.inf, 1.0, -np.inf],
                        [np.inf, (r * 0.95) ** 2, np.inf]),
                maxfev=4000,
            )
        return float(np.sqrt(abs(popt[1]))) * pixel_um
    except RuntimeError:
        return np.nan


def compute_imsd(acf: np.ndarray,
                 pixel_um: float = PIXEL_UM,
                 dt: float = DT,
                 lag_min: int = LAG_MIN,
                 lag_max: int = LAG_MAX) -> dict:
    """
    Extract iMSD from ACF stack.  Model: w²(τ) = w₀² + 4Dτ.

    Returns dict with keys: D, w0, slope, intercept, taus, w2, waists,
    valid_mask, r2.
    """
    max_lag = acf.shape[0] - 1
    lag_max = min(lag_max, max_lag)
    waists  = np.full(max_lag + 1, np.nan)
    for i in range(max_lag + 1):
        waists[i] = fit_stics_waist(acf[i], pixel_um=pixel_um,
                                    exclude_centre=(i == 0))
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
    D  = slope / 4.0
    w0 = float(np.sqrt(max(intercept, 0.0)))
    y_hat  = slope * taus[valid] + intercept
    ss_res = float(np.sum((w2[valid] - y_hat) ** 2))
    ss_tot = float(np.sum((w2[valid] - w2[valid].mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    result.update(D=D, w0=w0, slope=slope, intercept=intercept, r2=r2)
    return result


def compute_d_map(movie: np.ndarray,
                  tile_px: int   = TILE_PX,
                  pixel_um: float = PIXEL_UM,
                  dt: float       = DT,
                  max_lag: int    = MAX_LAG,
                  lag_min: int    = LAG_MIN,
                  lag_max: int    = LAG_MAX,
                  r2_min: float   = 0.2) -> np.ndarray:
    """
    Non-overlapping tiled iMSD D map.  Tiles with R² < r2_min are rejected
    (mixed-species or noisy w²(τ) curves).
    """
    T, Y, X  = movie.shape
    ny, nx   = Y // tile_px, X // tile_px
    D_map    = np.full((ny, nx), np.nan, dtype=np.float32)
    g_mean   = float(movie.mean())
    print(f"   Tiled D map: {ny}×{nx} tiles of {tile_px}×{tile_px} px "
          f"({tile_px * pixel_um:.1f} µm)")
    for iy in range(ny):
        for ix in range(nx):
            y0, x0 = iy * tile_px, ix * tile_px
            tile   = movie[:, y0: y0 + tile_px, x0: x0 + tile_px]
            if tile.mean() < 0.01 * g_mean:
                continue
            try:
                acf = compute_stics(tile, max_lag=max_lag,
                                    intensity_mask_frac=0)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    res = compute_imsd(acf, pixel_um=pixel_um, dt=dt,
                                       lag_min=lag_min, lag_max=lag_max)
                d, r2 = res["D"], res["r2"]
                if (np.isfinite(d) and np.isfinite(r2)
                        and r2 >= r2_min and 1e-5 < d < 10.0):
                    D_map[iy, ix] = d
            except Exception:
                pass
        print(f"     row {iy + 1}/{ny}", end="\r", flush=True)
    print()
    return D_map


# ═══════════════════════════════════════════════════════════════════════════════
# NEW: DoG spot detection + reconstruction
# ═══════════════════════════════════════════════════════════════════════════════

def detect_spots_per_frame(
    movie: np.ndarray,
    sigma_px: float = DEFAULT_SIGMA_PX,
    threshold_sigma: float = DEFAULT_THRESH_SIGMA,
    min_distance_px: int | None = None,
    detect_mode: str = "raw",
) -> np.ndarray:
    """
    Per-frame DoG spot detection.  No linking across frames.

    The DoG kernel (σ₁ = sigma_px, σ₂ = σ₁·√2) acts as a bandpass filter
    tuned to structures at the PSF scale.  Peaks in the DoG response that
    exceed mean + threshold_sigma·std (computed per frame) and are local
    maxima within a disk of radius min_distance_px are retained.

    Parameters
    ----------
    movie           : (T, Y, X) float32, bleach-corrected + background-subtracted
    sigma_px        : inner Gaussian σ in pixels; set to w₀/(2·pixel_um) so the
                      DoG peaks exactly at the PSF FWHM scale.
                      Default 1.5 px matches w₀ ≈ 0.28 µm at 0.094 µm/px.
    threshold_sigma : per-frame DoG threshold = mean + k·std.
                      k=5 gives ~0 false positives per frame in 998×998 images.
    min_distance_px : minimum centre-to-centre separation; default = 2·σ px.
    detect_mode     : 'raw'            — DoG on preprocessed intensity (default)
                      'positive_residual' — DoG on max(frame − temporal_mean, 0),
                        which preferentially detects transient positive excursions
                        above the pixel's own background; partly suppresses
                        persistently-bright immobile particles whose shot-noise
                        is symmetric around their mean.

    Returns
    -------
    spots : (N, 3) int32 array, columns = [frame, y, x]

    Notes
    -----
    DoG detection at σ ≈ 1.5 px naturally rejects fast diffusers (D > 0.1 µm²/s)
    because motion blur during the camera exposure (√(4Dt_exp)) broadens their
    PSF beyond the DoG passband.  However, moderately-fast species
    (D ≈ 0.1–0.2 µm²/s, blur ≈ PSF width) can still be detected, so the
    reconstructed spots movie may retain a multi-species iMSD shape similar to
    the raw movie.  The tiled D-map with R²>0.2 filter (r2_min parameter in
    compute_d_map) is the recommended way to isolate GEM-dominated tiles.
    """
    if min_distance_px is None:
        min_distance_px = max(2, int(round(2.0 * sigma_px)))

    sigma2    = sigma_px * np.sqrt(2.0)
    footprint = 2 * min_distance_px + 1

    # Pre-compute temporal mean for positive_residual mode
    if detect_mode == "positive_residual":
        mean_t = movie.mean(axis=0)  # (Y, X)
    else:
        mean_t = None

    rows = []
    T    = movie.shape[0]
    for t in range(T):
        if detect_mode == "positive_residual":
            frame = np.maximum(movie[t].astype(np.float32) - mean_t, 0.0)
        else:
            frame = movie[t].astype(np.float32)
        dog   = (ndimage.gaussian_filter(frame, sigma_px)
                 - ndimage.gaussian_filter(frame, sigma2))
        thresh = dog.mean() + threshold_sigma * dog.std()
        lmax   = ndimage.maximum_filter(dog, size=footprint)
        peaks  = (dog == lmax) & (dog >= thresh)
        ys, xs = np.where(peaks)
        for y, x in zip(ys, xs):
            rows.append((t, int(y), int(x)))

    if not rows:
        return np.empty((0, 3), dtype=np.int32)
    return np.array(rows, dtype=np.int32)


def reconstruct_spots_movie(
    shape: tuple,
    spots: np.ndarray,
    sigma_px: float = DEFAULT_SIGMA_PX,
) -> np.ndarray:
    """
    Stamp a unit-amplitude isotropic Gaussian (σ = sigma_px) at each detected
    spot location.  Overlapping stamps are summed.

    The reconstructed movie contains only PSF-scale objects and has the same
    spatial statistics as a dilute solution of identical fluorophores — making
    standard STICS theory fully applicable.

    Parameters
    ----------
    shape    : (T, Y, X) — must match the source movie shape
    spots    : (N, 3) int32 array from detect_spots_per_frame
    sigma_px : Gaussian σ in pixels for the stamp; should equal the sigma_px
               used during detection so the ACF w₀ matches the true PSF.

    Returns
    -------
    movie : (T, Y, X) float32
    """
    T, Y, X = shape
    out     = np.zeros(shape, dtype=np.float32)

    # Pre-build kernel on a (2r+1)² grid
    r        = int(np.ceil(3.5 * sigma_px))
    yy, xx   = np.mgrid[-r: r + 1, -r: r + 1]
    kernel   = np.exp(-(yy ** 2 + xx ** 2) / (2.0 * sigma_px ** 2)).astype(np.float32)

    for t, cy, cx in spots:
        # Source kernel window
        ky0 = max(0, r - cy);          ky1 = min(2 * r + 1, r + Y - cy)
        kx0 = max(0, r - cx);          kx1 = min(2 * r + 1, r + X - cx)
        # Destination image window
        iy0 = max(0, cy - r);          iy1 = min(Y, cy + r + 1)
        ix0 = max(0, cx - r);          ix1 = min(X, cx + r + 1)
        if ky1 > ky0 and kx1 > kx0:
            out[t, iy0:iy1, ix0:ix1] += kernel[ky0:ky1, kx0:kx1]

    return out


# ═══════════════════════════════════════════════════════════════════════════════
# Top-level analysis
# ═══════════════════════════════════════════════════════════════════════════════

def analyse_spots_movie(
    tiff_path: Path,
    tracking_D: float | None = None,
    out_dir: Path | None = None,
    pixel_um: float = PIXEL_UM,
    dt: float       = DT,
    tile_px: int    = TILE_PX,
    max_lag: int    = MAX_LAG,
    lag_min: int    = LAG_MIN,
    lag_max: int    = LAG_MAX,
    correct_drift_flag: bool = False,
    sigma_px: float = DEFAULT_SIGMA_PX,
    threshold_sigma: float = DEFAULT_THRESH_SIGMA,
    min_distance_px: int | None = None,
    detect_mode: str = "raw",
    r2_min: float = 0.2,
) -> dict:
    """
    Full spots-STICS pipeline on a single TIFF movie.

    Steps
    -----
    1. Load + preprocess (bleach, background, optional drift)
    2. DoG detection per frame (no linking)
    3. Reconstruct spots movie (Gaussian stamps)
    4. Global iMSD on reconstructed movie
    5. Tiled D map on reconstructed movie

    Returns dict with keys:
      imsd      — compute_imsd output dict (global iMSD)
      D_map     — (ny, nx) tiled D map in µm²/s
      spots     — (N, 3) detection array [frame, y, x]
      movie_pp  — preprocessed raw movie
      movie_spots — reconstructed spots movie
    """
    name  = Path(tiff_path).stem
    out   = out_dir or (Path(tiff_path).parent / "spots_stics_results")
    Path(out).mkdir(parents=True, exist_ok=True)

    # ── 1. Load + preprocess ─────────────────────────────────────────────────
    print(f"\n[1/5] Loading {name} …")
    movie_raw = load_tiff_stack(tiff_path)
    T, Y, X   = movie_raw.shape
    print(f"      {T} × {Y} × {X}  float32  "
          f"({T*Y*X*4/1e6:.0f} MB)")

    print("[2/5] Preprocessing …")
    movie_pp, shifts = preprocess(movie_raw, correct_drift_flag=correct_drift_flag)
    del movie_raw

    max_drift = float(np.hypot(shifts[:, 0], shifts[:, 1]).max())
    if correct_drift_flag:
        print(f"      max drift = {max_drift:.2f} px")

    # ── 2. Detect spots ──────────────────────────────────────────────────────
    print("[3/5] Detecting spots (DoG, no linking) …")
    spots = detect_spots_per_frame(
        movie_pp,
        sigma_px        = sigma_px,
        threshold_sigma = threshold_sigma,
        min_distance_px = min_distance_px,
        detect_mode     = detect_mode,
    )
    mean_per_frame = len(spots) / T if T > 0 else 0.0
    print(f"      {len(spots):,} detections  |  "
          f"{mean_per_frame:.1f} spots/frame  |  "
          f"σ={sigma_px:.1f} px  threshold={threshold_sigma:.1f}σ  "
          f"mode={detect_mode}")

    if len(spots) == 0:
        raise RuntimeError(
            "No spots detected.  Lower threshold_sigma or check preprocessing."
        )

    # ── 3. Reconstruct movie ─────────────────────────────────────────────────
    print("[4/5] Reconstructing spots movie …")
    movie_spots = reconstruct_spots_movie(movie_pp.shape, spots, sigma_px=sigma_px)
    nonzero_frac = float((movie_spots > 0).mean())
    print(f"      {nonzero_frac*100:.2f}% of pixels occupied  "
          f"(mean intensity {movie_spots.mean():.5f})")

    # ── 4. Global iMSD ───────────────────────────────────────────────────────
    print("[5/5] STICS + iMSD (global) …")
    movie_spots_crop = _fft_crop(movie_spots)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        acf  = compute_stics(movie_spots_crop, max_lag=max_lag,
                             intensity_mask_frac=0.0)
        imsd = compute_imsd(acf, pixel_um=pixel_um, dt=dt,
                            lag_min=lag_min, lag_max=lag_max)

    print(f"\n   Global iMSD (spots movie):")
    print(f"     D  = {imsd['D']:.5f} µm²/s")
    print(f"     w0 = {imsd['w0']:.4f} µm")
    print(f"     R² = {imsd['r2']:.4f}")
    if tracking_D is not None and np.isfinite(imsd['D']):
        print(f"     tracking D = {tracking_D:.5f}  ratio = {imsd['D']/tracking_D:.3f}×")

    # Print w²(τ) and D_eff table to show multi-species shape
    print(f"\n   {'lag':>4}  {'tau_s':>6}  {'w²':>9}  {'D_eff':>8}")
    taus_all = imsd["taus"]; w2_all = imsd["w2"]
    w2_0     = w2_all[0] if np.isfinite(w2_all[0]) else np.nan
    for i in range(min(21, len(taus_all))):
        tau = taus_all[i]
        w2  = w2_all[i]
        deff = (w2 - w2_0) / (4 * tau) if (tau > 0 and np.isfinite(w2) and np.isfinite(w2_0)) else np.nan
        mark = " ←" if (np.isfinite(deff) and tracking_D is not None
                        and abs(deff - tracking_D) < 0.015) else ""
        print(f"   {i:>4}  {tau:>6.2f}  {w2:>9.5f}  {deff:>8.4f}{mark}")

    # ── 5. Tiled D map ───────────────────────────────────────────────────────
    print("\n   Tiled D map (spots movie) …")
    D_map = compute_d_map(movie_spots, tile_px=tile_px,
                          pixel_um=pixel_um, dt=dt,
                          max_lag=max_lag, lag_min=lag_min, lag_max=lag_max,
                          r2_min=r2_min)

    n_valid = int(np.isfinite(D_map).sum())
    if n_valid > 0:
        med_D = float(np.nanmedian(D_map))
        print(f"   Valid tiles: {n_valid} / {D_map.size}")
        print(f"   Median D   = {med_D:.5f} µm²/s")
        print(f"   Mean D     = {np.nanmean(D_map):.5f} µm²/s")
        if tracking_D is not None:
            print(f"   Tracking D = {tracking_D:.5f}  ratio = {med_D/tracking_D:.3f}×")
    else:
        print("   No valid tiles — lower r2_min or check detection parameters.")

    # ── Save spot detections ─────────────────────────────────────────────────
    np.save(Path(out) / f"{name}_spots.npy", spots)
    print(f"\n   Detections → {out}/{name}_spots.npy")

    return dict(
        imsd=imsd, D_map=D_map, spots=spots,
        movie_pp=movie_pp, movie_spots=movie_spots,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Diagnostics + comparison plots
# ═══════════════════════════════════════════════════════════════════════════════

def plot_spots_overlay(movie_pp: np.ndarray,
                       spots: np.ndarray,
                       frame_idx: int = 0,
                       out_path: Path | None = None,
                       pixel_um: float = PIXEL_UM) -> None:
    """Show detected spots overlaid on the raw frame at frame_idx."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    frame    = movie_pp[frame_idx]
    t_spots  = spots[spots[:, 0] == frame_idx]

    vmin = np.percentile(frame, 1)
    vmax = np.percentile(frame, 99.5)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(frame, cmap="gray", vmin=vmin, vmax=vmax, origin="upper")
    for _, y, x in t_spots:
        circ = Circle((x, y), radius=3, fill=False, color="cyan",
                       linewidth=0.8, alpha=0.8)
        ax.add_patch(circ)
    ax.set_title(f"Frame {frame_idx}  —  {len(t_spots)} spots detected", fontsize=9)
    ax.set_xlabel("x (px)"); ax.set_ylabel("y (px)")
    ax.axis("off")
    fig.tight_layout()
    if out_path:
        fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"   Spots overlay → {out_path}")
    else:
        plt.show()


def plot_w2_comparison(imsd_raw: dict,
                       imsd_spots: dict,
                       tracking_D: float | None = None,
                       out_path: Path | None = None) -> None:
    """Overlay w²(τ) curves from raw-STICS and spots-STICS."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))

    for res, label, color in [
        (imsd_raw,   "raw STICS",    "steelblue"),
        (imsd_spots, "spots STICS",  "tomato"),
    ]:
        taus = res["taus"]
        w2   = res["w2"]
        vm   = res["valid_mask"]
        ax.plot(taus, w2, "o-", color=color, ms=4, lw=1.2,
                label=f"{label}  D={res['D']:.4f} µm²/s  R²={res['r2']:.3f}")
        # Fitted line
        if np.isfinite(res["D"]):
            t_fit = taus[vm]
            ax.plot(t_fit,
                    res["slope"] * t_fit + res["intercept"],
                    "--", color=color, lw=1.5, alpha=0.6)

    if tracking_D is not None:
        # Reference w²(τ) = w0² + 4·D_track·τ, using w0 from spots fit
        w0_ref = imsd_spots["w0"] if np.isfinite(imsd_spots["w0"]) else 0.28
        tref   = np.linspace(0, taus[-1], 100)
        ax.plot(tref, w0_ref ** 2 + 4 * tracking_D * tref,
                "k:", lw=1.5, label=f"tracking  D={tracking_D:.4f} µm²/s")

    ax.set_xlabel("τ  (s)")
    ax.set_ylabel("w²(τ)  (µm²)")
    ax.set_title("iMSD comparison: raw vs spots-STICS")
    ax.legend(fontsize=8)
    fig.tight_layout()
    if out_path:
        fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"   w² comparison → {out_path}")
    else:
        plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
# CLI diagnostic  (python spots_stics.py <tiff> [tracking_D])
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys, time
    import importlib, pathlib

    tiff   = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path("/tmp/Em1_crop.tif")
    track  = float(sys.argv[2]) if len(sys.argv) > 2 else 0.04374

    OUT = pathlib.Path("/tmp/spots_stics_out")
    OUT.mkdir(exist_ok=True)

    # ── run spots pipeline ──────────────────────────────────────────────────
    t0   = time.time()
    res  = analyse_spots_movie(tiff, tracking_D=track, out_dir=OUT,
                                correct_drift_flag=False)
    elapsed = time.time() - t0
    print(f"\n   Total elapsed: {elapsed:.0f}s")

    # ── compare with raw-STICS (re-use existing imaging_fcs.py) ────────────
    print("\n── Raw-STICS comparison (imaging_fcs.py) ──")
    sys.path.insert(0, str(pathlib.Path(__file__).parent))
    import imaging_fcs as fcs_raw

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mfft   = fcs_raw._fft_crop(res["movie_pp"])
        acf_r  = fcs_raw.compute_stics(mfft, max_lag=MAX_LAG,
                                        intensity_mask_frac=0.10)
        imsd_r = fcs_raw.compute_imsd(acf_r, pixel_um=PIXEL_UM, dt=DT,
                                       lag_min=LAG_MIN, lag_max=LAG_MAX)

    print(f"   Raw  iMSD: D={imsd_r['D']:.5f}  w0={imsd_r['w0']:.4f}  R²={imsd_r['r2']:.4f}")
    print(f"   Spots iMSD: D={res['imsd']['D']:.5f}  w0={res['imsd']['w0']:.4f}  R²={res['imsd']['r2']:.4f}")
    print(f"   Tracking D:  {track:.5f}")

    # ── plots ───────────────────────────────────────────────────────────────
    plot_spots_overlay(res["movie_pp"], res["spots"], frame_idx=0,
                       out_path=OUT / "spots_overlay_f0.png")
    plot_spots_overlay(res["movie_pp"], res["spots"], frame_idx=50,
                       out_path=OUT / "spots_overlay_f50.png")
    plot_w2_comparison(imsd_r, res["imsd"], tracking_D=track,
                       out_path=OUT / "w2_comparison.png")

    print(f"\n   Output → {OUT}/")
