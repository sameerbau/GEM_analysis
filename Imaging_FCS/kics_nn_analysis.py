"""
kics_nn_analysis.py — kICS and NN-MSD tracking-free diffusion analysis for GEM.

Method 1: kICS (k-space Image Correlation Spectroscopy)
---------------------------------------------------------
R(k,τ) = Re[<δF_k*(t)·δF_k(t+τ)>] / <|δF_k(t)|²>

where δF_k = F_k - <F_k>_t (temporal mean subtraction removes immobile fraction).

For single Brownian species: R(k,τ) = exp(-D·k²·τ).
PSF cancels in the ratio — no PSF knowledge needed.
At τ=0: R=1 by construction.

Method 2: NN-MSD (Nearest-Neighbour MSD, DANAE-style)
-------------------------------------------------------
Detect particles per frame via DoG; for lag τ find nearest neighbour in t+τ;
collect displacement r² values; fit MSD(τ) = 4Dτ + C and displacement distributions.

Usage
-----
    python kics_nn_analysis.py /path/to/movie.tif 0.04374
"""

import warnings
from pathlib import Path

import numpy as np
from scipy.optimize import nnls, minimize, Bounds, curve_fit
from scipy.spatial import cKDTree

from ddm_analysis import (
    load_tiff_stack,
    correct_bleaching,
    subtract_background,
    _fft_crop,
)
from spots_stics import detect_spots_per_frame

# ── constants ─────────────────────────────────────────────────────────────────
PIXEL_UM  = 0.094
DT        = 0.1
MAX_LAG   = 30
N_K_BINS  = 30


# ═══════════════════════════════════════════════════════════════════════════════
# kICS core
# ═══════════════════════════════════════════════════════════════════════════════

def compute_kics_acf(
    movie: np.ndarray,
    max_lag: int = MAX_LAG,
    pixel_um: float = PIXEL_UM,
    dt: float = DT,
    n_k_bins: int = N_K_BINS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute isotropic kICS autocorrelation function R(k,τ).

    R(k,τ) = Re[<δF_k*(t)·δF_k(t+τ)>_t] / <|δF_k(t)|²>_t

    Temporal mean subtraction (δF = F - <F>_t) removes immobile / constant
    background exactly.  The PSF amplitude cancels in the ratio, so R depends
    only on D and τ, not on PSF shape.

    Parameters
    ----------
    movie     : (T, Y, X) float32 — bleach + background corrected
    max_lag   : maximum lag in frames
    pixel_um  : pixel size in µm
    dt        : frame interval in s
    n_k_bins  : number of radial k bins

    Returns
    -------
    R_mat     : (max_lag, n_k_bins) float64, values in [-1, 1]
    k_centers : (n_k_bins,) bin centres in rad/µm
    taus      : (max_lag,) lag times in s
    """
    T, Y, X = movie.shape

    arr = movie.astype(np.float64)
    # Temporal mean subtraction — immobile fraction removed exactly
    mean_t = arr.mean(axis=0)           # (Y, X)
    dF     = arr - mean_t[np.newaxis]   # δF(r, t)

    F_all = np.fft.rfft2(dF)           # (T, Y, X//2+1) complex128

    # Denominator: time-averaged power spectrum <|δF_k|²>_t
    PS = (np.abs(F_all) ** 2).mean(axis=0)   # (Y, X//2+1)

    # Radial k map (rad/µm)
    ky    = np.fft.fftfreq(Y, d=pixel_um) * 2.0 * np.pi
    kx    = np.fft.rfftfreq(X, d=pixel_um) * 2.0 * np.pi
    KY, KX = np.meshgrid(ky, kx, indexing='ij')
    K_map  = np.sqrt(KY ** 2 + KX ** 2)       # (Y, X//2+1)

    k_lo = 2.0 * np.pi / (min(Y, X) * pixel_um) * 1.5
    k_hi = np.pi / pixel_um
    k_edges   = np.linspace(k_lo, k_hi, n_k_bins + 1)
    k_centers = 0.5 * (k_edges[:-1] + k_edges[1:])

    # Pre-compute masks and denominator per bin
    masks  = []
    PS_k   = np.zeros(n_k_bins, dtype=np.float64)
    for i in range(n_k_bins):
        m = (K_map >= k_edges[i]) & (K_map < k_edges[i + 1])
        masks.append(m)
        PS_k[i] = PS[m].mean() if m.sum() > 0 else np.nan

    taus  = np.arange(1, max_lag + 1, dtype=np.float64) * dt
    R_mat = np.zeros((max_lag, n_k_bins), dtype=np.float64)

    for tau in range(1, max_lag + 1):
        n = T - tau
        if n <= 0:
            break
        # Numerator: Re[<δF_k*(t)·δF_k(t+τ)>_t]
        cross = (np.conj(F_all[:n]) * F_all[tau: tau + n]).mean(axis=0)
        C_re  = np.real(cross)
        for ik, mask in enumerate(masks):
            if mask.sum() > 0 and np.isfinite(PS_k[ik]) and PS_k[ik] > 0:
                R_mat[tau - 1, ik] = C_re[mask].mean() / PS_k[ik]

    return R_mat, k_centers, taus


def fit_kics_per_k(
    R_mat: np.ndarray,
    k_centers: np.ndarray,
    taus: np.ndarray,
    k_min_fit: float = 6.0,
    k_max_fit: float = 14.0,
    lag_min: int = 1,
    lag_max: int = 20,
) -> dict:
    """
    Fit R(k,τ) = A·exp(-D·k²·τ) + B at each k bin independently.

    A should be close to 1, B close to 0 for a clean single-species system.
    D(k) should be flat in k for a single Brownian species.

    Returns dict with D_per_k, A_per_k, B_per_k, D_median, D_mean, D_sem.
    """
    n_k     = len(k_centers)
    D_per_k = np.full(n_k, np.nan)
    A_per_k = np.full(n_k, np.nan)
    B_per_k = np.full(n_k, np.nan)

    lag_min = max(1, lag_min)
    lag_max = min(lag_max, R_mat.shape[0])
    s_slice = slice(lag_min - 1, lag_max)
    tau_fit = taus[s_slice]

    for ik, k in enumerate(k_centers):
        if k < k_min_fit or k > k_max_fit:
            continue
        r = R_mat[s_slice, ik]
        if not np.all(np.isfinite(r)):
            continue

        def _model(tau, A, D, B):
            return A * np.exp(-D * k ** 2 * tau) + B

        r0  = float(r[0])
        p0  = [max(r0, 0.05), 0.044, max(1.0 - r0, 0.0)]
        blo = [0.0, 1e-6, -0.5]
        bhi = [2.0, k ** 2 * 20.0, 1.0]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                popt, _ = curve_fit(_model, tau_fit, r, p0=p0,
                                    bounds=(blo, bhi), maxfev=15000,
                                    method='trf')
            A, D, B = popt
            if A > 0 and D > 0:
                D_per_k[ik] = D
                A_per_k[ik] = A
                B_per_k[ik] = B
        except Exception:
            pass

    k_mask  = (k_centers >= k_min_fit) & (k_centers <= k_max_fit)
    D_valid = D_per_k[k_mask & np.isfinite(D_per_k)]

    return dict(
        D_per_k   = D_per_k,
        A_per_k   = A_per_k,
        B_per_k   = B_per_k,
        k_centers = k_centers,
        k_mask    = k_mask,
        taus      = taus,
        R_mat     = R_mat,
        D_median  = float(np.nanmedian(D_valid)) if len(D_valid) else np.nan,
        D_mean    = float(np.nanmean(D_valid))   if len(D_valid) else np.nan,
        D_sem     = float(np.std(D_valid) / np.sqrt(len(D_valid)))
                    if len(D_valid) > 1 else np.nan,
        n_valid_k = int(len(D_valid)),
    )


def fit_kics_two_component(
    R_mat: np.ndarray,
    k_centers: np.ndarray,
    taus: np.ndarray,
    k_min_fit: float = 6.0,
    k_max_fit: float = 14.0,
    lag_min: int = 3,
    lag_max: int = 30,
    D1_init: float = 0.044,
    D2_init: float = 0.005,
    D1_bounds: tuple = (0.01, 0.18),
    D2_bounds: tuple = (0.0005, 0.025),
) -> dict:
    """
    Two-component kICS fit across all k bins simultaneously.

    Model (per k bin):
        R(k,τ) = f₁(k)·exp(-D₁·k²·τ) + f₂(k)·exp(-D₂·k²·τ) + B(k)

    D₁ and D₂ are SHARED across k bins (D₁ > D₂ by convention).
    f₁(k), f₂(k), B(k) are non-negative and solved per-k via NNLS for each
    (D₁, D₂) candidate.  At τ=0 the model gives f₁+f₂+B = R(k,0) ≤ 1 by
    construction (NNLS enforces non-negativity; columns are ≤1 exponentials).

    Outer minimisation over (D₁, D₂) uses coarse 2-D grid search + L-BFGS-B.

    Default window k=6-14 rad/µm, lag_min=3 targets the GEM+slow residual pair:
    - At τ≥0.3s (lag 3) fast species (D≈0.45, τ_c≤0.06s at k≥6) has decayed.
    - D₁ ≈ D_GEM (0.01-0.18), D₂ ≈ D_slow (<0.025).

    Returns
    -------
    dict with D_GEM=D1, D_slow=D2, f1_per_k, f2_per_k, B_per_k,
    converged, residual, k_centers, k_mask, taus, R_mat.
    """
    lag_min = max(1, lag_min)
    lag_max = min(lag_max, R_mat.shape[0])
    s_slice = slice(lag_min - 1, lag_max)
    tau_fit = taus[s_slice]
    n_tau   = len(tau_fit)

    k_mask = (k_centers >= k_min_fit) & (k_centers <= k_max_fit)
    k_idx  = np.where(k_mask)[0]
    if len(k_idx) == 0:
        raise ValueError("No k bins in fit range.")

    R_fit = R_mat[s_slice][:, k_idx]   # (n_tau, n_k_in_range)

    # Per-k scale so all bins contribute equally to SSE
    scales = np.array([max(float(np.abs(R_fit[:, j]).max()), 1e-30)
                       for j in range(len(k_idx))])

    def total_sse(params):
        D1, D2 = params
        if D1 <= 1e-6 or D2 <= 1e-6 or D1 <= D2:
            return 1e12
        sse = 0.0
        for j, ik in enumerate(k_idx):
            k   = k_centers[ik]
            r   = R_fit[:, j]
            if not np.all(np.isfinite(r)):
                continue
            r_n = r / scales[j]
            e1  = np.exp(-D1 * k ** 2 * tau_fit)
            e2  = np.exp(-D2 * k ** 2 * tau_fit)
            # columns: [exp(-D1 k²τ), exp(-D2 k²τ), ones]
            # NNLS gives non-negative f1, f2, B
            X      = np.column_stack([e1, e2, np.ones(n_tau)])
            _, res = nnls(X, r_n)
            sse   += res ** 2
        return sse / len(k_idx)

    D1_grid = np.logspace(np.log10(D1_bounds[0] * 1.5),
                          np.log10(D1_bounds[1] * 0.9), 6)
    D2_grid = np.logspace(np.log10(D2_bounds[0] * 1.5),
                          np.log10(D2_bounds[1] * 0.9), 5)
    best_sse, best_p = np.inf, [D1_init, D2_init]
    for d1 in D1_grid:
        for d2 in D2_grid:
            if d1 <= d2:
                continue
            v = total_sse([d1, d2])
            if v < best_sse:
                best_sse, best_p = v, [d1, d2]

    result = minimize(
        total_sse,
        x0     = best_p,
        method = "L-BFGS-B",
        bounds = Bounds(lb=[D1_bounds[0], D2_bounds[0]],
                        ub=[D1_bounds[1], D2_bounds[1]]),
        options = dict(ftol=1e-14, gtol=1e-10, maxiter=2000),
    )
    D1_opt, D2_opt = result.x

    # Recover per-k amplitudes at optimal (D1, D2)
    n_centers = len(k_centers)
    f1_per_k  = np.full(n_centers, np.nan)
    f2_per_k  = np.full(n_centers, np.nan)
    B_per_k   = np.full(n_centers, np.nan)

    for j, ik in enumerate(k_idx):
        k  = k_centers[ik]
        r  = R_fit[:, j]
        if not np.all(np.isfinite(r)):
            continue
        e1 = np.exp(-D1_opt * k ** 2 * tau_fit)
        e2 = np.exp(-D2_opt * k ** 2 * tau_fit)
        X  = np.column_stack([e1, e2, np.ones(n_tau)])
        coeff, _ = nnls(X, r)
        f1_per_k[ik] = coeff[0]
        f2_per_k[ik] = coeff[1]
        B_per_k[ik]  = coeff[2]

    return dict(
        D1        = float(D1_opt),
        D2        = float(D2_opt),
        D_GEM     = float(D1_opt),
        D_slow    = float(D2_opt),
        f1_per_k  = f1_per_k,
        f2_per_k  = f2_per_k,
        B_per_k   = B_per_k,
        k_centers = k_centers,
        k_mask    = k_mask,
        taus      = taus,
        R_mat     = R_mat,
        residual  = float(result.fun),
        converged = bool(result.success),
        n_k_fit   = int(len(k_idx)),
    )


def plot_kics(
    fit1: dict,
    fit2: dict,
    tracking_D: float | None = None,
    movie_name: str = "",
    out_path: "Path | None" = None,
) -> None:
    """
    Four-panel kICS diagnostic figure.

    Panel 1: R(k,τ) curves at selected k values with single-component fits
    Panel 2: D(k) from single-component fit (analog of DDM D(q) plot)
    Panel 3: f1(k), f2(k) fractions from 2-component fit
    Panel 4: Normalised master curve (R-B)/(f1+f2) vs k²τ
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.cm import viridis

    k      = fit1["k_centers"]
    taus   = fit1["taus"]
    R      = fit1["R_mat"]
    mask1  = fit1["k_mask"]
    D_k    = fit1["D_per_k"]
    A_k    = fit1["A_per_k"]
    B_k1   = fit1["B_per_k"]

    D1     = fit2["D_GEM"]
    D2     = fit2["D_slow"]
    f1     = fit2["f1_per_k"]
    f2     = fit2["f2_per_k"]
    B_k2   = fit2["B_per_k"]
    mask2  = fit2["k_mask"]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(
        f"kICS analysis — {movie_name}\n"
        f"Single-comp D_med={fit1['D_median']:.4f} µm²/s    "
        f"Two-comp D_GEM={D1:.4f} µm²/s  D_slow={D2:.5f} µm²/s"
        + (f"    (tracking={tracking_D:.4f})" if tracking_D else ""),
        fontsize=10,
    )

    # ── Panel 1: R(k,τ) curves with single-component fits ────────────────────
    ax = axes[0, 0]
    valid_idx = np.where(mask1 & np.isfinite(D_k))[0]
    if len(valid_idx) >= 5:
        pct = np.percentile(np.arange(len(valid_idx)), [10, 30, 50, 70, 90]).astype(int)
        sel = valid_idx[np.clip(pct, 0, len(valid_idx) - 1)]
    else:
        sel = valid_idx
    colors = viridis(np.linspace(0.1, 0.9, max(len(sel), 1)))
    n_t    = min(len(taus), R.shape[0])
    for ik_abs, col in zip(sel, colors):
        ki = k[ik_abs]
        ax.plot(taus[:n_t], R[:n_t, ik_abs], "-", color=col, lw=1.2,
                label=f"k={ki:.1f}")
        if np.isfinite(D_k[ik_abs]) and np.isfinite(A_k[ik_abs]):
            r_fit = (A_k[ik_abs] * np.exp(-D_k[ik_abs] * ki ** 2 * taus[:n_t])
                     + B_k1[ik_abs])
            ax.plot(taus[:n_t], r_fit, "--", color=col, lw=0.8, alpha=0.7)
    ax.axhline(0, color="gray", lw=0.5, ls=":")
    ax.set_xlabel("τ  (s)")
    ax.set_ylabel("R(k,τ)")
    ax.set_title("kICS ACF curves + single-component fits")
    ax.legend(fontsize=7, ncol=2)

    # ── Panel 2: D(k) plot ────────────────────────────────────────────────────
    ax = axes[0, 1]
    fin = np.isfinite(D_k)
    ax.plot(k[~mask1 & fin], D_k[~mask1 & fin], "o", color="lightgray",
            ms=5, label="outside fit range")
    ax.plot(k[mask1 & fin], D_k[mask1 & fin], "o", color="steelblue",
            ms=6, label=f"fit range  median={fit1['D_median']:.4f} µm²/s")
    if np.isfinite(fit1["D_median"]):
        ax.axhline(fit1["D_median"], color="steelblue", lw=1.5, ls="--",
                   label=f"kICS D = {fit1['D_median']:.4f}")
    if tracking_D is not None:
        ax.axhline(tracking_D, color="red", lw=1.5, ls=":",
                   label=f"tracking D = {tracking_D:.4f}")
    k_lo_fit = float(k[mask1].min()) if mask1.any() else 6.0
    k_hi_fit = float(k[mask1].max()) if mask1.any() else 14.0
    ax.axvspan(k_lo_fit, k_hi_fit, alpha=0.08, color="steelblue")
    ax.set_xlabel("k  (rad/µm)")
    ax.set_ylabel("D(k)  (µm²/s)")
    ax.set_title("D(k) — flat region = single Brownian species")
    ax.legend(fontsize=8)
    ax.set_ylim(0, max(0.2, float(np.nanmax(D_k)) * 1.3))

    # ── Panel 3: f1(k), f2(k) from 2-component fit ───────────────────────────
    ax = axes[1, 0]
    dw = (k[1] - k[0]) * 0.4 if len(k) > 1 else 0.5
    v1 = mask2 & np.isfinite(f1)
    v2 = mask2 & np.isfinite(f2)
    ax.bar(k[v1] - dw, f1[v1], width=dw * 1.8, color="steelblue",
           alpha=0.75, label=f"f₁ (D_GEM={D1:.4f})")
    ax.bar(k[v2] + dw, f2[v2], width=dw * 1.8, color="darkorange",
           alpha=0.75, label=f"f₂ (D_slow={D2:.5f})")
    ax.set_xlabel("k  (rad/µm)")
    ax.set_ylabel("Fraction")
    ax.set_title("Two-component amplitudes per k bin")
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=8)

    # ── Panel 4: normalised master curve ──────────────────────────────────────
    ax = axes[1, 1]
    q_valid_idx = np.where(mask2 & np.isfinite(f1) & np.isfinite(f2))[0]
    cmap = viridis(np.linspace(0.1, 0.9, max(len(q_valid_idx), 1)))
    for ci, ik_abs in enumerate(q_valid_idx):
        ki   = k[ik_abs]
        r    = R[:n_t, ik_abs]
        denom = f1[ik_abs] + f2[ik_abs]
        if denom > 0 and np.isfinite(B_k2[ik_abs]):
            g_norm = (r - B_k2[ik_abs]) / denom
            k2tau  = ki ** 2 * taus[:n_t]
            ax.plot(k2tau, g_norm, ".", ms=2, alpha=0.35, color=cmap[ci])
    xref = np.linspace(0, 80, 200)
    ax.plot(xref, np.exp(-D1 * xref), "b-", lw=1.8,
            label=f"D_GEM={D1:.4f}")
    ax.plot(xref, np.exp(-D2 * xref), "r:", lw=1.5,
            label=f"D_slow={D2:.5f}")
    if tracking_D:
        ax.plot(xref, np.exp(-tracking_D * xref), "k--", lw=1.2,
                label=f"tracking={tracking_D:.4f}")
    ax.set_xlabel("k²τ  (rad²·s/µm²)")
    ax.set_ylabel("(R − B) / (f₁+f₂)")
    ax.set_title("Master curve (normalised)")
    ax.set_xlim(0, 80)
    ax.set_ylim(-0.15, 1.25)
    ax.legend(fontsize=8)

    fig.tight_layout()
    if out_path:
        fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
        import matplotlib.pyplot as _plt
        _plt.close(fig)
        print(f"   kICS plot → {out_path}")
    else:
        import matplotlib.pyplot as _plt
        _plt.show()


def analyse_kics(
    tiff_path: Path,
    tracking_D: float | None = None,
    out_dir: Path | None = None,
    pixel_um: float = PIXEL_UM,
    dt: float = DT,
    max_lag: int = MAX_LAG,
    n_k_bins: int = N_K_BINS,
    k_min_fit: float = 6.0,
    k_max_fit: float = 14.0,
    lag_min: int = 1,
    lag_max: int = 20,
    correct_bleaching_flag: bool = True,
) -> dict:
    """
    Full kICS pipeline on one TIFF movie.

    Steps
    -----
    1. Load + preprocess (bleach, background)
    2. Centre-crop to power-of-2 square
    3. Compute R(k,τ) for all k bins and lags
    4. Single-component fit D(k) per k bin
    5. Two-component fit (D_GEM + D_slow) shared across k bins
    6. Report results and save plots

    Returns dict with kics single-component fit (key 'fit1') and
    two-component fit (key 'fit2').
    """
    import time
    tiff_path = Path(tiff_path)
    name      = tiff_path.stem
    out       = Path(out_dir or tiff_path.parent / "kics_results")
    out.mkdir(parents=True, exist_ok=True)

    print(f"\n[kICS] {name}")
    print("[1/4] Loading + preprocessing …")
    movie_raw = load_tiff_stack(tiff_path)
    movie_pp  = correct_bleaching(movie_raw) if correct_bleaching_flag else movie_raw.copy()
    movie_pp  = subtract_background(movie_pp)
    del movie_raw
    movie_c   = _fft_crop(movie_pp)
    T, Y, X   = movie_c.shape
    print(f"      {T}×{Y}×{X}  (cropped to power-of-2)")

    print("[2/4] Computing R(k,τ) …")
    t0 = time.time()
    R_mat, k_centers, taus = compute_kics_acf(
        movie_c, max_lag=max_lag, pixel_um=pixel_um, dt=dt, n_k_bins=n_k_bins,
    )
    print(f"      done in {time.time()-t0:.1f}s")

    print(f"[3/4] Single-component fit over k={k_min_fit:.0f}–{k_max_fit:.0f} rad/µm …")
    fit1 = fit_kics_per_k(R_mat, k_centers, taus,
                          k_min_fit=k_min_fit, k_max_fit=k_max_fit,
                          lag_min=lag_min, lag_max=lag_max)

    print(f"\n   kICS D (median, k={k_min_fit:.0f}–{k_max_fit:.0f} rad/µm): "
          f"{fit1['D_median']:.5f} µm²/s  "
          f"(mean={fit1['D_mean']:.5f}, sem={fit1['D_sem']:.5f}, "
          f"n_k={fit1['n_valid_k']})")
    if tracking_D is not None and np.isfinite(fit1["D_median"]):
        print(f"   Tracking D:  {tracking_D:.5f}  ratio = {fit1['D_median']/tracking_D:.3f}×")

    print("\n   D(k) table:")
    print(f"   {'k (rad/µm)':>12}  {'τ_c_GEM(s)':>10}  {'τ_c_fast(s)':>12}  "
          f"{'D_fit (µm²/s)':>14}  in_range?")
    for ik, kv in enumerate(k_centers):
        tc_gem  = 1.0 / (0.044 * kv ** 2) if kv > 0 else np.inf
        tc_fast = 1.0 / (0.45  * kv ** 2) if kv > 0 else np.inf
        D       = fit1["D_per_k"][ik]
        flag    = "  ←" if fit1["k_mask"][ik] else ""
        Dstr    = f"{D:.4f}" if np.isfinite(D) else "  ---"
        print(f"   {kv:>12.2f}  {tc_gem:>10.3f}  {tc_fast:>12.4f}  {Dstr:>14}{flag}")

    k2_min, k2_max, lag2_min = 6.0, 14.0, 3
    print(f"\n[3b] Two-component kICS fit  "
          f"k={k2_min:.0f}–{k2_max:.0f} rad/µm, lag {lag2_min}–{max_lag} …")
    try:
        fit2 = fit_kics_two_component(
            R_mat, k_centers, taus,
            k_min_fit=k2_min, k_max_fit=k2_max,
            lag_min=lag2_min, lag_max=max_lag,
            D1_init=0.044, D2_init=0.005,
            D1_bounds=(0.01, 0.18), D2_bounds=(0.0005, 0.025),
        )
        print(f"   D_GEM  = {fit2['D_GEM']:.5f} µm²/s"
              + (f"   (ratio = {fit2['D_GEM']/tracking_D:.3f}×  "
                 f"tracking={tracking_D:.5f})"
                 if tracking_D else ""))
        print(f"   D_slow = {fit2['D_slow']:.5f} µm²/s")
        print(f"   converged={fit2['converged']}  residual={fit2['residual']:.4e}"
              f"  n_k={fit2['n_k_fit']}")
    except Exception as exc:
        print(f"   Two-component fit failed: {exc}")
        fit2 = None

    print("[4/4] Saving plots …")
    if fit2 is not None:
        plot_kics(fit1, fit2, tracking_D=tracking_D,
                  movie_name=name,
                  out_path=out / f"{name}_kics.png")

    return dict(
        movie_name = name,
        movie_pp   = movie_pp,
        R_mat      = R_mat,
        k_centers  = k_centers,
        taus       = taus,
        fit1       = fit1,
        fit2       = fit2,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# NN-MSD core
# ═══════════════════════════════════════════════════════════════════════════════

def detect_spots_all_frames(
    movie: np.ndarray,
    sigma_px: float = 1.5,
    threshold_sigma: float = 5.0,
    pixel_um: float = PIXEL_UM,
) -> list:
    """
    Detect spots in every frame using DoG (reuses spots_stics logic).

    Parameters
    ----------
    movie          : (T, Y, X) float32, bleach + background corrected
    sigma_px       : inner Gaussian σ in pixels (PSF scale)
    threshold_sigma: per-frame threshold = mean + k·std of DoG response
    pixel_um       : pixel size in µm (used to convert coords to µm)

    Returns
    -------
    spots_list : list of T arrays, each (N_f, 2) float64 of (y, x) in µm
    """
    T = movie.shape[0]
    # detect_spots_per_frame returns (N, 3) int32: [frame, y_px, x_px]
    raw = detect_spots_per_frame(
        movie,
        sigma_px        = sigma_px,
        threshold_sigma = threshold_sigma,
    )

    spots_list = [np.empty((0, 2), dtype=np.float64) for _ in range(T)]
    if len(raw) == 0:
        return spots_list

    for t in range(T):
        mask = raw[:, 0] == t
        yx_px = raw[mask, 1:3].astype(np.float64)
        spots_list[t] = yx_px * pixel_um

    return spots_list


def _auto_r_max(spots_list: list, pixel_um: float = PIXEL_UM) -> float:
    """
    Estimate r_max from mean inter-particle distance in the first 20 frames.

    r_max = min(0.5 µm, 0.3 × mean_inter_particle_distance)
    """
    n_frames = min(20, len(spots_list))
    densities = []
    for t in range(n_frames):
        pts = spots_list[t]
        if len(pts) > 1:
            # Approximate particle density from frame area and count
            densities.append(len(pts))

    if not densities:
        return 0.5

    mean_n   = float(np.mean(densities))
    # Estimate FOV area from first non-empty frame
    frame_um2 = None
    for t in range(n_frames):
        if len(spots_list[t]) > 0:
            pts = spots_list[t]
            span_y = pts[:, 0].max() - pts[:, 0].min() if len(pts) > 1 else 0.0
            span_x = pts[:, 1].max() - pts[:, 1].min() if len(pts) > 1 else 0.0
            if span_y > 0 and span_x > 0:
                frame_um2 = span_y * span_x
                break

    if frame_um2 is None or frame_um2 < 1.0:
        return 0.5

    density      = mean_n / frame_um2          # particles / µm²
    mean_ipd     = 1.0 / np.sqrt(density)      # mean inter-particle distance µm
    r_max        = min(0.5, 0.3 * mean_ipd)
    return max(r_max, 0.15)                    # floor at 0.15 µm


def compute_nn_displacements(
    spots_list: list,
    max_lag: int = MAX_LAG,
    dt: float = DT,
    r_max_um: float | None = None,
    pixel_um: float = PIXEL_UM,
) -> tuple[list, np.ndarray]:
    """
    For each lag τ = 1..max_lag frames, find nearest-neighbour displacement
    between particles in frame t and frame t+τ.

    For each particle in frame t the nearest particle in frame t+τ is found
    via cKDTree.  Pairs with distance > r_max_um are discarded to reject
    false cross-particle assignments.

    Parameters
    ----------
    spots_list : list of (N_f, 2) float64 arrays, coords in µm
    max_lag    : maximum lag in frames
    dt         : frame interval in s
    r_max_um   : maximum displacement to accept (µm). None → auto from density
    pixel_um   : pixel size in µm (passed to _auto_r_max if r_max_um is None)

    Returns
    -------
    r2_per_lag  : list of max_lag arrays, r2_per_lag[tau-1] = r² values (µm²)
    n_pairs_vec : (max_lag,) int64, number of accepted pairs per lag
    """
    T = len(spots_list)

    if r_max_um is None:
        r_max_um = _auto_r_max(spots_list, pixel_um=pixel_um)

    r2_per_lag  = [[] for _ in range(max_lag)]
    n_pairs_vec = np.zeros(max_lag, dtype=np.int64)

    for tau in range(1, max_lag + 1):
        r2_list = []
        for t in range(T - tau):
            pts_t    = spots_list[t]
            pts_ttau = spots_list[t + tau]
            if len(pts_t) == 0 or len(pts_ttau) == 0:
                continue
            tree = cKDTree(pts_ttau)
            dists, _ = tree.query(pts_t, k=1, distance_upper_bound=r_max_um)
            valid = dists < r_max_um
            r2_list.append(dists[valid] ** 2)

        if r2_list:
            r2_per_lag[tau - 1] = np.concatenate(r2_list).astype(np.float64)
        else:
            r2_per_lag[tau - 1] = np.empty(0, dtype=np.float64)
        n_pairs_vec[tau - 1] = len(r2_per_lag[tau - 1])

    return r2_per_lag, n_pairs_vec


def fit_displacement_distribution(
    r2_per_lag: list,
    taus: np.ndarray,
    max_lag_fit: int = 15,
    dt: float = DT,
) -> dict:
    """
    Fit displacement distributions to extract diffusion coefficients.

    Single-exponential fit per lag
    --------------------------------
    For 2-D Brownian motion r² ~ Exponential(MSD) where MSD = 4Dτ.
    p(r²) = (1/MSD) · exp(-r²/MSD) → log-likelihood maximised by MSD = mean(r²).
    D_per_tau(τ) = mean(r²) / (4τ).

    Global two-component fit
    ------------------------
    p(r²|τ) = f₁(τ)·(1/MSD₁)·exp(-r²/MSD₁) + f₂(τ)·(1/MSD₂)·exp(-r²/MSD₂)
    with MSD_i = 4·D_i·τ, D₁ > D₂ shared across τ; f₁+f₂=1 per τ.
    Optimised over (D₁, D₂) with outer grid + L-BFGS-B; inner NNLS for fractions.

    Parameters
    ----------
    r2_per_lag  : list of arrays from compute_nn_displacements
    taus        : (max_lag,) lag times in s
    max_lag_fit : maximum lag index to include in fits
    dt          : frame interval (used for axis labels only)

    Returns
    -------
    dict with D_per_tau, MSD_per_tau, n_pairs, D_mean, D_median,
    fit2c dict with D1, D2, f1_per_tau, f2_per_tau.
    """
    n_lag    = min(max_lag_fit, len(r2_per_lag))
    tau_fit  = taus[:n_lag]

    MSD_per_tau = np.full(n_lag, np.nan)
    D_per_tau   = np.full(n_lag, np.nan)
    n_pairs     = np.zeros(n_lag, dtype=np.int64)

    for i in range(n_lag):
        r2 = r2_per_lag[i]
        if len(r2) < 5:
            continue
        msd = float(r2.mean())
        MSD_per_tau[i] = msd
        D_per_tau[i]   = msd / (4.0 * tau_fit[i])
        n_pairs[i]     = len(r2)

    # Weighted mean D (weight by number of pairs)
    valid    = np.isfinite(D_per_tau) & (n_pairs > 0)
    wts      = n_pairs[valid].astype(np.float64)
    D_mean   = float(np.average(D_per_tau[valid], weights=wts)) if valid.any() else np.nan
    D_median = float(np.nanmedian(D_per_tau[valid])) if valid.any() else np.nan

    # Linear MSD(τ) = 4D·τ + C fit
    D_lin, C_lin = np.nan, np.nan
    if valid.sum() >= 3:
        try:
            def _lin(tau, D, C):
                return 4.0 * D * tau + C
            popt, _ = curve_fit(_lin, tau_fit[valid], MSD_per_tau[valid],
                                p0=[D_mean, 0.0],
                                bounds=([1e-6, -0.5], [5.0, 2.0]),
                                sigma=1.0 / np.sqrt(wts), maxfev=5000)
            D_lin, C_lin = float(popt[0]), float(popt[1])
        except Exception:
            pass

    # Two-component global fit over (D1, D2)
    def _neg_ll_two_comp(params, r2_list, taus_list):
        D1, D2 = params
        if D1 <= 0 or D2 <= 0 or D1 <= D2:
            return 1e12
        total = 0.0
        for r2, tau in zip(r2_list, taus_list):
            if len(r2) < 5:
                continue
            msd1 = 4.0 * D1 * tau
            msd2 = 4.0 * D2 * tau
            p1   = np.exp(-r2 / msd1) / msd1
            p2   = np.exp(-r2 / msd2) / msd2
            # Per-τ fraction via NNLS on stacked log-likelihood gradient
            # Approximate: minimise sum of squared residuals on CDF
            # Use NNLS on [p1_i, p2_i] to find (f1, f2) with f1+f2 = 1 constraint
            n    = len(r2)
            cols = np.column_stack([p1, p2])
            ones = np.ones(n)
            # Constrained: f1 + f2 = 1, so use substitution f2 = 1 - f1
            # Minimise ||f1·p1 + (1-f1)·p2 - 0||² is not right (no data target)
            # Instead: NNLS to find mixture that best fits empirical histogram
            hist_bins  = min(30, max(5, n // 10))
            r2_max_h   = float(r2.max())
            edges      = np.linspace(0.0, r2_max_h * 1.05, hist_bins + 1)
            counts, _  = np.histogram(r2, bins=edges)
            mid        = 0.5 * (edges[:-1] + edges[1:])
            widths     = np.diff(edges)
            h1         = np.exp(-mid / msd1) / msd1 * widths * n
            h2         = np.exp(-mid / msd2) / msd2 * widths * n
            X          = np.column_stack([h1, h2])
            coeff, res = nnls(X, counts.astype(np.float64))
            total     += res ** 2 / max(n, 1)
        return total / max(n_lag, 1)

    r2_list_fit  = [r2_per_lag[i] for i in range(n_lag) if len(r2_per_lag[i]) >= 5]
    taus_list_fit = [tau_fit[i]   for i in range(n_lag) if len(r2_per_lag[i]) >= 5]

    fit2c = dict(D1=np.nan, D2=np.nan, f1_per_tau=None, f2_per_tau=None,
                 converged=False, residual=np.nan)

    if len(r2_list_fit) >= 3 and np.isfinite(D_mean):
        D1_init = max(D_mean * 1.5, 0.03)
        D2_init = max(D_mean * 0.3, 0.005)
        D1_bounds2 = (0.01, 0.5)
        D2_bounds2 = (0.0005, 0.05)
        D1_grid = np.logspace(np.log10(max(D1_bounds2[0], D1_init * 0.5)),
                              np.log10(D1_bounds2[1]), 5)
        D2_grid = np.logspace(np.log10(D2_bounds2[0]),
                              np.log10(min(D2_bounds2[1], D1_init * 0.5)), 4)
        best_sse2, best_p2 = np.inf, [D1_init, D2_init]
        for d1 in D1_grid:
            for d2 in D2_grid:
                if d1 <= d2:
                    continue
                v = _neg_ll_two_comp([d1, d2], r2_list_fit, taus_list_fit)
                if v < best_sse2:
                    best_sse2, best_p2 = v, [d1, d2]

        try:
            res2 = minimize(
                _neg_ll_two_comp,
                x0     = best_p2,
                args   = (r2_list_fit, taus_list_fit),
                method = "L-BFGS-B",
                bounds = Bounds(lb=[D1_bounds2[0], D2_bounds2[0]],
                                ub=[D1_bounds2[1], D2_bounds2[1]]),
                options = dict(ftol=1e-12, gtol=1e-8, maxiter=1000),
            )
            D1_opt, D2_opt = res2.x

            # Recover per-tau fractions
            f1_t = np.full(n_lag, np.nan)
            f2_t = np.full(n_lag, np.nan)
            for i in range(n_lag):
                r2 = r2_per_lag[i]
                if len(r2) < 5:
                    continue
                tau    = tau_fit[i]
                msd1   = 4.0 * D1_opt * tau
                msd2   = 4.0 * D2_opt * tau
                n      = len(r2)
                hist_bins = min(30, max(5, n // 10))
                r2_max_h  = float(r2.max())
                edges     = np.linspace(0.0, r2_max_h * 1.05, hist_bins + 1)
                counts, _ = np.histogram(r2, bins=edges)
                mid       = 0.5 * (edges[:-1] + edges[1:])
                widths    = np.diff(edges)
                h1        = np.exp(-mid / msd1) / msd1 * widths * n
                h2        = np.exp(-mid / msd2) / msd2 * widths * n
                X         = np.column_stack([h1, h2])
                coeff, _  = nnls(X, counts.astype(np.float64))
                norm      = coeff[0] + coeff[1]
                if norm > 0:
                    f1_t[i] = coeff[0] / norm
                    f2_t[i] = coeff[1] / norm

            fit2c = dict(
                D1         = float(D1_opt),
                D2         = float(D2_opt),
                f1_per_tau = f1_t,
                f2_per_tau = f2_t,
                converged  = bool(res2.success),
                residual   = float(res2.fun),
            )
        except Exception as exc:
            fit2c["error"] = str(exc)

    return dict(
        D_per_tau   = D_per_tau,
        MSD_per_tau = MSD_per_tau,
        n_pairs     = n_pairs,
        taus        = tau_fit,
        D_mean      = D_mean,
        D_median    = D_median,
        D_lin       = D_lin,
        C_lin       = C_lin,
        fit2c       = fit2c,
    )


def plot_nn_msd(
    results: dict,
    r2_per_lag: list,
    tracking_D: float | None = None,
    movie_name: str = "",
    out_path: "Path | None" = None,
) -> None:
    """
    Four-panel NN-MSD diagnostic figure.

    Panel 1: r² histograms at selected τ values
    Panel 2: MSD(τ) = mean(r²) vs τ, fit line
    Panel 3: D_per_tau(τ) vs τ (bias diagnostic — fast species under-counted)
    Panel 4: N_pairs vs τ (pair count drop reveals escaping fast species)
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.cm import plasma

    taus        = results["taus"]
    MSD         = results["MSD_per_tau"]
    D_pt        = results["D_per_tau"]
    n_pairs     = results["n_pairs"]
    D_mean      = results["D_mean"]
    D_lin       = results["D_lin"]
    C_lin       = results["C_lin"]
    fit2c       = results["fit2c"]
    n_lag       = len(taus)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(
        f"NN-MSD analysis — {movie_name}\n"
        f"D_mean={D_mean:.4f} µm²/s  D_lin={D_lin:.4f} µm²/s"
        + (f"    (tracking={tracking_D:.4f})" if tracking_D else ""),
        fontsize=10,
    )

    # ── Panel 1: r² histograms at selected τ ──────────────────────────────────
    ax = axes[0, 0]
    valid_idx = [i for i in range(n_lag) if len(r2_per_lag[i]) >= 5]
    if valid_idx:
        pct = np.percentile(np.arange(len(valid_idx)), [10, 33, 66, 90]).astype(int)
        sel_i = [valid_idx[p] for p in np.clip(pct, 0, len(valid_idx) - 1)]
        colors = plasma(np.linspace(0.1, 0.9, len(sel_i)))
        for ci, i in enumerate(sel_i):
            r2 = r2_per_lag[i]
            tau = taus[i]
            r2_max = float(np.percentile(r2, 99))
            bins   = np.linspace(0, r2_max * 1.05, 30)
            counts, edges = np.histogram(r2, bins=bins, density=True)
            mid = 0.5 * (edges[:-1] + edges[1:])
            ax.step(mid, counts, where="mid", color=colors[ci], lw=1.2,
                    label=f"τ={tau:.2f}s")
            # Overlay single-exp fit
            msd = float(r2.mean())
            if msd > 0:
                xplot = np.linspace(0, r2_max * 1.05, 200)
                ax.plot(xplot, np.exp(-xplot / msd) / msd, "--",
                        color=colors[ci], lw=0.8, alpha=0.7)
    ax.set_xlabel("r²  (µm²)")
    ax.set_ylabel("p(r²)  (µm⁻²)")
    ax.set_title("r² histograms at selected lags")
    ax.legend(fontsize=7)

    # ── Panel 2: MSD(τ) vs τ ──────────────────────────────────────────────────
    ax = axes[0, 1]
    valid = np.isfinite(MSD)
    ax.plot(taus[valid], MSD[valid], "o", color="steelblue", ms=5,
            label="mean(r²)")
    t_ref = np.linspace(0, taus[valid].max() if valid.any() else 1.0, 200)
    if np.isfinite(D_lin) and np.isfinite(C_lin):
        ax.plot(t_ref, 4.0 * D_lin * t_ref + C_lin, "b-", lw=1.5,
                label=f"4Dτ+C  D={D_lin:.4f}")
    if tracking_D is not None:
        ax.plot(t_ref, 4.0 * tracking_D * t_ref, "k:", lw=1.5,
                label=f"tracking D={tracking_D:.4f}")
    if np.isfinite(fit2c.get("D1", np.nan)):
        D1 = fit2c["D1"]
        D2 = fit2c["D2"]
        ax.plot(t_ref, 4.0 * D1 * t_ref, "r--", lw=1.2, alpha=0.7,
                label=f"D1(GEM)={D1:.4f}")
        ax.plot(t_ref, 4.0 * D2 * t_ref, "g--", lw=1.2, alpha=0.7,
                label=f"D2(slow)={D2:.5f}")
    ax.set_xlabel("τ  (s)")
    ax.set_ylabel("MSD  (µm²)")
    ax.set_title("MSD(τ)")
    ax.legend(fontsize=8)
    ax.set_ylim(bottom=0)

    # ── Panel 3: D_per_tau vs τ ───────────────────────────────────────────────
    ax = axes[1, 0]
    valid_d = np.isfinite(D_pt)
    ax.plot(taus[valid_d], D_pt[valid_d], "s-", color="darkorange", ms=5,
            label="D(τ) = MSD/(4τ)")
    if np.isfinite(D_mean):
        ax.axhline(D_mean, color="darkorange", lw=1.5, ls="--",
                   label=f"D_mean={D_mean:.4f}")
    if tracking_D is not None:
        ax.axhline(tracking_D, color="red", lw=1.5, ls=":",
                   label=f"tracking={tracking_D:.4f}")
    ax.set_xlabel("τ  (s)")
    ax.set_ylabel("D(τ)  (µm²/s)")
    ax.set_title("D(τ) — should be flat for pure Brownian; bias toward slow at early τ")
    ax.legend(fontsize=8)
    ax.set_ylim(bottom=0)

    # ── Panel 4: N_pairs vs τ ─────────────────────────────────────────────────
    ax = axes[1, 1]
    ax.bar(taus, n_pairs, width=float(taus[1] - taus[0]) * 0.8 if len(taus) > 1 else 0.08,
           color="steelblue", alpha=0.7, label="N_pairs")
    ax.set_xlabel("τ  (s)")
    ax.set_ylabel("N pairs accepted")
    ax.set_title("Pair count — drop reveals fast species escaping r_max")
    ax.legend(fontsize=8)

    fig.tight_layout()
    if out_path:
        fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
        import matplotlib.pyplot as _plt
        _plt.close(fig)
        print(f"   NN-MSD plot → {out_path}")
    else:
        import matplotlib.pyplot as _plt
        _plt.show()


def analyse_nn_msd(
    tiff_path: Path,
    tracking_D: float | None = None,
    out_dir: Path | None = None,
    pixel_um: float = PIXEL_UM,
    dt: float = DT,
    max_lag: int = MAX_LAG,
    sigma_px: float = 1.5,
    threshold_sigma: float = 5.0,
    r_max_um: float | None = None,
    max_lag_fit: int = 15,
    correct_bleaching_flag: bool = True,
) -> dict:
    """
    Full NN-MSD pipeline on one TIFF movie.

    Steps
    -----
    1. Load + preprocess (bleach, background)
    2. DoG spot detection per frame (reuses spots_stics logic)
    3. Compute NN displacement r² for each lag
    4. Fit displacement distributions (single-exp per lag + global 2-component)
    5. Report results and save plots

    Returns dict with spots_list, r2_per_lag, n_pairs, fit results.
    """
    import time
    tiff_path = Path(tiff_path)
    name      = tiff_path.stem
    out       = Path(out_dir or tiff_path.parent / "nn_msd_results")
    out.mkdir(parents=True, exist_ok=True)

    print(f"\n[NN-MSD] {name}")
    print("[1/4] Loading + preprocessing …")
    movie_raw = load_tiff_stack(tiff_path)
    movie_pp  = correct_bleaching(movie_raw) if correct_bleaching_flag else movie_raw.copy()
    movie_pp  = subtract_background(movie_pp)
    del movie_raw
    T, Y, X   = movie_pp.shape
    print(f"      {T}×{Y}×{X}")

    print(f"[2/4] DoG spot detection (σ={sigma_px:.1f} px, thresh={threshold_sigma:.1f}σ) …")
    t0 = time.time()
    spots_list = detect_spots_all_frames(
        movie_pp,
        sigma_px        = sigma_px,
        threshold_sigma = threshold_sigma,
        pixel_um        = pixel_um,
    )
    n_total   = sum(len(s) for s in spots_list)
    mean_n    = n_total / T if T > 0 else 0.0
    print(f"      {n_total:,} detections  |  {mean_n:.1f} spots/frame  "
          f"(done in {time.time()-t0:.1f}s)")

    if n_total == 0:
        raise RuntimeError("No spots detected. Lower threshold_sigma.")

    if r_max_um is None:
        r_max_um = _auto_r_max(spots_list, pixel_um=pixel_um)
    print(f"      r_max = {r_max_um:.3f} µm")

    print(f"[3/4] Computing NN displacements (max_lag={max_lag}) …")
    t0 = time.time()
    r2_per_lag, n_pairs_vec = compute_nn_displacements(
        spots_list, max_lag=max_lag, dt=dt,
        r_max_um=r_max_um, pixel_um=pixel_um,
    )
    print(f"      done in {time.time()-t0:.1f}s")
    print(f"      Pairs at lag 1: {n_pairs_vec[0]:,}  "
          f"lag {max_lag}: {n_pairs_vec[max_lag-1]:,}")

    taus = np.arange(1, max_lag + 1, dtype=np.float64) * dt
    print(f"[3b] Fitting displacement distributions …")
    fit_res = fit_displacement_distribution(
        r2_per_lag, taus, max_lag_fit=max_lag_fit, dt=dt,
    )

    print(f"\n   NN-MSD results:")
    print(f"   D_mean   = {fit_res['D_mean']:.5f} µm²/s")
    print(f"   D_median = {fit_res['D_median']:.5f} µm²/s")
    print(f"   D_lin    = {fit_res['D_lin']:.5f} µm²/s  (MSD=4Dτ+C fit)")
    if tracking_D is not None and np.isfinite(fit_res["D_lin"]):
        print(f"   Tracking D: {tracking_D:.5f}  ratio = {fit_res['D_lin']/tracking_D:.3f}×")
    f2c = fit_res["fit2c"]
    if np.isfinite(f2c.get("D1", np.nan)):
        print(f"   2-comp D_GEM  = {f2c['D1']:.5f} µm²/s")
        print(f"   2-comp D_slow = {f2c['D2']:.5f} µm²/s")
        print(f"   converged={f2c['converged']}  residual={f2c['residual']:.4e}")

    print("\n   D(τ) table:")
    print(f"   {'lag':>4}  {'tau_s':>6}  {'N_pairs':>8}  {'MSD (µm²)':>11}  "
          f"{'D_eff (µm²/s)':>14}")
    for i in range(min(max_lag_fit, len(taus))):
        tau  = taus[i]
        msd  = fit_res["MSD_per_tau"][i]
        deff = fit_res["D_per_tau"][i]
        np_  = n_pairs_vec[i]
        msd_s  = f"{msd:.5f}" if np.isfinite(msd) else "  ---"
        deff_s = f"{deff:.5f}" if np.isfinite(deff) else "  ---"
        print(f"   {i+1:>4}  {tau:>6.2f}  {np_:>8}  {msd_s:>11}  {deff_s:>14}")

    print("[4/4] Saving plots …")
    plot_nn_msd(
        fit_res, r2_per_lag,
        tracking_D  = tracking_D,
        movie_name  = name,
        out_path    = out / f"{name}_nn_msd.png",
    )

    return dict(
        movie_name  = name,
        spots_list  = spots_list,
        r2_per_lag  = r2_per_lag,
        n_pairs_vec = n_pairs_vec,
        taus        = taus,
        r_max_um    = r_max_um,
        fit         = fit_res,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Top-level comparison
# ═══════════════════════════════════════════════════════════════════════════════

def compare_all_methods(
    tiff_path: Path,
    tracking_D: float | None = None,
    out_dir: Path | None = None,
    pixel_um: float = PIXEL_UM,
    dt: float = DT,
    max_lag: int = MAX_LAG,
    run_ddm: bool = True,
    run_kics: bool = True,
    run_nn_msd: bool = True,
) -> dict:
    """
    Run all tracking-free diffusion methods and print a comparison table.

    Loads DDM results if a pre-existing *_ddm_results.npz is found; otherwise
    runs DDM fresh.  Runs kICS and NN-MSD unconditionally if requested.

    Returns dict with keys 'ddm', 'kics', 'nn_msd', each being the result
    dict from the corresponding analyse_* function (or None if skipped).
    """
    import time
    tiff_path = Path(tiff_path)
    name      = tiff_path.stem
    out       = Path(out_dir or tiff_path.parent / "comparison_results")
    out.mkdir(parents=True, exist_ok=True)

    results = dict(ddm=None, kics=None, nn_msd=None)

    t_total = time.time()

    # ── DDM ──────────────────────────────────────────────────────────────────
    if run_ddm:
        from ddm_analysis import analyse_ddm
        print("\n" + "=" * 60)
        print("DDM")
        print("=" * 60)
        try:
            results["ddm"] = analyse_ddm(
                tiff_path, tracking_D=tracking_D,
                out_dir=out / "ddm", pixel_um=pixel_um, dt=dt,
                max_lag=max_lag,
            )
        except Exception as exc:
            print(f"   DDM failed: {exc}")

    # ── kICS ─────────────────────────────────────────────────────────────────
    if run_kics:
        print("\n" + "=" * 60)
        print("kICS")
        print("=" * 60)
        try:
            results["kics"] = analyse_kics(
                tiff_path, tracking_D=tracking_D,
                out_dir=out / "kics", pixel_um=pixel_um, dt=dt,
                max_lag=max_lag,
            )
        except Exception as exc:
            print(f"   kICS failed: {exc}")

    # ── NN-MSD ────────────────────────────────────────────────────────────────
    if run_nn_msd:
        print("\n" + "=" * 60)
        print("NN-MSD")
        print("=" * 60)
        try:
            results["nn_msd"] = analyse_nn_msd(
                tiff_path, tracking_D=tracking_D,
                out_dir=out / "nn_msd", pixel_um=pixel_um, dt=dt,
                max_lag=max_lag,
            )
        except Exception as exc:
            print(f"   NN-MSD failed: {exc}")

    # ── Comparison table ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("COMPARISON TABLE")
    print("=" * 60)
    print(f"   Movie : {name}")
    if tracking_D is not None:
        print(f"   Tracking D (reference) = {tracking_D:.5f} µm²/s")
    print()
    print(f"   {'Method':<22}  {'D (µm²/s)':>12}  {'Type':>18}  {'ratio/tracking':>14}")
    print("   " + "-" * 72)

    def _row(method_name, D_val, D_type):
        if not np.isfinite(D_val):
            print(f"   {method_name:<22}  {'---':>12}  {D_type:>18}  {'---':>14}")
            return
        ratio_s = f"{D_val/tracking_D:.3f}×" if tracking_D else "N/A"
        print(f"   {method_name:<22}  {D_val:>12.5f}  {D_type:>18}  {ratio_s:>14}")

    if results["ddm"] is not None:
        ddm = results["ddm"]
        _row("DDM single-comp", ddm.get("D_median", np.nan), "per-q median")
        if ddm.get("fit2") is not None:
            _row("DDM 2-comp D_GEM", ddm["fit2"].get("D_GEM", np.nan), "shared across q")
            _row("DDM 2-comp D_slow", ddm["fit2"].get("D_slow", np.nan), "shared across q")

    if results["kics"] is not None:
        kics = results["kics"]
        if kics.get("fit1") is not None:
            _row("kICS single-comp", kics["fit1"].get("D_median", np.nan), "per-k median")
        if kics.get("fit2") is not None:
            _row("kICS 2-comp D_GEM", kics["fit2"].get("D_GEM", np.nan), "shared across k")
            _row("kICS 2-comp D_slow", kics["fit2"].get("D_slow", np.nan), "shared across k")

    if results["nn_msd"] is not None:
        nn = results["nn_msd"]
        fit = nn.get("fit", {})
        _row("NN-MSD D_lin", fit.get("D_lin", np.nan), "MSD=4Dτ+C fit")
        _row("NN-MSD D_mean", fit.get("D_mean", np.nan), "weighted mean")
        f2c = fit.get("fit2c", {})
        if np.isfinite(f2c.get("D1", np.nan)):
            _row("NN-MSD 2-comp D_GEM", f2c.get("D1", np.nan), "shared across τ")
            _row("NN-MSD 2-comp D_slow", f2c.get("D2", np.nan), "shared across τ")

    if tracking_D is not None:
        print(f"\n   {'Tracking':22}  {tracking_D:>12.5f}  {'SPT reference':>18}  {'1.000×':>14}")

    print(f"\n   Total elapsed: {time.time()-t_total:.0f}s")
    print(f"   Output → {out}/")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys, time

    tiff       = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/Em1_crop.tif")
    tracking_D = float(sys.argv[2]) if len(sys.argv) > 2 else 0.04374
    out_dir    = Path("/tmp/kics_nn_out")
    out_dir.mkdir(exist_ok=True)

    t0 = time.time()
    compare_all_methods(
        tiff, tracking_D=tracking_D, out_dir=out_dir,
        pixel_um=PIXEL_UM, dt=DT, max_lag=MAX_LAG,
        run_ddm=False,
    )
    print(f"\n   Total: {time.time()-t0:.0f}s   Output → {out_dir}/")
