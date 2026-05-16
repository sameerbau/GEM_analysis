"""
jdd_analysis.py — Improved NN/jump-distance analysis + TICS for GEM diffusion.

Three complementary approaches that extend the basic NN-MSD:

1. Brightness-filtered NN-MSD
   GEM 40-mers are ~40× brighter than monomers/free GFP → select top-percentile
   spots by integrated intensity → NN-MSD on the GEM-enriched subset eliminates
   the D≈0.15 contamination seen in the unfiltered version.

2. Jump Distance Distribution (JDD) with EM
   Fit the full r² distribution at each lag (not just its mean) to a mixture
   of exponentials with shared D₁, D₂ across τ values.  The EM algorithm is
   robust to initial fractions and naturally separates species whose MSDs
   differ by ≥ 2–3×.  Uses all ~73 K pairs per lag for high statistical power.

3. TICS (Temporal Image Correlation Spectroscopy)
   G(τ) = ⟨δI(t)·δI(t+τ)⟩_t / ⟨I(t)⟩²_t  per tile.
   For 2-D Brownian with Gaussian PSF waist w₀:
       G(τ) = G₀ / (1 + τ/τ_D)     with τ_D = w₀² / (4D)
   Complementary to iMSD: uses the TEMPORAL decay (not spatial spread).
   Advantage: one scalar fit per tile, no PSF deconvolution needed if we
   treat τ_D as a free parameter and use w₀ only to convert to D.
   Returns a D-map analogous to the tiled iMSD D-map.

Relationship between these methods
------------------------------------
iMSD: tracks spatial width w²(τ) of STICS ACF → D from slope 4D
TICS: tracks temporal decay rate Γ = 1/τ_D of per-pixel ACF → D = w₀²/4τ_D
DDM:  tracks S(q,τ) = A(1-exp(-Dq²τ)) + B in k-space
kICS: tracks R(k,τ) = exp(-Dk²τ) — normalised DDM

All are related but the TICS D-map and JDD-EM give independent estimates.
"""

import warnings
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter
from scipy.optimize import curve_fit, minimize, Bounds

# ── constants ────────────────────────────────────────────────────────────────
PIXEL_UM   = 0.094
DT         = 0.1
W0_UM      = 0.28       # PSF 1/e² waist radius (= 2σ_PSF at 0.094 µm/px)
MAX_LAG    = 30
SIGMA_PX   = 1.5
THRESH_K   = 5.0
R_MAX_UM   = 0.50
TILE_SIZE  = 32         # pixels per TICS tile


# ═══════════════════════════════════════════════════════════════════════════════
# Shared helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _load_preprocess(tiff_path: Path, fft_crop: bool = False) -> np.ndarray:
    import tifffile
    movie = tifffile.imread(str(tiff_path)).astype(np.float32)
    if movie.ndim == 4:
        movie = movie[:, 0]
    # Bleach correction
    means = movie.mean(axis=(1, 2)).clip(min=1e-6)
    movie = movie * (means[0] / means)[:, None, None]
    # Background subtraction
    bg = np.percentile(movie, 5, axis=0)
    movie = np.maximum(movie - bg[np.newaxis], 0.0)
    if fft_crop:
        T, Y, X = movie.shape
        side = 1 << int(np.floor(np.log2(min(Y, X))))
        y0, x0 = (Y - side) // 2, (X - side) // 2
        movie = movie[:, y0: y0 + side, x0: x0 + side]
    return movie


def _detect_frame(frame: np.ndarray, sigma_px: float, thresh_k: float,
                  min_dist_px: int = 3) -> np.ndarray:
    """DoG detection. Returns (N, 2) int32 [y, x] in pixels."""
    s1  = gaussian_filter(frame.astype(np.float32), sigma=sigma_px)
    s2  = gaussian_filter(frame.astype(np.float32), sigma=sigma_px * 1.4142)
    dog = np.maximum(s1 - s2, 0.0)
    mn, std = dog.mean(), dog.std()
    if std < 1e-12:
        return np.zeros((0, 2), dtype=np.int32)
    thresh   = mn + thresh_k * std
    footprint = int(max(3, min_dist_px * 2 + 1))
    local_max = (maximum_filter(dog, size=footprint) == dog)
    peaks     = (dog >= thresh) & local_max
    ys, xs    = np.where(peaks)
    if len(ys) == 0:
        return np.zeros((0, 2), dtype=np.int32)
    return np.column_stack([ys, xs]).astype(np.int32)


def _detect_frame_log(frame: np.ndarray, sigma_px: float, thresh_k: float,
                      min_dist_px: int = 3) -> np.ndarray:
    """DoG on log(I+1) — compresses dynamic range, detects dim spots more easily."""
    log_frame = np.log1p(frame.astype(np.float32))
    return _detect_frame(log_frame, sigma_px, thresh_k, min_dist_px)


# ═══════════════════════════════════════════════════════════════════════════════
# Method 1: Brightness-filtered NN-MSD
# ═══════════════════════════════════════════════════════════════════════════════

def detect_spots_with_brightness(
    movie: np.ndarray,
    sigma_px: float = SIGMA_PX,
    threshold_sigma: float = THRESH_K,
    pixel_um: float = PIXEL_UM,
    radius_px: int = 3,
    use_log: bool = False,
) -> tuple[list, list]:
    """
    Detect spots per frame and extract integrated brightness within radius_px.

    Parameters
    ----------
    use_log : if True, run DoG on log(I+1) instead of I.
              Log detection compresses dynamic range → detects dim spots more
              easily but also increases contamination from fast/dim species.
              Hypothesis: log detection WORSENS NN-MSD by adding dim fast spots.

    Returns
    -------
    spots_um     : list of (N_f, 2) float32 arrays  [y, x] in µm
    brightness   : list of (N_f,) float32 arrays   integrated intensity (a.u.)
    """
    T, Y, X = movie.shape
    detect_fn = _detect_frame_log if use_log else _detect_frame
    spots_um   = []
    brightness = []
    for t in range(T):
        frame = movie[t]
        px    = detect_fn(frame, sigma_px, threshold_sigma)
        if len(px) == 0:
            spots_um.append(np.zeros((0, 2), dtype=np.float32))
            brightness.append(np.zeros(0, dtype=np.float32))
            continue
        # Integrated intensity within radius_px of each detected peak
        B = np.zeros(len(px), dtype=np.float32)
        for k, (py, px_) in enumerate(px):
            y0 = max(0, py - radius_px)
            y1 = min(Y, py + radius_px + 1)
            x0 = max(0, px_ - radius_px)
            x1 = min(X, px_ + radius_px + 1)
            B[k] = float(frame[y0: y1, x0: x1].sum())
        spots_um.append(px.astype(np.float32) * pixel_um)
        brightness.append(B)
    return spots_um, brightness


def filter_by_brightness(
    spots_um: list,
    brightness: list,
    lo_pct: float = 0.0,
    hi_pct: float = 100.0,
) -> tuple[list, dict]:
    """
    Keep only spots whose integrated brightness is in [lo_pct, hi_pct] percentile.

    GEM 40-mers are the BRIGHTEST spots (40 GFPs per complex).
    Use lo_pct=50 to select the top half by brightness → enriches for GEM.
    Use lo_pct=0, hi_pct=50 to select dim spots → enriches for fast/slow species.

    Returns filtered spots_um and statistics dict.
    """
    all_B = np.concatenate([B for B in brightness if len(B) > 0])
    B_lo  = float(np.percentile(all_B, lo_pct))
    B_hi  = float(np.percentile(all_B, hi_pct))

    filtered = []
    n_kept_list = []
    for spots, B in zip(spots_um, brightness):
        if len(spots) == 0:
            filtered.append(np.zeros((0, 2), dtype=np.float32))
            n_kept_list.append(0)
            continue
        mask = (B >= B_lo) & (B <= B_hi)
        filtered.append(spots[mask])
        n_kept_list.append(int(mask.sum()))

    stats = dict(
        B_lo         = B_lo,
        B_hi         = B_hi,
        n_kept_mean  = float(np.mean(n_kept_list)),
        n_total_mean = float(np.mean([len(s) for s in spots_um])),
        fraction_kept = float(np.mean(n_kept_list)) / float(np.mean([len(s) for s in spots_um]))
                        if any(len(s) > 0 for s in spots_um) else 0.0,
        B_median     = float(np.median(all_B)),
        B_p75        = float(np.percentile(all_B, 75)),
        B_p25        = float(np.percentile(all_B, 25)),
    )
    return filtered, stats


def nn_msd_from_spots(
    spots_um: list,
    max_lag: int,
    dt: float,
    r_max_um: float = R_MAX_UM,
) -> tuple[list, np.ndarray, np.ndarray]:
    """
    NN-MSD computation: for each lag τ, pair each particle in frame t with
    its nearest neighbour in frame t+τ within r_max.

    Returns
    -------
    r2_per_lag : list of arrays of r² values (µm²) per lag
    n_pairs    : (max_lag,) number of valid pairs
    taus       : (max_lag,) lag times in seconds
    """
    from scipy.spatial import cKDTree

    T = len(spots_um)
    taus = np.arange(1, max_lag + 1, dtype=np.float64) * dt
    r2_per_lag = []
    n_pairs    = np.zeros(max_lag, dtype=np.int64)

    for tau_idx in range(1, max_lag + 1):
        r2_list = []
        for t in range(T - tau_idx):
            p1 = spots_um[t]
            p2 = spots_um[t + tau_idx]
            if len(p1) == 0 or len(p2) == 0:
                continue
            tree = cKDTree(p2)
            dist, _ = tree.query(p1, k=1, distance_upper_bound=r_max_um)
            valid = dist < r_max_um
            if valid.any():
                r2_list.append(dist[valid] ** 2)
        r2_arr = np.concatenate(r2_list) if r2_list else np.empty(0)
        r2_per_lag.append(r2_arr.astype(np.float64))
        n_pairs[tau_idx - 1] = len(r2_arr)

    return r2_per_lag, n_pairs, taus


def msd_per_lag(r2_per_lag: list, taus: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    msd = np.array([r2.mean() if len(r2) >= 5 else np.nan for r2 in r2_per_lag])
    D_eff = np.where(np.isfinite(msd) & (taus > 0), msd / (4 * taus), np.nan)
    return msd, D_eff


# ═══════════════════════════════════════════════════════════════════════════════
# Method 2: Jump Distance Distribution — EM algorithm
# ═══════════════════════════════════════════════════════════════════════════════

def fit_jdd_em(
    r2_per_lag: list,
    taus: np.ndarray,
    n_components: int = 2,
    n_iter: int = 100,
    D_init: list | None = None,
    D_lo: float = 0.003,
    D_hi: float = 0.50,
    min_pairs: int = 20,
    max_lag_fit: int = 15,
    r_max: float = R_MAX_UM,
) -> dict:
    """
    Jump Distance Distribution via Expectation-Maximisation.

    Model (per lag τ):
        p(r² | D_j, τ) = (1 / 4D_j τ) · exp(−r² / 4D_j τ)   [2-D Brownian, exponential in r²]

    Mixture:
        p(r²) = Σ_j f_j(τ) · p(r² | D_j, τ)

    D_j is SHARED across all τ values.  f_j(τ) is FREE per lag.
    The EM iterates:
        E-step : responsibilities  w_{i,j} ∝ f_j · p(r²_i | D_j, τ)
        M-step : D_j = [Σ_{i,τ} w_{i,j} · r²_i / τ] / [4 · Σ_{i,τ} w_{i,j}]
                 f_j(τ) = Σ_i w_{i,j} / N_τ

    Truncation note (r_max bias):
    Pairs with r > r_max are excluded; the simple untruncated M-step underestimates
    D when MSD ≳ R² = r_max².  At max_lag_fit=15 (τ≤1.5 s) this produces a ~12 %
    overestimate for GEM (competing biases partially cancel in the mixture).
    Restricting to max_lag_fit ≤ 5 (τ ≤ 0.5 s) reduces truncation but also loses
    pairs and makes the slow component poorly constrained.  max_lag_fit=15 is the
    empirically best compromise.

    Parameters
    ----------
    n_components : 2 or 3 (GEM + slow, or fast + GEM + slow)
    D_init       : list of initial D values per component; defaults for GEM system
    max_lag_fit  : maximum lag index to include (τ = lag_index × dt)
    r_max        : NN truncation radius (µm) — only used for docstring context

    Returns
    -------
    dict with D (array), f_per_lag (n_comp × n_lag), log_likelihood, n_iter_done.
    """
    n_lag  = min(max_lag_fit, len(r2_per_lag))
    valid_lags = [i for i in range(n_lag) if len(r2_per_lag[i]) >= min_pairs]
    if not valid_lags:
        return dict(D=np.full(n_components, np.nan), converged=False)

    tau_vals   = np.array([taus[i] for i in valid_lags])
    r2_arrays  = [r2_per_lag[i].astype(np.float64) for i in valid_lags]

    # Initialise D
    if D_init is None:
        if n_components == 2:
            D_init = [0.10, 0.012]
        elif n_components == 3:
            D_init = [0.15, 0.044, 0.012]
        else:
            D_init = list(np.logspace(np.log10(D_hi * 0.8), np.log10(D_lo * 1.5), n_components))

    D = np.array(D_init[:n_components], dtype=np.float64)
    D = np.clip(D, D_lo, D_hi)
    D.sort()
    D = D[::-1]  # descending

    n_comp = n_components
    f = np.ones((n_comp, len(valid_lags))) / n_comp  # equal initial weights

    ll_prev = -np.inf
    for it in range(n_iter):
        D_num = np.zeros(n_comp)
        D_den = np.zeros(n_comp)
        ll    = 0.0

        new_f = np.zeros_like(f)

        for li, (tau, r2) in enumerate(zip(tau_vals, r2_arrays)):
            lam = 1.0 / (4.0 * D * tau)          # (n_comp,)  rate params
            # Log-likelihood per data point per component
            log_p = np.log(lam)[np.newaxis, :] - lam[np.newaxis, :] * r2[:, np.newaxis]  # (N, n_comp)
            log_f = np.log(f[:, li] + 1e-300)[np.newaxis, :]                              # (1, n_comp)
            log_resp = log_p + log_f                                                        # (N, n_comp)
            # Numerically stable softmax
            log_resp -= log_resp.max(axis=1, keepdims=True)
            resp = np.exp(log_resp)
            resp /= resp.sum(axis=1, keepdims=True)                                        # (N, n_comp)
            # Log-likelihood
            log_mix = np.log((np.exp(log_p) * f[:, li][np.newaxis, :]).sum(axis=1) + 1e-300)
            ll += log_mix.sum()
            # Accumulate M-step
            for j in range(n_comp):
                D_num[j] += (resp[:, j] * r2 / tau).sum()
                D_den[j] += resp[:, j].sum()
            new_f[:, li] = resp.mean(axis=0)

        # Update D
        D_new = np.where(D_den > 1e-12, D_num / (4.0 * D_den), D)
        D_new = np.clip(D_new, D_lo, D_hi)
        # Keep sorted descending
        order = np.argsort(D_new)[::-1]
        D_new = D_new[order]
        new_f = new_f[order]

        f = new_f
        converged = bool(np.max(np.abs(D_new - D)) < 1e-7 and abs(ll - ll_prev) < 1.0)
        D = D_new
        ll_prev = ll
        if converged:
            break

    return dict(
        D          = D,
        f_per_lag  = f,
        D_per_comp = {f"D{j+1}": float(D[j]) for j in range(n_comp)},
        log_likelihood = float(ll_prev),
        n_iter_done    = it + 1,
        converged      = converged,
        valid_lags     = valid_lags,
        taus_used      = tau_vals,
        n_components   = n_comp,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Method 3: TICS (Temporal Image Correlation Spectroscopy) per tile
# ═══════════════════════════════════════════════════════════════════════════════

def compute_tics_acf_tile(tile: np.ndarray, max_lag: int, dt: float) -> np.ndarray:
    """
    Compute temporal ACF G(τ) = ⟨δI(t)·δI(t+τ)⟩ / ⟨I⟩² for one tile.

    WARNING — widefield TICS limitation:
    The fit model G₀/(1+τ/τ_D) with τ_D = w₀²/(4D) is derived for single-point
    confocal FCS, where τ_D is the transit time through the PSF waist w₀.
    For widefield tile-mean TICS, intensity fluctuations arise from particles
    transiting the TILE (size L≫w₀), giving τ_tile = L²/(4π²D) ≫ τ_D.
    At the particle densities in GEM movies (~0.5 GEM/tile) the shot-noise floor
    is also high.  The resulting D estimates are unreliable; use DDM or iMSD
    instead.  Per-pixel TICS (with L=w₀) would require confocal optics.

    Returns G(τ) array of length max_lag.  G(0+) would be 1/N_avg particles
    in a single-species system.
    """
    T = tile.shape[0]
    I    = tile.reshape(T, -1).mean(axis=1)   # mean intensity per frame in tile
    dI   = I - I.mean()
    norm = I.mean() ** 2
    if norm < 1e-12:
        return np.full(max_lag, np.nan)
    G = np.zeros(max_lag)
    for lag in range(1, max_lag + 1):
        n   = T - lag
        if n <= 0:
            G[lag - 1] = np.nan
            continue
        G[lag - 1] = (dI[:n] * dI[lag: lag + n]).mean() / norm
    return G


def fit_tics_acf(G: np.ndarray, taus: np.ndarray,
                 lag_min: int = 1, lag_max: int = 20) -> dict:
    """
    Fit G(τ) = G₀ / (1 + τ/τ_D) + B  for 2-D Brownian diffusion.

    τ_D = w₀² / (4D)  →  D = w₀² / (4·τ_D)

    Returns D (using w₀=W0_UM), τ_D, G₀, B, r².
    """
    s_slice  = slice(lag_min - 1, min(lag_max, len(G)))
    tau_fit  = taus[s_slice]
    g_fit    = G[s_slice]
    valid    = np.isfinite(g_fit) & (g_fit > 0)
    if valid.sum() < 4:
        return dict(D=np.nan, tau_D=np.nan, G0=np.nan, B=np.nan, r2=np.nan)

    def _model(tau, G0, tau_D, B):
        return G0 / (1.0 + tau / tau_D) + B

    G_max = float(g_fit[valid].max())
    p0    = [G_max, float(tau_fit[valid][len(tau_fit[valid]) // 2]), 0.0]
    bounds = ([0.0, 1e-4, -G_max], [G_max * 10, 100.0, G_max])
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            popt, _ = curve_fit(_model, tau_fit[valid], g_fit[valid],
                                p0=p0, bounds=bounds, maxfev=10000)
        G0, tau_D, B = popt
        D = W0_UM ** 2 / (4.0 * tau_D) if tau_D > 0 else np.nan
        # R²
        y_pred = _model(tau_fit[valid], *popt)
        ss_res = ((g_fit[valid] - y_pred) ** 2).sum()
        ss_tot = ((g_fit[valid] - g_fit[valid].mean()) ** 2).sum()
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
        return dict(D=float(D), tau_D=float(tau_D), G0=float(G0), B=float(B), r2=float(r2))
    except Exception:
        return dict(D=np.nan, tau_D=np.nan, G0=np.nan, B=np.nan, r2=np.nan)


def compute_tics_dmap(
    movie: np.ndarray,
    pixel_um: float = PIXEL_UM,
    dt: float = DT,
    tile_size: int = TILE_SIZE,
    max_lag: int = MAX_LAG,
    lag_min: int = 1,
    lag_max: int = 20,
    r2_min: float = 0.2,
    w0_um: float = W0_UM,
) -> dict:
    """
    Compute TICS-based D-map: fit G(τ) = G₀/(1+τ/τ_D) per tile.

    R²>r2_min quality filter (same criterion as iMSD R²>0.2 filter).
    Returns D_map (µm²/s), τ_D_map, r2_map, tile coordinates.
    """
    global W0_UM
    W0_UM = w0_um   # update module-level constant

    T, Y, X   = movie.shape
    n_ty      = Y // tile_size
    n_tx      = X // tile_size
    taus      = np.arange(1, max_lag + 1, dtype=np.float64) * dt

    D_map   = np.full((n_ty, n_tx), np.nan)
    tD_map  = np.full((n_ty, n_tx), np.nan)
    r2_map  = np.full((n_ty, n_tx), np.nan)

    for iy in range(n_ty):
        for ix in range(n_tx):
            y0, x0 = iy * tile_size, ix * tile_size
            tile   = movie[:, y0: y0 + tile_size, x0: x0 + tile_size]
            G      = compute_tics_acf_tile(tile, max_lag=max_lag, dt=dt)
            res    = fit_tics_acf(G, taus, lag_min=lag_min, lag_max=lag_max)
            r2_map[iy, ix]  = res["r2"]
            if (res["r2"] > r2_min and np.isfinite(res["D"])
                    and 1e-4 < res["D"] < 10.0):
                D_map[iy, ix]  = res["D"]
                tD_map[iy, ix] = res["tau_D"]

    return dict(
        D_map   = D_map,
        tD_map  = tD_map,
        r2_map  = r2_map,
        D_median = float(np.nanmedian(D_map)),
        D_mean   = float(np.nanmean(D_map)),
        D_sem    = (float(np.nanstd(D_map) / np.sqrt(np.isfinite(D_map).sum()))
                    if np.isfinite(D_map).sum() > 1 else np.nan),
        n_valid  = int(np.isfinite(D_map).sum()),
        n_total  = n_ty * n_tx,
        tile_size = tile_size,
        pixel_um  = pixel_um,
        w0_um     = w0_um,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Plots
# ═══════════════════════════════════════════════════════════════════════════════

def plot_brightness_comparison(
    taus: np.ndarray,
    results: dict,
    tracking_D: float | None = None,
    movie_name: str = "",
    out_path: "Path | None" = None,
) -> None:
    """Compare intensity histogram + D_eff(τ) for unfiltered vs filtered NN-MSD."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    D_GEM_line = tracking_D
    title = f"Brightness-filtered NN-MSD — {movie_name}"
    if D_GEM_line:
        title += f"  (tracking D={D_GEM_line:.4f} µm²/s)"
    fig.suptitle(title, fontsize=10)

    # ── Panel 1: Brightness histogram ─────────────────────────────────────────
    ax = axes[0, 0]
    all_B = results["all_brightness"]
    ax.hist(all_B, bins=80, density=True, color="steelblue", alpha=0.7)
    for label, pct, col in [("top 50%", 50, "red"), ("top 25%", 75, "orange"),
                             ("top 10%", 90, "green")]:
        thresh = float(np.percentile(all_B, pct))
        ax.axvline(thresh, color=col, lw=1.5, ls="--", label=f"{label} (>{thresh:.0f})")
    ax.set_xlabel("Integrated spot intensity (a.u.)")
    ax.set_ylabel("Density")
    ax.set_title("Spot brightness histogram")
    ax.legend(fontsize=8)
    ax.set_xlim(0, float(np.percentile(all_B, 99.5)))

    # ── Panel 2: D_eff(τ) for each filter level ───────────────────────────────
    ax = axes[0, 1]
    colors = {"all": "gray", "top50": "steelblue", "top25": "darkorange",
              "top10": "green", "log": "purple"}
    labels = {"all": "Unfiltered", "top50": "Bright 50%",
              "top25": "Bright 25%", "top10": "Bright 10%", "log": "Log-detect"}
    for key, col in colors.items():
        if key not in results:
            continue
        D_eff = results[key]["D_eff"]
        n_pairs = results[key]["n_pairs"]
        valid = np.isfinite(D_eff) & (n_pairs >= 5)
        if valid.any():
            ax.semilogy(taus[valid], D_eff[valid], "o-", color=col, ms=4, lw=1.2,
                        label=f"{labels[key]}  n={int(n_pairs[valid].mean()):.0f}/lag")
    if D_GEM_line:
        ax.axhline(D_GEM_line, color="red", ls=":", lw=1.5,
                   label=f"tracking D={D_GEM_line:.4f}")
    ax.set_xlabel("τ  (s)"); ax.set_ylabel("D_eff(τ) = MSD/4τ  (µm²/s)")
    ax.set_title("D_eff(τ) profile: brightness filter comparison")
    ax.legend(fontsize=7, ncol=2)
    ax.set_ylim(0.005, 0.5)

    # ── Panel 3: MSD(τ) absolute values ───────────────────────────────────────
    ax = axes[1, 0]
    for key, col in colors.items():
        if key not in results:
            continue
        msd = results[key]["msd"]
        valid = np.isfinite(msd)
        if valid.any():
            ax.plot(taus[valid], msd[valid], "o-", color=col, ms=4, lw=1.2,
                    label=labels[key])
    tau_ref = np.linspace(0, taus[-1], 100)
    if D_GEM_line:
        ax.plot(tau_ref, 4 * D_GEM_line * tau_ref, "r:", lw=1.5,
                label=f"4D_tracking×τ")
    ax.set_xlabel("τ  (s)"); ax.set_ylabel("MSD  (µm²)")
    ax.set_title("MSD(τ) for each brightness filter")
    ax.legend(fontsize=7, ncol=2); ax.set_xlim(0)

    # ── Panel 4: N_pairs vs τ ─────────────────────────────────────────────────
    ax = axes[1, 1]
    for key, col in colors.items():
        if key not in results:
            continue
        n = results[key]["n_pairs"]
        ax.semilogy(taus, n, "o-", color=col, ms=3, lw=1.0, label=labels[key])
    ax.set_xlabel("τ  (s)"); ax.set_ylabel("N valid NN pairs")
    ax.set_title("Pair statistics vs lag")
    ax.legend(fontsize=7)

    fig.tight_layout()
    if out_path:
        fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
        import matplotlib.pyplot as _plt; _plt.close(fig)
        print(f"   Brightness comparison plot → {out_path}")
    else:
        import matplotlib.pyplot as _plt; _plt.show()


def plot_jdd_em(
    jdd_result: dict,
    r2_per_lag: list,
    taus: np.ndarray,
    tracking_D: float | None = None,
    movie_name: str = "",
    out_path: "Path | None" = None,
) -> None:
    """JDD-EM diagnostic: r² histograms with component fits, D values, fractions."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.cm import viridis

    D   = jdd_result["D"]
    f   = jdd_result["f_per_lag"]
    n_c = jdd_result["n_components"]
    vl  = jdd_result["valid_lags"]
    tv  = jdd_result["taus_used"]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    comp_labels = [f"D{j+1}={D[j]:.4f}" for j in range(n_c)]
    fig.suptitle(
        f"JDD-EM ({n_c}-component) — {movie_name}\n" +
        "  ".join(comp_labels) +
        (f"   tracking={tracking_D:.4f}" if tracking_D else ""),
        fontsize=10,
    )

    colors = ["steelblue", "darkorange", "green"]

    # ── Panel 1: r² histograms at selected lags ───────────────────────────────
    ax = axes[0, 0]
    sel = vl[::max(1, len(vl) // 4)][:4]  # at most 4 lags
    hcolors = viridis(np.linspace(0.1, 0.9, len(sel)))
    for li, hcol in zip(sel, hcolors):
        r2 = r2_per_lag[li]
        tau = taus[li]
        if len(r2) < 5:
            continue
        bins  = np.linspace(0, min(float(r2.max()), 0.5), 50)
        ax.hist(r2, bins=bins, density=True, alpha=0.4, color=hcol,
                label=f"τ={tau:.1f}s  n={len(r2)}")
        li_idx = vl.index(li) if li in vl else -1
        if li_idx >= 0:
            x = np.linspace(0, bins[-1], 200)
            p_mix = np.zeros_like(x)
            for j in range(n_c):
                msd_j = 4 * D[j] * tau
                p_j   = (1.0 / msd_j) * np.exp(-x / msd_j)
                p_mix += f[j, li_idx] * p_j
            ax.plot(x, p_mix, "-", color=hcol, lw=1.5)
    ax.set_xlabel("r²  (µm²)"); ax.set_ylabel("p(r²)")
    ax.set_title("r² histograms + JDD-EM mixture fits")
    ax.legend(fontsize=7)

    # ── Panel 2: D(τ) effective vs EM components ─────────────────────────────
    ax = axes[0, 1]
    msd_all = np.array([r2.mean() if len(r2) >= 5 else np.nan for r2 in r2_per_lag])
    D_eff_all = np.where(np.isfinite(msd_all) & (taus > 0), msd_all / (4 * taus), np.nan)
    ax.semilogy(taus[np.isfinite(D_eff_all)], D_eff_all[np.isfinite(D_eff_all)],
                "k--", ms=4, lw=1.2, label="D_eff(τ) = MSD/4τ")
    for j in range(n_c):
        ax.axhline(D[j], color=colors[j], lw=1.8, ls="-",
                   label=f"D{j+1}={D[j]:.4f} µm²/s")
    if tracking_D:
        ax.axhline(tracking_D, color="red", lw=1.5, ls=":",
                   label=f"tracking={tracking_D:.4f}")
    ax.set_xlabel("τ  (s)"); ax.set_ylabel("D  (µm²/s)")
    ax.set_title("EM-fitted D components vs D_eff(τ) profile")
    ax.legend(fontsize=8); ax.set_ylim(0.003, 0.5)

    # ── Panel 3: mixing fractions f_j(τ) ─────────────────────────────────────
    ax = axes[1, 0]
    for j in range(n_c):
        ax.plot(tv, f[j], "o-", color=colors[j], ms=5, lw=1.2,
                label=comp_labels[j])
    ax.set_xlabel("τ  (s)"); ax.set_ylabel("Fraction f_j(τ)")
    ax.set_title("Component fractions per lag (from EM)")
    ax.legend(fontsize=8); ax.set_ylim(-0.05, 1.05)

    # ── Panel 4: r² CDF at τ=0.1s, 0.5s with model overlay ──────────────────
    ax = axes[1, 1]
    for li, lc in zip([0, 4], ["steelblue", "darkorange"]):
        if li >= len(r2_per_lag): continue
        r2 = r2_per_lag[li]
        tau = taus[li]
        if len(r2) < 5: continue
        r2_sorted = np.sort(r2)
        cdf_emp = np.arange(1, len(r2_sorted) + 1) / len(r2_sorted)
        ax.plot(r2_sorted, cdf_emp, "-", color=lc, lw=1.5, label=f"τ={tau:.1f}s data")
        # Model CDF
        x = np.linspace(0, r2_sorted[-1], 200)
        li_idx = vl.index(li) if li in vl else -1
        if li_idx >= 0:
            cdf_mix = np.zeros_like(x)
            for j in range(n_c):
                msd_j = 4 * D[j] * tau
                cdf_mix += f[j, li_idx] * (1 - np.exp(-x / msd_j))
            ax.plot(x, cdf_mix, "--", color=lc, lw=1.0)
    ax.set_xlabel("r²  (µm²)"); ax.set_ylabel("CDF")
    ax.set_title("r² CDF: data vs JDD-EM model")
    ax.legend(fontsize=8); ax.set_xlim(0)

    fig.tight_layout()
    if out_path:
        fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
        import matplotlib.pyplot as _plt; _plt.close(fig)
        print(f"   JDD-EM plot → {out_path}")
    else:
        import matplotlib.pyplot as _plt; _plt.show()


def plot_tics_dmap(
    tics_res: dict,
    tracking_D: float | None = None,
    movie_name: str = "",
    out_path: "Path | None" = None,
) -> None:
    """TICS D-map and histogram."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    D_map  = tics_res["D_map"]
    r2_map = tics_res["r2_map"]
    D_med  = tics_res["D_median"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle(f"TICS D-map — {movie_name}  (w₀={tics_res['w0_um']:.3f} µm, "
                 f"tile={tics_res['tile_size']}×{tics_res['tile_size']} px)", fontsize=10)

    valid_D = D_map[np.isfinite(D_map)]

    # D-map
    ax = axes[0]
    vmax = float(np.nanpercentile(D_map, 95)) if valid_D.size else 0.1
    im = ax.imshow(D_map, vmin=0, vmax=vmax, cmap="hot_r", origin="upper")
    plt.colorbar(im, ax=ax, label="D (µm²/s)")
    ax.set_title(f"D-map  median={D_med:.4f} µm²/s  (R²>{0.2})")

    # R²-map
    ax = axes[1]
    im2 = axes[1].imshow(r2_map, vmin=0, vmax=1, cmap="RdYlGn", origin="upper")
    plt.colorbar(im2, ax=ax, label="R²")
    ax.set_title("Fit R²-map")

    # Histogram
    ax = axes[2]
    if valid_D.size > 0:
        ax.hist(valid_D, bins=30, color="steelblue", alpha=0.8, density=True)
    ax.axvline(D_med, color="steelblue", lw=2, ls="--",
               label=f"TICS median={D_med:.4f}")
    if tracking_D:
        ax.axvline(tracking_D, color="red", lw=1.5, ls=":",
                   label=f"tracking={tracking_D:.4f}")
    ax.set_xlabel("D  (µm²/s)"); ax.set_ylabel("Density")
    ax.set_title(f"D distribution  n={tics_res['n_valid']}/{tics_res['n_total']} tiles")
    ax.legend(fontsize=9)

    fig.tight_layout()
    if out_path:
        fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
        import matplotlib.pyplot as _plt; _plt.close(fig)
        print(f"   TICS D-map plot → {out_path}")
    else:
        import matplotlib.pyplot as _plt; _plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
# Top-level runners
# ═══════════════════════════════════════════════════════════════════════════════

def run_brightness_nn_msd(
    tiff_path: Path,
    tracking_D: float | None = None,
    out_dir: Path | None = None,
    pixel_um: float = PIXEL_UM,
    dt: float = DT,
    max_lag: int = 20,
    r_max_um: float = R_MAX_UM,
    sigma_px: float = SIGMA_PX,
    threshold_sigma: float = THRESH_K,
) -> dict:
    """Brightness-filtered NN-MSD pipeline."""
    import time
    tiff_path = Path(tiff_path)
    name = tiff_path.stem
    out  = Path(out_dir or tiff_path.parent / "jdd_out")
    out.mkdir(parents=True, exist_ok=True)

    print(f"\n[Brightness-filtered NN-MSD] {name}")
    movie = _load_preprocess(tiff_path, fft_crop=True)
    T, Y, X = movie.shape
    print(f"   {T}×{Y}×{X}")

    print("   Detecting spots + extracting brightness …")
    t0 = time.time()
    spots_um, brightness = detect_spots_with_brightness(
        movie, sigma_px=sigma_px, threshold_sigma=threshold_sigma,
        pixel_um=pixel_um, radius_px=3, use_log=False,
    )
    counts = np.array([len(s) for s in spots_um])
    all_B  = np.concatenate([B for B in brightness if len(B) > 0])
    print(f"   {counts.mean():.0f}±{counts.std():.0f} spots/frame  {time.time()-t0:.1f}s")
    print(f"   Brightness: median={np.median(all_B):.0f}  p25={np.percentile(all_B,25):.0f}"
          f"  p75={np.percentile(all_B,75):.0f}  p90={np.percentile(all_B,90):.0f}")

    # Also detect with log-I
    print("   Detecting spots with log(I) …")
    spots_log, brightness_log = detect_spots_with_brightness(
        movie, sigma_px=sigma_px, threshold_sigma=threshold_sigma,
        pixel_um=pixel_um, radius_px=3, use_log=True,
    )
    counts_log = np.array([len(s) for s in spots_log])
    print(f"   log-detect: {counts_log.mean():.0f}±{counts_log.std():.0f} spots/frame")

    taus = np.arange(1, max_lag + 1, dtype=np.float64) * dt

    def run_nn(spot_list, label):
        r2pl, npairs, _ = nn_msd_from_spots(spot_list, max_lag, dt, r_max_um)
        msd, D_eff = msd_per_lag(r2pl, taus)
        return dict(msd=msd, D_eff=D_eff, n_pairs=npairs, r2_per_lag=r2pl)

    print("   Computing NN-MSD for each filter level …")
    results = {"all_brightness": all_B}

    t0 = time.time()
    results["all"] = run_nn(spots_um, "all")
    print(f"   Unfiltered NN-MSD: {time.time()-t0:.1f}s")

    for label, lo, hi in [("top50", 50, 100), ("top25", 75, 100), ("top10", 90, 100)]:
        filt, stats = filter_by_brightness(spots_um, brightness, lo_pct=lo, hi_pct=hi)
        results[label] = run_nn(filt, label)
        fc = stats['n_kept_mean']
        print(f"   {label}: {fc:.0f} spots/frame  D_eff(τ=0.5s)="
              f"{results[label]['D_eff'][4]:.4f} µm²/s")

    results["log"] = run_nn(spots_log, "log")

    # Report D_eff at key lags
    print("\n   D_eff(τ) comparison [µm²/s]:")
    print(f"   {'τ':>6}  {'all':>8}  {'top50':>8}  {'top25':>8}  {'top10':>8}  {'log':>8}")
    for i in [0, 2, 4, 9, 14, 19]:
        if i >= max_lag: continue
        tau = taus[i]
        row = f"   {tau:>6.1f}s"
        for key in ["all", "top50", "top25", "top10", "log"]:
            D = results.get(key, {}).get("D_eff", np.full(max_lag, np.nan))
            row += f"  {D[i]:>8.4f}" if i < len(D) and np.isfinite(D[i]) else f"  {'  ---':>8}"
        if tracking_D:
            row += f"   (tracking={tracking_D:.4f})"
        print(row)

    print("\n   Saving plot …")
    plot_brightness_comparison(taus, results, tracking_D=tracking_D,
                                movie_name=name,
                                out_path=out / f"{name}_brightness_nn.png")
    return results


def run_jdd_em(
    tiff_path: Path,
    tracking_D: float | None = None,
    out_dir: Path | None = None,
    pixel_um: float = PIXEL_UM,
    dt: float = DT,
    max_lag: int = 20,
    r_max_um: float = R_MAX_UM,
    n_components: int = 3,
    sigma_px: float = SIGMA_PX,
    threshold_sigma: float = THRESH_K,
) -> dict:
    """Full JDD-EM pipeline."""
    import time
    tiff_path = Path(tiff_path)
    name = tiff_path.stem
    out  = Path(out_dir or tiff_path.parent / "jdd_out")
    out.mkdir(parents=True, exist_ok=True)

    print(f"\n[JDD-EM {n_components}-component] {name}")
    movie = _load_preprocess(tiff_path, fft_crop=True)

    print("   Detecting spots …")
    t0 = time.time()
    spots_um, _ = detect_spots_with_brightness(
        movie, sigma_px=sigma_px, threshold_sigma=threshold_sigma, pixel_um=pixel_um)
    counts = np.array([len(s) for s in spots_um])
    print(f"   {counts.mean():.0f}±{counts.std():.0f} spots/frame  {time.time()-t0:.1f}s")

    taus = np.arange(1, max_lag + 1, dtype=np.float64) * dt

    print(f"   Computing NN displacements (r_max={r_max_um} µm) …")
    t0 = time.time()
    r2_per_lag, n_pairs, _ = nn_msd_from_spots(spots_um, max_lag, dt, r_max_um)
    print(f"   Pairs at lag 1: {n_pairs[0]}  lag {max_lag}: {n_pairs[-1]}"
          f"  {time.time()-t0:.1f}s")

    print(f"   Running EM ({n_components} components, max_lag_fit=15) …")
    t0 = time.time()
    jdd_res = fit_jdd_em(r2_per_lag, taus, n_components=n_components,
                          max_lag_fit=15, n_iter=100, r_max=r_max_um)
    print(f"   EM done in {time.time()-t0:.1f}s  ({jdd_res['n_iter_done']} iterations"
          f"  converged={jdd_res['converged']})")

    D = jdd_res["D"]
    for j in range(n_components):
        ratio_str = f"  ratio={D[j]/tracking_D:.3f}×" if tracking_D else ""
        print(f"   D{j+1} = {D[j]:.5f} µm²/s{ratio_str}")

    print("   Saving JDD-EM plot …")
    plot_jdd_em(jdd_res, r2_per_lag, taus, tracking_D=tracking_D,
                movie_name=name, out_path=out / f"{name}_jdd_em.png")
    jdd_res["r2_per_lag"] = r2_per_lag
    jdd_res["taus"] = taus
    return jdd_res


def run_tics(
    tiff_path: Path,
    tracking_D: float | None = None,
    out_dir: Path | None = None,
    pixel_um: float = PIXEL_UM,
    dt: float = DT,
    tile_size: int = TILE_SIZE,
    max_lag: int = MAX_LAG,
    lag_min: int = 1,
    lag_max: int = 20,
    w0_um: float = W0_UM,
    r2_min: float = 0.2,
) -> dict:
    """Full TICS D-map pipeline."""
    import time
    tiff_path = Path(tiff_path)
    name = tiff_path.stem
    out  = Path(out_dir or tiff_path.parent / "jdd_out")
    out.mkdir(parents=True, exist_ok=True)

    print(f"\n[TICS] {name}")
    movie = _load_preprocess(tiff_path, fft_crop=False)
    T, Y, X = movie.shape
    n_tiles = (Y // tile_size) * (X // tile_size)
    print(f"   {T}×{Y}×{X}  tile={tile_size}×{tile_size}  n_tiles={n_tiles}"
          f"  w₀={w0_um:.3f} µm")

    t0 = time.time()
    tics_res = compute_tics_dmap(
        movie, pixel_um=pixel_um, dt=dt, tile_size=tile_size,
        max_lag=max_lag, lag_min=lag_min, lag_max=lag_max,
        r2_min=r2_min, w0_um=w0_um,
    )
    print(f"   Done in {time.time()-t0:.1f}s")
    print(f"   TICS D_median = {tics_res['D_median']:.5f} µm²/s"
          + (f"  ratio={tics_res['D_median']/tracking_D:.3f}×" if tracking_D else ""))
    print(f"   Valid tiles: {tics_res['n_valid']}/{tics_res['n_total']}"
          f"  (R²>{r2_min})")

    print("   Saving TICS D-map plot …")
    plot_tics_dmap(tics_res, tracking_D=tracking_D, movie_name=name,
                   out_path=out / f"{name}_tics_dmap.png")
    return tics_res


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys, time
    inp        = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/Em1_crop.tif")
    tracking_D = float(sys.argv[2]) if len(sys.argv) > 2 else 0.04374
    mode       = sys.argv[3] if len(sys.argv) > 3 else "all"   # bright | jdd | tics | all

    tiffs = sorted(list(inp.glob("*.tif")) + list(inp.glob("*.tiff"))) if inp.is_dir() else [inp]
    if not tiffs:
        sys.exit(f"[JDD] No .tif files found in {inp}")

    for tiff in tiffs:
        out_dir = tiff.parent / "jdd_out"
        out_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        if mode in ("bright", "all"):
            run_brightness_nn_msd(tiff, tracking_D=tracking_D, out_dir=out_dir)
        if mode in ("jdd", "all"):
            run_jdd_em(tiff, tracking_D=tracking_D, out_dir=out_dir, n_components=3)
        if mode in ("tics", "all"):
            run_tics(tiff, tracking_D=tracking_D, out_dir=out_dir)
        print(f"\n   Total: {time.time()-t0:.0f}s   Output → {out_dir}/")
