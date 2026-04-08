# GEM Mitosis Analysis Pipeline

Analysis of GEM (Genetically Encoded Multimeric nanoparticle) particle
tracking data to compare cytoplasmic diffusion in mitotic vs interphase
cells, characterise cell-to-cell and within-cell variability, and
correlate diffusion with cell morphology.

---

## Overview

The pipeline processes TrackMate trajectory CSVs and Cellpose segmentation
masks through 8 scripts, from raw trajectories to statistical reporting.

```
Raw data (Traj_*.csv + *_cp_masks.png)
        │
        ▼
1_cell_trajectory_classifier.py   ── classify trajectories → cells
        │
        ▼
2_diffusion_per_cell.py           ── MSD fitting, per-cell D statistics
        │
        ▼
3_circularity_diffusion_correlation.py  ── mitotic/interphase classification
        │
        ▼
5_batch_pipeline.py               ── run steps 1-3 for all experiments,
        │                            pool into experiments/ subfolders
        ▼
4_pool_experiments.py             ── pool single experiment (legacy)
        │
        ▼
6_within_cell_variability.py      ── D_iqr, D_std, D_cv vs morphology
        │
        ▼
7_variance_decomposition.py       ── nested ANOVA: η² decomposition
        │
        ▼
8_confound_check.py               ── bootstrap + partial correlation
                                     to validate area-diffusivity result
```

---

## Input files (per experiment)

| File | Description |
|------|-------------|
| `Traj_<name>.csv` | TrackMate trajectory table (columns: Trajectory, Frame, x, y) |
| `<name>.nd2membrane image.tif` | Membrane channel image (background display) |
| `<name>.nd2membrane image_cp_masks.png` | Cellpose label image (uint16, each cell = unique integer) |

All four experiments (`set1_3em_001` – `set1_3em_004`) are in this folder.

---

## Scripts

### `1_cell_trajectory_classifier.py`
Assigns each GEM trajectory to the cell it belongs to by looking up the
Cellpose mask at the trajectory's mean (x, y) position (majority vote).
Computes per-cell shape metrics (area, perimeter, circularity) from
`skimage.measure.regionprops`.

**Outputs:** `cell_trajectory_summary.csv`, `cell_trajectories.pkl`,
`cell_trajectory_classification.png`, `cell_overview_distributions.png`

---

### `2_diffusion_per_cell.py`
Fits a linear MSD model (MSD = 4Dt + offset) to each trajectory using
`scipy.optimize.curve_fit`. Applies quality filters (minimum trajectory
length, minimum trajectories per cell, minimum cell area). Aggregates to
per-cell statistics: D_median, D_mean, D_std, D_iqr, D_p25, D_p75.

**Key parameters:**
- `DT = 0.1 s/frame`
- `PIXEL_TO_MICRON = 0.094 µm/px`
- `MIN_TRAJ_LENGTH = 10 frames`
- `MIN_TRAJ_PER_CELL = 5`
- `MSD_FIT_FRACTION = 0.8` (use first 80% of trajectory for fitting)

**Outputs:** `diffusion_per_cell.csv`, `diffusion_per_traj.csv`,
`diffusion_per_cell.pkl`, `diffusion_qc.png`, `diffusion_per_cell.png`

---

### `3_circularity_diffusion_correlation.py`
Classifies each cell as mitotic (circularity ≥ 0.70) or interphase
(circularity < 0.70). Computes Pearson correlation between D and
circularity/area, and Mann-Whitney U test comparing mitotic vs interphase.

**Outputs:** `cell_classification.csv`, `correlation_diffusion.png`,
`mitotic_vs_interphase.png`, `mitotic_cell_comparison.png`, `step3_stats.txt`

---

### `5_batch_pipeline.py`
Batch wrapper that runs steps 1–3 for each experiment and places outputs
in per-experiment subfolders under `experiments/`. Then pools all
experiments via the step 4 logic.

**Adding new experiments:** add an entry to the `EXPERIMENTS` list at the
top of the script:
```python
EXPERIMENTS = [
    ("set1_3em_001", SCRIPT_DIR / "Traj_set1_3em_001_crop.tif.csv",
                     SCRIPT_DIR / "set1_3em_001.nd2membrane image.tif",
                     SCRIPT_DIR / "set1_3em_001.nd2membrane image_cp_masks.png",
                     True),   # True = precomputed, skip re-processing
    ("set1_3em_005", Path("/path/to/Traj_set1_3em_005.csv"),
                     Path("/path/to/set1_3em_005.tif"),
                     Path("/path/to/set1_3em_005_cp_masks.png"),
                     False),  # False = run full pipeline
]
```

**Outputs:** `experiments/<exp_name>/` subfolders +
`pooled_results_v2/pooled_cells.csv`, `pooled_stats.txt`, figures

---

### `4_pool_experiments.py`
Original single-experiment pooling script (legacy). Discovers experiments
from sub-folder layout or explicit `EXPERIMENT_CSVS` list. Produces the
`pooled_results/` folder.

---

### `6_within_cell_variability.py`
Computes three within-cell variability metrics from `pooled_cells.csv`:
- **D_iqr** — interquartile range of per-trajectory D values
- **D_std** — standard deviation
- **D_cv** — coefficient of variation (D_std / D_mean)

Correlates each metric with circularity and area separately for mitotic
and interphase cells.

**Outputs (→ `pooled_results_v2/`):**
`variability_vs_shape.png`, `variability_mitotic_vs_interphase.png`,
`variability_mitotic_cells.png`, `variability_stats.txt`

---

### `7_variance_decomposition.py`
Nested ANOVA on log(D), following **Hubatsch et al. 2023**
(*Biophys J*, PMC10027447). Partitions total variance in log(D) into:

| Level | η² (all cells) |
|-------|---------------|
| Between-experiment | 2.3% |
| Between-cell (within experiment) | 18.6% |
| Within-cell (track-to-track) | **79.1%** |

Analysis is repeated separately for mitotic (n=40) and interphase (n=104)
cells. Also correlates per-cell mean log(D) and within-cell SD of log(D)
with morphology metrics.

**Outputs (→ `pooled_results_v2/`):**
`variance_decomposition.png`, `variance_within_cell_vs_shape.png`,
`variance_decomposition_stats.txt`

---

### `8_confound_check.py`
Validates the cell-area → diffusivity correlation against the potential
confound that larger cells may simply have more trajectories and therefore
more stable mean estimates. Three complementary tests:

1. **Confound audit** — Spearman correlations among area, n_traj, mean log(D)
2. **Bootstrap downsample** — subsample k_min=5 trajectories per cell
   (2000 iterations), recompute mean log(D), re-test correlation
3. **Partial Spearman correlation** — r(area, log D | n_traj)

**Outputs (→ `pooled_results_v2/`):**
`confound_check.png`, `confound_check_stats.txt`

---

## Results summary (4 pooled experiments, 144 cells, 1,111 trajectories)

### Cell counts
| Experiment | Mitotic | Interphase |
|------------|---------|------------|
| set1_3em_001 | 8 | 32 |
| set1_3em_002 | 6 | 4 |
| set1_3em_003 | 16 | 23 |
| set1_3em_004 | 10 | 45 |
| **Total** | **40** | **104** |

Circularity threshold for mitotic classification: **≥ 0.70**

### Mitotic vs interphase diffusivity
- Mitotic median D = 0.0211 µm²/s
- Interphase median D = 0.0200 µm²/s
- Fold change = 0.95× — **no significant difference** (Mann-Whitney p = 0.98)
- Per-experiment direction is inconsistent, suggesting the 0.70 circularity
  threshold may capture non-mitotic rounded cells in some experiments

### Variance decomposition (nested ANOVA on log D)
The dominant source of variability is **within individual cells**:
- Within-cell (track-to-track): **~79%** of total log(D) variance
- Between-cell: ~17–19%
- Between-experiment: ~2–6%

This mirrors the finding of Hubatsch et al. 2023 in yeast.

### Cell size → diffusivity (robust finding)
Larger cells diffuse faster. This holds across all three validation tests:

| Test | Mitotic ρ | p | Interphase ρ | p |
|------|-----------|---|--------------|---|
| Raw Spearman | 0.44 | 0.004 ** | 0.24 | 0.013 * |
| Bootstrap k=5 (median) | 0.40 [0.28–0.49] | <0.001 *** | 0.22 [0.13–0.30] | <0.001 *** |
| Partial (controlling n_traj) | 0.42 | 0.007 ** | 0.25 | 0.010 ** |

### Circularity → diffusivity
No significant correlation in either cell type (all p > 0.17).

### Within-cell variability vs morphology
- Within-cell spread (D_iqr) correlates with area in mitotic cells
  (Spearman ρ = 0.50, p = 0.001 ***) — larger mitotic cells show
  greater trajectory-to-trajectory heterogeneity
- No circularity relationship detected

---

## Output folder structure

```
GEM_mitosis/
├── experiments/
│   ├── set1_3em_001/
│   │   ├── diffusion_per_cell.csv
│   │   └── cell_classification.csv
│   ├── set1_3em_002/
│   │   ├── cell_trajectories.pkl
│   │   ├── cell_trajectory_summary.csv
│   │   ├── diffusion_per_cell.csv
│   │   ├── diffusion_per_traj.csv
│   │   └── cell_classification.csv
│   ├── set1_3em_003/   (same structure)
│   └── set1_3em_004/   (same structure)
│
└── pooled_results_v2/
    ├── pooled_cells.csv                      ← all 144 cells, one row each
    ├── pooled_stats.txt
    ├── pooled_mitotic_vs_interphase.png
    ├── pooled_mitotic_cell_comparison.png
    ├── pooled_correlation_diffusion.png
    ├── variability_stats.txt
    ├── variability_vs_shape.png
    ├── variability_mitotic_vs_interphase.png
    ├── variability_mitotic_cells.png
    ├── variance_decomposition_stats.txt
    ├── variance_decomposition.png
    ├── variance_within_cell_vs_shape.png
    ├── confound_check_stats.txt
    └── confound_check.png
```

---

## Dependencies

```
numpy, pandas, scipy, matplotlib, scikit-image, scikit-learn,
tifffile, Pillow, pickle
```

Install with:
```bash
pip install numpy pandas scipy matplotlib scikit-image scikit-learn tifffile Pillow
```

---

## Running the full pipeline

```bash
cd GEM_mitosis/

# Process all experiments and pool (Steps 1-4)
python 5_batch_pipeline.py

# Variability analysis (Step 6)
python 6_within_cell_variability.py

# Nested ANOVA decomposition (Step 7)
python 7_variance_decomposition.py

# Confound check (Step 8)
python 8_confound_check.py
```

Steps 6–8 all read from `pooled_results_v2/pooled_cells.csv` produced by
step 5, so they can be re-run independently after pooling.
