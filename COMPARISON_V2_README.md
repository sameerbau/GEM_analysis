# Diffusion Dataset Comparison V2 - User Guide

## Overview

`compare_diffusion_datasets_v2.py` is an enhanced comparison tool that integrates with `diffusion_analyzer_v4_validation.py` to provide comprehensive multi-sample, multi-metric analysis with quality-aware filtering.

## Key Enhancements Over V1

### 1. **Hierarchical Sample Organization**
Now supports sample-level organization:
```
Sample_WT/
  ├── analyzed_image001.pkl
  ├── analyzed_image002.pkl
  └── analyzed_image003.pkl
Sample_Mutant/
  ├── analyzed_image001.pkl
  └── analyzed_image002.pkl
```

### 2. **Quality-Based Filtering**
Leverages quality flags (PASS/WARNING/FAIL) from v4_validation:
- **PASS**: High-quality trajectories (R² ≥ 0.8, reasonable D, low uncertainty)
- **WARNING**: Acceptable with caveats (R² 0.7-0.8 or minor issues)
- **FAIL**: Unreliable (poor fit, unrealistic D, high localization error dominance)

### 3. **Multi-Mode Comparison**
Compare the same samples under different quality filters to assess robustness:
- **All trajectories**: Includes everything
- **PASS only**: Most stringent, highest quality
- **PASS + WARNING**: Balanced approach

### 4. **Multi-Metric Analysis**
Compare across 5 key metrics:
- **D**: Diffusion coefficient (μm²/s)
- **σ_loc**: Localization error (nm)
- **R²**: Fit quality
- **Track length**: Trajectory duration (frames)
- **CI width**: Uncertainty in D measurement

### 5. **Enhanced Visualizations**
- Quality distribution comparison (stacked bar charts)
- Multi-metric violin plots
- Quality-coded scatter plots (D vs R², D vs σ_loc, etc.)
- Effect size heatmaps across metrics
- Detailed diffusion comparisons (KDE, CDF, box plots)

## Global Configuration

Edit these parameters at the top of the script:

```python
# Quality filtering mode
QUALITY_FILTER_MODE = 'all'  # Options: 'all', 'pass_only', 'pass_warning', 'multi_mode'

# Additional filters (applied after quality filtering)
MIN_R_SQUARED = 0.7
MIN_TRACK_LENGTH = 10
MAX_DIFFUSION_COEFFICIENT = None  # None = no limit

# Statistical parameters
ALPHA = 0.05
SUBSAMPLE_SIZE = 100
N_SUBSAMPLES = 50
```

### Quality Filter Modes Explained

| Mode | Description | When to Use |
|------|-------------|-------------|
| `'all'` | Include all trajectories | Initial exploration, maximum data |
| `'pass_only'` | Only PASS quality | Most stringent, publication quality |
| `'pass_warning'` | PASS + WARNING | Balanced, good data quality |
| `'multi_mode'` | Compare all three | Assess robustness across filters |

## Usage

### Basic Comparison (Single Mode)

1. **Set quality mode** in the script:
   ```python
   QUALITY_FILTER_MODE = 'pass_only'
   ```

2. **Run the script**:
   ```bash
   python compare_diffusion_datasets_v2.py
   ```

3. **Enter sample information**:
   ```
   Enter number of samples to compare: 2

   --- Sample 1 ---
   Enter directory for sample 1: /path/to/Sample_WT/
   Enter name for sample 1: Wild Type

   --- Sample 2 ---
   Enter directory for sample 2: /path/to/Sample_Mutant/
   Enter name for sample 2: Mutant
   ```

### Multi-Mode Comparison

1. **Set mode to multi_mode**:
   ```python
   QUALITY_FILTER_MODE = 'multi_mode'
   ```

2. **Run the script** (same as above)

3. **Review outputs** organized by mode:
   ```
   comparison_v2_20250119_123456/
   ├── mode_all/
   │   ├── diffusion_detailed_comparison.png
   │   ├── quality_distribution_comparison.png
   │   └── dataset_summary.csv
   ├── mode_pass_only/
   │   ├── diffusion_detailed_comparison.png
   │   └── dataset_summary.csv
   ├── mode_pass_warning/
   │   └── ...
   └── cross_mode_summary.csv  ← Compare results across modes
   ```

## Output Files

### CSV Files

#### `dataset_summary.csv`
Summary statistics for each sample:
- Sample name, quality mode, number of files/trajectories
- Quality counts (PASS/WARNING/FAIL before filtering)
- Median/mean/std for D, σ_loc, R², track length

#### `comparison_SampleA_vs_SampleB.csv`
Detailed pairwise comparison for all metrics:
- Statistical tests (Mann-Whitney U, KS test)
- Effect sizes (Cliff's delta, Cohen's d)
- Means, medians, standard deviations
- p-values and significance flags

#### `quality_distribution_comparison.csv`
Chi-square test results for quality distribution differences

#### `cross_mode_summary.csv` (multi_mode only)
Comparison of results across quality filtering modes

### Plots

#### Quality Analysis
- **`quality_distribution_comparison.png`**: Stacked bar charts showing PASS/WARNING/FAIL counts per sample

#### Multi-Metric Comparison
- **`multi_metric_violin_comparison.png`**: Violin plots for all 5 metrics across samples

#### Quality-Coded Scatter Plots
- **`quality_coded_scatter_plots.png`**:
  - D vs R² (fit quality)
  - D vs σ_loc (localization error)
  - D vs track length
  - D vs CI width (uncertainty)
  - Points colored by quality (green=PASS, orange=WARNING, red=FAIL)

#### Diffusion Detailed Comparison
- **`diffusion_detailed_comparison.png`**:
  - KDE (kernel density estimate)
  - CDF (cumulative distribution)
  - Box plots
  - Histograms

#### Effect Size Analysis
- **`effect_size_heatmap.png`**: Heatmap showing Cliff's delta across all metrics

## Interpretation Guide

### 1. Quality Distribution Comparison

**Check first:** Are quality distributions similar across samples?

```
Sample_WT:    PASS=80%, WARNING=15%, FAIL=5%
Sample_Mut:   PASS=40%, WARNING=30%, FAIL=30%
```

⚠️ **WARNING**: If distributions differ significantly (p < 0.05), differences in D might reflect data quality, not biology!

**Recommendation**: Use `'pass_only'` or `'pass_warning'` mode to ensure fair comparison.

### 2. Multi-Metric Comparison

Look for consistency across metrics:

| Scenario | Interpretation |
|----------|----------------|
| D differs, σ_loc similar | Genuine biological difference |
| D differs, σ_loc also differs | May be tracking quality issue |
| D similar, R² differs |Fit quality issue, not biological |

### 3. Effect Sizes

Statistical significance (p < 0.05) doesn't always mean biological importance!

| Cliff's Delta | Interpretation | Action |
|---------------|----------------|--------|
| < 0.147 | Negligible | Samples essentially the same |
| 0.147-0.33 | Small | Minor difference, may not be meaningful |
| 0.33-0.474 | Medium | Moderate difference, worth investigating |
| > 0.474 | Large | Strong difference, likely biological |

### 4. Multi-Mode Comparison

If result changes across modes:

```
Mode            Significant?   Effect Size
all             Yes (p=0.001)  Medium
pass_only       No  (p=0.12)   Small
pass_warning    Yes (p=0.03)   Small
```

**Interpretation**: Difference driven by low-quality trajectories. Not robust.

**Robust result example**:
```
Mode            Significant?   Effect Size
all             Yes (p<0.001)  Large
pass_only       Yes (p<0.001)  Large
pass_warning    Yes (p<0.001)  Large
```

## Workflow Recommendations

### For Initial Exploration
1. Set `QUALITY_FILTER_MODE = 'multi_mode'`
2. Run comparison to see if results are robust
3. Check quality distribution differences

### For Publication
1. Check quality distributions are similar
2. Use `'pass_only'` or `'pass_warning'` mode
3. Report both statistical significance AND effect size
4. Show multi-mode results in supplement

### If Quality Differs Between Samples
1. Report quality metrics in main text
2. Use `'pass_only'` for fairest comparison
3. Consider: Is the quality difference itself meaningful?
   - If treatment affects tracking quality → biological signal
   - If imaging conditions differed → technical artifact

## Integration with Pipeline

This tool works with outputs from:

```
2diffusion_analyzer_v4_validation.py
  ↓ produces analyzed_*.pkl files
  ↓ containing quality flags
compare_diffusion_datasets_v2.py
  ↓ loads hierarchical samples
  ↓ filters by quality
  ↓ compares across metrics
```

### Complete Analysis Pipeline

```bash
# Step 1: Analyze each sample with v4_validation
python 2diffusion_analyzer_v4_validation.py
# (process Sample_WT/ files)

python 2diffusion_analyzer_v4_validation.py
# (process Sample_Mutant/ files)

# Step 2: Compare samples
# Option A: Single quality mode
python compare_diffusion_datasets_v2.py  # with QUALITY_FILTER_MODE = 'pass_only'

# Option B: Multi-mode (recommended first time)
python compare_diffusion_datasets_v2.py  # with QUALITY_FILTER_MODE = 'multi_mode'
```

## Advanced Customization

### Custom Quality Filters

Edit the `filter_trajectories_by_quality()` function:

```python
def filter_trajectories_by_quality(trajectories, quality_mode='all',
                                   min_r_squared=MIN_R_SQUARED,
                                   min_track_length=MIN_TRACK_LENGTH,
                                   max_diffusion=MAX_DIFFUSION_COEFFICIENT):
    # Add custom logic here
    # Example: Exclude trajectories with high σ_loc
    if traj.get('sigma_loc', 0) * 1000 > 50:  # > 50 nm
        continue
```

### Additional Metrics

To add a new metric (e.g., radius of gyration):

1. Extract it in `load_sample_hierarchical()`:
   ```python
   'radius_gyration': np.array([t.get('radius_of_gyration', np.nan) for t in all_trajectories])
   ```

2. Add to `compare_datasets_multi_metric()`:
   ```python
   'metrics': ['D', 'sigma_loc', 'r_squared', 'track_length', 'D_CI_width', 'radius_gyration']
   ```

3. Add plotting configuration:
   ```python
   metrics_info = {
       # ... existing metrics ...
       'radius_gyration': {'label': 'Radius of gyration (μm)', 'log': False}
   }
   ```

## Troubleshooting

### No files found
```
Warning: No analyzed_*.pkl files found
```
**Solution**: Ensure you're pointing to directories containing `analyzed_*.pkl` files from v4_validation, not raw trajectory files.

### Insufficient data after filtering
```
Warning: No trajectories passed filtering
```
**Solution**:
- Check `MIN_R_SQUARED` and `MIN_TRACK_LENGTH` aren't too stringent
- Try `quality_mode='all'` to see unfiltered data
- Review v4_validation quality reports

### Quality distributions differ
```
WARNING: Quality distributions differ significantly (p=0.001)
```
**Solution**:
- Use `'pass_only'` mode for fairest comparison
- Report quality difference in results
- Investigate: imaging conditions, sample preparation, tracking parameters

### All comparisons non-significant
**Possible causes**:
- Insufficient sample size (need more trajectories)
- High variability within samples
- Genuinely no difference

**Check**:
- Effect sizes (may have meaningful effect without significance)
- Quality-coded scatter plots (visual patterns)
- Individual trajectory distributions

## Citation and Version

**Version**: 2.0
**Date**: 2025-01-19
**Integrates with**: diffusion_analyzer_v4_validation.py
**Enhances**: 9python compare_diffusion_datasets.py (original)

## Questions?

For issues or feature requests, refer to the main repository documentation.
