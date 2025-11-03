# Alpha Exponent Analyzer for Diffusion Analysis

## Overview

The **11alpha_exponent_analyzer.py** script calculates the **alpha exponent** from single particle tracking trajectories to characterize diffusion types.

### What is Alpha (α)?

The alpha exponent comes from the generalized Mean Square Displacement equation:

```
MSD(τ) = 4*D*τ^α
```

**Physical Interpretation:**
- **α = 1**: Normal (Brownian) diffusion
- **α > 1**: Super-diffusion (ballistic motion, active transport)
- **α < 1**: Sub-diffusion (confined, hindered, or crowded environments)
- **α ≈ 0**: Fully confined motion

---

## Features

✅ **Log-log linear regression** (primary method - most stable)
✅ **Power-law nonlinear fitting** (validation method)
✅ **Automatic diffusion type classification**
✅ **Quality filtering** (R², track length, alpha range)
✅ **Comprehensive diagnostic plots**
✅ **Alpha vs Diffusion Coefficient correlation plot**
✅ **CSV and Pickle export** with all fitting parameters
✅ **Statistical summaries** per file

---

## Input Requirements

**Input files:** `analyzed_*.pkl` files generated from `2diffusion_analyzer.py`

Each pickle file should contain:
```python
{
    'trajectories': [
        {
            'x': array([...]),  # μm coordinates
            'y': array([...]),  # μm coordinates
            'D': float,         # Diffusion coefficient (optional)
            'id': trajectory_id
        },
        ...
    ]
}
```

---

## Configuration Parameters

Edit these global parameters in the script to match your data:

```python
# Time step between frames (seconds)
DT = 0.1

# Minimum trajectory length for analysis (frames)
MIN_TRACK_LENGTH = 10

# Minimum number of time lags needed for alpha fitting
MIN_TIME_LAGS = 4

# Maximum lag to use for fitting (fraction of trajectory length)
MAX_LAG_FRACTION = 0.3

# Minimum R² for accepting alpha fit
ALPHA_FIT_MIN_R2 = 0.6

# Alpha range for quality filtering
ALPHA_MIN = 0.0
ALPHA_MAX = 2.5
```

### Recommended Settings for Short Trajectories (10-25 frames)

The default settings are optimized for short trajectories:
- `MIN_TRACK_LENGTH = 10` (minimum usable length)
- `MIN_TIME_LAGS = 4` (at least 4 points needed for reliable fit)
- `MAX_LAG_FRACTION = 0.3` (uses first 30% of trajectory)
  - 10 frames → 3 lag points
  - 15 frames → 4 lag points
  - 20 frames → 6 lag points
  - 25 frames → 7 lag points

---

## Usage

### Method 1: Interactive Mode

```bash
python 11alpha_exponent_analyzer.py
```

The script will prompt you for:
1. **Input path** (single file or directory with multiple analyzed_*.pkl files)
2. **Output directory** (press Enter to use same as input)

### Method 2: Modify Script for Batch Processing

Edit the `main()` function to hardcode paths:

```python
def main():
    input_path = "/path/to/your/analyzed_files/"
    output_dir = "/path/to/output/"

    files_to_process = glob.glob(os.path.join(input_path, "analyzed_*.pkl"))

    for file_path in files_to_process:
        results = analyze_file(file_path, output_dir)
        if results:
            export_results_csv(results, output_dir)
            export_results_pickle(results, output_dir)
            create_diagnostic_plots(results, output_dir)
            create_alpha_vs_diffusion_plot(results, output_dir)
```

---

## Output Files

For each input file `analyzed_filename.pkl`, the following outputs are generated:

### 1. **CSV Files**

#### `filename_alpha_results.csv`
Per-trajectory results with columns:
- `trajectory_id`: Trajectory identifier
- `track_length`: Number of frames
- `alpha`: Alpha exponent (from log-log fit)
- `alpha_err`: Standard error of alpha
- `alpha_r_squared`: Fit quality (R²)
- `alpha_p_value`: Statistical significance
- `D_generalized`: Generalized diffusion coefficient
- `D_original`: Original diffusion coefficient from step 2
- `diffusion_type`: Classification (normal, sub-diffusion, super-diffusion, confined)
- `radius_of_gyration`: Spatial extent of trajectory
- `n_time_lags`: Number of time lags used for fitting

If `CALCULATE_BOTH_METHODS = True`:
- `alpha_powerlaw`: Alpha from power-law fit
- `alpha_powerlaw_err`: Error estimate
- `alpha_powerlaw_r_squared`: Power-law fit quality

#### `filename_alpha_summary.csv`
Statistical summary:
- `n_trajectories`: Total analyzed
- `alpha_mean`: Mean alpha value
- `alpha_median`: Median alpha value
- `alpha_std`: Standard deviation
- `alpha_sem`: Standard error of the mean
- `D_original_mean`: Mean original D
- `D_generalized_mean`: Mean generalized D
- `normal_diffusion_%`: Percentage of normal diffusion
- `sub_diffusion_%`: Percentage of sub-diffusion
- `super_diffusion_%`: Percentage of super-diffusion
- `confined_%`: Percentage of confined motion

### 2. **Pickle File**

#### `alpha_analyzed_filename.pkl`
Complete results dictionary containing:
- All trajectory-level results
- Statistical summaries
- Diffusion type counts
- Analysis parameters used

This file can be used as input for the multi-condition comparison script.

### 3. **Diagnostic Plots**

#### `filename_alpha_diagnostic_plots.png`
Comprehensive multi-panel figure (20x12 inches, 300 DPI) showing:

1. **Alpha distribution histogram** - Overall distribution with mean and normal diffusion reference
2. **Diffusion type pie chart** - Percentage breakdown by classification
3. **Fit quality vs Alpha** - R² scatter plot to assess fit reliability
4. **Alpha vs Track length** - Check for length-dependent biases
5. **Alpha vs Diffusion coefficient** - Main correlation plot (color-coded by alpha)
6. **Generalized vs Original D** - Comparison of D estimates
7. **Example MSD fits** - 6 representative trajectories showing log-log plots with fits

#### `filename_alpha_vs_diffusion.png`
**HIGH-QUALITY CORRELATION PLOT** (16x6 inches, 300 DPI) with two panels:

**Panel 1:** Scatter plot with diffusion type color-coding
- X-axis: Diffusion coefficient D (log scale)
- Y-axis: Alpha exponent
- Colors: Green (normal), Yellow (sub-diffusion), Blue (super-diffusion), Red (confined)
- Reference line at α = 1

**Panel 2:** Hexbin density map
- Shows clustering patterns in α-D space
- Useful for identifying population heterogeneity

---

## Interpretation Guide

### Alpha Values

| Alpha Range | Diffusion Type | Physical Meaning | Example Systems |
|-------------|----------------|------------------|-----------------|
| α < 0.2 | Confined | Trapped in small domain | Membrane corrals, cages |
| 0.2 ≤ α < 0.9 | Sub-diffusion | Hindered motion | Crowded cytoplasm, gels |
| 0.9 ≤ α ≤ 1.1 | Normal | Brownian diffusion | Free particles in solution |
| α > 1.1 | Super-diffusion | Active/ballistic motion | Motor-driven transport, flow |

### Fit Quality Criteria

**Good fit:**
- R² ≥ 0.7
- α within physically reasonable range (0.0-2.5)
- Sufficient time lags (≥4 points)

**Warning signs:**
- R² < 0.6: Noisy data or complex motion
- α > 2.0: Check for tracking errors or artifacts
- α < 0.1: May be too confined to measure accurately

### Alpha vs Diffusion Coefficient

The correlation between α and D reveals important information:

- **No correlation**: Homogeneous population with consistent diffusion type
- **Positive correlation**: Faster particles show more super-diffusive behavior
- **Negative correlation**: Faster particles are more confined (unusual, check data)
- **Multiple clusters**: Heterogeneous populations with distinct diffusion mechanisms

---

## Quality Filtering

Trajectories are excluded if:
1. Track length < `MIN_TRACK_LENGTH` frames
2. Insufficient time lags (< `MIN_TIME_LAGS`)
3. Alpha fit R² < `ALPHA_FIT_MIN_R2`
4. Alpha outside range [`ALPHA_MIN`, `ALPHA_MAX`]

The script reports:
- Number successfully analyzed
- Number failed/filtered
- Reason for exclusion

---

## Method Details

### Log-Log Linear Regression (Primary Method)

**Equation:**
```
log₁₀(MSD) = log₁₀(4*D) + α * log₁₀(τ)
```

**Advantages:**
- Numerically stable
- Less sensitive to noise
- Standard approach in SPT field
- Direct linear regression (fast, reliable)

**Fitting:**
- Takes logarithm of both MSD and time lag
- Performs linear regression to extract slope (α) and intercept
- Calculates R² for fit quality
- Estimates standard error of α

### Power-Law Fitting (Validation Method)

**Equation:**
```
MSD(τ) = 4*D*τ^α
```

**Advantages:**
- Direct physical model
- Can handle non-linearities better in some cases

**Disadvantages:**
- Less stable for noisy data
- Requires good initial guesses
- Slower optimization

**Note:** This method is optional (`CALCULATE_BOTH_METHODS = True`) and used primarily to validate log-log results.

---

## Troubleshooting

### Problem: "No trajectories passed quality filters!"

**Solutions:**
- Lower `ALPHA_FIT_MIN_R2` (try 0.5 or 0.4)
- Lower `MIN_TRACK_LENGTH` if you have very short trajectories
- Lower `MIN_TIME_LAGS` (minimum is 3)
- Check if trajectories are too short (need at least 10 frames)

### Problem: Most trajectories show α > 1.5

**Possible causes:**
- Tracking errors (check trajectory quality)
- Active motion or flow in the system
- Time step `DT` is incorrect
- Pixel-to-micron `CONVERSION` is wrong in earlier steps

### Problem: R² values are very low (< 0.5)

**Possible causes:**
- Very noisy data
- Mixed diffusion modes within single trajectories
- Trajectories too short
- Non-power-law behavior (e.g., directed motion transitions)

**Solutions:**
- Use longer trajectories if possible
- Check localization precision in tracking
- Consider more advanced analysis (segmented MSD, etc.)

### Problem: Alpha vs D plot shows no clear pattern

This is **often normal**! It means:
- Diffusion type is independent of speed (common)
- Homogeneous population

**Meaningful patterns to look for:**
- Clustering: Multiple populations
- Correlation: Relationship between speed and diffusion mode

---

## Integration with Pipeline

### Full Pipeline Workflow

```
Step 1: Load trajectories
  └─ 1Traj_load_v1.py
     Input:  CSV files (TrackMate format)
     Output: tracked_*.pkl

Step 2: Calculate diffusion coefficients
  └─ 2diffusion_analyzer.py
     Input:  tracked_*.pkl
     Output: analyzed_*.pkl

Step 11: Calculate alpha exponents  ← NEW!
  └─ 11alpha_exponent_analyzer.py
     Input:  analyzed_*.pkl
     Output: alpha_analyzed_*.pkl, CSV, plots

Step 12: Compare conditions (coming soon)
  └─ 11compare_alpha_across_conditions.py
     Input:  Multiple alpha_analyzed_*.pkl files
     Output: Statistical comparisons, multi-condition plots
```

---

## Example Terminal Output

```
======================================================================
ALPHA EXPONENT ANALYZER FOR SINGLE PARTICLE TRACKING
======================================================================

Found 3 file(s) to process

======================================================================
Loaded data from: analyzed_control.pkl

Analyzing alpha exponents for: control
Total trajectories: 450

Successfully analyzed: 387
Failed or filtered: 63

=== Alpha Analysis Summary ===
Mean alpha: 0.987 ± 0.018
Median alpha: 0.976

Diffusion type distribution:
  confined: 2.3% (n=9)
  sub-diffusion: 18.6% (n=72)
  normal: 73.4% (n=284)
  super-diffusion: 5.7% (n=22)

Exported CSV: control_alpha_results.csv
Exported summary: control_alpha_summary.csv
Exported pickle: alpha_analyzed_control.pkl
Saved diagnostic plots: control_alpha_diagnostic_plots.png
Saved alpha vs D plot: control_alpha_vs_diffusion.png

======================================================================
ANALYSIS COMPLETE!
======================================================================
```

---

## Scientific References

**Recommended reading on anomalous diffusion:**

1. **MSD power-law fitting:**
   - Saxton & Jacobson (1997) "Single-particle tracking: Applications to membrane dynamics" *Annu Rev Biophys Biomol Struct*

2. **Anomalous diffusion classification:**
   - Metzler et al. (2014) "Anomalous diffusion models and their properties" *Phys Rep*

3. **Alpha exponent interpretation:**
   - Manzo & Garcia-Parajo (2015) "A review of progress in single particle tracking" *Rep Prog Phys*

4. **SPT analysis best practices:**
   - Chenouard et al. (2014) "Objective comparison of particle tracking methods" *Nat Methods*

---

## Next Steps

After running the alpha analyzer:

1. **Review diagnostic plots** to assess fit quality
2. **Check CSV summaries** for overall statistics
3. **Compare alpha distributions** across experimental conditions
4. **Use the alpha_analyzed_*.pkl files** for multi-condition comparisons

**Coming soon:** `11compare_alpha_across_conditions.py` for statistical comparison across multiple experimental groups!

---

## Contact & Support

For questions or issues with the alpha analyzer:
- Check that input files are properly formatted
- Review the troubleshooting section above
- Verify that global parameters match your experimental setup

**Good luck with your analysis!** 🎉
