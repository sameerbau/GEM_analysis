# Methods Summary: GEM Analysis Pipelines

## Overview

This document describes the methodological approaches used in two complementary single-particle tracking analysis pipelines: (1) the main GEM (Generalized Ensemble Method) analysis pipeline for characterizing particle diffusion and motion dynamics, and (2) the ROI (Region Of Interest)-based classification pipeline for spatially-resolved diffusion analysis.

---

## 1. Main GEM Analysis Pipeline

### 1.1 Trajectory Data Processing

Single-particle trajectories were extracted from microscopy videos using TrackMate (Tinevez et al., 2017) or similar tracking software and exported as CSV files containing particle positions and frame numbers. Raw trajectory data were processed using a custom Python pipeline (1Traj_load_v1.py) that performed the following operations:

**Coordinate Conversion and Preprocessing:**
- Pixel coordinates were converted to physical units using a calibrated conversion factor (0.094 μm/pixel)
- Trajectories were filtered to include only tracks with minimum length of 10 frames
- Frame-to-frame displacements (Δx, Δy) and squared displacements (Δr²) were calculated for each trajectory
- Individual trajectory mean squared displacement (MSD) curves were computed

**Quality Control:**
- Trajectory counts and length distributions were analyzed to assess tracking quality
- Diagnostic plots showing MSD curves and trajectory statistics were generated for each dataset
- Only trajectories meeting minimum length criteria were retained for subsequent analysis

### 1.2 Diffusion Coefficient Estimation

Diffusion coefficients were calculated using linear regression on MSD curves (2diffusion_analyzer_v4_validation.py). The analysis implemented several methodological refinements across multiple versions (V1-V4):

**Core Method:**
The MSD was modeled as:
```
MSD(τ) = 4Dτ + 4σ²
```
where D is the diffusion coefficient, τ is the time lag, and σ² represents localization error. Linear regression was performed on the first 80% of each MSD curve (typically 8-11 time points) to extract D from the slope.

**Statistical Refinements:**

*Version 1 (Localization Error Reporting):*
- Explicitly calculated and reported localization error (σ_loc) from the MSD intercept
- Provided transparency in measurement precision

*Version 2 (Bootstrap Confidence Intervals):*
- Implemented bootstrap resampling (1000 iterations) to estimate 95% confidence intervals for D
- Generated robust, non-parametric error estimates without distributional assumptions

*Version 3 (Adaptive Time Lag Filtering):*
- Applied adaptive tau filtering with maximum lag = trajectory length/4
- Prevented poor averaging at long time lags where few data points contribute

*Version 4 (Automated Quality Control):*
- Implemented comprehensive quality validation with automatic flagging
- Quality metrics included:
  - Fit quality (R² ≥ 0.8 threshold)
  - Localization error dominance assessment
  - Diffusion coefficient range validation
  - Bootstrap confidence interval width evaluation
- Trajectories classified as PASS, WARNING, or FAIL based on quality criteria

**Output:**
For each trajectory, the following parameters were calculated:
- Diffusion coefficient (D, μm²/s)
- 95% confidence interval for D (bootstrap method)
- Localization error (σ_loc, μm)
- Fit quality (R²)
- Quality flag (PASS/WARNING/FAIL)

### 1.3 Statistical Analysis and Comparison

**Multi-Condition Comparisons:**
Diffusion coefficient distributions across experimental conditions were compared using non-parametric statistical methods (6Get_median_diffusion_v2.py):

*Outlier Detection:*
- Modified Z-score method (Iglewicz & Hoaglin, 1993) with threshold of 3.5
- Robust to non-normal distributions
- Outliers flagged but retained in visualizations

*Statistical Tests:*
- **Mann-Whitney U test:** Pairwise comparisons between conditions
- **Kolmogorov-Smirnov test:** Distribution shape comparisons
- **Cliff's Delta:** Non-parametric effect size measure with interpretation:
  - |δ| < 0.147: Negligible
  - 0.147 ≤ |δ| < 0.33: Small
  - 0.33 ≤ |δ| < 0.474: Medium
  - |δ| ≥ 0.474: Large

*Confidence Intervals:*
- Bootstrap resampling (1000 iterations) for median and percentile estimates
- 95% confidence intervals reported for all summary statistics

**Data Visualization:**
Results were presented using multiple complementary visualizations:
- Median bar plots with 95% confidence intervals
- Cumulative distribution functions (CDFs)
- Violin plots showing distribution shapes
- Histograms with outlier identification
- All figures generated at publication quality (300 DPI)

### 1.4 Anomalous Diffusion Characterization

**Alpha Exponent Analysis:**
The anomalous diffusion exponent (α) was calculated using generalized MSD analysis (11alpha_exponent_analyzer.py):

```
MSD(τ) = 4Dτ^α
```

**Fitting Procedure:**
- MSD calculated at multiple time lags using the first 30% of each trajectory
- Log-log linear regression: log(MSD) = log(4D) + α·log(τ)
- Power-law fitting for validation
- Only fits with R² ≥ 0.6 retained
- Alpha values constrained to physically reasonable range [0.0, 2.5]

**Diffusion Type Classification:**
Based on alpha exponent values, trajectories were classified as:
- **Confined (α ≈ 0):** Fully restricted motion
- **Sub-diffusive (0 < α < 0.9):** Hindered, crowded, or viscoelastic environments
- **Normal Brownian (0.9 ≤ α ≤ 1.1):** Free diffusion
- **Super-diffusive (α > 1.1):** Active transport or ballistic motion

**Multi-Condition Alpha Comparison:**
Alpha distributions across experimental conditions were compared using robust statistical methods (11compare_alpha_across_conditions.py):

*Statistical Tests:*
- **Kruskal-Wallis H-test:** Overall comparison across multiple conditions
- **Mann-Whitney U test:** Pairwise comparisons with Bonferroni correction
- **Cliff's Delta:** Effect size for pairwise differences
- Bootstrap confidence intervals (1000 resamples) for all metrics

*Comprehensive Visualizations:*
Nine-panel comparison figures included:
- Box plots and violin plots
- Cumulative distribution functions
- Mean ± 95% CI bar plots
- Diffusion type distribution pie charts
- Correlation plots (α vs. D)
- Histogram overlays
- Sample size comparisons
- Statistical summary panels

### 1.5 Temporal Consistency and Noise Analysis

**Noise Characterization:**
Measurement consistency was assessed using temporal partitioning methods (7Noise_calculation_v2.py, 8Noise_calculation_bootstrap.py):

- Trajectories partitioned into smaller chunks
- Movies divided into temporal segments
- Coefficient of variation (CV) calculated across chunks
- Trend analysis to detect systematic drift
- Bootstrap validation for confidence intervals
- Quality recommendations based on noise levels

### 1.6 Specialized Motion Analysis

**Velocity Autocorrelation:**
Directional persistence was quantified by analyzing velocity autocorrelation functions:
```
C_v(τ) = ⟨v(t)·v(t+τ)⟩ / ⟨v²(t)⟩
```
- Correlation decay times measured
- Persistence lengths calculated
- Multi-condition comparisons performed

**Angle Autocorrelation:**
Turning behavior characterized through angle autocorrelation analysis:
```
C_θ(τ) = ⟨cos[θ(t+τ) - θ(t)]⟩
```
- Directional persistence quantified
- Crossing times (when correlation reaches zero) calculated
- Comparison of reorientation dynamics across conditions

**Two-Point Microrheology:**
Bulk viscoelastic properties estimated from correlated particle pair motion:
- Cross-correlation functions calculated for particle pairs
- Spatial correlation ranges determined
- Material properties inferred from collective motion
- Required multiple particles within same field of view

### 1.7 Key Experimental Parameters

**Standard Configuration:**
- Time step between frames (Δt): 0.1 s
- Pixel-to-micron conversion: 0.094 μm/pixel
- Minimum trajectory length: 10 frames
- MSD fitting range: First 80% of trajectory or first 11 points
- Alpha fitting range: First 30% of trajectory
- R² threshold for diffusion analysis: 0.8
- R² threshold for alpha analysis: 0.6
- Modified Z-score outlier threshold: 3.5
- Bootstrap iterations: 1000
- Significance level: α = 0.05

---

## 2. ROI-Based Classification Pipeline

### 2.1 Overview and Purpose

The ROI-based classification pipeline enables spatially-resolved diffusion analysis by segregating trajectories based on their location relative to user-defined regions of interest. This approach is particularly useful for comparing diffusion properties in different cellular compartments (e.g., nucleus vs. cytoplasm, inside vs. outside cells, or between specific organelles).

### 2.2 ROI Definition and Loading

**ROI Creation:**
Regions of interest were manually defined in ImageJ/Fiji by outlining desired areas as polygon ROIs. Multiple ROIs per image were supported. ROI definitions were saved as ImageJ ROI ZIP files.

**Coordinate System Integration:**
The pipeline (1 IJ ROI loader_within_outside.py) integrated ImageJ ROI coordinates with trajectory data through the following steps:

1. **ROI File Parsing:** ImageJ ROI ZIP files were loaded using the read_roi Python library
2. **Coordinate Transformation:** Trajectory coordinates (in micrometers) were converted to pixel coordinates to match ROI definitions
3. **Scaling Consistency:** The same pixel-to-micron conversion factor (0.094 μm/pixel) was applied consistently across both pipelines

**Spatial Parameters:**
```
Pixel-to-micron conversion: 0.094 μm/pixel
X offset: 0.0 pixels (default)
Y offset: 0.0 pixels (default)
Minimum trajectories per ROI: 1
```

### 2.3 Trajectory Classification

**Point-in-Polygon Testing:**
Trajectories were assigned to ROIs using computational geometry algorithms:

- Ray casting method for point-in-polygon determination
- Each trajectory point tested against all ROI boundaries
- Trajectory classified based on majority of points (>50% threshold)
- Three classification categories:
  - **Inside ROI:** Trajectory predominantly within defined ROI(s)
  - **Outside ROI:** Trajectory predominantly outside all ROIs
  - **Unassigned:** Ambiguous trajectories (edge cases)

**Multi-ROI Support:**
When multiple ROIs were defined:
- Each trajectory assigned to a specific ROI (roi_0001, roi_0002, etc.)
- Combined "inside_roi" group aggregating all ROI-assigned trajectories
- Combined "outside_roi" group for all external trajectories
- Enables both individual ROI analysis and inside vs. outside comparisons

### 2.4 ROI-Specific Diffusion Analysis

**Per-ROI Statistics:**
Diffusion coefficients were calculated separately for each ROI classification using methods identical to the main pipeline (Section 1.2):

For each ROI group:
- Individual trajectory diffusion coefficients calculated
- Bootstrap confidence intervals estimated
- Quality control filters applied
- Distribution statistics computed (median, quartiles, mean, SD)

**Spatial Visualization:**
ROI classifications were visualized through:
- Trajectory overlay plots showing ROI boundaries and assigned trajectories
- Color-coded trajectories by ROI assignment
- Heatmaps showing spatial distribution of diffusion coefficients (4 diffusion_heatmap_generator.py)

### 2.5 Integration with Main Analysis Pipeline

**Single-Condition Integration:**
ROI-classified trajectories were reformatted for compatibility with the main pipeline (roi_to_pipeline_integration.py):

1. **Data Splitting:** Trajectories separated into "inside_roi" and "outside_roi" groups
2. **Format Conversion:** ROI classification output converted to main pipeline format (tracked_*.pkl)
3. **Directory Organization:** Standardized folder structure created:
   ```
   pipeline_integrated/
   ├── inside_roi/tracked_inside_roi.pkl
   ├── outside_roi/tracked_outside_roi.pkl
   ├── roi_classification_images/
   └── integration_summary.txt
   ```
4. **Analysis Continuation:** Standard pipeline steps (diffusion analysis, alpha analysis, etc.) applied to each ROI group independently

**Batch Multi-Condition Processing:**
For experiments with multiple conditions, batch integration was performed (roi_to_pipeline_batch.py):

*Simple Mode (Inside vs. Outside):*
- Two subdirectories per condition
- Trajectories grouped as "inside all ROIs" vs. "outside all ROIs"
- Enables statistical comparison of inside vs. outside populations

*Multi-ROI Mode:*
- Individual directory for each ROI (roi_0001, roi_0002, etc.)
- Combined inside/outside directories maintained
- Supports both inter-ROI comparisons and inside vs. outside analysis
- Example output structure:
  ```
  batch_output/
  ├── condition1/
  │   ├── roi_0001/tracked_roi_0001.pkl
  │   ├── roi_0002/tracked_roi_0002.pkl
  │   ├── inside_roi/tracked_inside_roi.pkl
  │   └── outside_roi/tracked_outside_roi.pkl
  └── condition2/[similar structure]
  ```

### 2.6 Statistical Comparison of Spatial Regions

**Inter-ROI Statistical Analysis:**
Following integration with the main pipeline, ROI-specific diffusion properties were compared using identical statistical methods as multi-condition comparisons (Section 1.3):

- Mann-Whitney U tests for pairwise ROI comparisons
- Kolmogorov-Smirnov tests for distribution shape differences
- Cliff's Delta effect sizes
- Bootstrap confidence intervals
- Multiple testing correction (Bonferroni) when comparing >2 ROIs

**Spatial Heterogeneity Analysis:**
Heatmap generation (4 diffusion_heatmap_generator.py) provided visual assessment of:
- Diffusion coefficient spatial distribution
- Regions of high vs. low mobility
- Spatial organization of diffusion properties
- Correlation of diffusion with cellular structures

### 2.7 Data Flow and File Formats

**ROI Classification Output (roi_trajectory_data.pkl):**
Contains:
- ROI assignments for each trajectory
- ROI-specific trajectory lists
- Per-ROI summary statistics
- Coordinate transformation parameters

**Integration Output (tracked_*.pkl per ROI):**
Standard main pipeline format containing:
- Trajectory coordinates (x, y in μm)
- Time points (seconds)
- Displacements (Δx, Δy, Δr²)
- Individual trajectory MSD data
- Trajectory length statistics

### 2.8 Quality Control and Parameter Consistency

**Critical Parameter Matching:**
To ensure valid comparisons, the following parameters were maintained consistently across both pipelines:
- Time step (Δt): 0.1 s
- Pixel-to-micron conversion: 0.094 μm/pixel
- Minimum trajectory length: 10 frames
- MSD fitting parameters: Identical to main pipeline
- Statistical thresholds: R² ≥ 0.8, outlier detection threshold = 3.5

**Validation Checks:**
- Trajectory counts verified before and after integration
- Coordinate transformation accuracy assessed through visual inspection
- ROI assignment consistency checked across processing steps
- Statistical power evaluated for each ROI (sufficient trajectory counts)

---

## 3. Experimental Design Considerations

### 3.1 Sample Size and Statistical Power

**Minimum Requirements:**
- At least 100 trajectories per condition recommended for robust statistics
- Minimum 30 trajectories per ROI for meaningful comparisons
- Bootstrap methods provide robust estimates even with modest sample sizes

**Reporting:**
All analyses report:
- Number of trajectories analyzed (n)
- Trajectory length distributions
- Quality control filtering statistics (trajectories excluded)

### 3.2 Quality Metrics and Filtering

**Multi-Level Quality Control:**
1. **Trajectory Level:** Minimum length, no gaps in tracking
2. **Fit Quality:** R² thresholds for linear regression
3. **Physical Reasonableness:** D and α within expected ranges
4. **Localization Error:** Assessment of measurement precision
5. **Statistical Validation:** Bootstrap CI width evaluation

**Transparent Reporting:**
- Quality flags (PASS/WARNING/FAIL) reported for each trajectory
- Outliers identified but retained in visualizations
- Filtering criteria explicitly documented in output files

### 3.3 Software and Implementation

**Dependencies:**
- Python 3.7+
- NumPy, SciPy: Numerical computing and statistics
- Matplotlib, Seaborn: Data visualization
- Pandas: Data manipulation
- read_roi: ImageJ ROI file parsing
- scikit-learn: Additional statistical tools

**Computational Performance:**
- Typical processing time: <1 minute per 1000 trajectories
- Bootstrap operations: ~10-30 seconds per condition
- Batch processing supported for high-throughput experiments

---

## 4. Summary of Methodological Innovations

The GEM analysis pipelines incorporate several methodological advances over traditional diffusion analysis approaches:

1. **Robust Error Estimation:** Bootstrap confidence intervals provide non-parametric error estimates without distributional assumptions

2. **Automated Quality Control:** Multi-level validation with explicit flagging ensures only reliable measurements contribute to conclusions

3. **Anomalous Diffusion Characterization:** Alpha exponent analysis distinguishes different motion types and provides richer characterization than D alone

4. **Spatial Resolution:** ROI-based classification enables spatially-resolved analysis while maintaining full statistical rigor

5. **Transparent Reporting:** Explicit documentation of localization error, fit quality, and quality flags ensures reproducibility

6. **Multi-Modal Comparison:** Integration of multiple comparison methods (statistical tests, effect sizes, visualization) provides robust evidence for differences

7. **Comprehensive Validation:** Noise analysis, temporal consistency checks, and methodological comparisons (overlapping vs. non-overlapping MSD) ensure measurement reliability

---

## 5. Data Availability and Reproducibility

**Code Availability:**
All analysis scripts are available in the GEM_analysis repository with detailed documentation and example datasets.

**Analysis Parameters:**
All critical parameters are explicitly documented in script headers and output files, ensuring full reproducibility.

**Output Documentation:**
Each analysis step generates:
- Processed data files with metadata
- Summary statistics in CSV format
- Publication-quality figures with embedded parameters
- Analysis logs documenting filtering and quality control decisions

---

## References

Iglewicz, B., & Hoaglin, D. C. (1993). How to detect and handle outliers. ASQC Quality Press.

Tinevez, J. Y., Perry, N., Schindelin, J., Hoopes, G. M., Reynolds, G. D., Laplantine, E., ... & Eliceiri, K. W. (2017). TrackMate: An open and extensible platform for single-particle tracking. Methods, 115, 80-90.

---

**Document Version:** 1.0
**Last Updated:** 2025-11-21
**Pipeline Version:** Main Pipeline v4, ROI Pipeline v2
