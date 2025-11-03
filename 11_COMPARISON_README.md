# Multi-Condition Alpha Exponent Comparison

## Quick Start Guide

This script compares alpha exponent values across multiple experimental conditions (e.g., control vs treatment, different time points, etc.).

---

## Usage

### Method 1: Pattern Matching (Recommended)

Best when your files follow a naming pattern:

```bash
python 11compare_alpha_across_conditions.py
```

**Example:**
```
Choose mode (1=Manual, 2=Pattern): 2
Enter base directory: C:\Data\Alpha_Results\
How many conditions? 3

Condition 1:
  Name: Control
  Pattern: alpha_analyzed_control_*.pkl

Condition 2:
  Name: Treatment_5min
  Pattern: alpha_analyzed_treat_5min_*.pkl

Condition 3:
  Name: Treatment_10min
  Pattern: alpha_analyzed_treat_10min_*.pkl
```

### Method 2: Manual File Selection

When files don't follow a pattern:

```
Choose mode (1=Manual, 2=Pattern): 1
How many conditions? 2

Condition 1:
  Name: Control
  Files: C:\Data\file1.pkl, C:\Data\file2.pkl, C:\Data\file3.pkl

Condition 2:
  Name: Treatment
  Files: C:\Data\treatment_a.pkl, C:\Data\treatment_b.pkl
```

---

## Output Files

### 1. `alpha_summary_by_condition.csv`

Summary statistics for each condition:

| Column | Description |
|--------|-------------|
| condition | Condition name |
| n_trajectories | Total trajectories analyzed |
| alpha_mean | Mean alpha value |
| alpha_median | Median alpha value |
| alpha_std | Standard deviation |
| alpha_sem | Standard error of mean |
| alpha_ci_lower | Lower 95% CI bound |
| alpha_ci_upper | Upper 95% CI bound |
| n_normal, n_sub_diffusion, etc. | Counts by diffusion type |
| pct_normal, pct_sub_diffusion, etc. | Percentages by type |

### 2. `alpha_pairwise_comparisons.csv`

Statistical comparison for each pair of conditions:

| Column | Description |
|--------|-------------|
| condition_1, condition_2 | Conditions being compared |
| mean_alpha_1, mean_alpha_2 | Mean alpha values |
| diff_mean | Difference in means |
| mann_whitney_u | Mann-Whitney U statistic |
| p_value | Statistical significance |
| significant | TRUE if p < 0.05 |
| cliffs_delta | Effect size (non-parametric) |
| effect_size_interpretation | Small/medium/large |

### 3. `alpha_comparison_all_conditions.png`

Comprehensive 9-panel figure showing:
1. **Box plot** - Distribution with outliers
2. **Violin plot** - Density distribution
3. **CDF comparison** - Cumulative distribution
4. **Mean ± CI** - Bar plot with error bars
5. **Diffusion type distribution** - Stacked percentage bars
6. **Histogram overlay** - Direct comparison
7. **Sample sizes** - Number of trajectories
8. **Alpha vs D by condition** - Correlation scatter
9. **Statistical summary** - Text panel with test results

### 4. `alpha_pairwise_comparison_matrix.png`

(Created when comparing 2-5 conditions)

- Diagonal: Histograms for each condition
- Off-diagonal: Scatter plots with p-values and effect sizes

---

## Interpreting Results

### Statistical Tests

**Kruskal-Wallis Test** (Overall comparison):
- Tests if ANY conditions differ
- p < 0.05 → At least one condition is significantly different

**Mann-Whitney U Test** (Pairwise):
- Tests specific pairs
- p < 0.05 → This pair is significantly different

**Cliff's Delta** (Effect size):
- Measures practical significance
- Interpretation:
  - |δ| < 0.147: Negligible
  - |δ| < 0.33: Small
  - |δ| < 0.474: Medium
  - |δ| ≥ 0.474: Large

### Example Interpretation

```
Condition 1 (Control): α = 0.98 ± 0.02
Condition 2 (Treatment): α = 0.75 ± 0.03

Pairwise comparison:
  p = 0.0001 (significant)
  Cliff's δ = 0.52 (large effect)

Interpretation:
Treatment significantly REDUCES alpha, indicating a shift
from normal diffusion toward sub-diffusion. The large effect
size suggests this is a biologically meaningful change.
```

---

## File Organization Tips

### Recommended naming convention:

```
alpha_analyzed_[condition]_[replicate].pkl

Examples:
  alpha_analyzed_control_rep1.pkl
  alpha_analyzed_control_rep2.pkl
  alpha_analyzed_control_rep3.pkl
  alpha_analyzed_treatment_rep1.pkl
  alpha_analyzed_treatment_rep2.pkl
```

Then use pattern matching:
- Condition "Control": `alpha_analyzed_control_*.pkl`
- Condition "Treatment": `alpha_analyzed_treatment_*.pkl`

---

## Quality Control

The script automatically filters trajectories based on:
- **Minimum R² = 0.6** (alpha fit quality)
- **Minimum track length = 10 frames**

You can modify these in the script:
```python
MIN_R_SQUARED = 0.6
MIN_TRACK_LENGTH = 10
```

---

## Common Use Cases

### 1. Drug Treatment Comparison

```
Conditions:
  - Control (vehicle)
  - Drug_1uM
  - Drug_10uM
  - Drug_100uM

Research question: Does drug concentration affect diffusion mode?
Look for: Dose-dependent changes in mean alpha
```

### 2. Time Course

```
Conditions:
  - T0min
  - T5min
  - T10min
  - T30min

Research question: How does diffusion evolve over time?
Look for: Temporal trends in alpha values
```

### 3. Genotype Comparison

```
Conditions:
  - WT
  - Mutant_A
  - Mutant_B
  - Rescue

Research question: Do mutations alter diffusion properties?
Look for: Differences in diffusion type distribution
```

### 4. Subcellular Region Comparison

```
Conditions:
  - Nuclear
  - Cytoplasmic
  - Membrane

Research question: Does diffusion vary by location?
Look for: Compartment-specific alpha signatures
```

---

## Troubleshooting

### Problem: "Need at least 2 conditions with valid data"

**Solutions:**
- Check file paths are correct
- Verify files contain alpha_analyzed data
- Ensure files passed quality filters

### Problem: All p-values are non-significant

**Possible reasons:**
- Conditions truly don't differ (biological result!)
- Sample size too small (need more trajectories)
- High variability within conditions

**Solutions:**
- Increase replicates
- Check effect sizes (Cliff's Delta) - may have practical significance
- Look at diffusion type distributions instead

### Problem: Huge differences in sample sizes

**Impact:**
- Statistical tests handle unequal n
- But very small groups (<10) may be unreliable

**Solutions:**
- Increase MIN_TRAJECTORIES_PER_CONDITION
- Collect more data for undersampled conditions

---

## Advanced: Bootstrapping

The script uses **1000 bootstrap samples** to calculate:
- 95% confidence intervals for means
- Robust error estimates

This is more reliable than simple SEM when:
- Data is non-normal
- Sample sizes are unequal
- Distributions are skewed

Bootstrap parameters:
```python
N_BOOTSTRAP = 1000  # Number of resamples
BOOTSTRAP_CI = 95   # Confidence interval %
```

---

## Publication-Ready Output

The generated plots are:
- **High resolution**: 300 DPI
- **Large format**: 20x12 inches (9 panels)
- **Color-coded**: Consistent colors across panels
- **Statistical annotations**: P-values and effect sizes included

You can directly insert these into manuscripts or presentations!

---

## Complete Workflow Example

```bash
# Step 1: Analyze individual files
python 11alpha_exponent_analyzer.py
  Input: analyzed_*.pkl files
  Output: alpha_analyzed_*.pkl files

# Step 2: Compare conditions
python 11compare_alpha_across_conditions.py
  Input: Multiple alpha_analyzed_*.pkl files
  Output: Statistical comparison and plots
```

---

## Tips for Best Results

1. **Use consistent acquisition parameters** across conditions
   - Same frame rate (DT)
   - Same pixel size (CONVERSION)
   - Same tracking settings

2. **Include biological replicates**
   - At least 3 replicates per condition
   - More is better for statistical power

3. **Balance sample sizes**
   - Aim for similar numbers of trajectories per condition
   - If imbalanced, report effect sizes alongside p-values

4. **Check assumptions**
   - Look at distributions (histograms, CDFs)
   - Ensure quality filtering is appropriate
   - Verify no batch effects between replicates

---

## Next Steps After Analysis

Based on your results:

**If conditions DON'T differ significantly:**
- Check your hypothesis
- May be a genuine null result
- Consider alternative hypotheses

**If conditions DO differ significantly:**
- Examine which diffusion types changed (pie charts)
- Check if differences are concentration/dose dependent
- Correlate with other measurements (if available)
- Design follow-up experiments to test mechanisms

**If results are mixed:**
- Look at pairwise comparisons
- Some conditions may cluster together
- May reveal unexpected relationships

---

Good luck with your comparisons! 🎉
