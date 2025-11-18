# ROI Batch Integration Guide with Multi-ROI Support

## Overview

This guide explains how to use the batch processing tool to integrate multiple ROI-classified datasets with the main diffusion analysis pipeline. The batch tool supports both simple mode (inside/outside only) and multi-ROI mode (separate analysis for each ROI).

## Features

- **Batch Processing**: Process multiple conditions in one run
- **Multi-ROI Mode**: Create separate tracked files for each individual ROI
- **Automatic Organization**: Proper directory structure for pipeline integration
- **Parameter Validation**: Ensures consistency with main pipeline
- **Progress Tracking**: Clear feedback during processing
- **Summary Reports**: Detailed reports of all processed conditions

## Quick Start

### Basic Batch Processing (Simple Mode)

```bash
python roi_to_pipeline_batch.py
```

When prompted:
1. Enter directory containing ROI classification results
2. Select mode: `1` for Simple mode (inside/outside only)
3. Confirm output directory

### Multi-ROI Mode

```bash
python roi_to_pipeline_batch.py
```

When prompted:
1. Enter directory containing ROI classification results
2. Select mode: `2` for Multi-ROI mode
3. Confirm output directory

## Processing Modes

### 1. Simple Mode (Inside/Outside Only)

Creates two subdirectories per condition:
- `inside_roi/` - All trajectories within any ROI
- `outside_roi/` - All trajectories not assigned to any ROI (unassigned)

**Use when:** You want to compare trajectories inside vs outside ROIs as a group.

**Output structure:**
```
batch_output/
├── condition1/
│   ├── inside_roi/
│   │   └── tracked_inside_roi.pkl
│   ├── outside_roi/
│   │   └── tracked_outside_roi.pkl
│   └── roi_classification_images/
│       ├── rois.png
│       └── assigned_trajectories.png
├── condition2/
│   ├── inside_roi/
│   │   └── tracked_inside_roi.pkl
│   └── ...
└── batch_integration_summary.txt
```

### 2. Multi-ROI Mode

Creates a separate subdirectory for EACH individual ROI, plus combined inside/outside directories.

**Use when:** You want to analyze each ROI independently and compare specific ROIs.

**Output structure:**
```
batch_output/
├── condition1/
│   ├── roi_0001/
│   │   └── tracked_roi_0001.pkl
│   ├── roi_0002/
│   │   └── tracked_roi_0002.pkl
│   ├── roi_0003/
│   │   └── tracked_roi_0003.pkl
│   ├── unassigned/
│   │   └── tracked_unassigned.pkl
│   ├── inside_roi/
│   │   └── tracked_inside_roi.pkl  (combined roi_0001 + roi_0002 + roi_0003)
│   ├── outside_roi/
│   │   └── tracked_outside_roi.pkl  (same as unassigned)
│   └── roi_classification_images/
├── condition2/
│   └── ...
└── batch_integration_summary.txt
```

## Usage Examples

### Example 1: Process Multiple Experimental Conditions

You have ROI classification results for 3 conditions:
```
data/
├── control/
│   └── roi_trajectory_data.pkl
├── treatment1/
│   └── roi_trajectory_data.pkl
└── treatment2/
    └── roi_trajectory_data.pkl
```

**Steps:**
```bash
python roi_to_pipeline_batch.py
# Enter: data/
# Mode: 1 (Simple)
# Output: data/pipeline_batch_integrated/
```

**Result:**
```
data/pipeline_batch_integrated/
├── control/
│   ├── inside_roi/tracked_inside_roi.pkl
│   └── outside_roi/tracked_outside_roi.pkl
├── treatment1/
│   ├── inside_roi/tracked_inside_roi.pkl
│   └── outside_roi/tracked_outside_roi.pkl
└── treatment2/
    ├── inside_roi/tracked_inside_roi.pkl
    └── outside_roi/tracked_outside_roi.pkl
```

### Example 2: Analyze Multiple Cell Types (Multi-ROI)

You have ROI classification where each ROI represents a different cell type:
```
data/experiment1/
└── roi_trajectory_data.pkl  (with roi_0001=cellA, roi_0002=cellB, roi_0003=cellC)
```

**Steps:**
```bash
python roi_to_pipeline_batch.py
# Enter: data/
# Mode: 2 (Multi-ROI)
# Output: data/pipeline_batch_integrated/
```

**Result:**
```
data/pipeline_batch_integrated/
└── experiment1/
    ├── roi_0001/tracked_roi_0001.pkl  (cellA trajectories)
    ├── roi_0002/tracked_roi_0002.pkl  (cellB trajectories)
    ├── roi_0003/tracked_roi_0003.pkl  (cellC trajectories)
    ├── unassigned/tracked_unassigned.pkl
    ├── inside_roi/tracked_inside_roi.pkl  (all cells combined)
    └── outside_roi/tracked_outside_roi.pkl
```

Now you can:
- Analyze each cell type independently using the main pipeline
- Compare cellA vs cellB vs cellC
- Compare all cells (inside_roi) vs outside

## Running Main Pipeline After Integration

### For Simple Mode

```bash
# Navigate to each condition
cd batch_output/condition1/inside_roi/
python ../../../2traj_analyze_v1.py
# Continue with steps 3-11...

cd ../outside_roi/
python ../../../2traj_analyze_v1.py
# Continue with steps 3-11...
```

### For Multi-ROI Mode

```bash
# Process each ROI individually
cd batch_output/condition1/roi_0001/
python ../../../2traj_analyze_v1.py
# Continue with steps 3-11...

cd ../roi_0002/
python ../../../2traj_analyze_v1.py
# Continue with steps 3-11...

# Also process combined inside/outside
cd ../inside_roi/
python ../../../2traj_analyze_v1.py
# Continue with steps 3-11...
```

## Automated Pipeline Execution (Advanced)

You can create a simple script to run the main pipeline on all conditions:

```bash
#!/bin/bash
# run_pipeline_batch.sh

OUTPUT_DIR="batch_output"

# Find all tracked_*.pkl files
for pkl_file in $(find "$OUTPUT_DIR" -name "tracked_*.pkl"); do
    # Get directory containing the pkl file
    pkl_dir=$(dirname "$pkl_file")

    echo "Processing: $pkl_dir"

    # Run Step 2 in that directory
    cd "$pkl_dir"
    python ../../../2traj_analyze_v1.py

    # Run subsequent steps (3-11) here...

    cd -
done

echo "Batch pipeline execution complete!"
```

## Batch Summary Report

After processing, a summary report is generated: `batch_integration_summary.txt`

**Contents:**
- Processing timestamp
- Mode used (Simple or Multi-ROI)
- Parameters (DT, CONVERSION, MIN_TRACK_LENGTH)
- Per-condition trajectory counts
- Files created
- Next steps

**Example:**
```
ROI-to-Pipeline Batch Integration Summary
======================================================================
Generated: 2025-11-18 15:30:45
Mode: Multi-ROI

Parameters:
  DT = 0.1 s
  CONVERSION = 0.094 μm/pixel
  MIN_TRACK_LENGTH = 10 frames

Processed Conditions:
----------------------------------------------------------------------

control:
  roi_0001: 45 trajectories
  roi_0002: 38 trajectories
  unassigned: 12 trajectories
  inside_roi: 83 trajectories
  outside_roi: 12 trajectories
  Files created: 5

treatment1:
  roi_0001: 52 trajectories
  roi_0002: 41 trajectories
  unassigned: 8 trajectories
  inside_roi: 93 trajectories
  outside_roi: 8 trajectories
  Files created: 5

======================================================================
Total conditions processed: 2
Total tracked_*.pkl files created: 10

Next Steps:
  1. Navigate to each condition directory
  2. For each ROI subdirectory, run main pipeline Step 2:
     cd <condition>/<roi_dir>/
     python ../../2traj_analyze_v1.py
  3. Continue with subsequent pipeline steps (3, 4, 5, ...) for each ROI
  4. Use comparison scripts to compare different ROI conditions
```

## Comparison Strategies

### Simple Mode Comparisons

Use existing comparison scripts:
```bash
# Compare inside vs outside for one condition
python 11compare_alpha_across_conditions.py
# Select: condition1/inside_roi and condition1/outside_roi

# Compare same region across conditions
python 11compare_alpha_across_conditions.py
# Select: condition1/inside_roi, condition2/inside_roi, condition3/inside_roi
```

### Multi-ROI Mode Comparisons

More comparison possibilities:
```bash
# Compare different ROIs within same condition
python 11compare_alpha_across_conditions.py
# Select: condition1/roi_0001, condition1/roi_0002, condition1/roi_0003

# Compare same ROI across conditions
python 11compare_alpha_across_conditions.py
# Select: condition1/roi_0001, condition2/roi_0001, condition3/roi_0001

# Compare specific ROI vs combined inside
python 11compare_alpha_across_conditions.py
# Select: condition1/roi_0001 vs condition1/inside_roi
```

## Troubleshooting

### No roi_trajectory_data.pkl Files Found

**Problem:** Script reports "No roi_trajectory_data.pkl files found"

**Solutions:**
- Verify you ran ROI classification first
- Check the directory path is correct
- Ensure files are named exactly `roi_trajectory_data.pkl`

### Parameter Mismatch Warning

**Problem:** Warning about PIXEL_TO_MICRON mismatch

**Solution:**
- Update ROI classification scripts to use PIXEL_TO_MICRON = 0.094
- Or update CONVERSION in main pipeline if you're certain 0.09 is correct
- The integration will continue but results may be inconsistent

### Empty Tracked Files

**Problem:** tracked_*.pkl files contain no trajectories

**Possible causes:**
- MIN_TRACK_LENGTH filter too strict
- Original ROI classification had no trajectories assigned
- All trajectories filtered out during processing

**Solutions:**
- Check MIN_TRACK_LENGTH parameter (line 37 in roi_to_pipeline_batch.py)
- Verify original ROI classification results
- Check trajectory lengths in source data

### Memory Issues with Large Datasets

**Problem:** Script crashes or runs out of memory

**Solutions:**
- Process conditions one at a time using single-condition script
- Reduce the number of trajectories per condition
- Use a machine with more RAM

## Performance Tips

1. **Parallel Processing**: Process independent conditions on different machines/cores
2. **Incremental Processing**: Process and analyze one condition at a time
3. **Disk Space**: Ensure sufficient space (~2-5x the size of input data)
4. **File Organization**: Keep ROI classification results organized by condition

## Parameter Reference

### Critical Parameters (Must Match Main Pipeline)

| Parameter | Value | Location | Purpose |
|-----------|-------|----------|---------|
| DT | 0.1 s | Line 37 | Time step between frames |
| CONVERSION | 0.094 μm/px | Line 38 | Pixel to micron conversion |
| MIN_TRACK_LENGTH | 10 frames | Line 39 | Minimum trajectory length |

### ROI Classification Parameters (Must Be Consistent)

| Parameter | Recommended Value | Location |
|-----------|-------------------|----------|
| PIXEL_TO_MICRON | 0.094 μm/px | ROI classification scripts |
| X_OFFSET | 0.0 px | ROI classification scripts |
| Y_OFFSET | 0.0 px | ROI classification scripts |

## Advanced Features

### Custom Condition Naming

The script automatically extracts condition names from directory structure. To customize:

1. Organize ROI results in named directories:
```
experiments/
├── control_day1/
│   └── roi_trajectory_data.pkl
└── treatment_day1/
    └── roi_trajectory_data.pkl
```

2. Condition names will be: `control_day1` and `treatment_day1`

### Selective Processing

To process only specific conditions, organize them in a separate directory:
```
selected/
├── condition_A/ → ../all_data/condition_A/
└── condition_C/ → ../all_data/condition_C/
```

Then run batch processing on `selected/`

### Integration with Other Tools

The tracked_*.pkl files are compatible with:
- All main pipeline scripts (Steps 2-11)
- Velocity analysis tools
- Angle analysis tools
- Two-point rheology analysis
- Custom analysis scripts that read tracked format

## Comparison with Single-Condition Script

| Feature | Single (`roi_to_pipeline_integration.py`) | Batch (`roi_to_pipeline_batch.py`) |
|---------|-------------------------------------------|-------------------------------------|
| Conditions | 1 | Multiple |
| Multi-ROI | No | Yes |
| User Input | Interactive | Interactive |
| Progress | Per-step | Per-condition |
| Summary | Single condition | All conditions |
| Use Case | Testing, single analysis | Production, multiple experiments |

## Next Steps

After successful batch integration:

1. **Verify Output**: Check summary report and sample a few tracked_*.pkl files
2. **Test Pipeline**: Run Step 2 on one condition to verify compatibility
3. **Automate**: Create shell scripts for batch pipeline execution
4. **Analyze**: Use main pipeline and comparison tools for analysis
5. **Document**: Record condition names, ROI meanings, and analysis parameters

## Files Created

- `roi_to_pipeline_batch.py` - Main batch processing script
- `batch_integration_summary.txt` - Processing report (per run)
- `tracked_*.pkl` - Pipeline-compatible trajectory files (per ROI/condition)
- `roi_classification_images/` - Preserved ROI visualizations (per condition)

## See Also

- `ROI_INTEGRATION_GUIDE.md` - Single-condition integration guide
- `test_roi_integration.py` - Test suite for verification
- Main pipeline documentation (Steps 1-11)
