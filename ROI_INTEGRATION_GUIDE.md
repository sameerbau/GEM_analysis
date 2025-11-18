# ROI Classification to Main Pipeline Integration Guide

## Overview

This guide explains how to integrate ROI-based classification results with the main diffusion analysis pipeline, allowing you to reuse all existing analysis tools for inside/outside ROI comparisons.

**Available Integration Tools:**
- **Single Condition** (`roi_to_pipeline_integration.py`) - This guide - Process one condition at a time
- **Batch Processing** (`roi_to_pipeline_batch.py`) - See [ROI_BATCH_INTEGRATION_GUIDE.md](ROI_BATCH_INTEGRATION_GUIDE.md) - Process multiple conditions with multi-ROI support

Choose batch processing if you have multiple conditions or need multi-ROI analysis.

## Integration Workflow

```
┌─────────────────────────────────────────────────────────────┐
│ Step 1: ROI Classification                                  │
│ Run: 1 IJ ROI loader_within_outside.py                      │
│ Input: ImageJ ROI file + trajectory/analyzed pickle files   │
│ Output: roi_trajectory_data.pkl                             │
└─────────────────────────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 2: Integration (NEW!)                                  │
│ Run: roi_to_pipeline_integration.py                         │
│ Input: roi_trajectory_data.pkl                              │
│ Output: tracked_inside_roi.pkl + tracked_outside_roi.pkl    │
└─────────────────────────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 3: Main Pipeline Analysis                              │
│ Run: 2traj_analyze_v1.py → Step 3 → Step 4 → ... → Step 11 │
│ Input: tracked_*.pkl files                                  │
│ Output: All standard analysis results                       │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Run ROI Classification (Existing Script)

```bash
python "Roi based classification/Over a folder/1 IJ ROI loader_within_outside.py"
```

This will create a directory with:
- `roi_trajectory_data.pkl` - ROI classification results
- `rois.png` - ROI visualization
- `assigned_trajectories.png` - Trajectory assignments

### 2. Run Integration Script (NEW!)

```bash
python roi_to_pipeline_integration.py
```

When prompted:
1. Enter path to `roi_trajectory_data.pkl`
2. Confirm or change output directory
3. Review parameter warnings (if any)

Output directory structure:
```
pipeline_integrated/
├── inside_roi/
│   └── tracked_inside_roi.pkl
├── outside_roi/
│   └── tracked_outside_roi.pkl
├── roi_classification_images/
│   ├── rois.png
│   └── assigned_trajectories.png
└── integration_summary.txt
```

### 3. Run Main Pipeline on Each Condition

**For Inside ROI:**
```bash
cd pipeline_integrated/inside_roi/
python ../../2traj_analyze_v1.py
# Continue with steps 3-11...
```

**For Outside ROI:**
```bash
cd pipeline_integrated/outside_roi/
python ../../2traj_analyze_v1.py
# Continue with steps 3-11...
```

### 4. Compare Results

Use existing comparison scripts (e.g., `11compare_alpha_across_conditions.py`) to compare inside vs outside ROI results.

## Important Parameters

### ⚠️ Parameter Consistency

**CRITICAL:** The following parameters must match across all scripts:

| Parameter | Main Pipeline | ROI Classification | Integration Script |
|-----------|---------------|--------------------|--------------------|
| DT | 0.1 s | N/A | 0.1 s |
| CONVERSION | 0.094 μm/px | N/A | 0.094 μm/px |
| PIXEL_TO_MICRON | N/A | 0.09 μm/px | N/A |

**⚠️ Current Issue:** There's a mismatch between `CONVERSION = 0.094` (main pipeline) and `PIXEL_TO_MICRON = 0.09` (ROI classification). This should be resolved by updating the ROI classification scripts to use `PIXEL_TO_MICRON = 0.094`.

### Recommended Fix

Update these files to use `PIXEL_TO_MICRON = 0.094`:
- `Roi based classification/Over a folder/1 IJ ROI loader_within_outside.py`
- `Roi based classification/1 IJ ROI loader_within_outside.py`

Change line ~51 from:
```python
PIXEL_TO_MICRON = 0.09  # µm/pixel
```
to:
```python
PIXEL_TO_MICRON = 0.094  # µm/pixel
```

## File Format Reference

### tracked_*.pkl Format (Main Pipeline Compatible)

```python
{
    'trajectories': [
        {
            'id': trajectory_id,
            'x': np.array([...]),      # x coordinates in μm
            'y': np.array([...]),      # y coordinates in μm
            'time': np.array([...]),   # time in seconds
            'dx': np.array([...]),     # x displacements
            'dy': np.array([...]),     # y displacements
            'dr2': np.array([...])     # squared displacements
        },
        ...
    ],
    'trajectory_lengths': [...],
    'msd_data': [np.array([...]), ...],
    'time_data': [np.array([...]), ...]
}
```

### roi_trajectory_data.pkl Format (ROI Classification Output)

```python
{
    'roi_assignments': {
        'roi_0001': [traj_indices],
        'roi_0002': [traj_indices],
        'unassigned': [traj_indices]
    },
    'roi_trajectories': {
        'roi_0001': [trajectory_dicts],
        'unassigned': [trajectory_dicts]
    },
    'roi_statistics': {
        'roi_0001': {
            'n': count,
            'mean_D': value,
            'median_D': value,
            ...
        }
    },
    'coordinate_transform': {
        'pixel_to_micron': 0.09,
        'scale_factor': 11.11,
        'x_offset': 0.0,
        'y_offset': 0.0
    }
}
```

## Benefits of Integration

1. **Code Reuse:** All existing analysis scripts (Steps 2-11) work without modification
2. **No Duplication:** Don't need to reimplement analysis for ROI-based data
3. **Easy Comparison:** Use existing comparison tools to compare inside vs outside ROI
4. **Consistent Analysis:** Same parameters and methods for all conditions
5. **Preserved Images:** ROI classification visualizations are kept for reference

## Troubleshooting

### "No trajectories found in ROI data"
- Check that ROI classification completed successfully
- Verify the `roi_trajectory_data.pkl` file is not empty

### "Parameter mismatch detected"
- Update ROI classification scripts to use `PIXEL_TO_MICRON = 0.094`
- Or modify the integration script if you're certain the current value is correct

### "File not found" errors
- Check file paths are correct
- Ensure ROI classification completed before running integration

### Integration script produces empty pickle files
- Check `MIN_TRACK_LENGTH` parameter (default: 10 frames)
- Verify trajectories have sufficient length in original data

## Next Steps

After integration is complete, you can:
1. Run any analysis from Steps 2-11 of the main pipeline
2. Use specialized analysis scripts (velocity, angle, rheology)
3. Compare inside vs outside ROI using existing comparison tools
4. Generate publication-quality figures using existing plotting scripts

## Files Created by This Integration

- `roi_to_pipeline_integration.py` - Main integration script
- `ROI_INTEGRATION_GUIDE.md` - This guide
- Output: `tracked_inside_roi.pkl` and `tracked_outside_roi.pkl` per condition

## Future Enhancements

Coming soon:
- Batch processing version for multiple conditions
- Automated pipeline execution after integration
- Multi-ROI support (separate tracked files for each ROI)
