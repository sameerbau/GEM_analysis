# GEM_analysis
GEM based analysis


Main Analysis Pipeline (Numbered Scripts 1-10)

The core workflow processes trajectory data through these sequential steps:
1. Trajectory Loading (1Traj_load_v1.py)

    Input: CSV files with trajectory data (TrackMate format or similar)
    Function: Loads and preprocesses trajectory files with robust error handling
    Output: Processed trajectory data saved as .pkl files
    Key Features:
        Multiple delimiter support
        Pixel-to-μm conversion (0.094 μm/pixel default)
        MSD calculation for individual trajectories
        Diagnostic plots

2. Diffusion Analysis (2diffusion_analyzer.py)

    Input: Processed .pkl files from step 1
    Function: Calculates diffusion coefficients using MSD curve fitting
    Output: Analyzed trajectories with D values
    Key Metrics:
        Diffusion coefficient (D) via linear MSD fitting: MSD = 4*D*t + offset
        Radius of gyration
        Track quality metrics (R² values)

3. Data Pooling (3Traj_data_pooler.py)

    Input: Multiple analyzed .pkl files
    Function: Combines data from multiple experiments
    Output:
        Pooled statistics across datasets
        Comparison plots (violin plots, correlations)
        CSV exports for further analysis
    Features: Quality filtering (R² > 0.8, minimum track length)

6. Median Diffusion Analysis (6Get_median_diffusion_v2.py)

    Input: Analyzed trajectory files
    Function: Publication-quality visualization and statistical analysis
    Output:
        Bar graphs of median diffusion coefficients
        CDFs and histograms
        Statistical comparisons (Mann-Whitney U, Kolmogorov-Smirnov tests)
        Effect size calculations (Cliff's delta)
        Outlier detection using Modified Z-Score

7. Noise Characterization (7Noise_calculation_v2.py)

    Input: Processed trajectory data
    Function: Assesses measurement consistency and noise
    Methods:
        Trajectory partitioning: Splits tracks into smaller chunks
        Temporal chunking: Divides movie into time segments
    Output: Consistency metrics, CV values, trend analysis

8. Bootstrap Noise Analysis (8Noise_calculation_bootstrap.pu.py)

    Statistical validation using bootstrap methods

9-10. Advanced Validation

    Dataset comparison and advanced diffusion validation

Specialized Analysis Modules
A. Velocity Autocorrelation (Velocity autocorrelation/)

    Purpose: Measures how velocity correlations decay over time
    Key Scripts:
        5Velocity_autocorrel.py: Main calculation
        5Compare_Velocity_autocorrel.py: Compare datasets
        velocity_autocorrelation_validation.py: Validation tools
    Output: Velocity autocorrelation functions, correlation times

B. Angle Autocorrelation (Angle autocorrelation/)

    Purpose: Analyzes directional persistence in particle motion
    Key Scripts:
        4angle_autocorrelation.py: Main calculation
        Comparision_angle_autocorel.py: Dataset comparison
        angle_autocorrelation_scenarios.py: Scenario generation for validation
    Output: Angle correlation functions, crossing times (measure of persistence)

C. Two-Point Rheology (Two point rheology/)

    Purpose: Extract bulk viscoelastic properties from correlated motion of particle pairs
    Key Scripts:
        1two_point_rheology.py: Main TPM analysis
        2tpm_comparison.py: Compare conditions
        3tpm_visualizer.py: Visualization tools
    Output:
        Cross-correlation functions
        Longitudinal and transverse correlation components
        Viscoelastic modulus estimations
    Requirements: Multiple particles in same field of view

D. ROI-Based Classification (Roi based classification/)

    Purpose: Spatially segregate analysis based on regions of interest (e.g., inside vs outside cells)
    Key Scripts:
        0 diffusion_analysis_launcher.py: Workflow launcher
        1 IJ ROI loader.py: Load ImageJ ROI files
        2 Roi diffusion analyser.py: Calculate diffusion per ROI
        3 advanced_diffusion_stats.py: Statistical analysis
        4 diffusion_heatmap_generator.py: Spatial visualization
    Output:
        Diffusion coefficients segregated by spatial region
        Statistical comparisons between ROIs
        Heatmaps showing spatial distribution
    Tools: ImageJ integration for ROI definition

Key Analysis Parameters (Commonly Used)

    Time step (DT): 0.05-0.1 seconds
    Pixel conversion: 0.094 μm/pixel
    Min track length: 10-15 frames
    MSD fitting: First 4-11 points (up to 80% of curve)
    Quality threshold: R² > 0.8

Typical Workflow

    Track particles (external tool like TrackMate) → CSV files
    Load & process (1Traj_load_v1.py) → .pkl files
    Analyze diffusion (2diffusion_analyzer.py) → D values
    Pool data (optional, 3Traj_data_pooler.py) → Combined statistics
    Generate publication plots (6Get_median_diffusion_v2.py)
    Specialized analyses:
        Velocity/angle autocorrelation for motion characterization
        Two-point rheology for material properties
        ROI-based for spatial heterogeneity
    Noise validation (7Noise_calculation_v2.py)
