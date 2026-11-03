# NMR Analysis and Machine Learning Pipelines

Tools for processing and analyzing one-dimensional nuclear magnetic resonance (NMR) spectra, building compound calibration curves, and evaluating machine-learning workflows on **Standard** and **Pure Shift** NMR data.

## Repository Structure

| Directory | Scope |
| --- | --- |
| `Machine_Learning/` | Dataset preparation, model scripts, data augmentation, and experimental results |
| `Calibration_Pipeline/` | Concentration extraction, spectral peak visualization, and calibration curves |
| `RMN_Pipeline/` | Initial calibration workflow tested on glucose |
| `V2_RMN_Pipeline/` | 1D JCAMP-DX processing, TSP referencing, compound-specific integration, normalization, and CSV export |

## Machine Learning

### Scripts

- `scripts/` : Builds the CSV tables used in the machine-learning workflow.
- `new_scripts/` : Contains model-related scripts and their associated execution commands.
- `aug3.sh` : Runs data augmentation experiments. The script was adapted for the noise levels and thresholds used in each experiment and is available alongside the corresponding results.

### Experiments and Results

Results use the following naming conventions:

| Suffix / Label | Interpretation |
| --- | --- |
| `Std` | Standard NMR |
| `PS` | Pure Shift NMR |
| `no_noise` | Noise level reduced by a factor of 2 |
| `high_noise` | Noise level increased by a factor of 2 |
| `Std_d2`, `PS_d2` | Threshold reduced by a factor of 2 |
| `Stdx2`, `PSx2` | Threshold increased by a factor of 2 |

Additional result sets combine noise and threshold variations following the same conventions.

## Calibration Pipeline

The calibration workflow extracts compound concentrations, visualizes relevant spectral peaks, and generates calibration curves for pure compounds.

### Concentration Extraction

`extract_compound` reads the Excel concentration table and extracts the required compound concentrations.

### Spectral Peak Visualization

`plot_new.py` plots the relevant peak(s) for each compound spectrum to support comparisons between Standard and Pure Shift NMR.

**Output:** `results/plots_new/`

### Calibration Curves

The calibration scripts generate curves for individual pure compounds.

**Output:** `results/calibration/`

Integration regions are defined separately for each compound and acquisition method (Standard or Pure Shift).

**Metadata:** `data/metadata/`

## RMN Pipeline

The pipeline processes **1D NMR spectra in JCAMP-DX format** and supports:

1. Spectral plotting and analysis.
2. Chemical-shift referencing using **TSP**.
3. Targeted, compound-specific signal integration.
4. Normalization.
5. CSV export of processed results.

## Data and Output Conventions

- **Input spectra:** JCAMP-DX (`.dx`) in the V2 pipeline.
- **Concentration data:** Excel tables in the calibration pipeline.
- **Integration metadata:** `Calibration_Pipeline/data/metadata/`.
- **Diagnostic plots:** `Calibration_Pipeline/results/plots_new/`.
- **Calibration outputs:** `Calibration_Pipeline/results/calibration/`.
- **Machine-learning results:** Directories identified by acquisition type, noise setting, and threshold setting.

