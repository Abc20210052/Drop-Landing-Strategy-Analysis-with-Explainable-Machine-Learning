# Drop Landing Strategy Analysis with Explainable Machine Learning

This repository contains the analysis code for a biomechanical study of drop landing (DL) strategies under different landing heights and fatigue states. The project uses time-series lower-limb biomechanical features and explainable machine learning to identify and interpret landing strategy differences.

The project analyzes sagittal-plane hip, knee, and ankle joint angles and moments, together with vertical ground reaction force (vGRF), collected during drop landing tasks at 40 cm and 80 cm before and after fatigue.

## Study Overview

Drop landing is a complex neuromuscular control process affected by both landing height and fatigue. This project investigates how these two factors influence lower-limb landing strategies and evaluates whether explainable machine learning can be used to identify and interpret those strategy changes.

The analysis includes:

- Classification models for distinguishing landing conditions（Cross-validation and subject-based train-test splitting）.
- Time-series Cohen's d effect size analysis for biomechanical group differences.
- SHAP-based feature importance and feature interaction analysis.

## Data Format

The scripts expect four CSV files:

| File | Condition |
| --- | --- |
| `DL_40.csv` | 40 cm drop landing, non-fatigued |
| `DL_80.csv` | 80 cm drop landing, non-fatigued |
| `DL_P40.csv` | 40 cm drop landing, post-fatigue |
| `DL_P80.csv` | 80 cm drop landing, post-fatigue |

Each CSV is expected to contain one trial per row. The current scripts assume:

- 60 rows per condition: 20 participants x 3 repetitions.
- 1407 columns per sample.
- 7 biomechanical variables x 201 time points.
- Every three consecutive rows belong to the same participant.

Feature column layout:

| Variable | Column range, zero-indexed |
| --- | --- |
| Hip Angle | `0:201` |
| Hip Moment | `201:402` |
| Knee Angle | `402:603` |
| Knee Moment | `603:804` |
| Ankle Angle | `804:1005` |
| Ankle Moment | `1005:1206` |
| Vertical GRF | `1206:1407` |

Note: the scripts currently use hard-coded Windows paths such as `F:\DL_40.csv`. Before running the code on another machine, update the `file_paths` variable in each script or modify the code to use relative paths such as `data/DL_40.csv`.

## Classification Tasks

The main analyses use four binary classification or comparison tasks:

| Task | Interpretation |
| --- | --- |
| `DL_40_vs_DL__80` | Height effect without fatigue |
| `DL_P40_vs_DL_P80` | Height effect after fatigue |
|  `DL_40_vs_DL_P40` | Fatigue effect at 40 cm |
| `DL_80_vs_DL_P80` | Fatigue effect at 80 cm |

## Repository Structure

| Script | Purpose |
| --- | --- |
| `gridsearch_RF.py` | Hyperparameter search for Random Forest. |
| `gridsearch_SVM.py` | Hyperparameter search for SVM. |
| `gridsearch_MLP.py` | Hyperparameter search for MLP |
| `gridsearch_GBM.py` | Hyperparameter search for LightGBM|
| `DL_RF.py` | Trains and evaluates Random Forest classifiers for the four binary tasks. |
| `DL_SVM.py` | Trains and evaluates RBF-kernel SVM classifiers. |
| `DL_MLP.py` | Trains and evaluates MLP classifiers. |
| `DL_LGBM.py` | Trains and evaluates LightGBM classifiers and saves model, parameter, and feature-importance outputs. |
| `DL_LSTM.py` | Runs LSTM-based/1DCNN-LSTM cross-validation classification. |
| `DL_Cohen's d.py` | Performs time-series effect size analysis and generates heatmaps, normalized difference plots, and CSV summaries. |
| `DL_SHAP.py` | Computes cross-fold SHAP feature importance using LightGBM. |
| `DL_SHAP_interaction.py` | Computes and visualizes SHAP interaction patterns among biomechanical variables. |


## Environment

Recommended Python version: Python 3.9 or newer.

Install the main dependencies:

```bash
pip install numpy pandas scipy scikit-learn matplotlib seaborn lightgbm shap tensorflow openpyxl
```

TensorFlow is only required for `DL_LSTM.py`. If you do not plan to run the LSTM analysis, TensorFlow can be omitted.

## Usage

1. Place the four CSV files in an accessible directory.

2. Update `file_paths` in each script. For example:

```python
file_paths = [
    r"data/DL_40.csv",
    r"data/DL_80.csv",
    r"data/DL_P40.csv",
    r"data/DL_P80.csv",
]
```


3. Optionally run hyperparameter searches:

```bash
python gridsearch_RF.py
python gridsearch_SVM.py
python gridsearch_MLP.py
python gridsearch_GBM.py
```

4. Run the classification models:

```bash
python DL_RF.py
python DL_SVM.py
python DL_MLP.py
python DL_LGBM.py
python DL_LSTM.py
```

5. Run the effect size analysis:

```bash
python "DL_Cohen's d.py"
```


6. Run SHAP explainability analyses:

```bash
python DL_SHAP.py
python DL_SHAP_interaction.py
```

## Outputs

The scripts save figures and statistical summaries as CSV, PNG, SVG, and PDF files. Main output folders include:

- `Biomechanics_Feature_Analysis/`
- `SVM__Analysis_Results_4to1/`
- `RandomForest_Analysis_Results_4to1/`
- `LightGBM_Analysis_Results_4to1/`
- `LSTM/1DCNN-LSTM_Analysis_Results_4to1/`
- `SHAP_CrossFold_Results/`
- `SHAP_CrossFold_Interaction_Results/`


## Data Availability

The raw biomechanical data are not included in this repository by default.  According to institutional ethics and privacy requirements, data cannot be shared temporarily. If necessary, the data used in this study can be obtained through communication with the corresponding author.

