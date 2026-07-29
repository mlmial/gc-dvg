# Granger-Causality Analysis of Defective Viral Genomes (DVGs)

This repository contains the analysis pipeline for identifying DVGs that are Granger-causality related to virus cultivation metrics like the infectious virus concentration. The analysis applies Granger-causality testing, creates negative controls, OLS forecasting validation, and DVG ranking.

## Notebooks

The analysis is split across four notebooks, intended to be run in order:

1. **01_granger_causality_analysis_dvg_classification.ipynb** — Granger-causality testing between DVG and cultivation time series, including stationarity checks, model fitting, and candidate classification.
2. **02_granger_causality_analysis_negative_control.ipynb** — Negative control analysis using randomized time series to validate the Granger-causality results.
3. **03_validation_ols_forecasting.ipynb** — Rolling-window OLS forecasting to independently validate identified causal DVG candidates.
4. **04_create_dvg_ranks.ipynb** — Ranking of DVG candidates by MAE and SSR metrics.

Each notebook writes every table and figure it exports below its own
`output_prefix`, and reads the outputs of the notebooks before it.

## Shared code

Routines used by more than one notebook live in `utils/` rather than being
copy-pasted between them:

| Module | Contents |
| --- | --- |
| `utils/granger.py` | stationarity transform, Granger-causality matrix, Benjamini-Hochberg correction, label classification, summary table |
| `utils/granger_prediction.py` | restricted/full OLS prediction per candidate and the per-candidate fit plots |
| `utils/metrics.py` | forecast-error metrics (MAE/MAPE/DTW/SSR variants) and the statistical tests and effect sizes |
| `utils/plotting.py` | colour conventions per Granger label, significance brackets, the two swarmplot builders |
| `utils/ols_forecasting.py` | rolling-window OLS forecasting used by notebook 03 |
| `pelz_datasets.py` | loading of the Pelz 2021 read counts and cultivation values |

## Setup

Create and activate the conda environment:

```bash
conda env create -f environment.yaml
conda activate gc_dvg
```
## Environment

All analyses were run in Jupyter notebooks under Python 3.9 (conda environment
`gc_dvg`, see `environment.yml`) on x86_64 Linux (openSUSE Tumbleweed,
kernel 6.10.3).

The environment file and notebooks also build and run on macOS (Intel),
though the results reported in the main manuscript were produced on Linux.

​```bash
conda env create -f environment.yml
conda activate gc_dvg
​```

Then open and run the notebooks in order with Jupyter Notebook (01 through 04).

## Granger-causality analysis

Labels are assigned to each DVG based on the following procedure:

1. **Test for stationarity.** Each time series is tested with the Augmented
   Dickey-Fuller test (`autolag='AIC'`, `regression='ct'`,
   `maxlag=int(√n / timepoints)`). If the p-value is ≥ 0.05, the series is
   differenced, and this is repeated until the p-value is < 0.05.

2. **Construct the Granger-causality matrix (GCM).** The transformed
   (stationary) time series are arranged as the rows and columns of the GCM.

3. **Perform Granger-causality tests for each cell of the GCM.** Each cell is
   filled with the p-values from the tests (`params_ftest`, `ssr_ftest`,
   `ssr_chi2test`, `lrtest`).

4. **Extract Granger-related DVGs**, where `CV` denotes the infectious virus
   concentration:

   | Label            | Condition                                        |
   | ---------------- | ------------------------------------------------ |
   | Granger-causing  | `GCM[X, CV] < 0.05`                               |
   | Granger-caused   | `GCM[CV, X] < 0.05`                               |
   | Bi-directional   | `GCM[X, CV] < 0.05` **and** `GCM[CV, X] < 0.05`  |
   | Non-related      | neither condition holds                          |

5. **BH-correction.** The p-values are corrected for multiple testing using the
   Benjamini-Hochberg procedure.

## Forecasting validation

The forecasting validation of each DVG follows this procedure:

1. **Split the data set into expanding-window folds.** The time series is
   partitioned into `n = 8` non-overlapping test folds of 3 time points each,
   constructed from the end backwards. For each fold, all preceding data is
   assigned as the training set.

2. **Fit OLS models for each DVG and fold.**

   | Model      | Formula                     |
   | ---------- | --------------------------- |
   | Restricted | `y_t ~ y_{t-1}`             |
   | Full       | `y_t ~ y_{t-1} + DVG_{t-1}` |

3. **Forecast the test window iteratively.** Both models predict the test window
   step by step, with each prediction serving as input for the next time step.

4. **Calculate MAE per fold.** The mean absolute error, `MAE = |y_true - ŷ|`, is
   computed for each predicted time point.

5. **Aggregate across folds.** The median MAE across all folds is computed per
   DVG as its summary performance metric.

6. **Statistical comparison.** Median MAE values are compared across
   Granger-causality labels and shuffled controls using pairwise Mann–Whitney U
   tests (Bonferroni-corrected) with Cliff's δ effect sizes.