# Granger-Causality Analysis of Defective Viral Genomes (DVGs)

This repository contains the analysis pipeline for identifying DVGs that Granger-cause changes in viral cultivation metrics. The analysis applies Granger-causality testing, negative controls, OLS forecasting validation, and DVG ranking.

## Notebooks

The analysis is split across four notebooks, intended to be run in order:

1. **01_granger_causality_analysis_dvg_classification.ipynb** — Granger-causality testing between DVG and cultivation time series, including stationarity checks, model fitting, and candidate classification.
2. **02_granger_causality_analysis_negative_control.ipynb** — Negative control analysis using randomized time series to validate the Granger-causality results.
3. **03_validation_ols_forecasting.ipynb** — Rolling-window OLS forecasting to independently validate identified causal DVG candidates.
4. **04_rank_dvgs.ipynb** — Ranking of DVG candidates by MAE and SSR metrics.

## Setup

Create and activate the conda environment:

```bash
conda env create -f environment.yaml
conda activate gc_dvg
```

## Running the Notebooks

```bash
jupyter notebook
```

Then open and run the notebooks in order (01 through 04).
