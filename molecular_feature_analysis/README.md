This folder is a self-contained reproduction package for the three results shown in the main publication:

- `results/segment_counts_4panel.png`: segment-composition plot;
- `results/dataset_overview.csv`: dataset overview table;
- `results/feature_summary.csv`: balanced feature-summary table.

The script also writes a Word report and intermediate convergence files to `results/`.

## Run

From this folder, install the small dependency set once and run:

```sh
python3 -m pip install -r requirements.txt
./run.sh
```

Alternatively, run `python3 generate_results.py` directly.

The analysis uses the same random seed (`42`), 100 summary iterations, 1000 convergence iterations, segment order, PR8 sequences, and statistical tests as the original repository workflow.

## Expected key values

After a successful run, the overview should contain 82, 136, 164, and 1702 events for bi-directional, caused, causing, and non-related respectively. The feature-summary median p-values should be approximately `0.003035`, `0.130992`, `0.314925`, and `0.999972`.
