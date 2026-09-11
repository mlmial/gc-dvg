"""Core Granger-causality routines shared by notebooks 01 and 02.

Extracted verbatim from the notebooks; the analysis is unchanged.

* :func:`make_stationary` - iterative differencing until the ADF test passes.
* :func:`granger_causation_matrix_fixed_lag` - pairwise Granger tests at a fixed lag.
* :func:`correct_p_values_bh_separately` / :func:`calculate_critical_pval_bh` -
  Benjamini-Hochberg correction applied per cultivation-value direction.
* :func:`granger_labels` - assign the causing / caused / bi-directional /
  non-related labels straight from the p-value matrix.
* :func:`create_summary_df` - per-DVG summary table written to CSV.
"""

import types

import numpy as np
import pandas as pd
from IPython.display import display
from statsmodels.stats.multitest import multipletests
from statsmodels.tsa.stattools import adfuller, grangercausalitytests

__all__ = [
    "make_stationary",
    "granger_causation_matrix_fixed_lag",
    "correct_p_values_bh_separately",
    "calculate_critical_pval_bh",
    "granger_labels",
    "create_summary_df",
]


def make_stationary(input_df, max_diff=10, autolag=None, max_lag=None, regression="c"):
    """Transform each column (candidate) to a stationary series by iterative differencing.

    Args:
        input_df (pd.DataFrame): rows are timepoints; columns are candidates/variables.
        max_diff (int): maximum differencing level. Defaults to 10.
        autolag (str | None): lag selection for ADF. Defaults to 'AIC'.
        max_lag (int | None): max lags for ADF. Defaults to sqrt(n_timepoints), then clamped.
        regression (str): ADF regression trend ('c', 'ct', 'ctt', 'nc'). Defaults to 'c'.

    Returns:
        df_stationary (pd.DataFrame): final stationary series (may be differenced) for stationary columns.
        df_nonstationary (pd.DataFrame): original non-stationary columns.
        df_diff_nonstationary (pd.DataFrame): last differenced data for non-stationary columns (np.diff, padded).
        diff_levels (pd.Series): differencing level used per column (NaN for non-stationary/error).
        lags (pd.Series): ADF-selected lag per column (NaN for non-stationary/error).
        pvals (pd.Series): final ADF p-value per column (NaN on error).
        status (pd.Series): "stationary", "non-stationary", or "error".
    """
    df = input_df.copy()
    n_timepoints = len(df)

    if autolag is None:
        autolag = "AIC"
    if max_lag is None:
        max_lag = int(np.sqrt(max(n_timepoints, 1)))
    # Clamp to feasible range
    max_lag = max(1, min(max_lag, max(1, n_timepoints // 2 - 1)))

    stationary_data = {}
    nonstationary_cols = []
    diff_nonstationary_dict = {}
    diff_levels = {}
    lags = {}
    pvals = {}
    status = {}

    for col in df.columns:
        series = pd.to_numeric(df[col], errors="coerce")
        # Drop NaNs for testing but keep original index for reconstruction
        ts_data = series.dropna().to_numpy()
        if len(ts_data) < 3 or len(ts_data) <= max_lag:
            # Too short for ADF with this max_lag
            nonstationary_cols.append(col)
            diff_nonstationary_dict[col] = np.diff(series.to_numpy()) if len(series) > 1 else np.array([])
            diff_levels[col] = np.nan
            lags[col] = np.nan
            pvals[col] = np.nan
            status[col] = "error"
            continue

        diff_level = 0
        last_data = ts_data.copy()

        def run_adf(arr):
            with np.errstate(divide='ignore', invalid='ignore'):
                return adfuller(arr, autolag=autolag, regression=regression, maxlag=max_lag)

        # Initial ADF
        try:
            adf_result = run_adf(last_data)
            p_value = adf_result[1]
        except Exception:
            nonstationary_cols.append(col)
            diff_nonstationary_dict[col] = np.diff(series.to_numpy()) if len(series) > 1 else np.array([])
            diff_levels[col] = np.nan
            lags[col] = np.nan
            pvals[col] = np.nan
            status[col] = "error"
            continue

        if p_value <= 0.05:
            # Already stationary; keep original cleaned series, realign to original index
            stationary_series = series
            stationary_data[col] = stationary_series
            diff_levels[col] = diff_level
            lags[col] = adf_result[2]
            pvals[col] = p_value
            status[col] = "stationary"
            continue

        # Iterative differencing
        became_stationary = False
        while diff_level < max_diff:
            diff_data = np.diff(last_data)
            diff_level += 1
            # Need at least 3 points to run ADF reasonably
            if len(diff_data) < 3 or len(diff_data) <= max_lag:
                # Can't test further; mark non-stationary
                break
            try:
                adf_result = run_adf(diff_data)
                p_value = adf_result[1]
            except Exception:
                break

            if p_value <= 0.05:
                # Store differenced series aligned to original index (pad front with NaN)
                padded = np.concatenate(([np.nan] * diff_level, diff_data))
                # Trim/pad to original length
                padded = padded[:n_timepoints] if len(padded) >= n_timepoints else np.pad(
                    padded, (0, n_timepoints - len(padded)), constant_values=np.nan
                )
                stationary_series = pd.Series(padded, index=df.index)
                stationary_data[col] = stationary_series
                diff_levels[col] = diff_level
                lags[col] = adf_result[2]
                pvals[col] = p_value
                status[col] = "stationary"
                became_stationary = True
                break

            last_data = diff_data

        if not became_stationary:
            nonstationary_cols.append(col)
            # Keep the last differenced data for inspection
            diff_nonstationary_dict[col] = np.diff(series.to_numpy()) if len(series) > 1 else np.array([])
            diff_levels[col] = np.nan
            lags[col] = np.nan
            pvals[col] = np.nan
            status[col] = "non-stationary"

    # Assemble outputs
    df_stationary = pd.DataFrame(stationary_data, index=df.index) if stationary_data else pd.DataFrame(index=df.index)
    df_nonstationary = df[nonstationary_cols].copy() if nonstationary_cols else pd.DataFrame(index=df.index)

    if diff_nonstationary_dict:
        padded = {}
        for col, arr in diff_nonstationary_dict.items():
            # np.diff gives length n-1; pad with leading NaN to align to n
            padded_arr = np.concatenate(([np.nan], arr))
            if len(padded_arr) < n_timepoints:
                padded_arr = np.pad(padded_arr, (0, n_timepoints - len(padded_arr)), constant_values=np.nan)
            else:
                padded_arr = padded_arr[:n_timepoints]
            padded[col] = padded_arr
        df_diff_nonstationary = pd.DataFrame(padded, index=df.index)
    else:
        df_diff_nonstationary = pd.DataFrame(index=df.index)

    return types.SimpleNamespace(
        df_stationary=df_stationary,
        df_nonstationary=df_nonstationary,
        df_diff_nonstationary=df_diff_nonstationary,
        diff_levels=pd.Series(diff_levels),
        lags=pd.Series(lags),
        pvals=pd.Series(pvals),
        status=pd.Series(status),
    )


def granger_causation_matrix_fixed_lag(data, variables, cultivation_values=[],test='ssr_chi2test', plot=False,fixlag=3, maxlag=3, debug=False):
    test_results = {}
    err_dips = {}
    max_lags = {}
    # creating a dataframe with the same dimensions as number of variables entered, assigned to the variable 'X_train'
    X_train = pd.DataFrame(np.nan, index=variables, columns=variables)
    
    # loops through the columns and the indexes
    for c in X_train.columns:
        for r in X_train.index:
            #skip if we're not looking at any cultivation variable
            if r == c:
                continue
            go_on = False
            if c in cultivation_values:
                go_on = True
            if r in cultivation_values:
                go_on = True
            
            if go_on == False:
                continue
            # conducts a granger causality test on a variable row and column using the 'maxlag' variable; assigns to 
            # variable test_result
            try:
                test_result = grangercausalitytests(data[[r, c]], maxlag=[maxlag], verbose=False)
            except Exception as e:
                if debug:
                  print(f'Error on {r} and {c}, {data[[r, c]]}: {e}')
                err_dips[(r,c)] = e
                X_train.loc[r, c] = None
                max_lags[(r,c)] = None
                continue
            test_results[(r,c)] = test_result    
            # locates the test result in the tuple 'test_result' and rounds the number by 4 digits; assigns to 'p_values'
            p_value = round(test_result[fixlag][0][test][1], 4)
            max_lags[(r,c)] = fixlag
            X_train.loc[r, c] = p_value
            
    # rename the row and column names based on the relationship
    X_train.columns = [var + '_x' for var in variables]
    X_train.index = [var + '_y' for var in variables]
    return X_train, test_results, max_lags, err_dips


def correct_p_values_bh_separately(matrix, cult_columns=5, alpha=0.05, debug=False):
    """
    Applies Benjamini-Hochberg correction cultivation-column-wise:
    1. DVGs -> each Virus Concentration column (row i, DVG columns to the right)
    2. Each Virus Concentration column -> DVGs (DVG rows, column i)

    Parameters:
        matrix (pd.DataFrame): A pandas DataFrame where each cell contains a p-value.
        cult_columns (int): Number of columns corresponding to virus concentration variables.
        alpha (float): significance level for BH.
        debug (bool): if True, print and display slices being corrected.

    Returns:
        pd.DataFrame: Same shape with BH-adjusted p-values.
    """
    corrected_matrix = matrix.copy()

    for col_i in range(cult_columns):
        col_name = matrix.columns[col_i].replace("_x", "")

        # DVG -> col_i (row col_i, DVG columns to the right)
        dvg_to_col = matrix.iloc[col_i, cult_columns:].values
        if debug:
            print(f"\n[DVG -> {col_name}] slice BEFORE correction:")
            display(matrix.iloc[col_i:col_i+1, cult_columns:])
        mask = ~np.isnan(dvg_to_col)
        if mask.any():
            _, corrected_vals, _, _ = multipletests(dvg_to_col[mask], alpha=alpha, method="fdr_bh")
            tmp = dvg_to_col.copy()
            tmp[mask] = corrected_vals
            corrected_matrix.iloc[col_i, cult_columns:] = tmp
            if debug:
                print(f"[DVG -> {col_name}] slice AFTER correction:")
                display(corrected_matrix.iloc[col_i:col_i+1, cult_columns:])

        # col_i -> DVGs (all DVG rows below, column col_i)
        col_to_dvg = matrix.iloc[cult_columns:, col_i].values
        if debug:
            print(f"\n[{col_name} -> DVGs] slice BEFORE correction:")
            display(matrix.iloc[cult_columns:, col_i:col_i+1])
        mask = ~np.isnan(col_to_dvg)
        if mask.any():
            _, corrected_vals, _, _ = multipletests(col_to_dvg[mask], alpha=alpha, method="fdr_bh")
            tmp = col_to_dvg.copy()
            tmp[mask] = corrected_vals
            corrected_matrix.iloc[cult_columns:, col_i] = tmp
            if debug:
                print(f"[{col_name} -> DVGs] slice AFTER correction:")
                display(corrected_matrix.iloc[cult_columns:, col_i:col_i+1])
    return corrected_matrix


def _bh_critical_pval(pvals, alpha=0.05, debug=False):
    """Return the BH critical p-value for a 1D array (ignoring NaNs)."""
    p = np.sort(np.asarray(pvals)[~np.isnan(pvals)])
    m = p.size
    if m == 0:
        if debug:
            print("No valid p-values found.")
        return None
    thresholds = alpha * (np.arange(1, m+1) / m)
    hits = np.where(p <= thresholds)[0]
    if debug:
      # create df
      tmp_df = pd.DataFrame({'p-values': p, 'thresholds': thresholds})
      tmp_df['hits'] = np.where(p <= thresholds, 'hit', 'miss')
      display(tmp_df)
    if hits.size == 0:
      if debug:
          print("No hits found.")
      return None
    kmax = hits.max()
    if debug:
      print(f"Critical p-value for BH correction at rank {kmax}: {p[kmax]}")
    return p[kmax]   # the largest p-value that still passes


def calculate_critical_pval_bh(matrix, cult_columns=5, alpha=0.05, debug=False):
    """
    For each cultivation column i (0..cult_columns-1), return:
      - 'dvg_to_<colname>': BH critical p for DVGs -> <colname>
        (row i, DVG columns to the right)
      - '<colname>_to_dvg': BH critical p for <colname> -> DVGs
        (DVG rows below, column i)
    """
    crit_dct = {}
    for i, col_name in enumerate(matrix.columns[:cult_columns]):
        clean = col_name.replace('_x', '')

        # DVG -> col_i  (row i, DVG columns)
        dvg_to_col_slice = matrix.iloc[i, cult_columns:]
        if debug:
            print(f"\n[DVG -> {clean}] slice:")
            display(dvg_to_col_slice.to_frame().T)
        dvg_to_col = dvg_to_col_slice.values
        crit_dct[f"dvg_to_{clean}"] = _bh_critical_pval(dvg_to_col, alpha, debug)

        # col_i -> DVGs  (DVG rows, column i)
        col_to_dvg_slice = matrix.iloc[cult_columns:, i]
        if debug:
            print(f"\n[{clean} -> DVGs] slice:")
            display(col_to_dvg_slice.to_frame())
        col_to_dvg = col_to_dvg_slice.values
        crit_dct[f"{clean}_to_dvg"] = _bh_critical_pval(col_to_dvg, alpha, debug)

    return crit_dct


def granger_labels(gc_matrix, cultivation_value, alpha=0.05, bh_critical_pvals=None):
    """Granger label per DVG for one cultivation value, read straight off the matrix.

    Cell ``[Y_y, X_x]`` of the matrix holds the p-value for "X Granger-causes Y",
    so for a DVG and a cultivation value CV:

    * ``p_causing`` = ``[CV_y, DVG_x]`` -- the DVG improves the CV forecast
    * ``p_caused``  = ``[DVG_y, CV_x]`` -- the CV improves the DVG forecast

    Both significant gives ``bi-directional``, one gives ``causing`` / ``caused``,
    neither gives ``non-related``. A NaN p-value (the test errored) leaves that
    direction undecided; a DVG undecided in both directions gets ``None`` and is
    dropped from the summary table.

    Args:
        gc_matrix (pd.DataFrame): p-values, rows ``<var>_y``, columns ``<var>_x``.
        cultivation_value (str): the cultivation variable to label against.
        alpha (float): significance threshold when no BH cut-offs are given.
        bh_critical_pvals (dict | None): output of
            :func:`calculate_critical_pval_bh`; supplies a separate critical
            p-value per direction. ``None`` for that direction means BH found no
            significant test, so nothing in it counts as significant.

    Returns:
        pd.Series: DVG key -> label (object dtype, ``None`` where undecided).
    """
    if bh_critical_pvals is None:
        thr_causing = thr_caused = alpha
    else:
        thr_causing = bh_critical_pvals[f'dvg_to_{cultivation_value}']
        thr_caused = bh_critical_pvals[f'{cultivation_value}_to_dvg']
    # BH found nothing in this direction -> no p-value can be significant
    thr_causing = -np.inf if thr_causing is None else thr_causing
    thr_caused = -np.inf if thr_caused is None else thr_caused

    # set_axis returns a new Series, so gc_matrix is never touched
    p_causing = gc_matrix.loc[f'{cultivation_value}_y'].set_axis(
        [c.removesuffix('_x') for c in gc_matrix.columns])
    p_caused = gc_matrix[f'{cultivation_value}_x'].set_axis(
        [r.removesuffix('_y') for r in gc_matrix.index])

    sig_causing, sig_caused = p_causing <= thr_causing, p_caused <= thr_caused
    # NaN is neither significant nor plainly non-significant, hence both masks
    plain_causing, plain_caused = p_causing > thr_causing, p_caused > thr_caused

    # a plain None scalar would be coerced to NaN, so seed with a list of None
    labels = pd.Series([None] * len(p_causing.index), index=p_causing.index, dtype=object)
    labels[plain_causing & plain_caused] = 'non-related'
    labels[sig_caused] = 'caused'
    labels[sig_causing] = 'causing'
    labels[sig_causing & sig_caused] = 'bi-directional'
    return labels.drop(index=cultivation_value, errors='ignore')


def create_summary_df(gc_matrix,
                      gc_test_results,
                      gc_max_lag,
                      diff_level_dct,
                      readcounts_data,
                      time_series_data,
                      output_file,
                      bh_critical_pvals=None,
                      alpha=0.05,
                      stat_tests=('ssr_ftest', 'ssr_chi2test', 'lrtest'),
                      cultivation_values=('vrna', 'tcid50', 'plaque_assay', 'log_ha', 'ha_titer')):
    """Per-DVG summary table of Granger labels, test p-values and lags.

    Pass ``bh_critical_pvals`` to label against the Benjamini-Hochberg critical
    p-values instead of a flat ``alpha``; pass a BH-corrected ``gc_matrix`` to
    label against corrected p-values at ``alpha``.

    Returns the table and writes it to ``output_file``.
    """
    stat_tests = list(stat_tests)
    cultivation_values = list(cultivation_values)

    labels = {cv: granger_labels(gc_matrix, cv, alpha, bh_critical_pvals)
              for cv in cultivation_values}

    # keep the DVGs that got a label for at least one cultivation value
    labelled = set().union(*(s.dropna().index for s in labels.values()))
    summary_df = readcounts_data[readcounts_data['key'].isin(labelled)][['key']].copy()

    # look the labels up through plain dicts: Series.map turns a missing label
    # into NaN, and the "no label -> no statistics" check below needs None
    for cv in cultivation_values:
        lookup = labels[cv].to_dict()
        summary_df[f'{cv}_granger_label'] = [lookup.get(k) for k in summary_df['key']]

    summary_df['max_diff'] = summary_df['key'].map(diff_level_dct).fillna(0)

    blank = dict.fromkeys(['max_lag', 'cross_corr'] + stat_tests)

    def granger_stats(key, cultivation_value):
        """Test p-values, selected lag and cross-correlation for one DVG."""
        tr_key = (cultivation_value, key)          # always read the "causing" direction
        if tr_key not in gc_test_results or tr_key not in gc_max_lag:
            return blank
        max_lag = gc_max_lag[tr_key]
        tests = gc_test_results[tr_key][max_lag][0]
        return {
            'max_lag': max_lag,
            'cross_corr': time_series_data[key].corr(
                time_series_data[cultivation_value].shift(-max_lag)),
            **{test: tests[test][1] for test in stat_tests},
        }

    for cv in cultivation_values:
        stats = pd.DataFrame(
            [granger_stats(key, cv) if pd.notna(label) else blank
             for key, label in zip(summary_df['key'], summary_df[f'{cv}_granger_label'])],
            index=summary_df.index,
            columns=['max_lag', 'cross_corr'] + stat_tests,
        )
        summary_df[f'{cv}_max_lag'] = stats['max_lag']
        summary_df[f'{cv}_cross_corr'] = stats['cross_corr']
        for test in stat_tests:
            summary_df[f'{cv}_{test}'] = stats[test]
        summary_df[f'{cv}_significant_pvals'] = (
            summary_df[[f'{cv}_{test}' for test in stat_tests]].lt(0.05).sum(axis=1))

    summary_df['granger_score'] = (
        summary_df[[f'{cv}_significant_pvals' for cv in cultivation_values]].sum(axis=1)
        - summary_df['max_diff'])

    summary_df = summary_df.sort_values(by=['granger_score', 'max_diff'],
                                        ascending=[False, True])
    summary_df.to_csv(output_file, index=False)
    return summary_df
