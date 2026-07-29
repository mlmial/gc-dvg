"""Forecast-error metrics and statistical tests shared by notebooks 01-03.

Every function here was previously copy-pasted into two or three notebooks.

Note on ``cliffs_delta``: notebooks 01/02 used an O(n*m) double loop and
notebooks 02/03 an O(n log m) searchsorted implementation. They compute the
same statistic, so only the vectorised one is kept.
"""

from itertools import combinations

import numpy as np
import pandas as pd
import scikit_posthocs as sp
from scipy.signal import argrelextrema
from scipy.stats import kruskal, mannwhitneyu, pearsonr, shapiro, ttest_ind
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from statsmodels.stats.multitest import multipletests
from tslearn.metrics import dtw_path

__all__ = [
    # forecast-error metrics
    "nrsme", "mase", "find_local_extrema",
    "calculate_weighted_mae", "calculate_weighted_mape",
    "dtw_pearson", "dtw_mape",
    # effect sizes and tests
    "cohens_d", "cliffs_delta", "epsilon_squared_kruskal", "kendalls_w",
    "test_statistical_difference", "pairwise_mwu_cliffs", "pairwise_stats_tests",
    # formatting helpers
    "sym_matrix", "get_pval_symbol", "better_percentage",
]

#: Human-readable names for the effect sizes reported next to significance bars.
EFFECT_SIZE_NAMES = {"mann-whitney": "Cliff's delta", "t-test": "Cohen's d"}

#: Short symbols used in significance-bar annotations.
EFFECT_SIZE_SYMBOLS = {
    "cliffs_delta": "\u03b4",
    "rank_biserial": "Rank-biserial Correlation",
    "epsilon_squared": "Rank Epsilon Squared",
    "kendalls_w": "Kendall\u2019s W",
}


# ----------------------------------------------------------------------------
# forecast-error metrics
# ----------------------------------------------------------------------------


def nrsme(y_true, y_pred):
    return root_mean_squared_error(y_true, y_pred, squared=False) / (np.max(y_true) - np.min(y_true))


def mase(y_true, y_pred):
    # Calculate naive forecast (shifted by one time step)
    naive_forecast = np.roll(y_true, shift=1)
    # Remove the first element to align with y_true[1:]
    naive_forecast = naive_forecast[1:]
    
    # Calculate MAE of the model
    mae_model = mean_absolute_error(y_true, y_pred)
    # Calculate MAE of the naive forecast
    mae_naive = mean_absolute_error(y_true[1:], naive_forecast)
    
    return mae_model / mae_naive


def find_local_extrema(y, order=1):
    """
    Find indices of local minima and maxima in the array.
    
    Parameters:
    - y: Array of values.
    - order: How many points on each side to use for the comparison to consider a point a local minimum or maximum.
    
    Returns:
    - minima_indices: Indices of local minima.
    - maxima_indices: Indices of local maxima.
    """
    minima_indices = argrelextrema(y, np.less, order=order)[0]
    maxima_indices = argrelextrema(y, np.greater, order=order)[0]
    
    # Check the first and last points manually
    if len(y) > 1:
        if y[0] < y[1]:
            minima_indices = np.insert(minima_indices, 0, 0)
        if y[0] > y[1]:
            maxima_indices = np.insert(maxima_indices, 0, 0)
        if y[-1] < y[-2]:
            minima_indices = np.append(minima_indices, len(y) - 1)
        if y[-1] > y[-2]:
            maxima_indices = np.append(maxima_indices, len(y) - 1)
    
    return minima_indices, maxima_indices


def calculate_weighted_mae(y_true, y_pred, order=1, weight_factor=3):
    """
    Calculate the Weighted Mean Absolute Error (WMAE) with higher weights on local minima and maxima.
    
    Parameters:
    - y_true: Array of true values.
    - y_pred: Array of predicted values.
    - order: Order for finding local minima and maxima.
    - weight_factor: Weight assigned to local minima and maxima.
    
    Returns:
    - wmae: Weighted Mean Absolute Error.
    """
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    # Find local minima and maxima
    minima_indices, maxima_indices = find_local_extrema(y_true_arr, order)
    
    # Initialize weights to 1
    weights = np.ones_like(y_true_arr)
    
    # Assign higher weights to local minima and maxima
    weights[minima_indices] = weight_factor
    weights[maxima_indices] = weight_factor
    
    # Calculate weighted MAE
    wmae = np.sum(weights * np.abs(y_true_arr - y_pred_arr) / np.sum(weights))
    
    return wmae


def calculate_weighted_mape(y_true, y_pred, order=1, weight_factor=3):
    """
    Calculate the Weighted Mean Absolute Percentage Error (WMAPE) with higher weights on local minima and maxima.
    
    Parameters:
    - y_true: Array of true values.
    - y_pred: Array of predicted values.
    - order: Order for finding local minima and maxima.
    - weight_factor: Weight assigned to local minima and maxima.
    
    Returns:
    - wmape: Weighted Mean Absolute Percentage Error.
    """
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    # Find local minima and maxima
    minima_indices, maxima_indices = find_local_extrema(y_true_arr, order)
    
    # Initialize weights to 1
    weights = np.ones_like(y_true_arr)
    
    # Assign higher weights to local minima and maxima
    weights[minima_indices] = weight_factor
    weights[maxima_indices] = weight_factor
    
    # Calculate WMAPE
    # To avoid division by zero, we add a small epsilon value to y_true
    epsilon = 1e-10
    wmape = np.sum(weights * np.abs((y_true_arr - y_pred_arr) / (y_true_arr + epsilon))) / np.sum(weights)
    
    return wmape


def dtw_pearson(y_true, y_pred):
    """
    Calculate the Dynamic Time Warping (DTW) aligned Pearson correlation coefficient between two time series.
    
    Parameters:
    - y_true: Array of true values.
    - y_pred: Array of predicted values.
    
    Returns:
    - pearson_corr: The Pearson correlation coefficient between the aligned (warped) y_true and y_pred.
    """
    y_true = np.array(y_true).reshape(-1, 1)
    y_pred = np.array(y_pred).reshape(-1, 1)
    path, dtw_distance = dtw_path(y_true, y_pred)
    aligned_true = np.array([y_true[i] for i, j in path]).flatten()
    aligned_pred = np.array([y_pred[j] for i, j in path]).flatten()
    pearson_corr = pearsonr(aligned_true, aligned_pred)[0]
    return pearson_corr


def dtw_mape(y_true, y_pred, epsilon=1e-6):
    """
    Calculate the DTW-aligned Mean Absolute Percentage Error (MAPE) between two time series.
    
    Parameters:
    - y_true: Array of true values.
    - y_pred: Array of predicted values.
    - epsilon: Small constant to avoid division by zero in MAPE calculation.
    
    Returns:
    - dtw_distance: The DTW distance between y_true and y_pred.
    - dtw_aligned_mape: The MAPE calculated on the DTW-aligned series.
    """
    y_true = np.array(y_true).reshape(-1, 1)
    y_pred = np.array(y_pred).reshape(-1, 1)
    
    # Calculate DTW path and distance
    path, dtw_distance = dtw_path(y_true, y_pred)
    
    # Extract aligned series
    aligned_true = np.array([y_true[i] for i, j in path]).flatten()
    aligned_pred = np.array([y_pred[j] for i, j in path]).flatten()
    
    # Calculate MAPE on aligned series
    dtw_aligned_mape = np.mean(np.abs((aligned_true - aligned_pred) / (aligned_true + epsilon))) * 100
    
    return dtw_aligned_mape


# ----------------------------------------------------------------------------
# effect sizes and statistical tests
# ----------------------------------------------------------------------------


def sym_matrix(levels, vals, all_levels, fill=np.nan, diag=1.0):
    """Symmetric matrix from pairwise values aligned to all_levels order."""
    n = len(all_levels)
    M = np.full((n, n), fill, dtype=float if isinstance(fill, (float, int, np.floating)) else object)
    if diag is not None:
        np.fill_diagonal(M, diag)
    k = 0
    for i in range(len(levels)):
        for j in range(i + 1, len(levels)):
            li, lj = levels[i], levels[j]
            v = vals[k]; k += 1
            if li in all_levels and lj in all_levels:
                ii, jj = all_levels.index(li), all_levels.index(lj)
                M[ii, jj] = M[jj, ii] = v
    return pd.DataFrame(M, index=all_levels, columns=all_levels)


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """Cliff's delta δ = P(x>y)-P(x<y), O((nx+ny)log ny)."""
    x = np.asarray(x); y = np.asarray(y)
    nx, ny = len(x), len(y)
    if nx == 0 or ny == 0: return np.nan
    ys = np.sort(y)
    less = np.searchsorted(ys, x, side="left")           # y < x
    greater = ny - np.searchsorted(ys, x, side="right")  # y > x
    return float((less.sum() - greater.sum()) / (nx * ny))


def epsilon_squared_kruskal(H: float, k: int, n: int) -> float:
    """Rank epsilon squared for Kruskal–Wallis."""
    if n <= k or k < 2: return np.nan
    return (H - k + 1) / (n - k)


def kendalls_w(df_long: pd.DataFrame, subject_col: str, group_col: str, value_col: str, levels: list) -> float:
    """
    Kendall's W for repeated-measures (Friedman setting).
    Uses subjects (rows) × conditions (columns) complete cases only.
    """
    if subject_col is None:
        return np.nan
    sub = df_long[[subject_col, group_col, value_col]].copy()
    sub = sub[sub[group_col].isin(levels)]
    wide = sub.pivot_table(index=subject_col, columns=group_col, values=value_col, aggfunc="mean")
    wide = wide.reindex(columns=levels)
    wide = wide.dropna(axis=0, how="any")  # complete blocks only
    n, m = wide.shape  # n subjects, m conditions
    if n < 2 or m < 2:
        return np.nan
    ranks = wide.rank(axis=1, method="average")  # rank within each subject
    Rj = ranks.sum(axis=0)                       # rank sums per condition
    Rbar = n * (m + 1) / 2.0
    S = ((Rj - Rbar) ** 2).sum()
    W = 12 * S / (m**2 * (n**3 - n))
    return float(W)


def cohens_d(group1, group2):
    # Calculate means and standard deviations
    mean1, mean2 = np.mean(group1), np.mean(group2)
    std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
    
    # Pooled standard deviation
    pooled_std = np.sqrt(((len(group1) - 1) * std1 ** 2 + (len(group2) - 1) * std2 ** 2) / (len(group1) + len(group2) - 2))
    
    # Cohen's d
    return (mean1 - mean2) / pooled_std


def test_statistical_difference(group1, group2, alpha=0.05):
    """
    Tests if two groups are statistically different from each other and returns effect size.

    Parameters:
    - group1, group2: The two groups to be compared.
    - alpha: Significance level for the tests.

    Returns:
    - A tuple containing:
    - p-value from the test
    - Effect size (Cohen's d for t-test, Cliff's delta for Mann-Whitney U test)
    - Test type ('t-test' or 'mann-whitney')
    """
    
    # Test for normality
    shapiro_test_group1 = shapiro(group1)
    shapiro_test_group2 = shapiro(group2)
    
    if shapiro_test_group1.pvalue > alpha and shapiro_test_group2.pvalue > alpha:
        # If both groups are normal, use T-test
        t_stat, t_pvalue = ttest_ind(list(group1), list(group2))
        
        # Calculate Cohen's d
        d = cohens_d(group1, group2)
        
        return t_pvalue, d, 't-test'
    else:
        # If either group is not normal, use Mann-Whitney U test
        mw_stat, mw_pvalue = mannwhitneyu(group1, group2)
        
        # Calculate Cliff's delta
        delta = cliffs_delta(group1, group2)
        
        return mw_pvalue, delta, 'mann-whitney'


def pairwise_mwu_cliffs(df, value_col, group_col, label_order, mw_adjust='bonferroni'):
    """Pairwise Mann-Whitney U (two-sided) + Cliff's delta over the given groups.
    Returns (mannwhitney_p_adjusted, cliffs_delta) as label x label DataFrames."""
    g = df[[group_col, value_col]].dropna(subset=[value_col])
    levels = [lab for lab in label_order if lab in g[group_col].unique()]
    pairs = list(combinations(levels, 2))
    mw_p, deltas = [], []
    for a, b in pairs:
        x = g.loc[g[group_col] == a, value_col].to_numpy()
        y = g.loc[g[group_col] == b, value_col].to_numpy()
        if x.size == 0 or y.size == 0:
            mw_p.append(np.nan); deltas.append(np.nan)
        else:
            mw_p.append(mannwhitneyu(x, y, alternative="two-sided", method="auto").pvalue)
            deltas.append(cliffs_delta(x, y))
    if mw_adjust:
        valid = [p for p in mw_p if not np.isnan(p)]
        if valid:
            _, p_adj, _, _ = multipletests(valid, method=mw_adjust)
            it = iter(p_adj)
            mw_p = [next(it) if not np.isnan(p) else np.nan for p in mw_p]
    mw_mat    = sym_matrix(levels, mw_p,   label_order, fill=np.nan, diag=1.0)
    delta_mat = sym_matrix(levels, deltas, label_order, fill=np.nan, diag=0.0)
    return mw_mat, delta_mat


def pairwise_stats_tests(
    df: pd.DataFrame,
    value_col: str,
    group_col: str = "label",
    dpi_col: str = "dpi",
    mw_adjust: str = "holm",
    dunn_adjust: str = "holm",
    label_order: list = ("restricted", "causing", "bi-directional", "caused", "non-related"),
    subject_col: str = None
) -> dict:
    """
    For each dpi, returns:
      - 'mannwhitney_p_uncorrected' : pairwise MW p-value matrix
      - 'mannwhitney_p'             : pairwise MW p-value matrix (adjusted by mw_adjust)
      - 'dunn_p'                    : Dunn's post-hoc p-value matrix (adjusted by dunn_adjust)
      - 'cliffs_delta'              : pairwise Cliff's δ matrix
      - 'rank_biserial'             : pairwise rank-biserial matrix (identical to δ)
      - 'epsilon_squared'           : scalar ε² for Kruskal–Wallis (overall effect)
      - 'kendalls_w'                : scalar Kendall's W (requires subject_col; else NaN)
    All matrices are aligned to label_order; missing groups are NaN off-diagonal, 1.0 on diagonal.
    """
    results = {}
    data = df[[dpi_col, group_col, value_col] + ([subject_col] if subject_col else [])].dropna(subset=[value_col])

    for dpi_val, g in data.groupby(dpi_col):
        levels = [lab for lab in label_order if lab in g[group_col].unique()]
        if len(levels) < 2:
            continue

        # ----- pairwise Mann–Whitney + δ (and r_b) -----
        pairs = list(combinations(levels, 2))
        mw_p, deltas = [], []
        for a, b in pairs:
            x = g.loc[g[group_col] == a, value_col].to_numpy()
            y = g.loc[g[group_col] == b, value_col].to_numpy()
            if x.size == 0 or y.size == 0:
                mw_p.append(np.nan); deltas.append(np.nan)
            else:
                res = mannwhitneyu(x, y, alternative="two-sided", method="auto")
                mw_p.append(res.pvalue)
                deltas.append(cliffs_delta(x, y))

        mw_uncorr = sym_matrix(levels, mw_p, label_order, fill=np.nan, diag=1.0)
        if mw_adjust:
            valid = [p for p in mw_p if not np.isnan(p)]
            if valid:
                _, p_adj, _, _ = multipletests(valid, method=mw_adjust)
                it = iter(p_adj)
                mw_p_adj = [next(it) if not np.isnan(p) else np.nan for p in mw_p]
            else:
                mw_p_adj = mw_p
            mw_adj = sym_matrix(levels, mw_p_adj, label_order, fill=np.nan, diag=1.0)
        else:
            mw_adj = mw_uncorr.copy()

        delta_mat = sym_matrix(levels, deltas, label_order, fill=np.nan, diag=0.0)
        rank_biserial_mat = delta_mat.copy()  # r_b == δ for MW comparisons

        # ----- Dunn's post-hoc (adjusted) -----
        dunn = sp.posthoc_dunn(g, val_col=value_col, group_col=group_col, p_adjust=dunn_adjust)
        dunn = dunn.reindex(index=label_order, columns=label_order)

        # ----- Global ε² from Kruskal–Wallis -----
        arrays = [g.loc[g[group_col] == lab, value_col].to_numpy() for lab in levels]
        if all(len(a) > 0 for a in arrays) and len(arrays) >= 2:
            H, _ = kruskal(*arrays)
            eps2 = epsilon_squared_kruskal(H, k=len(arrays), n=sum(len(a) for a in arrays))
        else:
            eps2 = np.nan

        # ----- Kendall's W (optional; repeated-measures only) -----
        W = kendalls_w(g if subject_col else g.assign(__dummy=1), subject_col, group_col, value_col, levels) if subject_col else np.nan

        results[dpi_val] = {
            "mannwhitney_p_uncorrected": mw_uncorr,
            "mannwhitney_p": mw_adj,
            "dunn_p": dunn,
            "cliffs_delta": delta_mat,
            "rank_biserial": rank_biserial_mat,
            "epsilon_squared": eps2,
            "kendalls_w": W,
        }

    return results


# ----------------------------------------------------------------------------
# formatting helpers
# ----------------------------------------------------------------------------


def get_pval_symbol(pval):
    if pval < 0.0001:
        return '****'
    elif pval < 0.001:
        return '***'
    elif pval < 0.01:
        return '**'
    elif pval < 0.05:
        return '*'
    else:
        return ''


def better_percentage(group, val):
    n = len(group)
    perc = round(len(group[group < val]) / n * 100)
    return f'{perc}%\nbetter'
