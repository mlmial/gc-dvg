"""Granger-model forecasting and the per-candidate fit plots.

Shared by notebooks 01 and 02, which previously held near-identical copies.
:func:`plot_single_candidate` keeps both notebooks' looks behind ``style``:
``"detailed"`` is notebook 01's publication panel (SSR annotations, fixed axis
limits, hi-res), ``"simple"`` is notebook 02's plain control-data plot.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, root_mean_squared_error
from tslearn.metrics import dtw

from utils.metrics import (
    calculate_weighted_mae,
    calculate_weighted_mape,
    dtw_mape,
    nrsme,
)
from utils.plotting import granger_label_color_map, lighten_color

__all__ = ["run_granger_prediction", "plot_single_candidate", "plot_restricted_only"]


def run_granger_prediction(
    summary_df,
    gc_max_lag,
    gc_test_results,
    plot_dips,
    log_long_data,
    restricted_pred_index=None,
    include_ssr=True,
    figsize=(8,6),
    ylabel='log10(PFU)',
    ylabel_pos=None,
    xlabel='Time post infection (days)',
    cultivation_value_label='plaque_assay',
    cultivation_value_names={'plaque_assay': 'PFU'},
    retransform=False,
    yscale='linear',
    dpi=300,
    linewidth=4,           # line width for all series
    markersize=7,          # marker size for all series
    markeredgewidth=2,      # marker edge width for the model lines,
    plot_legend=False,
    plot_title=True
):
    # Initialize data structures
    columns = ['key', 'granger_label', 'optimal_lag', 'diff_level', 'ssr_chi2test_pval', 'significant_pvals',
                'mape', 'mae', 'rmse', 'nrsme', 'mse', 'dtw', 'wmae', 'wmape', 'relative_mape',
                'relative_rmse', 'relative_mae', 'relative_wmae', 'relative_wmape', 'relative_dtw',
                'relative_mse', 'dtw_aligned_mape', 'ssr', 'weighted_ssr', 'pred_data']

    gc_prediction_summary_tmp = pd.DataFrame(columns=columns)
    ref_ols_performance_dct_tmp = {}
    full_preds = {}

    # Prepare dataframe
    summary_df_cp = summary_df.copy()
    summary_df_cp[f'{cultivation_value_label}_granger_label'] = summary_df_cp[f'{cultivation_value_label}_granger_label'].fillna('non-related')

    if retransform:
      # retransform all time series data
      log_long_data = np.power(10, log_long_data)
    restricted_pred_done = False
    for idx, row in summary_df_cp.iterrows():
        dip = row.key
        opt_lag = gc_max_lag[(cultivation_value_label, dip)]
        full_pred = gc_test_results[cultivation_value_label, dip][opt_lag][1][1].predict()
        if retransform:
          full_pred = np.power(10, full_pred)

        if not restricted_pred_done:
          restricted_pred_index = idx if restricted_pred_index is None else restricted_pred_index
          restricted_pred = gc_test_results[cultivation_value_label, dip][opt_lag][1][0].predict()
          if retransform:
            # Retransform predictions if necessary
            restricted_pred = np.power(10, restricted_pred)
          restricted_pred_done = True

        full_preds[dip] = full_pred
        actual_value = log_long_data[cultivation_value_label]
        actual_value_comp = actual_value.iloc[opt_lag:]

        dpis = log_long_data.index.values

        metrics = {
              'mape': mean_absolute_percentage_error(actual_value_comp, full_pred),
              'mae': mean_absolute_error(actual_value_comp, full_pred),
              'mse': root_mean_squared_error(actual_value_comp, full_pred)**2
          }

        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['nrsme'] = metrics['rmse'] / np.mean(actual_value)
        metrics['dtw'] = dtw(actual_value_comp, full_pred)
        metrics['wmae'] = calculate_weighted_mae(actual_value_comp, full_pred)
        metrics['wmape'] = calculate_weighted_mape(actual_value_comp, full_pred)

        # Relative errors
        rel_metrics = {
            'relative_mape': metrics['mape'] / mean_absolute_percentage_error(actual_value_comp, restricted_pred),
            'relative_rmse': metrics['rmse'] / root_mean_squared_error(actual_value_comp, restricted_pred),
            'relative_mae': metrics['mae'] / mean_absolute_error(actual_value_comp, restricted_pred),
            'relative_wmae': metrics['wmae'] / calculate_weighted_mae(actual_value_comp, restricted_pred),
            'relative_wmape': metrics['wmape'] / calculate_weighted_mape(actual_value_comp, restricted_pred),
            'relative_dtw': metrics['dtw'] / dtw(actual_value_comp, restricted_pred),
            'relative_mse': metrics['mse'] / root_mean_squared_error(actual_value_comp, restricted_pred)**2
        }

        ssr_value = np.sum((actual_value_comp - full_pred) ** 2) if include_ssr else None
        weighted_ssr = np.sum(((actual_value_comp - full_pred)/max(actual_value_comp)) ** 2) / len(full_pred) if include_ssr else None

        gc_prediction_summary_tmp.loc[len(gc_prediction_summary_tmp)] = [
            dip,
            row[f'{cultivation_value_label}_granger_label'], opt_lag, row['max_diff'], row[f'{cultivation_value_label}_ssr_chi2test'],
            row[f'{cultivation_value_label}_significant_pvals'], metrics['mape'], metrics['mae'], metrics['rmse'],
            metrics['nrsme'], metrics['mse'], metrics['dtw'], metrics['wmae'], metrics['wmape'],
            rel_metrics['relative_mape'], rel_metrics['relative_rmse'], rel_metrics['relative_mae'],
            rel_metrics['relative_wmae'], rel_metrics['relative_wmape'], rel_metrics['relative_dtw'],
            rel_metrics['relative_mse'], dtw_mape(actual_value_comp, full_pred),
            ssr_value if include_ssr else None,
            weighted_ssr if include_ssr else None,
            (dpis[opt_lag:], full_pred)
        ]

        # Save reference metrics for restricted prediction
        if idx == restricted_pred_index:
            for metric in ['mape', 'mae', 'rmse', 'nrsme', 'mse', 'dtw', 'wmae', 'wmape']:
                ref_ols_performance_dct_tmp[metric.upper()] = metrics[metric]
            ref_ols_performance_dct_tmp['DTW_ALIGNED_MAPE'] = dtw_mape(actual_value_comp, restricted_pred)
            if include_ssr:
                ref_ols_performance_dct_tmp['SSR'] = np.sum((actual_value_comp - restricted_pred) ** 2)
                ref_ols_performance_dct_tmp['WEIGHTED_SSR'] = np.sum(((actual_value_comp - restricted_pred)/actual_value_comp) ** 2) / len(restricted_pred)

        # Plot if dip is in plot_dips
        if dip in plot_dips:
            granger_label = row[f'{cultivation_value_label}_granger_label']
            full_color = granger_label_color_map[granger_label]
            fig, ax = plt.subplots(figsize=(figsize), dpi=dpi)
            ax.plot(dpis, actual_value,
                    label=f'Data',
                    marker='D', markersize=max(1,markersize-1), color='black',
                    linewidth=linewidth)
            ax.plot(dpis[opt_lag:], restricted_pred, label='Restricted\nmodel',
                    marker='^', markersize=markersize,
                    color='#e09312',
                    markeredgecolor='#e09312',
                    markerfacecolor="#e1bf83",
                    markeredgewidth=markeredgewidth,
                    linewidth=linewidth, linestyle=(0,(1,1)))
            ax.plot(dpis[opt_lag:], full_pred, label='Full\nmodel',
                    marker='o', markersize=markersize,
                    color=full_color,
                    markeredgecolor=full_color,
                    markerfacecolor=lighten_color(full_color, 0.5),
                    markeredgewidth=markeredgewidth,
                    linewidth=linewidth, linestyle='dashed')
            if plot_title:
              ax.set_title(f'Prediction with Granger-{row[f"{cultivation_value_label}_granger_label"]}\n{dip} (lag={opt_lag})')
            ax.set_xlabel(xlabel)
            if ylabel_pos:
              ax.set_ylabel(ylabel, y=ylabel_pos, ha='left')
            else:
              ax.set_ylabel(ylabel, ha='center')

            # --- x-axis: major ticks 0-22 step 2, minor ticks between ---
            from matplotlib.ticker import MultipleLocator, AutoMinorLocator

            ax.xaxis.set_major_locator(MultipleLocator(2))
            ax.xaxis.set_minor_locator(AutoMinorLocator(2))

            # --- y-axis: end at 10 ---
            ax.set_ylim(0, 10)
            if plot_legend:
              ax.legend(loc='center left', bbox_to_anchor=(1.05, 0.6), ncol=3, frameon=False)
            ax.set_yscale(yscale)
            # fig.text(1.1, -0.05, f'SSR$_{{full}}$={ssr_value:.2f}',
            #   transform=ax.transAxes,
            #   verticalalignment='top',
            #   bbox=dict(boxstyle='square', facecolor=full_color, alpha=0.5))
            # fig.text(1.1, 0.1, f'SSR$_{{restricted}}$={ref_ols_performance_dct_tmp["SSR"]:.2f}',
            #   transform=ax.transAxes,
            #   verticalalignment='top',
            #   bbox=dict(boxstyle='square', facecolor='#e09312', alpha=0.5))
            plt.show()

    # For color correction later
    gc_prediction_summary_tmp['norm_ssr_chi2test_pval'] = gc_prediction_summary_tmp['ssr_chi2test_pval'] / 0.1

    # Group the DataFrame by 'granger_label'
    grouped = gc_prediction_summary_tmp.groupby('granger_label')
    gc_prediction_summary_tmp['group_norm_ssr_chi2test_pval'] = gc_prediction_summary_tmp['ssr_chi2test_pval']
    # Normalize the 'norm_ssr_chi2test' column within each group
    for label, group in grouped:
        max_value = group['group_norm_ssr_chi2test_pval'].max()  # Get the maximum value in the group
        gc_prediction_summary_tmp.loc[group.index, 'group_norm_ssr_chi2test_pval'] = group['group_norm_ssr_chi2test_pval'] / max_value  # Normalize

    return gc_prediction_summary_tmp, ref_ols_performance_dct_tmp


def plot_single_candidate(row,
                          gc_prediction_summary_df,
                          gc_test_results,
                          log_long_data,
                          cultivation_value_label='plaque_assay',
                          dpis=None,
                          figsize=(7, 5),
                          xlabel='Time post infection (days)',
                          ylabel='log10(PFU)',
                          yscale='linear',
                          opt_lag=2,
                          title_prefix='',
                          style='simple',
                          markersize=7,
                          linewidth=2,
                          markeredgewidth=1.5,
                          dpi=800):
  """Plot the restricted and full model fits for one DVG candidate.

  ``style='detailed'`` reproduces notebook 01's figure (SSR annotation boxes,
  x-limit 0-22, y-limit 0-10, minor ticks, left-aligned title, dpi=800);
  ``style='simple'`` reproduces notebook 02's plainer control-data figure.

  The data objects are passed in explicitly -- the original default arguments
  were evaluated at definition time against notebook globals.
  """
  if dpis is None:
    dpis = log_long_data.index.values
  fits = gc_test_results[(cultivation_value_label, row.key)][row.plaque_assay_max_lag][1]
  restricted_pred = fits[0].predict()
  full_pred = fits[1].predict()
  granger_label = row[f'{cultivation_value_label}_granger_label']
  full_color = granger_label_color_map[granger_label]
  actual_value = log_long_data[cultivation_value_label]
  dip = row.key
  print(gc_prediction_summary_df[gc_prediction_summary_df.key == dip].iloc[0]['ssr'])

  if style == 'simple':
    fig, ax = plt.subplots(figsize=(figsize))
    ax.plot(dpis, actual_value, label=f'{cultivation_value_label}', marker='D', color='black')
    ax.plot(dpis[opt_lag:], restricted_pred, label='Restricted\nmodel prediction', marker='o', color='#e09312', linewidth=4)
    ax.plot(dpis[opt_lag:], full_pred, label=f'Full model\nprediction with\n{dip}', marker='o', color=full_color, linewidth=4)
    ax.set_title(f'{title_prefix}Prediction with Granger-{granger_label} DIP {dip.replace("_","-")} (lag={opt_lag})')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend([r'$\mathregular{c_{virus}}$', 'Restricted model', 'Full model'], loc='center left', bbox_to_anchor=(1, 0.5), ncol=3)
    ax.set_yscale(yscale)
    plt.show()
    return

  restr_ssr = fits[0].ssr
  full_ssr = fits[1].ssr

  fig, ax = plt.subplots(figsize=(figsize), dpi=dpi)
  ax.plot(dpis, actual_value,
          label=f'Data',
          marker='D', markersize=max(1, markersize-1), color='black',
          linewidth=linewidth)
  ax.plot(dpis[opt_lag:], restricted_pred, label='Restricted\nmodel',
          marker='^', markersize=markersize,
          color='#e09312',
          markeredgecolor='#e09312',
          markerfacecolor="#e1bf83",
          markeredgewidth=markeredgewidth,
          linewidth=linewidth, linestyle=(0,(1,1)))
  ax.plot(dpis[opt_lag:], full_pred, label=f'Full model',
          marker='o', markersize=markersize,
          color=full_color,
          markeredgecolor=full_color,
          markerfacecolor=lighten_color(full_color, 0.5),
          markeredgewidth=markeredgewidth,
          linewidth=linewidth, linestyle='dashed')
  ax.set_xlabel(xlabel)
  ax.set_ylabel(ylabel)
  ax.set_ylim(0, 10)
  ax.set_xlim(0, 22)
  ax.xaxis.set_major_locator(MultipleLocator(2))
  ax.xaxis.set_minor_locator(AutoMinorLocator(2))
  ax.set_title(f'{title_prefix}Granger-{granger_label} {dip} (lag={opt_lag})', pad=20, loc='left')

  ax.legend(loc='center left', bbox_to_anchor=(1.05, 0.6), frameon=False)
  ax.set_yscale(yscale)
  fig.text(1.1, -0.1, f'SSR$_{{full}}$={full_ssr:.2f}',
            transform=ax.transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='square', facecolor=full_color, alpha=0.5))
  fig.text(1.1, 0.1, f'SSR$_{{restricted}}$={restr_ssr:.2f}',
            transform=ax.transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='square', facecolor='#e09312', alpha=0.5))
  plt.show()


def plot_restricted_only(row,
                          gc_prediction_summary_df,
                          gc_test_results,
                          log_long_data,
                          cultivation_value_label='plaque_assay',
                          dpis=None,
                          figsize=(7, 5),
                          xlabel='Time post infection (days)',
                          ylabel='log10(PFU)',
                          yscale='linear',
                          opt_lag=2,
                          title_prefix=''):
  if dpis is None:
    dpis = log_long_data.index.values
  restricted_pred = gc_test_results[(cultivation_value_label, row.key)][row.plaque_assay_max_lag][1][0].predict()
  actual_value = log_long_data[cultivation_value_label]
  dip = row.key

  fig, ax = plt.subplots(figsize=(figsize))
  ax.plot(dpis, actual_value, label=f'{cultivation_value_label}', marker='D', color='black')
  ax.plot(dpis[opt_lag:], restricted_pred, label='Restricted\nmodel prediction', marker='o', color='#e09312', linewidth=4)
  ax.set_title(f'{title_prefix}Restricted model prediction (lag={opt_lag})')
  ax.set_xlabel(xlabel)
  ax.set_ylabel(ylabel)
  ax.legend([r'$\mathregular{c_{virus}}$', 'Restricted model'], loc='center left', bbox_to_anchor=(1, 0.5), ncol=2)
  ax.set_yscale(yscale)
  plt.show()
