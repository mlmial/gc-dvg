"""Shared plotting helpers and colour conventions for notebooks 01-04.

Colour conventions
------------------
Granger labels map to a fixed colour family everywhere:
``causing`` blue, ``caused`` red, ``bi-directional`` purple,
``restricted`` orange, everything else (``non-related``, ``shuffled``,
``random``, ``bootstrapped``) grey.

The two swarmplot builders were previously copy-pasted with small style drifts
between notebooks. They are unified here; the style differences are exposed as
keyword arguments so every notebook keeps rendering exactly as before.
"""

import matplotlib.colors as mc
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.gridspec import GridSpec

from utils.metrics import better_percentage

__all__ = [
    "granger_label_color_map", "lighten_color", "lighten",
    "get_main_color", "granger_color",
    "get_corrected_color", "get_corrected_color_name", "get_corrected_color_v2",
    "add_sig_bar_strip",
    "make_performance_swarmplot", "make_performance_swarmplot_with_stats",
]

#: Colormap family per Granger-causality label.
LABEL_CMAP = {
    "caused": "Reds",
    "bi-directional": "Purples",
    "causing": "Blues",
    "restricted": "Oranges",
    "non-related": "Greys",
    "shuffled": "Greys",
    "random": "Greys",
    "bootstrapped": "Greys",
}

#: Short effect-size symbols used inside significance brackets.
EFFECT_SIZE_SYMBOLS_SHORT = {"cliffs_delta": "\u03b4", "rank_biserial": "r_b"}


# ----------------------------------------------------------------------------
# colours
# ----------------------------------------------------------------------------


granger_label_color_map = {
  'non-related': 'gray',
  'causing': 'tab:blue',
  'caused': 'tab:red',
  'bi-directional': 'tab:purple'
}


def lighten_color(color, amount=0.5):
    """Blend `color` toward white. amount=0 -> original, amount=1 -> white."""
    r, g, b = mc.to_rgb(color)
    return (r + (1 - r) * amount, g + (1 - g) * amount, b + (1 - b) * amount)


#: Alias kept for notebook 03, which imported this under a shorter name.
lighten = lighten_color


def get_main_color(granger_label):
    if granger_label == 'caused':
        return 'tab:red'
    elif granger_label == 'bi-directional':
        return 'tab:purple'
    elif granger_label == 'causing':
        return 'tab:blue'
    else:
        return 'grey'


def granger_color(label):
    if label == 'caused':
        return 'tab:red'
    elif label == 'bi-directional':
        return 'tab:purple'
    elif label == 'causing':
        return 'tab:blue'
    else:
        return 'gray'


def get_corrected_color(row, column="granger_score", invert=True,
                        label_key="granger_label", strict=False):
    """Shade a row by ``column`` within the colour family of its Granger label.

    ``label_key`` selects the column holding the label (notebooks 01/02 use
    ``granger_label``, notebook 03 uses ``label``). With ``strict=True`` an
    unknown label raises instead of silently falling back to grey.
    """
    label = row[label_key]
    cmap_name = LABEL_CMAP.get(label)
    if cmap_name is None:
        if strict:
            raise ValueError(f"Unrecognized label '{label}' for color mapping.")
        cmap_name = "Greys"
    cmap = plt.get_cmap(cmap_name)
    if invert:
        cmap = cmap.reversed()
    return cmap(row[column])


def get_corrected_color_name(row, column="granger_score", invert=True,
                             label_key="granger_label"):
    """Name of the colormap :func:`get_corrected_color` would use for ``row``."""
    base = LABEL_CMAP.get(row[label_key], "Greys")
    return base + ("_r" if invert else "")


def _assert_label_colors(df, label_col="label", name_col="corrected_color_name"):
    """Sanity-check that every label got the colormap it is supposed to get."""
    for label, expected in LABEL_CMAP.items():
        wrong = df[(df[label_col] == label) & (df[name_col] != expected + "_r")]
        assert wrong.empty, f"{len(wrong)} rows labelled '{label}' got the wrong colormap"


def add_sig_bar_strip(ax_sig, x1, x2, y, h, text, linewidth=1, fontsize=None, bold=False):
    """Draw a significance bracket in the top strip (``ax_sig`` spans y in [0, 1]).

    ``bold=True`` (adjusted p < alpha) gives a heavier line and bold text.
    ``fontsize=None`` leaves the text at the current rcParams size.
    """
    ax_sig.plot([x1, x1, x2, x2], [y, y + h, y + h, y], color="k",
                lw=linewidth * (2.0 if bold else 1.0), clip_on=False)
    kwargs = {} if fontsize is None else {"fontsize": fontsize}
    ax_sig.text((x1 + x2) / 2, y + h, text, ha="center", va="bottom",
                fontweight=("bold" if bold else "normal"), **kwargs)


def get_corrected_color_v2(row, label_column, categories=[None,None,None], column='granger_score', invert=True):
    if row[label_column] == categories[0]:
        base_color = plt.get_cmap('Reds')
    elif row[label_column] == categories[1]:
        base_color = plt.get_cmap('Purples')
    elif row[label_column] == categories[2]:
        base_color = plt.get_cmap('Blues')
    else:
        base_color = plt.get_cmap('Greys')

    if invert == False:
        base_color = base_color.reversed()
        
    # Return a lighter shade based on the granger_score (the lower the score, the lighter the shade)
    return base_color(row[column])


# ----------------------------------------------------------------------------
# swarmplots
# ----------------------------------------------------------------------------


def make_performance_swarmplot(dip_forecast_summary,
                                ref_performance_dct,
                                performance_metric='mape',
                                forecasted_value = 'pfu',
                                title = 'Granger-related DI vRNAs predictive power over PFU\n',
                                ylabel='Mean Absolute Percentage Error (MAPE)',
                                dot_color_ref = 'diff_level_norm',
                                inverted_colors = True, # the higher the value, the darker the color
                                upper_border = -10,
                                ymax = None,
                                ymin = 0,
                                yscale = 'linear',
                                p_val_pos = 0,
                                figsize=(8, 4),
                                better_perc = False,
                                markersize=18,
                                linewidth=4,
                                dotsize=5,
                                dpi=None,          # nb02 renders at dpi=800
                                title_pad=15,      # nb02 uses 10
                                median_halo=False, # nb02 draws a yellow halo behind the median
                                included_dvgs = None
                                ): 
    dip_forecast_summary['corrected_color'] = dip_forecast_summary.apply(get_corrected_color, args=(dot_color_ref, inverted_colors), axis=1)
    
    if included_dvgs is not None:
        dip_forecast_summary = dip_forecast_summary[dip_forecast_summary.key.isin(included_dvgs)]
        
    # Create the swarm plot
    fig, axs = plt.subplots(nrows=1, figsize=figsize, dpi=dpi)
    ax = axs
    ax = sns.swarmplot(x='granger_label', 
                        y=performance_metric, 
                        hue='key',
                        data=dip_forecast_summary, 
                        order = ['causing', 'bi-directional', 'caused', 'non-related'],
                        palette=dip_forecast_summary['corrected_color'].tolist(),
                        size=dotsize, ax=ax, legend=False, linewidth=0.2
    )
    # Set title and other plot parameters
    ax.set_title(title, pad=title_pad)
    ax.set_xlabel('Granger-causality label', labelpad=10)

    ax.set_ylabel(ylabel)
    xmin, xmax = ax.get_xlim()
    ax.hlines(y=ref_performance_dct[performance_metric.upper()], xmin=xmin, xmax=xmax, label='restricted model performance', color='#e09312', linewidth=linewidth, zorder=10)
    ax.legend(loc='lower right')
    if ymax is None:
        ymax = ref_performance_dct[performance_metric.upper()]*3
    ax.set_ylim(ymin, ymax)
    ax.set_yscale(yscale)
    #ax.invert_yaxis()
    ax.tick_params(top=False, labeltop=False, bottom=True, labelbottom=True)

    xs, xticklabels = ax.get_xticks(), ax.get_xticklabels()

    # Loop through the x-axis categories and plot the medians as markers
    for x, xticklabel in zip(xs, xticklabels):
        label = xticklabel.get_text()
        
        # Calculate median
        median_value = dip_forecast_summary[dip_forecast_summary.granger_label == label][performance_metric].median()
        
        # Plot the median as a horizontal marker (-)
        if median_halo:
            ax.plot(x, median_value, marker='+', color='black', markersize=markersize, zorder=11, label=f'{label} Median', markeredgewidth=linewidth)
            ax.plot(x, median_value, marker='+', color='yellow', markersize=markersize, zorder=1, label=f'{label} Median')
        else:
            ax.plot(x, median_value, marker='+', color='black', markersize=markersize, zorder=11, label=f'{label} Median')
        
        ax.text(x, p_val_pos, f'n={len(dip_forecast_summary[dip_forecast_summary.granger_label == label])}', ha='center', va='bottom', color='black', zorder=10)

        if better_perc:
            ax.text(x, upper_border, 
                better_percentage(dip_forecast_summary[dip_forecast_summary.granger_label == label][performance_metric], 
                                        ref_performance_dct[performance_metric.upper()]), 
                    ha='center', va='bottom', color='black', zorder=10)
    
    fig.tight_layout()
    return fig, ax


def make_performance_swarmplot_with_stats(
        dip_forecast_summary, ref_performance_dct,
        performance_metric='ssr', forecasted_value='plaque_assay',
        title='', ylabel='SSR', dot_color_ref='group_norm_ssr_chi2test_pval',
        inverted_colors=True, ymax=210, ymin=80, yscale='linear', p_val_pos=0,
        figsize=(12,6), markersize=18, linewidth=3, dotsize=3,
        stats_matrix=None, stat_test_name=None, effect_size_matrix=None,
        effect_size_name=None, alpha=0.05, dpi=300,
        order=('causing','bi-directional','caused','non-related','shuffled','bootstrapped'),
        x='label', plot_reference=True,
        granger_label_col='granger_label', key_col='key',
        color_label_col='color_label', title_offset=0.98,
        upper_border=0.05,        # accepted for call compatibility; spacing only
        palette=None,             # nb03 supplies a fixed DVG -> colour mapping
        validate_colors=False,    # nb03 asserts label/colormap consistency
        median_color='#FFFB00',   # nb03 uses '#EEFF00'
        median_inner_edge_width=None,
        effect_size_symbols=None,
        legend_loc='lower left', legend_bbox=(0, 1.02),
        legend_ncol=2, legend_frameon=False):
    effect_size_symbols = EFFECT_SIZE_SYMBOLS_SHORT if effect_size_symbols is None else effect_size_symbols
    order = list(order)
    df = dip_forecast_summary.copy()
    # accept native nb01/nb02 schema
    if 'label' not in df.columns and granger_label_col in df.columns:
        df = df.rename(columns={granger_label_col: 'label'})
    if 'dvg' not in df.columns and key_col in df.columns:
        df = df.rename(columns={key_col: 'dvg'})
    df['dvg'] = df['dvg'].astype(str)

    # color label: use the dedicated column if provided, else fall back to x-axis label
    if color_label_col not in df.columns:
        df[color_label_col] = df['label']

    df = df.sort_values(by=dot_color_ref, ascending=inverted_colors)
    df['corrected_color']      = df.apply(get_corrected_color,      args=(dot_color_ref, inverted_colors, color_label_col, True), axis=1)
    df['corrected_color_name'] = df.apply(get_corrected_color_name, args=(dot_color_ref, inverted_colors, color_label_col), axis=1)

    fig = plt.figure(figsize=figsize, dpi=dpi)
    if stats_matrix is not None:
        gs = GridSpec(nrows=2, ncols=1, height_ratios=[7, 14], hspace=0.0)
        ax_sig = fig.add_subplot(gs[0]); ax = fig.add_subplot(gs[1])
    else:
        ax = fig.add_subplot(1, 1, 1)

    if validate_colors:
        _assert_label_colors(df)
    assert df['dvg'].is_unique, "dvg (hue) keys must be unique across all swarms"
    if palette is None:
        palette = df.set_index('dvg')['corrected_color'].to_dict()
    else:
        # fill in any DVG the caller's palette does not cover
        missing = set(df['dvg'].unique()) - set(palette)
        if missing:
            palette.update(df[df['dvg'].isin(missing)].groupby('dvg')['corrected_color'].first().to_dict())

    sns.swarmplot(x=x, y=performance_metric, hue='dvg', data=df, order=order,
                  palette=palette, size=dotsize, ax=ax, legend=False, linewidth=0.2)

    ax.set_xlabel('Granger-causality label', labelpad=10)
    ax.set_ylabel(ylabel)

    if plot_reference:
        xmin, xmax = ax.get_xlim()
        ax.hlines(y=ref_performance_dct[performance_metric.upper()], xmin=xmin, xmax=xmax,
                  label='restricted model', color='#e09312', linewidth=linewidth, zorder=10)
        ax.legend(loc=legend_loc, bbox_to_anchor=legend_bbox, ncol=legend_ncol, frameon=legend_frameon)

    if ymax is None:
        ymax = ref_performance_dct[performance_metric.upper()] * 3
    ax.set_ylim(ymin, ymax); ax.set_yscale(yscale)
    ax.tick_params(top=False, labeltop=False, bottom=True, labelbottom=True)

    for xpos, xticklabel in zip(ax.get_xticks(), ax.get_xticklabels()):
        lab = xticklabel.get_text()
        sub = df[df['label'] == lab][performance_metric]
        if len(sub):
            ax.plot(xpos, sub.median(), marker='+', color=median_color, markeredgecolor='black', markersize=markersize, markeredgewidth=3, zorder=11)
            inner = {} if median_inner_edge_width is None else {'markeredgewidth': median_inner_edge_width}
            ax.plot(xpos, sub.median(), marker='+', color=median_color, markersize=markersize, zorder=15, **inner)
        ax.text(xpos, p_val_pos, f"n={len(df[df[x] == lab])}", ha='center', va='bottom', color='black', zorder=10)

    if stats_matrix is not None:
        ax_sig.set_xlim(ax.get_xlim()); ax_sig.set_ylim(0, 1); ax_sig.axis('off')
        stats_mat = stats_matrix.reindex(index=order, columns=order)
        eff_mat   = effect_size_matrix.reindex(index=order, columns=order) if effect_size_matrix is not None else None
        x_pos = {lbl: xp for lbl, xp in zip(order, ax.get_xticks())}
        pairs = []
        for i, a in enumerate(order):
            for j, b in enumerate(order):
                if j <= i:
                    continue
                p = stats_mat.loc[a, b]
                try:
                    show = np.isfinite(p) and (p <= 1.0)   # print bracket for every real pair
                    bold = np.isfinite(p) and (p < alpha) # bold only when significant
                except Exception:
                    show = bold = False
                if show:
                    eff_txt = ""
                    if eff_mat is not None:
                        eff = eff_mat.loc[a, b]
                        if isinstance(eff, (int, float, np.floating)) and np.isfinite(eff):
                            sym = effect_size_symbols.get(effect_size_name, effect_size_name or 'effect')
                            eff_txt = f"{sym}={eff:.2f}"
                    pairs.append((a, b, eff_txt, bool(bold)))
        if pairs:
            pairs.sort(key=lambda t: abs(x_pos[t[0]] - x_pos[t[1]]))
            base, step, h = 0.10, 0.24, 0.04
            levels_spans = []
            for a, b, eff_txt, bold in pairs:
                xa, xb = x_pos[a], x_pos[b]
                lo, hi = sorted((xa, xb))
                level = None
                for li, (u_lo, u_hi) in enumerate(levels_spans):
                    if hi < u_lo or lo > u_hi:
                        level = li; levels_spans[li] = (min(u_lo, lo), max(u_hi, hi)); break
                if level is None:
                    level = len(levels_spans); levels_spans.append((lo, hi))
                add_sig_bar_strip(ax_sig, xa, xb, base + level*step, h, eff_txt, bold=bold)
    fig.suptitle(title, y=title_offset)
    fig.tight_layout(rect=[0, 0, 1, 1] if stats_matrix is not None else None)
    return fig, ax, df
