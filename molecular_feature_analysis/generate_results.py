import csv
import os
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib
import numpy as np
import pandas as pd
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt
from matplotlib.ticker import FuncFormatter
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
OUTFILE = RESULTS / "custom_v2_analysis_report.docx"
DATASETS = {
    "bi-directional": ROOT / "data" / "custom_v2" / "bi_directional_dataset.csv",
    "caused": ROOT / "data" / "custom_v2" / "caused_dataset.csv",
    "causing": ROOT / "data" / "custom_v2" / "causing_dataset.csv",
    "non-related": ROOT / "data" / "custom_v2" / "non_related_dataset.csv",
}
GROUP_ORDER = ["bi-directional", "caused", "causing", "non-related"]
GROUP_COLORS = {
    "bi-directional": "#8A2BE2",
    "causing": "#2F80ED",
    "caused": "#E45756",
    "non-related": "#9E9E9E",
}
SEGMENTS = ["PB2", "PB1", "PA", "HA", "NP", "NA", "M", "NS"]
SUMMARY_REPS = 100
CONVERGENCE_REPS = 1000
ITERATION_THRESHOLDS = (20, 40, 80, 100, 200, 500, 1000)
RNG_SEED = 42
NUCLEOTIDES = ("A", "C", "G", "U")


def get_sequence(strain: str, segment: str) -> str:
    """Read one bundled FASTA and return its RNA sequence."""
    fasta = ROOT / "data" / strain / f"{segment}.fasta"
    return "".join(
        line.strip().replace("T", "U")
        for line in fasta.read_text().splitlines()
        if line and not line.startswith(">")
    )


def preprocess(strain: str, df: pd.DataFrame, thresh: int) -> pd.DataFrame:
    """Add the sequence columns used by the original comparison workflow."""
    if thresh > 1:
        df = df[df["NGS_read_count"] >= thresh].copy()
    df["Segment"] = df["Segment"].replace({"NS1": "NS", "NS2": "NS"})
    df["full_seq"] = df["Segment"].map(lambda segment: get_sequence(strain, segment))
    df["deleted_sequence"] = [
        get_sequence(strain, row.Segment)[int(row.Start) : int(row.End) - 1]
        for row in df.itertuples(index=False)
    ]
    return df


def calculate_direct_repeat(seq: str, start: int, end: int, window: int = 5) -> tuple[int, str]:
    start_window = seq[start - window : start]
    end_window = seq[end - 1 - window : end - 1]
    if start_window == end_window:
        return len(start_window), start_window
    if len(seq) < end:
        return 0, "_"
    counter = 0
    for i in range(len(end_window) - 1, -1, -1):
        if start_window[i] == end_window[i]:
            counter += 1
        else:
            break
    overlap = start_window[i + 1 : window]
    return counter, overlap or "_"


def count_direct_repeats_overall(df: pd.DataFrame, seq: str) -> tuple[dict, dict]:
    counts = {i: 0 for i in range(6)}
    overlaps = {}
    for row in df.itertuples(index=False):
        repeat_length, overlap = calculate_direct_repeat(seq, int(row.Start), int(row.End))
        counts[repeat_length] += 1
        overlaps[overlap] = overlaps.get(overlap, 0) + 1
    return counts, overlaps


def create_nucleotide_ratio_matrix(df: pd.DataFrame, column: str) -> pd.DataFrame:
    sequence_matrix = df[column].str.split("", expand=True).drop(columns=[0])
    sequence_matrix = sequence_matrix.drop(columns=[sequence_matrix.columns[-1]])
    return pd.DataFrame(
        {nucleotide: sequence_matrix.eq(nucleotide).mean(axis=0) for nucleotide in NUCLEOTIDES}
    )


def set_default_font(document: Document) -> None:
    styles = document.styles
    normal = styles["Normal"]
    normal.font.name = "Arial"
    normal.font.size = Pt(10.5)


def add_heading(document: Document, text: str, level: int = 1) -> None:
    document.add_heading(text, level=level)


def add_paragraph(document: Document, text: str) -> None:
    document.add_paragraph(text)


def load_custom_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        keep_default_na=False,
        dtype={"Segment": "string", "Start": "int64", "End": "int64"},
    )
    df["Segment"] = df["Segment"].replace({"NS1": "NS", "NS2": "NS"})
    df["NGS_read_count"] = 1
    return df[["Segment", "Start", "End", "NGS_read_count"]]


def preprocessed_datasets() -> Dict[str, pd.DataFrame]:
    return {name: preprocess("PR8", load_custom_dataset(path).copy(), 1) for name, path in DATASETS.items()}


def segment_counts(df: pd.DataFrame) -> pd.Series:
    return df["Segment"].value_counts().reindex(SEGMENTS, fill_value=0)


def dvg_lengths(df: pd.DataFrame) -> np.ndarray:
    return (df["full_seq"].str.len() - df["deleted_sequence"].str.len()).to_numpy(dtype=int)


def segment_lengths() -> Dict[str, int]:
    return {
        "PB2": 2341,
        "PB1": 2341,
        "PA": 2233,
        "HA": 1775,
        "NP": 1565,
        "NA": 1413,
        "M": 1027,
        "NS": 890,
    }


def dvg_deletion_lengths(df: pd.DataFrame) -> np.ndarray:
    lengths = segment_lengths()
    return np.array([lengths[row.Segment] - (int(row.End) - int(row.Start)) for row in df.itertuples(index=False)])


def build_nucleotide_context_df(df: pd.DataFrame, strain: str = "PR8", isize: int = 5) -> pd.DataFrame:
    rows = []
    for row in df.itertuples(index=False):
        seg = "NS" if row.Segment in {"NS1", "NS2"} else row.Segment
        seq = get_sequence(strain, seg)
        start = int(row.Start)
        end = int(row.End)
        deleted_sequence = seq[start : end - 1]
        seq_head = seq[:start]
        seq_foot = seq[end - 1 :]
        seq_before_start = seq_head[-isize:]
        seq_after_start = deleted_sequence[:isize]
        seq_before_end = deleted_sequence[-isize:]
        seq_after_end = seq_foot[:isize]
        rows.append(
            {
                "Segment": seg,
                "Start": start,
                "End": end,
                "seq_around_deletion_junction": seq_before_start
                + seq_after_start
                + seq_before_end
                + seq_after_end,
            }
        )
    return pd.DataFrame(rows)


def get_repeat_counts(df: pd.DataFrame) -> Dict[int, int]:
    counts: Dict[int, int] = {}
    for segment in SEGMENTS:
        df_s = df[df["Segment"] == segment]
        if len(df_s) == 0:
            continue
        seq = get_sequence("PR8", segment)
        repeat_counts, _ = count_direct_repeats_overall(df_s, seq)
        for key, value in repeat_counts.items():
            counts[key] = counts.get(key, 0) + int(value)
    return counts


def get_nucleotide_counts(df: pd.DataFrame) -> np.ndarray:
    nuc_df = build_nucleotide_context_df(df)
    matrix = create_nucleotide_ratio_matrix(nuc_df, "seq_around_deletion_junction")
    n_samples = len(nuc_df)
    out = []
    for pos in matrix.index:
        for nuc in matrix.columns:
            out.append(int(round(matrix.loc[pos, nuc] * n_samples)))
    return np.array(out)


def summarize_prefix(values: Sequence[float], n_reps: int) -> tuple[float, float, int]:
    arr = np.asarray(values[:n_reps], dtype=float)
    sig = arr < 0.05
    return float(np.median(arr)), float(np.mean(sig)), int(np.sum(sig))


def chi2_stat_from_segment_counts(segment_counts: np.ndarray, totals: np.ndarray) -> float:
    table = np.column_stack([segment_counts, totals - segment_counts]).astype(float)
    row_sums = table.sum(axis=1, keepdims=True)
    col_sums = table.sum(axis=0, keepdims=True)
    expected = row_sums @ col_sums / table.sum()
    mask = expected > 0
    return float(np.sum(((table - expected) ** 2)[mask] / expected[mask]))


def permutation_pvalue(segment_counts: np.ndarray, totals: np.ndarray, n_perm: int = 2000, seed: int = 42) -> float:
    rng = np.random.default_rng(seed)
    total_events = int(totals.sum())
    flags = np.zeros(total_events, dtype=int)
    flags[: int(segment_counts.sum())] = 1

    observed = chi2_stat_from_segment_counts(segment_counts, totals)
    perm_stats = np.empty(n_perm, dtype=float)
    group_sizes = totals.astype(int)

    for i in range(n_perm):
        rng.shuffle(flags)
        start = 0
        perm_counts = np.empty(len(group_sizes), dtype=int)
        for j, size in enumerate(group_sizes):
            perm_counts[j] = int(flags[start : start + size].sum())
            start += size
        perm_stats[i] = chi2_stat_from_segment_counts(perm_counts, totals)

    return float((np.sum(perm_stats >= observed) + 1) / (n_perm + 1))


def dominant_segments(df: pd.DataFrame) -> str:
    counts = df["Segment"].value_counts()
    counts = counts[counts > 0]
    if counts.empty:
        return ""
    top = counts.sort_values(ascending=False)
    total = int(top.sum())
    selected: List[str] = []
    cumulative = 0
    for segment, count in top.items():
        selected.append(segment)
        cumulative += int(count)
        if len(selected) >= 2 and cumulative / total >= 0.9:
            break
        if len(selected) == 3:
            break
    return ", ".join(selected)


def summarise_lengths(raw_df: pd.DataFrame) -> float:
    pre = preprocess("PR8", raw_df.copy(), 1)
    lengths = (pre["full_seq"].str.len() - pre["deleted_sequence"].str.len()).to_numpy()
    return float(np.median(lengths))


def compute_balanced_pvalues(sampled: Dict[str, pd.DataFrame]) -> Dict[str, float]:
    seg_table = np.vstack([segment_counts(sampled[name]).to_numpy(dtype=int) for name in GROUP_ORDER])
    seg_table = seg_table[:, seg_table.sum(axis=0) > 0]
    seg_p = stats.chi2_contingency(seg_table)[1]

    len_groups = [dvg_deletion_lengths(sampled[name]) for name in GROUP_ORDER]
    len_p = stats.kruskal(*len_groups)[1]

    rep_by_name = {name: get_repeat_counts(sampled[name]) for name in GROUP_ORDER}
    rep_keys = sorted(set().union(*[d.keys() for d in rep_by_name.values()]))
    rep_table = np.vstack([[int(rep_by_name[name].get(k, 0)) for k in rep_keys] for name in GROUP_ORDER])
    rep_table = rep_table[:, rep_table.sum(axis=0) > 0]
    rep_p = stats.chi2_contingency(rep_table)[1]

    nuc_by_name = {name: get_nucleotide_counts(sampled[name]) for name in GROUP_ORDER}
    nuc_table = np.vstack([nuc_by_name[name] for name in GROUP_ORDER])
    nuc_table = nuc_table[:, nuc_table.sum(axis=0) > 0]
    nuc_p = stats.chi2_contingency(nuc_table)[1]

    return {
        "Segment composition": float(seg_p),
        "DVG length": float(len_p),
        "Direct repeats": float(rep_p),
        "Nucleotide enrichment/context": float(nuc_p),
    }


def run_balanced_sampling(raw_dfs: Dict[str, pd.DataFrame], n_reps: int) -> tuple[pd.DataFrame, int]:
    min_n = min(len(df) for df in raw_dfs.values())
    rng = np.random.default_rng(RNG_SEED)
    records = []

    for rep in range(n_reps):
        sampled = {
            name: df.sample(
                n=min_n,
                replace=False,
                random_state=int(rng.integers(0, 1_000_000_000)),
            )
            for name, df in raw_dfs.items()
        }
        pvalues = compute_balanced_pvalues(sampled)
        for feature, pvalue in pvalues.items():
            records.append(
                {
                    "iteration": rep + 1,
                    "Feature": feature,
                    "pvalue": float(pvalue),
                    "significant": float(pvalue) < 0.05,
                }
            )

    return pd.DataFrame(records), min_n


def build_balanced_summary_from_records(records: pd.DataFrame, n_reps: int) -> pd.DataFrame:
    summary_rows = []
    interpretations = {
        "Segment composition": "The strongest and most stable difference. PB1 and PA dominate all groups, while non-related also carries noticeable PB2 and minor NS/HA/NP signal.",
        "DVG length": "Length differences remain present but are less stable after balancing, suggesting that some of the unbalanced contrast is driven by sample-size effects.",
        "Direct repeats": "No robust separation is retained after balancing; short repeat classes appear broadly conserved across the four datasets.",
        "Nucleotide enrichment/context": "The junction-proximal nucleotide pattern is highly conserved across groups, with no stable group-specific enrichment signal.",
    }
    feature_order = [
        ("Viral segment composition", "Segment composition"),
        ("", "DVG length"),
        ("", "Direct repeats"),
        ("", "Nucleotide enrichment/context"),
    ]

    for molecular_feature, feature in feature_order:
        values = records.loc[records["Feature"] == feature, "pvalue"].tolist()
        median_p, frac_sig, sig_count = summarize_prefix(values, n_reps)
        summary_rows.append(
            {
                "Molecular feature": molecular_feature,
                "Feature": feature,
                "Median p-value": median_p,
                "Significant runs": f"{(sig_count / n_reps) * 100:.2f}%",
                "Interpretation": interpretations[feature],
            }
        )

    return pd.DataFrame(summary_rows).fillna("")


def build_convergence_summary(records: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature in records["Feature"].unique():
        feature_records = records[records["Feature"] == feature].sort_values("iteration")
        values = feature_records["pvalue"].tolist()
        prev_frac = None
        for n_reps in ITERATION_THRESHOLDS:
            median_p, frac_sig, sig_count = summarize_prefix(values, n_reps)
            delta = np.nan if prev_frac is None else frac_sig - prev_frac
            rows.append(
                {
                    "Feature": feature,
                    "n_reps": n_reps,
                    "significant_count": sig_count,
                    "frac_sig": frac_sig,
                    "pct_sig": frac_sig * 100.0,
                    "median_p": median_p,
                    "delta_frac_sig": delta,
                    "delta_pct_sig": delta * 100.0 if not np.isnan(delta) else np.nan,
                    "abs_delta_pct_sig": abs(delta) * 100.0 if not np.isnan(delta) else np.nan,
                }
            )
            prev_frac = frac_sig
    return pd.DataFrame(rows)


def plot_convergence_curve(ax: plt.Axes, df: pd.DataFrame, title: str, color: str) -> None:
    ax.plot(
        df["n_reps"],
        df["frac_sig"],
        marker="o",
        markersize=5.5,
        linewidth=2.4,
        color=color,
    )
    ax.set_xscale("log")
    ax.set_title(title, fontweight="semibold", fontsize=11)
    ax.set_xticks(list(ITERATION_THRESHOLDS))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{int(value):d}" if value >= 1 else ""))
    ax.set_ylim(0, 1)
    ax.axhline(0.05, linestyle="--", linewidth=1.0, color="0.55", alpha=0.45, zorder=0)
    ax.grid(True, which="major", alpha=0.22, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_convergence_grid(convergence_df: pd.DataFrame, save_path: Path) -> None:
    feature_specs = [
        ("Segment composition", "Segment composition", "#8A2BE2"),
        ("DVG length", "DVG length", "#2F80ED"),
        ("Direct repeats", "Direct repeats", "#E45756"),
        ("Nucleotide enrichment /\ncontext", "Nucleotide enrichment/context", "#4E9F3D"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14.5, 9.2), sharex=True, sharey=True)
    axes_flat = axes.ravel()

    for ax, (display_title, feature_key, color) in zip(axes_flat, feature_specs):
        feature_df = convergence_df[convergence_df["Feature"] == feature_key].sort_values("n_reps")
        plot_convergence_curve(ax, feature_df, display_title, color)
        ax.tick_params(axis="both", labelsize=10)

    axes[1, 0].set_xlabel("")
    axes[1, 1].set_xlabel("")
    axes[0, 0].set_ylabel("")
    axes[1, 0].set_ylabel("")

    fig.suptitle(
        "Convergence of balanced tests across increasing iterations",
        fontweight="semibold",
        fontsize=15,
        y=0.98,
    )
    fig.supxlabel("Number of iterations", fontsize=13, y=0.04)
    fig.supylabel("Cumulative proportion of p < 0.05", fontsize=13, x=0.03)
    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.1, top=0.9, wspace=0.22, hspace=0.32)
    fig.savefig(save_path, dpi=320, bbox_inches="tight")
    plt.close(fig)


def build_dataset_table_rows(raw_dfs: Dict[str, pd.DataFrame]) -> List[tuple]:
    rows = []
    for name in GROUP_ORDER:
        raw_df = raw_dfs[name]
        rows.append(
            (
                name,
                str(len(raw_df)),
                dominant_segments(raw_df),
                f"{summarise_lengths(raw_df):.1f} nt",
            )
        )
    return rows


def dataset_overview_df(raw_dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    return pd.DataFrame(
        build_dataset_table_rows(raw_dfs),
        columns=["Dataset", "Events", "Dominant segments", "Median DVG length"],
    )


def build_balanced_summary(raw_dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    min_n = min(len(df) for df in raw_dfs.values())
    rng = np.random.default_rng(RNG_SEED)

    records = []
    for _ in range(SUMMARY_REPS):
        sampled = {
            name: df.sample(
                n=min_n,
                replace=False,
                random_state=int(rng.integers(0, 1_000_000_000)),
            )
            for name, df in raw_dfs.items()
        }

        seg_table = np.vstack([segment_counts(sampled[name]).to_numpy(dtype=int) for name in GROUP_ORDER])
        seg_table = seg_table[:, seg_table.sum(axis=0) > 0]
        seg_p = stats.chi2_contingency(seg_table)[1]

        len_groups = [dvg_deletion_lengths(sampled[name]) for name in GROUP_ORDER]
        len_p = stats.kruskal(*len_groups)[1]

        rep_by_name = {name: get_repeat_counts(sampled[name]) for name in GROUP_ORDER}
        rep_keys = sorted(set().union(*[d.keys() for d in rep_by_name.values()]))
        rep_table = np.vstack([[int(rep_by_name[name].get(k, 0)) for k in rep_keys] for name in GROUP_ORDER])
        rep_table = rep_table[:, rep_table.sum(axis=0) > 0]
        rep_p = stats.chi2_contingency(rep_table)[1]

        nuc_by_name = {name: get_nucleotide_counts(sampled[name]) for name in GROUP_ORDER}
        nuc_table = np.vstack([nuc_by_name[name] for name in GROUP_ORDER])
        nuc_table = nuc_table[:, nuc_table.sum(axis=0) > 0]
        nuc_p = stats.chi2_contingency(nuc_table)[1]

        records.append({"Feature": "Segment composition", "pvalue": float(seg_p)})
        records.append({"Feature": "DVG length", "pvalue": float(len_p)})
        records.append({"Feature": "Direct repeats", "pvalue": float(rep_p)})
        records.append({"Feature": "Nucleotide enrichment/context", "pvalue": float(nuc_p)})

    summary_rows = []
    for molecular_feature, feature, interpretation in [
        (
            "Viral segment composition",
            "Segment composition",
            "The strongest and most stable difference. PB1 and PA dominate all groups, while non-related also carries noticeable PB2 and minor NS/HA/NP signal.",
        ),
        (
            "",
            "DVG length",
            "Length differences remain present but are less stable after balancing, suggesting that some of the unbalanced contrast is driven by sample-size effects.",
        ),
        (
            "",
            "Direct repeats",
            "No robust separation is retained after balancing; short repeat classes appear broadly conserved across the four datasets.",
        ),
        (
            "",
            "Nucleotide enrichment/context",
            "The junction-proximal nucleotide pattern is highly conserved across groups, with no stable group-specific enrichment signal.",
        ),
    ]:
        values = np.array([r["pvalue"] for r in records if r["Feature"] == feature], dtype=float)
        summary_rows.append(
            {
                "Molecular feature": molecular_feature,
                "Feature": feature,
                "Median p-value": float(np.median(values)),
                "Significant runs": f"{(np.sum(values < 0.05) / SUMMARY_REPS) * 100:.2f}%",
                "Interpretation": interpretation,
            }
        )
    return pd.DataFrame(summary_rows).fillna("")


def p_to_stars(pvalue: float) -> str:
    if pvalue < 0.001:
        return "***"
    if pvalue < 0.01:
        return "**"
    if pvalue < 0.05:
        return "*"
    return "n.s."


def plot_segment_composition(raw_dfs: Dict[str, pd.DataFrame]) -> Path:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.size": 18,
            "axes.titlesize": 34,
            "axes.labelsize": 24,
            "xtick.labelsize": 22,
            "ytick.labelsize": 22,
            "legend.fontsize": 19,
            "legend.title_fontsize": 21,
            "axes.linewidth": 1.2,
            "xtick.major.width": 1.2,
            "ytick.major.width": 1.2,
        }
    )

    count_df = pd.DataFrame({name: segment_counts(df) for name, df in raw_dfs.items()}).T.loc[GROUP_ORDER]
    prop_df = count_df.div(count_df.sum(axis=1), axis=0) * 100

    segment_pvalues = []
    totals = count_df.sum(axis=1).to_numpy(dtype=int)
    for idx, segment in enumerate(SEGMENTS):
        segment_counts_arr = count_df[segment].to_numpy(dtype=int)
        pvalue = permutation_pvalue(segment_counts_arr, totals, n_perm=2000, seed=42 + idx)
        segment_pvalues.append(pvalue)

    fig, ax_bar = plt.subplots(figsize=(19.5, 9.6))
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.12, top=0.81)

    x = np.arange(len(SEGMENTS))
    n_groups = len(GROUP_ORDER)
    width = 0.18
    offsets = (np.arange(n_groups) - (n_groups - 1) / 2) * width

    for i, name in enumerate(GROUP_ORDER):
        ax_bar.bar(
            x + offsets[i],
            prop_df.loc[name, SEGMENTS].to_numpy(),
            width=width,
            label=name,
            color=GROUP_COLORS[name],
            edgecolor="black",
            linewidth=1.1,
        )

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(SEGMENTS)
    ax_bar.set_ylabel("DVGs per dataset (%)")
    ax_bar.set_ylim(0, max(100, float(prop_df.to_numpy().max()) * 1.25))
    ax_bar.tick_params(axis="both", labelsize=22)
    legend = ax_bar.legend(
        title="Granger-labeled datasets",
        loc="upper right",
        ncol=2,
        frameon=True,
        borderpad=1.0,
        labelspacing=0.7,
        handletextpad=0.9,
        columnspacing=1.8,
    )
    legend.get_title().set_fontsize(21)
    for text in legend.get_texts():
        text.set_fontsize(19)
    fig.suptitle("Segment composition across Granger-labeled datasets", fontweight="bold", fontsize=34, y=0.985)
    fig.text(
        0.5,
        0.895,
        "Asterisks indicate segment-wise significance from a permutation chi-square test across groups "
        "(* p<0.05, ** p<0.01, *** p<0.001).",
        ha="center",
        va="top",
        fontsize=20,
    )

    max_height = float(prop_df.max().max())
    for idx, (segment, pvalue) in enumerate(zip(SEGMENTS, segment_pvalues)):
        stars = p_to_stars(pvalue)
        y = float(prop_df[segment].max()) + 2.2
        ax_bar.text(
            x[idx],
            y,
            stars,
            ha="center",
            va="bottom",
            fontsize=20,
            fontweight="bold",
            color="black" if stars != "n.s." else "0.35",
        )
    ax_bar.set_ylim(0, max(100, max_height + 10))

    RESULTS.mkdir(parents=True, exist_ok=True)
    out_file = RESULTS / "segment_counts_4panel.png"
    out_pdf = RESULTS / "segment_counts_4panel.pdf"
    out_svg = RESULTS / "segment_counts_4panel.svg"
    fig.savefig(out_file, dpi=300, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(out_svg, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return out_file


def add_table(document: Document, headers: List[str], rows: List[tuple]) -> None:
    table = document.add_table(rows=1, cols=len(headers))
    table.style = "Table Grid"
    for idx, header in enumerate(headers):
        table.rows[0].cells[idx].text = header
    for row in rows:
        cells = table.add_row().cells
        for idx, value in enumerate(row):
            cells[idx].text = str(value)


def add_image(document: Document, image_path: Path, caption: str) -> None:
    paragraph = document.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.add_run().add_picture(str(image_path), width=Inches(6.5))
    caption_p = document.add_paragraph()
    caption_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = caption_p.add_run(caption)
    run.italic = True


def main() -> None:
    raw_dfs = {name: load_custom_dataset(path) for name, path in DATASETS.items()}
    records_df, _ = run_balanced_sampling(raw_dfs, CONVERGENCE_REPS)
    balanced_summary = build_balanced_summary_from_records(records_df, SUMMARY_REPS)
    overview_df = dataset_overview_df(raw_dfs)
    plot_path = plot_segment_composition(raw_dfs)

    RESULTS.mkdir(parents=True, exist_ok=True)
    balanced_summary.to_csv(RESULTS / "feature_summary.csv", index=False, na_rep="")
    overview_df.to_csv(RESULTS / "dataset_overview.csv", index=False)
    records_df.to_csv(RESULTS / "feature_pvalues_by_iteration.csv", index=False)

    convergence_df = build_convergence_summary(records_df)
    convergence_df.to_csv(RESULTS / "feature_convergence_summary.csv", index=False)

    conv_dir = RESULTS / "feature_convergence_plots"
    conv_dir.mkdir(parents=True, exist_ok=True)
    plot_convergence_grid(convergence_df, conv_dir / "convergence_grid_2x2.png")

    document = Document()
    set_default_font(document)

    title = document.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = title.add_run("Custom v2 dataset comparison report")
    run.bold = True
    run.font.size = Pt(18)

    subtitle = document.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = subtitle.add_run("Comparison of bi-directional, caused, causing, and non-related DelVG datasets")
    run.italic = True
    run.font.size = Pt(11)

    document.add_paragraph(
        "This report re-runs the PR8 custom comparison workflow on the four custom_v2 datasets. "
        "Because these files only contain Segment/Start/End coordinates, the analysis focuses on "
        "segment composition and DVG length. The balanced summary table below uses 100 repetitions."
    )

    add_heading(document, "Dataset overview", level=1)
    add_table(
        document,
        ["Dataset", "Events", "Dominant segments", "Median DVG length"],
        [tuple(row) for row in overview_df.itertuples(index=False, name=None)],
    )

    add_heading(document, "Feature summary", level=1)
    add_table(
        document,
        ["Feature", "Median p-value", "Significant runs", "Interpretation"],
        [
            (
                row["Feature"],
                f'{row["Median p-value"]:.6f}',
                row["Significant runs"],
                row["Interpretation"],
            )
            for _, row in balanced_summary.iterrows()
        ],
    )

    add_heading(document, "Segment composition", level=1)
    add_paragraph(
        document,
        "The plot below reproduces the PR8-style segment composition figure for the four custom_v2 datasets.",
    )
    add_image(
        document,
        plot_path,
        "Segment composition across Granger datasets. Stars indicate chi-square test significance across groups.",
    )

    document.save(str(OUTFILE))

    print(OUTFILE)
    print(RESULTS / "dataset_overview.csv")
    print(RESULTS / "feature_summary.csv")
    print(plot_path)


if __name__ == "__main__":
    main()
