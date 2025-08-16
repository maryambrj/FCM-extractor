import os
import math
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def find_scoring_csvs(root_dir: str) -> List[Tuple[str, str, str]]:
    """
    Walk the `fcm_outputs` directory to find all scoring CSV files and return a list of
    tuples: (model_name, dataset_id, csv_path).
    """
    results: List[Tuple[str, str, str]] = []
    if not os.path.isdir(root_dir):
        return results

    for model_name in sorted(os.listdir(root_dir)):
        model_dir = os.path.join(root_dir, model_name)
        if not os.path.isdir(model_dir):
            continue

        # Each dataset folder (e.g., BD006)
        for dataset_id in sorted(os.listdir(model_dir)):
            dataset_dir = os.path.join(model_dir, dataset_id)
            if not os.path.isdir(dataset_dir):
                continue

            expected_filename = f"{dataset_id}_scoring_results.csv"
            csv_path = os.path.join(dataset_dir, expected_filename)
            if os.path.isfile(csv_path):
                results.append((model_name, dataset_id, csv_path))

    return results


def load_all_f1(records: List[Tuple[str, str, str]]) -> pd.DataFrame:
    """
    Load all CSVs (expects column named 'F1') and return a dataframe with columns:
    at least ['model', 'dataset', 'F1'] and, when available, other metrics like
    TP, FP, FN, PP, threshold, tp_scale, pp_scale, gt_nodes, gt_edges, gen_nodes, gen_edges.
    """
    rows = []
    for model_name, dataset_id, csv_path in records:
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue

        # Heuristics: prefer 'F1' column, otherwise try lowercase variants
        f1_col = None
        for candidate in [
            "F1",
            "f1",
            "f1_score",
            "F1_score",
        ]:
            if candidate in df.columns:
                f1_col = candidate
                break
        if f1_col is None:
            continue

        # If multiple rows exist, take the first row's F1 (assuming one summary per file)
        try:
            f1_value = float(df.iloc[0][f1_col])
        except Exception:
            continue

        if math.isnan(f1_value):
            continue

        row = {"model": model_name, "dataset": dataset_id, "F1": f1_value}

        # Try to extract additional numeric metrics if present
        optional_cols = [
            "TP",
            "PP",
            "FP",
            "FN",
            "threshold",
            "tp_scale",
            "pp_scale",
            "gt_nodes",
            "gt_edges",
            "gen_nodes",
            "gen_edges",
        ]
        for col in optional_cols:
            if col in df.columns:
                try:
                    row[col] = float(df.iloc[0][col])
                except Exception:
                    pass

        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=["model", "dataset", "F1"])

    return pd.DataFrame(rows)


def _ci95_from_mean_std_n(mean_val: float, std_val: float, n: int) -> Tuple[float, float]:
    if n <= 1 or std_val <= 0:
        return (np.nan, np.nan)
    try:
        from math import sqrt
        import scipy.stats as st  # type: ignore

        t_crit = float(st.t.ppf(0.975, df=n - 1))
        margin = t_crit * std_val / sqrt(n)
    except Exception:
        from math import sqrt

        z_crit = 1.96
        margin = z_crit * std_val / sqrt(n)
    return (mean_val - margin, mean_val + margin)


def _bootstrap_ci_mean(series: pd.Series, n_bootstrap: int = 10000, alpha: float = 0.05, seed: int = 42) -> Tuple[float, float]:
    """Percentile bootstrap CI for the mean of the data in `series`."""
    s = pd.to_numeric(series.dropna().astype(float), errors="coerce").dropna()
    n = int(s.shape[0])
    if n == 0:
        return (np.nan, np.nan)
    if n == 1:
        val = float(s.iloc[0])
        return (val, val)
    rng = np.random.default_rng(seed)
    # Vectorized bootstrap resampling
    idx = rng.integers(0, n, size=(n_bootstrap, n))
    samples = s.values[idx]
    boot_means = samples.mean(axis=1)
    lower = float(np.quantile(boot_means, alpha / 2))
    upper = float(np.quantile(boot_means, 1 - alpha / 2))
    return (lower, upper)


def compute_group_stats(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=[
                group_col,
                "count",
                "mean",
                "median",
                "std",
                "min",
                "max",
                "ci95_low",
                "ci95_high",
            ]
        )

    grp = df.groupby(group_col)["F1"]
    stats = grp.agg(
        count="count",
        mean="mean",
        median="median",
        std=lambda s: float(s.std(ddof=1)) if s.shape[0] > 1 else 0.0,
        min="min",
        max="max",
    ).reset_index()

    ci_low_list: List[float] = []
    ci_high_list: List[float] = []
    boot_low_list: List[float] = []
    boot_high_list: List[float] = []
    for _, row in stats.iterrows():
        ci_low, ci_high = _ci95_from_mean_std_n(float(row["mean"]), float(row["std"]), int(row["count"]))
        ci_low_list.append(ci_low)
        ci_high_list.append(ci_high)
        # Bootstrap CI based on original group values
        group_values = grp.get_group(row[group_col]) if row[group_col] in grp.groups else pd.Series([], dtype=float)
        b_low, b_high = _bootstrap_ci_mean(group_values)
        boot_low_list.append(b_low)
        boot_high_list.append(b_high)

    stats["ci95_low"] = ci_low_list
    stats["ci95_high"] = ci_high_list
    stats["boot_ci95_low"] = boot_low_list
    stats["boot_ci95_high"] = boot_high_list
    return stats


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def plot_box_by_model(df: pd.DataFrame, out_dir: str) -> None:
    if df.empty:
        return
    # Order models by median F1
    order = (
        df.groupby("model")["F1"].median().sort_values(ascending=False).index.tolist()
    )

    plt.figure(figsize=(12, 6))
    df.boxplot(column="F1", by="model", grid=False, vert=True, rot=45)
    plt.suptitle("")
    plt.title("F1 distribution by model")
    plt.xlabel("Model")
    plt.ylabel("F1 score")
    # Reorder x-ticks according to median if possible
    try:
        ax = plt.gca()
        labels = [t.get_text() for t in ax.get_xticklabels()]
        mapping = {label: i for i, label in enumerate(labels)}
        # If labels differ from desired order, rebuild plot with pandas is non-trivial; keep default
    except Exception:
        pass
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "boxplot_f1_by_model.png"), dpi=200)
    plt.close()


def plot_bar_means_with_ci(stats_df: pd.DataFrame, out_dir: str) -> None:
    if stats_df.empty or "mean" not in stats_df.columns:
        return
    # Sort by mean descending
    stats_sorted = stats_df.sort_values("mean", ascending=False)

    x = np.arange(stats_sorted.shape[0])
    means = stats_sorted["mean"].values.astype(float)

    lower_errors: List[float] = []
    upper_errors: List[float] = []
    for _, row in stats_sorted.iterrows():
        mean_val = float(row["mean"]) if not pd.isna(row["mean"]) else 0.0
        ci_low = float(row.get("ci95_low", np.nan))
        ci_high = float(row.get("ci95_high", np.nan))
        if not np.isnan(ci_low) and not np.isnan(ci_high):
            lower_errors.append(max(0.0, mean_val - ci_low))
            upper_errors.append(max(0.0, ci_high - mean_val))
        else:
            lower_errors.append(0.0)
            upper_errors.append(0.0)

    yerr_arr = np.vstack([lower_errors, upper_errors]) if len(lower_errors) else None

    plt.figure(figsize=(12, 6))
    plt.bar(x, means, yerr=yerr_arr, capsize=4, color="#4C72B0")
    plt.xticks(x, stats_sorted.iloc[:, 0].astype(str).tolist(), rotation=45, ha="right")
    plt.ylabel("Mean F1 score")
    plt.title("Mean F1 by model (95% CI)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "bar_mean_f1_by_model.png"), dpi=200)
    plt.close()


def plot_bar_means_with_bootstrap_ci(stats_df: pd.DataFrame, out_dir: str) -> None:
    if stats_df.empty or "mean" not in stats_df.columns or "boot_ci95_low" not in stats_df.columns:
        return
    stats_sorted = stats_df.sort_values("mean", ascending=False)
    x = np.arange(stats_sorted.shape[0])
    means = stats_sorted["mean"].values.astype(float)

    lower_errors: List[float] = []
    upper_errors: List[float] = []
    for _, row in stats_sorted.iterrows():
        mean_val = float(row["mean"]) if not pd.isna(row["mean"]) else 0.0
        ci_low = float(row.get("boot_ci95_low", np.nan))
        ci_high = float(row.get("boot_ci95_high", np.nan))
        if not np.isnan(ci_low) and not np.isnan(ci_high):
            lower_errors.append(max(0.0, mean_val - ci_low))
            upper_errors.append(max(0.0, ci_high - mean_val))
        else:
            lower_errors.append(0.0)
            upper_errors.append(0.0)

    yerr_arr = np.vstack([lower_errors, upper_errors]) if len(lower_errors) else None

    plt.figure(figsize=(12, 6))
    plt.bar(x, means, yerr=yerr_arr, capsize=4, color="#55A868")
    plt.xticks(x, stats_sorted.iloc[:, 0].astype(str).tolist(), rotation=45, ha="right")
    plt.ylabel("Mean F1 score")
    plt.title("Mean F1 by model (bootstrap 95% CI)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "bar_mean_f1_by_model_bootstrap.png"), dpi=200)
    plt.close()


def plot_bar_means_with_both_ci(stats_df: pd.DataFrame, out_dir: str) -> None:
    """
    Overlay plot: Bars for means; two errorbar layers for normal (t-based) and
    bootstrap percentile 95% CIs.
    """
    required_cols = {"mean", "ci95_low", "ci95_high", "boot_ci95_low", "boot_ci95_high"}
    if stats_df.empty or not required_cols.issubset(stats_df.columns):
        return

    stats_sorted = stats_df.sort_values("mean", ascending=False)
    x = np.arange(stats_sorted.shape[0])
    means = stats_sorted["mean"].values.astype(float)

    # Normal CI errors
    n_lower = []
    n_upper = []
    # Bootstrap CI errors
    b_lower = []
    b_upper = []
    for _, row in stats_sorted.iterrows():
        m = float(row["mean"]) if not pd.isna(row["mean"]) else 0.0
        # Normal
        n_lo = float(row.get("ci95_low", np.nan))
        n_hi = float(row.get("ci95_high", np.nan))
        if not np.isnan(n_lo) and not np.isnan(n_hi):
            n_lower.append(max(0.0, m - n_lo))
            n_upper.append(max(0.0, n_hi - m))
        else:
            n_lower.append(0.0)
            n_upper.append(0.0)
        # Bootstrap
        b_lo = float(row.get("boot_ci95_low", np.nan))
        b_hi = float(row.get("boot_ci95_high", np.nan))
        if not np.isnan(b_lo) and not np.isnan(b_hi):
            b_lower.append(max(0.0, m - b_lo))
            b_upper.append(max(0.0, b_hi - m))
        else:
            b_lower.append(0.0)
            b_upper.append(0.0)

    plt.figure(figsize=(12, 6))
    # Bars for means
    plt.bar(x, means, color="#A6CEE3")
    # Overlay errorbars: normal (darker), bootstrap (orange) with slight cap differences
    plt.errorbar(
        x,
        means,
        yerr=[n_lower, n_upper],
        fmt="none",
        ecolor="#1F78B4",
        elinewidth=1.5,
        capsize=5,
        label="Normal 95% CI",
    )
    plt.errorbar(
        x,
        means,
        yerr=[b_lower, b_upper],
        fmt="none",
        ecolor="#FF7F00",
        elinewidth=1.5,
        capsize=3,
        label="Bootstrap 95% CI",
    )

    plt.xticks(x, stats_sorted.iloc[:, 0].astype(str).tolist(), rotation=45, ha="right")
    plt.ylabel("Mean F1 score")
    plt.title("Mean F1 by model – Normal vs Bootstrap 95% CIs")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "bar_mean_f1_by_model_overlay.png"), dpi=200)
    plt.close()


def plot_heatmap(df: pd.DataFrame, out_dir: str) -> None:
    if df.empty:
        return
    pivot = df.pivot_table(index="dataset", columns="model", values="F1", aggfunc="first")
    if pivot.empty:
        return

    plt.figure(figsize=(max(8, 0.4 * pivot.shape[1] + 4), max(6, 0.3 * pivot.shape[0] + 3)))
    # Simple heatmap using imshow; annotate axes with labels
    data = pivot.values.astype(float)
    im = plt.imshow(data, aspect="auto", cmap="viridis")
    plt.colorbar(im, fraction=0.046, pad=0.04, label="F1 score")
    plt.xticks(ticks=np.arange(pivot.shape[1]), labels=pivot.columns.astype(str), rotation=45, ha="right")
    plt.yticks(ticks=np.arange(pivot.shape[0]), labels=pivot.index.astype(str))
    plt.title("F1 heatmap (datasets × models)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "heatmap_f1_datasets_by_models.png"), dpi=200)
    plt.close()


def plot_per_dataset_bars(df: pd.DataFrame, out_dir: str) -> None:
    """For each dataset, plot a bar chart of F1 across models."""
    if df.empty:
        return
    target_dir = os.path.join(out_dir, "per_dataset_bars")
    ensure_dir(target_dir)
    for dataset_id, sub in df.groupby("dataset"):
        sub_sorted = sub.sort_values("F1", ascending=False)
        x = np.arange(sub_sorted.shape[0])
        plt.figure(figsize=(12, 5))
        plt.bar(x, sub_sorted["F1"].values.astype(float), color="#55A868")
        plt.xticks(x, sub_sorted["model"].astype(str).tolist(), rotation=45, ha="right")
        plt.ylabel("F1 score")
        plt.title(f"F1 by model – {dataset_id}")
        plt.tight_layout()
        plt.savefig(os.path.join(target_dir, f"bar_f1_by_model_{dataset_id}.png"), dpi=200)
        plt.close()


def plot_per_model_combined_metrics(df: pd.DataFrame, out_dir: str) -> None:
    """
    For each model, create a combined plot that shows mean counts (TP, FP, FN, PP)
    as bars on the left y-axis and mean F1 on the right y-axis.
    """
    if df.empty:
        return

    required_count_cols = ["TP", "FP", "FN", "PP"]
    if not any(col in df.columns for col in required_count_cols):
        return

    target_dir = os.path.join(out_dir, "per_model_combined_metrics")
    ensure_dir(target_dir)

    for model_name, sub in df.groupby("model"):
        # Compute means for metrics
        means = {}
        for col in required_count_cols:
            if col in sub.columns:
                try:
                    means[col] = float(sub[col].astype(float).mean())
                except Exception:
                    means[col] = np.nan
            else:
                means[col] = np.nan
        f1_mean = float(sub["F1"].astype(float).mean()) if "F1" in sub.columns else np.nan

        labels = required_count_cols
        values = [means[c] for c in labels]

        fig, ax1 = plt.subplots(figsize=(8, 5))
        ax1.bar(labels, values, color=["#4C72B0", "#C44E52", "#DD8452", "#55A868"])
        ax1.set_ylabel("Mean count (TP/FP/FN/PP)")
        ax1.set_ylim(bottom=0)
        ax1.set_title(f"Combined metrics – {model_name}")

        ax2 = ax1.twinx()
        ax2.plot([-0.5, len(labels) - 0.5], [f1_mean, f1_mean], color="#8172B2", linestyle="--", label="Mean F1")
        ax2.scatter([len(labels) / 2], [f1_mean], color="#8172B2")
        ax2.set_ylabel("Mean F1")
        ax2.set_ylim(0, 1)

        ax1.grid(axis="y", linestyle=":", alpha=0.5)
        fig.tight_layout()
        fig.savefig(os.path.join(target_dir, f"combined_metrics_{model_name}.png"), dpi=200)
        plt.close(fig)


def plot_metric_correlation_heatmap(df: pd.DataFrame, out_dir: str) -> None:
    """Compute and plot the correlation heatmap among numeric metrics including F1."""
    if df.empty:
        return
    metric_cols = [
        col
        for col in [
            "F1",
            "TP",
            "FP",
            "FN",
            "PP",
            "threshold",
            "tp_scale",
            "pp_scale",
            "gt_nodes",
            "gt_edges",
            "gen_nodes",
            "gen_edges",
        ]
        if col in df.columns
    ]
    if len(metric_cols) < 2:
        return
    sub = df[metric_cols].copy()
    for c in metric_cols:
        sub[c] = pd.to_numeric(sub[c], errors="coerce")
    corr = sub.corr(method="spearman")

    plt.figure(figsize=(1.5 * len(metric_cols) + 2, 1.2 * len(metric_cols) + 2))
    im = plt.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(im, fraction=0.046, pad=0.04, label="Spearman corr")
    plt.xticks(ticks=np.arange(len(metric_cols)), labels=metric_cols, rotation=45, ha="right")
    plt.yticks(ticks=np.arange(len(metric_cols)), labels=metric_cols)
    plt.title("Correlation heatmap across metrics")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "correlation_heatmap_metrics.png"), dpi=200)
    plt.close()


def plot_best_model_per_dataset(df: pd.DataFrame, out_dir: str) -> None:
    """
    Single chart: x-axis datasets, bar height = best F1, bar color = best model.
    Legend maps colors to models.
    """
    if df.empty or "F1" not in df.columns:
        return

    # Compute best per dataset
    best = (
        df.sort_values(["dataset", "F1"], ascending=[True, False])
        .groupby("dataset", as_index=False)
        .first()[["dataset", "model", "F1"]]
    )
    if best.empty:
        return

    # Stable ordering by dataset id
    best = best.sort_values("dataset")

    # Assign distinct colors per model
    models = best["model"].unique().tolist()
    # Use tab20 colors then cycle if needed
    cmap = plt.get_cmap("tab20")
    colors = [cmap(i % 20) for i in range(len(models))]
    model_to_color = {m: colors[i] for i, m in enumerate(models)}

    x = np.arange(best.shape[0])
    heights = best["F1"].astype(float).values
    bar_colors = [model_to_color[m] for m in best["model"].tolist()]

    plt.figure(figsize=(max(12, 0.6 * len(x) + 4), 6))
    plt.bar(x, heights, color=bar_colors)
    plt.xticks(x, best["dataset"].astype(str).tolist(), rotation=45, ha="right")
    plt.ylabel("Best F1 per dataset")
    plt.title("Best model per dataset (color-coded by model)")

    # Build legend
    from matplotlib.patches import Patch

    handles = [Patch(facecolor=model_to_color[m], label=m) for m in models]
    plt.legend(handles=handles, title="Model", bbox_to_anchor=(1.02, 1), loc="upper left")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "best_model_per_dataset.png"), dpi=200)
    plt.close()


def plot_per_model_all_datasets(df: pd.DataFrame, out_dir: str) -> None:
    """For each model, bar plot of F1 across all datasets."""
    if df.empty or "F1" not in df.columns:
        return
    target_dir = os.path.join(out_dir, "per_model_bars")
    ensure_dir(target_dir)

    for model_name, sub in df.groupby("model"):
        sub = sub.sort_values("dataset")
        x = np.arange(sub.shape[0])
        plt.figure(figsize=(max(10, 0.5 * len(x) + 3), 5))
        plt.bar(x, sub["F1"].astype(float).values, color="#4C72B0")
        plt.xticks(x, sub["dataset"].astype(str).tolist(), rotation=45, ha="right")
        plt.ylim(0, 1)
        plt.ylabel("F1 score")
        plt.title(f"F1 across datasets – {model_name}")
        plt.tight_layout()
        plt.savefig(os.path.join(target_dir, f"bar_f1_by_dataset_{model_name}.png"), dpi=200)
        plt.close()


def main() -> None:
    repo_root = os.path.dirname(os.path.abspath(__file__))
    outputs_root = os.path.join(repo_root, "fcm_outputs")
    analysis_out = os.path.join(repo_root, "fcm_outputs_analysis")
    ensure_dir(analysis_out)

    csv_records = find_scoring_csvs(outputs_root)
    df = load_all_f1(csv_records)

    if df.empty:
        print("No F1 scores found. Exiting.")
        return

    # Save raw aggregated data
    df.to_csv(os.path.join(analysis_out, "all_f1_scores.csv"), index=False)

    # Per-model stats
    per_model = compute_group_stats(df, "model")
    per_model.to_csv(os.path.join(analysis_out, "per_model_stats.csv"), index=False)

    # Per-dataset stats
    per_dataset = compute_group_stats(df, "dataset")
    per_dataset.to_csv(os.path.join(analysis_out, "per_dataset_stats.csv"), index=False)

    # Plots
    plot_box_by_model(df, analysis_out)
    plot_bar_means_with_ci(per_model, analysis_out)
    plot_bar_means_with_bootstrap_ci(per_model, analysis_out)
    plot_bar_means_with_both_ci(per_model, analysis_out)
    plot_heatmap(df, analysis_out)
    plot_per_dataset_bars(df, analysis_out)
    plot_per_model_combined_metrics(df, analysis_out)
    plot_metric_correlation_heatmap(df, analysis_out)
    plot_best_model_per_dataset(df, analysis_out)
    plot_per_model_all_datasets(df, analysis_out)

    # Extra: Best model per dataset CSV
    best_rows = []
    for dataset_id, sub in df.groupby("dataset"):
        best = sub.sort_values("F1", ascending=False).iloc[0]
        best_rows.append({
            "dataset": dataset_id,
            "best_model": best["model"],
            "best_F1": float(best["F1"]),
        })
    pd.DataFrame(best_rows).to_csv(os.path.join(analysis_out, "best_model_per_dataset.csv"), index=False)

    print(f"Wrote analysis to: {analysis_out}")


if __name__ == "__main__":
    main()


