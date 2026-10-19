"""Exploratory Data Analysis for the heart disease dataset.

This script is notebook-style and can be run interactively or as a regular Python file.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd  # noqa: F401
import seaborn as sns

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.model_pipeline import DATA_PATH, load_heart_data  # noqa: E402


def run_eda() -> None:
    df = load_heart_data(DATA_PATH)
    print(df.head())
    print(df.describe())
    print(df["target"].value_counts(normalize=True))

    output_dir = ROOT_DIR / "docs" / "screenshots"
    output_dir.mkdir(parents=True, exist_ok=True)

    numeric_cols = [
        col for col in df.select_dtypes(include="number").columns if col != "target"
    ]

    sns.set_theme(style="whitegrid")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    df["age"].plot.hist(ax=axes[0, 0], bins=20, edgecolor="black")
    axes[0, 0].set_title("Age distribution")
    df["chol"].plot.hist(ax=axes[0, 1], bins=20, edgecolor="black")
    axes[0, 1].set_title("Cholesterol distribution")
    df["thalach"].plot.hist(ax=axes[1, 0], bins=20, edgecolor="black")
    axes[1, 0].set_title("Maximum heart rate")
    sns.countplot(data=df, x="target", ax=axes[1, 1])
    axes[1, 1].set_title("Class balance")
    plt.tight_layout()
    fig.savefig(output_dir / "eda_summary.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    correlation = df.corr(numeric_only=True)
    corr_fig, corr_ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(correlation, cmap="coolwarm", annot=False, ax=corr_ax)
    corr_ax.set_title("Feature correlation matrix")
    corr_fig.tight_layout()
    corr_fig.savefig(output_dir / "feature_correlation.png", dpi=300, bbox_inches="tight")
    plt.close(corr_fig)

    box_fig, box_ax = plt.subplots(figsize=(14, 8))
    sns.boxplot(data=df[numeric_cols], orient="h", palette="Set2", ax=box_ax)
    box_ax.set_title("Box plots for continuous features")
    box_fig.tight_layout()
    box_fig.savefig(output_dir / "feature_boxplots.png", dpi=300, bbox_inches="tight")
    plt.close(box_fig)

    selected_features = ["age", "chol", "thalach", "oldpeak"]
    target_box_fig, target_box_ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(
        data=df.melt(id_vars="target", value_vars=selected_features),
        x="variable",
        y="value",
        hue="target",
        ax=target_box_ax,
    )
    target_box_ax.set_title("Feature distributions by heart disease target")
    target_box_box = target_box_ax.get_figure()
    target_box_box.tight_layout()
    target_box_box.savefig(output_dir / "target_boxplots.png", dpi=300, bbox_inches="tight")
    plt.close(target_box_box)

    scatter_fig, scatter_axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.scatterplot(data=df, x="age", y="chol", hue="target", palette="deep", ax=scatter_axes[0])
    scatter_axes[0].set_title("Age vs cholesterol by target")
    sns.scatterplot(data=df, x="oldpeak", y="thalach", hue="target", palette="deep", ax=scatter_axes[1])
    scatter_axes[1].set_title("Oldpeak vs maximum heart rate by target")
    scatter_fig.tight_layout()
    scatter_fig.savefig(output_dir / "feature_scatterplots.png", dpi=300, bbox_inches="tight")
    plt.close(scatter_fig)

    pairplot_cols = ["age", "chol", "thalach", "oldpeak", "target"]
    pairplot = sns.pairplot(df[pairplot_cols], hue="target", diag_kind="kde", height=2.2)
    pairplot.fig.tight_layout()
    pairplot.fig.savefig(output_dir / "feature_pairplot.png", dpi=300, bbox_inches="tight")
    plt.close(pairplot.fig)

    q1 = df[numeric_cols].quantile(0.25)
    q3 = df[numeric_cols].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outlier_counts = {
        col: int(((df[col] < lower[col]) | (df[col] > upper[col])).sum())
        for col in numeric_cols
    }
    outlier_summary = {k: v for k, v in outlier_counts.items() if v > 0}
    print("Outlier summary (IQR rule):")
    for col, count in outlier_summary.items():
        print(f"- {col}: {count} values outside expected range")
    if not outlier_summary:
        print("- No strong outlier signals detected by the IQR rule.")

    print(f"EDA plots saved to: {output_dir}")


if __name__ == "__main__":
    run_eda()
