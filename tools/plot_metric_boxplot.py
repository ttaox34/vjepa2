# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Utility to compare per-frame metrics across multiple CSV files using box plots.

import argparse
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def parse_args():
    parser = argparse.ArgumentParser(description="Plot box plots for per-frame metric CSVs.")
    parser.add_argument("csv", nargs="+", help="Paths to CSV files produced by evaluation scripts.")
    parser.add_argument(
        "--labels",
        nargs="*",
        default=None,
        help="Optional labels for each CSV; defaults to basenames.",
    )
    parser.add_argument(
        "--value-column",
        default="loss",
        help="Column name containing the metric values (default: 'loss').",
    )
    parser.add_argument(
        "--output",
        default="metric_plot.png",
        help="Output image filename (default: metric_plot.png).",
    )
    parser.add_argument(
        "--title",
        default="Per-frame Metric Comparison",
        help="Title for the plot.",
    )
    parser.add_argument(
        "--ylab",
        default=None,
        help="Optional Y-axis label; uses value-column name if omitted.",
    )
    parser.add_argument(
        "--plot-type",
        choices=["box", "violin"],
        default="box",
        help="Plot style to use (box or violin).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.labels and len(args.labels) != len(args.csv):
        raise ValueError("Number of labels must match number of CSV files.")

    values: List[pd.Series] = []
    labels: List[str] = []
    value_col = args.value_column

    for idx, csv_path in enumerate(args.csv):
        csv_file = Path(csv_path)
        if not csv_file.exists():
            raise FileNotFoundError(csv_path)
        df = pd.read_csv(csv_file)
        if value_col not in df.columns:
            raise ValueError(f"Column '{value_col}' not found in {csv_path}; available columns: {df.columns.tolist()}")
        values.append(df[value_col].dropna())
        if args.labels:
            labels.append(args.labels[idx])
        else:
            labels.append(csv_file.stem)


    plt.figure(figsize=(max(6, len(values) * 1.5), 6))
    if args.plot_type == "box":
        sns.boxplot(data=values, orient="v")
        plt.xticks(range(len(labels)), labels)
    else:
        plot_data = []
        plot_labels = []
        for label, series in zip(labels, values):
            plot_data.extend(series.tolist())
            plot_labels.extend([label] * len(series))
        sns.violinplot(x=plot_labels, y=plot_data, cut=0, inner="quartile")
    plt.title(args.title)
    plt.ylabel(args.ylab or value_col)
    plt.grid(axis="y", linestyle="--", alpha=0.3)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved box plot to {output_path}")


if __name__ == "__main__":
    main()
