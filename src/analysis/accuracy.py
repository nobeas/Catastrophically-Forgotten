"""Plotting helpers for training/validation metrics."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from src.data import get_plotting_color


def plot_results(results_dict, num_classes=10, ax=None):
    """Plot losses and accuracies across training epochs."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 3.5))

    chance = 100 / num_classes
    plotted = False
    acc_ax = None

    for result_type in ["losses", "accuracies"]:
        for dataset in ["train", "valid"]:
            key = f"avg_{dataset}_{result_type}"
            if key not in results_dict:
                continue

            if result_type == "losses":
                ylabel = "Loss"
                plot_ax = ax
                ls = None
            else:
                if acc_ax is None:
                    acc_ax = ax.twinx()
                    acc_ax.spines[["right"]].set_visible(True)
                    acc_ax.axhline(chance, ls="dashed", color="k", alpha=0.8)
                    acc_ax.set_ylim(-5, 105)
                ylabel = "Accuracy (%)"
                plot_ax = acc_ax
                ls = "dashed"

            data = results_dict[key]
            plot_ax.plot(data, ls=ls, label=dataset, alpha=0.8, color=get_plotting_color(dataset))
            plot_ax.set_ylabel(ylabel)
            plotted = True

    if not plotted:
        raise RuntimeError("No data found to plot.")

    ax.legend(loc="center left")
    ax.set_xticks(range(len(data)))
    ax.set_xticklabels([f"{int(e)}" for e in range(len(data))])
    ax.set_title("Performance across learning")
    ax.set_xlabel("Epoch")
    return ax


def plot_scores_per_class(results_dict, num_classes=10, ax=None):
    """Plot accuracy per class for the final epoch."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 3))

    ax.set_prop_cycle(None)
    for s, dataset in enumerate(["train", "valid"]):
        correct_by_class = results_dict[f"{dataset}_correct_by_class"]
        seen_by_class = results_dict[f"{dataset}_seen_by_class"]
        xs = []
        ys = []
        for i, total in seen_by_class.items():
            xs.append(i + 0.3 * (s - 0.5))
            ys.append(np.nan if total == 0 else 100 * correct_by_class[i] / total)

        avg_key = f"avg_{dataset}_accuracies"
        if avg_key in results_dict:
            ax.axhline(results_dict[avg_key][-1], ls="dashed", alpha=0.8, color=get_plotting_color(dataset))

        ax.bar(xs, ys, label=dataset, width=0.3, alpha=0.8, color=get_plotting_color(dataset))

    ax.set_xticks(range(num_classes))
    ax.set_xlabel("Class")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Class scores")
    ax.set_ylim(-5, 105)
    chance = 100 / num_classes
    ax.axhline(chance, ls="dashed", color="k", alpha=0.8)
    ax.legend()
    return ax
