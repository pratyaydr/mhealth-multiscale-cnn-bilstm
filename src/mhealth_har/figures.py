"""Publication figures derived only from saved original aggregate evidence."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def make_figures(evidence: str | Path, destination: str | Path) -> list[Path]:
    """Render the original hybrid confusion counts, history, and observed errors."""
    evidence, destination = Path(evidence), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    hybrid = evidence / "cnn_multiscale_bilstm_11class"
    paths = []
    with plt.rc_context({"font.size": 11, "axes.titlesize": 15, "figure.facecolor": "white"}):
        matrix = pd.read_csv(hybrid / "confusion_matrix_raw.csv", index_col=0)
        counts = matrix.to_numpy()
        normalized = counts / counts.sum(axis=1, keepdims=True)
        fig, ax = plt.subplots(figsize=(11, 9), layout="constrained")
        im = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
        names = [f"{i + 1}. {name}" for i, name in enumerate(matrix.index)]
        ax.set(
            xticks=np.arange(12),
            yticks=np.arange(12),
            yticklabels=names,
            xticklabels=np.arange(1, 13),
            xlabel="Predicted activity ID",
            ylabel="True activity",
            title="Hybrid model | recorded 12-class test confusion matrix",
        )
        for i in range(12):
            for j in range(12):
                if counts[i, j]:
                    ax.text(
                        j,
                        i,
                        f"{counts[i, j]}\n{normalized[i, j]:.0%}",
                        ha="center",
                        va="center",
                        color="white" if normalized[i, j] > 0.5 else "#173c63",
                        fontsize=9,
                    )
        fig.colorbar(im, ax=ax, label="Fraction of true-class test windows", shrink=0.8)
        path = destination / "hybrid_confusion_matrix.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(path)
        history = pd.read_csv(hybrid / "training_history.csv")
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), layout="constrained")
        for ax, keys, title, ylabel in (
            (axes[0], ("loss", "val_loss"), "Regularized cross-entropy", "Loss"),
            (axes[1], ("accuracy", "val_accuracy"), "Window classification accuracy", "Accuracy"),
        ):
            for key, label, color in zip(keys, ("Training", "Validation"), ("#235789", "#b65d20")):
                ax.plot(history.Epoch, history[key], label=label, color=color, linewidth=2)
            ax.set(title=title, xlabel="Epoch", ylabel=ylabel)
            ax.legend(frameon=False)
            ax.grid(alpha=0.18)
            if ylabel == "Accuracy":
                ax.set_ylim(0, 1.02)
        fig.suptitle("Hybrid model | original training history")
        path = destination / "hybrid_training_history.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(path)
        errors = [
            (matrix.index[i] + " → " + matrix.columns[j], int(counts[i, j]))
            for i in range(12)
            for j in range(12)
            if i != j and counts[i, j]
        ]
        fig, ax = plt.subplots(figsize=(10, 4), layout="constrained")
        ax.barh([e[0] for e in errors][::-1], [e[1] for e in errors][::-1], color="#235789")
        ax.set(
            xlabel="Misclassified test windows (count)",
            xlim=(0, 52),
            title="All 70 hybrid test errors | original confusion counts",
        )
        for i, (_, value) in enumerate(errors[::-1]):
            ax.text(value + 0.6, i, str(value), va="center")
        ax.spines[["top", "right"]].set_visible(False)
        path = destination / "hybrid_error_analysis.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(path)
    return paths
