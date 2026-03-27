from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def save_figure(fig, save_path: Path, dpi: int = 200) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

def save_training_curves(history, model_name: str, output_dir: Path) -> None:
    """
    Training loss curve
    Training metric curve

    Tries binary_accuracy first, then precision, then recall
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    history_dict = history.history

    # Loss curve
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history_dict.get("loss", []), label="train_loss")
    if "val_loss" in history_dict:
        ax.plot(history_dict["val_loss"], label="val_loss")
    ax.set_title(f"Training Loss - {model_name}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    save_figure(fig, output_dir / f"training_loss.png")

    # Metric curve
    metric_name = None
    for candidate in ["binary_accuracy", "precision", "recall"]:
        if candidate in history_dict:
            metric_name = candidate
            break

    if metric_name is not None:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(history_dict[metric_name], label=f"train_{metric_name}")
        val_metric_name = f"val_{metric_name}"
        if val_metric_name in history_dict:
            ax.plot(history_dict[val_metric_name], label=val_metric_name)

        ax.set_title(f"Training Metric - {model_name} ({metric_name})")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric_name)
        ax.legend()
        save_figure(fig, output_dir / f"training_{metric_name}.png")

def plot_per_label_f1(
    per_label_df: pd.DataFrame,
    model_name: str,
    save_path: Path,
) -> None:
    df = per_label_df.sort_values("f1", ascending=True)

    fig, ax = plt.subplots(figsize=(10, max(8, len(df) * 0.35)))
    ax.barh(df["label"], df["f1"])
    ax.set_title(f"Per-Label F1 - {model_name}")
    ax.set_xlabel("F1")
    ax.set_ylabel("Emotion")
    save_figure(fig, save_path)

def plot_truth_table_chart(
    truth_table_df: pd.DataFrame,
    model_name: str,
    save_path: Path,
) -> None:
    """
    One big chart showing TP / FP / FN / TN for all labels
    """
    df = truth_table_df.set_index("label")[["TP", "FP", "FN", "TN"]]

    fig, ax = plt.subplots(figsize=(10, max(8, len(df) * 0.35)))
    im = ax.imshow(df.values, aspect="auto")

    ax.set_title(f"Per-Label Truth Table Counts - {model_name}")
    ax.set_xlabel("Count Type")
    ax.set_ylabel("Emotion")

    ax.set_xticks(np.arange(len(df.columns)))
    ax.set_xticklabels(df.columns)
    ax.set_yticks(np.arange(len(df.index)))
    ax.set_yticklabels(df.index)

    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            ax.text(j, i, str(df.iloc[i, j]), ha="center", va="center", fontsize=8)

    fig.colorbar(im, ax=ax)
    save_figure(fig, save_path)