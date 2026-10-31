"""Publication-quality Matplotlib & Seaborn plotting utilities for notebooks and Lab 8 report."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image

from src.config import CLASS_NAMES, EXTERNAL_TEST_DIR, FIGURES_DIR, SPLIT_DATA_DIR

sns.set_theme(style="whitegrid", context="notebook", font_scale=1.05)


def _save_fig(fig: plt.Figure, save_name: str | None) -> None:
    if save_name:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        out_path = FIGURES_DIR / save_name
        fig.savefig(out_path, dpi=180, bbox_inches="tight")


def plot_sample_images(
    split_dir: Path = SPLIT_DATA_DIR,
    samples_per_class: int = 5,
    save_name: str | None = "00_dataset_samples.png",
) -> None:
    """Display sample cat and dog images from the prepared training split."""
    fig, axes = plt.subplots(2, samples_per_class, figsize=(2.8 * samples_per_class, 5.6))
    for row_idx, cls_name in enumerate(CLASS_NAMES):
        files = sorted((split_dir / "train" / cls_name).glob("*.jpg"))[:samples_per_class]
        for col_idx, img_path in enumerate(files):
            ax = axes[row_idx, col_idx]
            with Image.open(img_path) as img:
                ax.imshow(img)
                w, h = img.size
            ax.set_title(f"{cls_name[:-1].upper()} ({w}×{h})\n{img_path.name}", fontsize=9)
            ax.axis("off")

    fig.suptitle("Приклади підготовлених зображень із навчальної вибірки (Dogs vs. Cats)", fontsize=13, y=1.02)
    plt.tight_layout()
    _save_fig(fig, save_name)
    plt.show()


def plot_training_curves(
    record: dict[str, Any],
    title_prefix: str | None = None,
    save_name: str | None = None,
) -> None:
    """Plot side-by-side Accuracy and Binary Cross-Entropy Loss curves with best epoch marker."""
    hist = record["history"]
    epochs = np.arange(1, len(hist["loss"]) + 1)
    best_ep = int(record.get("best_epoch", int(np.argmin(hist["val_loss"]) + 1)))
    label_title = title_prefix or f"{record['display_name']} (Dropout = {record['dropout_rate']})"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))

    ax1.plot(epochs, hist["accuracy"], "o-", label="Train Accuracy", linewidth=2, markersize=4)
    ax1.plot(epochs, hist["val_accuracy"], "s-", label="Val Accuracy", linewidth=2, markersize=4)
    ax1.axvline(best_ep, color="crimson", linestyle="--", alpha=0.7, label=f"Best Epoch ({best_ep})")
    ax1.set_title(f"{label_title} — Точність (Accuracy)")
    ax1.set_xlabel("Епоха")
    ax1.set_ylabel("Accuracy")
    ax1.legend(loc="lower right")

    ax2.plot(epochs, hist["loss"], "o-", label="Train Loss", linewidth=2, markersize=4)
    ax2.plot(epochs, hist["val_loss"], "s-", label="Val Loss", linewidth=2, markersize=4)
    ax2.axvline(best_ep, color="crimson", linestyle="--", alpha=0.7, label=f"Best Val Loss ({best_ep})")
    ax2.set_title(f"{label_title} — Функція втрат (Loss)")
    ax2.set_xlabel("Епоха")
    ax2.set_ylabel("Binary Cross-Entropy")
    ax2.legend(loc="upper right")

    plt.tight_layout()
    _save_fig(fig, save_name)
    plt.show()


def plot_dropout_comparison(
    records: list[dict[str, Any]],
    save_name: str | None = "01_vgg16_dropout_comparison.png",
) -> None:
    """Plot comprehensive 2x2 analysis of how Dropout controls overfitting in VGG16."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    palette = sns.color_palette("tab10", n_colors=len(records))

    for color, rec in zip(palette, records, strict=True):
        do_rate = rec["dropout_rate"]
        hist = rec["history"]
        epochs = np.arange(1, len(hist["loss"]) + 1)
        acc_gap = np.array(hist["accuracy"]) - np.array(hist["val_accuracy"])

        axes[0, 0].plot(epochs, hist["val_loss"], "o-", color=color, linewidth=2, markersize=4, label=f"Dropout={do_rate}")
        axes[0, 1].plot(epochs, hist["val_accuracy"], "s-", color=color, linewidth=2, markersize=4, label=f"Dropout={do_rate}")
        axes[1, 0].plot(epochs, acc_gap * 100.0, "^-", color=color, linewidth=2, markersize=4, label=f"Dropout={do_rate}")

    axes[0, 0].set_title("Оціночна помилка (Validation Loss) за епохами")
    axes[0, 0].set_xlabel("Епоха")
    axes[0, 0].set_ylabel("Val Loss (нижче = краще)")
    axes[0, 0].legend()

    axes[0, 1].set_title("Оціночна точність (Validation Accuracy) за епохами")
    axes[0, 1].set_xlabel("Епоха")
    axes[0, 1].set_ylabel("Val Accuracy (вище = краще)")
    axes[0, 1].legend()

    axes[1, 0].axhline(0.0, color="gray", linestyle="--", linewidth=1)
    axes[1, 0].set_title("Розрив перенавчання: Train Acc − Val Acc (%)")
    axes[1, 0].set_xlabel("Епоха")
    axes[1, 0].set_ylabel("Overfitting Gap (%)")
    axes[1, 0].legend()

    labels = [f"DO={r['dropout_rate']}" for r in records]
    x_pos = np.arange(len(records))
    test_accs = [r["test_metrics"]["accuracy"] * 100.0 for r in records]
    gaps = [r["overfitting_acc_gap"] * 100.0 for r in records]

    ax_bar = axes[1, 1]
    bars = ax_bar.bar(x_pos - 0.18, test_accs, width=0.35, label="Test Accuracy (%)", color="#2b7bba")
    ax_bar2 = ax_bar.twinx()
    bars2 = ax_bar2.bar(x_pos + 0.18, gaps, width=0.35, label="Overfitting Gap (%)", color="#e06666", alpha=0.85)

    ax_bar.set_xticks(x_pos)
    ax_bar.set_xticklabels(labels)
    ax_bar.set_ylim(min(test_accs) - 1.5, 100.0)
    ax_bar.set_ylabel("Test Accuracy (%)", color="#2b7bba")
    ax_bar2.set_ylabel("Final Overfitting Gap (%)", color="#c0392b")
    ax_bar2.grid(False)
    ax_bar.set_title("Підсумок: Test Accuracy проти Overfitting Gap")

    for b in bars:
        ax_bar.annotate(
            f"{b.get_height():.2f}%",
            (b.get_x() + b.get_width() / 2, b.get_height()),
            ha="center",
            va="bottom",
            fontsize=8.5,
        )

    lines_1, labels_1 = ax_bar.get_legend_handles_labels()
    lines_2, labels_2 = ax_bar2.get_legend_handles_labels()
    ax_bar.legend(lines_1 + lines_2, labels_1 + labels_2, loc="lower left")

    fig.suptitle("Дослідження впливу гіперпараметра Dropout на перенавчання (VGG16)", fontsize=14, y=1.01)
    plt.tight_layout()
    _save_fig(fig, save_name)
    plt.show()


def plot_confusion_matrix(
    cm: list[list[int]] | np.ndarray,
    model_title: str,
    save_name: str | None = None,
) -> None:
    """Render an annotated confusion matrix heatmap."""
    cm_arr = np.asarray(cm, dtype=int)
    cm_pct = cm_arr / np.maximum(cm_arr.sum(axis=1, keepdims=True), 1) * 100.0
    annot = np.array(
        [[f"{cm_arr[i, j]}\n({cm_pct[i, j]:.1f}%)" for j in range(2)] for i in range(2)]
    )

    fig, ax = plt.subplots(figsize=(5.5, 4.6))
    sns.heatmap(
        cm_arr,
        annot=annot,
        fmt="",
        cmap="Blues",
        xticklabels=["Cat (0)", "Dog (1)"],
        yticklabels=["Cat (0)", "Dog (1)"],
        cbar=False,
        ax=ax,
    )
    ax.set_xlabel("Передбачений клас (Predicted)")
    ax.set_ylabel("Істинний клас (Actual)")
    ax.set_title(f"Матриця помилок — {model_title}")
    plt.tight_layout()
    _save_fig(fig, save_name)
    plt.show()


def plot_external_predictions_grid(
    df_preds: pd.DataFrame,
    model_title: str,
    external_dir: Path = EXTERNAL_TEST_DIR,
    save_name: str | None = None,
) -> None:
    """Visualize predictions on 10 external unseen cat & dog images."""
    n = len(df_preds)
    cols = 5
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.1 * cols, 3.4 * rows))
    axes_flat = np.atleast_1d(axes).ravel()

    for idx, (_, row) in enumerate(df_preds.iterrows()):
        ax = axes_flat[idx]
        img_path = external_dir / row["rel_path"]
        with Image.open(img_path) as img:
            ax.imshow(img)
        color = "#1b7837" if row["correct"] else "#c51b7d"
        status = "OK" if row["correct"] else "ERR"
        ax.set_title(
            f"[{status}] True: {row['true_class'][:-1]} → Pred: {row['pred_class'][:-1]}\n"
            f"Conf: {row['confidence']*100:.1f}%",
            color=color,
            fontsize=9.5,
            fontweight="bold",
        )
        ax.axis("off")

    for extra_ax in axes_flat[n:]:
        extra_ax.axis("off")

    acc = df_preds["correct"].mean() * 100.0
    fig.suptitle(
        f"{model_title} — Тест на нових зовнішніх зображеннях: {acc:.1f}% ({int(df_preds['correct'].sum())}/{n})",
        fontsize=13,
        y=1.02,
    )
    plt.tight_layout()
    _save_fig(fig, save_name)
    plt.show()


def plot_models_comparison(
    df_summary: pd.DataFrame,
    save_name: str | None = "05_models_comparison_charts.png",
) -> None:
    """Plot comparative bar charts across all evaluated architectures."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    palette = sns.color_palette("viridis", n_colors=len(df_summary))

    # 1. Test Accuracy
    bars1 = axes[0].bar(df_summary["Model"], df_summary["Test Accuracy (%)"], color=palette)
    axes[0].set_title("Точність на тестовій вибірці (%)")
    axes[0].set_ylim(max(70.0, df_summary["Test Accuracy (%)"].min() - 5.0), 100.5)
    axes[0].tick_params(axis="x", rotation=20)
    for b in bars1:
        axes[0].annotate(
            f"{b.get_height():.2f}%",
            (b.get_x() + b.get_width() / 2, b.get_height()),
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # 2. Model File Size (MB)
    bars2 = axes[1].bar(df_summary["Model"], df_summary["Size (MB)"], color=palette)
    axes[1].set_title("Розмір збереженої моделі (МБ)")
    axes[1].tick_params(axis="x", rotation=20)
    for b in bars2:
        axes[1].annotate(
            f"{b.get_height():.1f}",
            (b.get_x() + b.get_width() / 2, b.get_height()),
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # 3. Total Pipeline Time (s)
    bars3 = axes[2].bar(df_summary["Model"], df_summary["Total Time (s)"], color=palette)
    axes[2].set_title("Час навчання / екстракції (сек)")
    axes[2].tick_params(axis="x", rotation=20)
    for b in bars3:
        axes[2].annotate(
            f"{b.get_height():.0f}s",
            (b.get_x() + b.get_width() / 2, b.get_height()),
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.suptitle("Порівняльний аналіз моделей (Базова CNN з Лаб 7 vs Transfer Learning)", fontsize=14, y=1.03)
    plt.tight_layout()
    _save_fig(fig, save_name)
    plt.show()

