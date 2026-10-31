"""Publication-quality Matplotlib & Seaborn plotting utilities for Lab 9 notebooks and Word report."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image

from .config import CLASS_NAMES, EXTERNAL_TEST_DIR, FIGURES_DIR, SPLIT_DATA_DIR

sns.set_theme(style="whitegrid", context="notebook", font_scale=1.04)


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

    fig.suptitle(
        "Приклади валідованих зображень із навчальної вибірки (Dogs vs. Cats)",
        fontsize=13,
        y=1.02,
    )
    plt.tight_layout()
    _save_fig(fig, save_name)
    plt.show()


def plot_two_stage_curves(
    record: dict[str, Any],
    title_prefix: str | None = None,
    save_name: str | None = None,
) -> None:
    """Plot combined Stage 1 (Frozen Warm-up) + Stage 2 (Fine-Tuning) Accuracy and Loss trajectories."""
    ft_hist = record["history"]
    warmup_hist = record.get("warmup_history")
    label_title = title_prefix or f"{record['display_name']} Fine-Tuning"

    min_ft_ep = int(np.argmin(ft_hist["val_loss"]) + 1)
    recorded_best_ep = int(record.get("best_epoch", min_ft_ep))
    shown_ft_ep = recorded_best_ep if recorded_best_ep >= 1 else min_ft_ep
    ep_marker_label = (
        f"Best FT Epoch ({shown_ft_ep})"
        if recorded_best_ep >= 1
        else f"Min FT Val Loss (Ep {shown_ft_ep})"
    )

    if warmup_hist and len(warmup_hist.get("loss", [])) > 0:
        w_len = len(warmup_hist["loss"])
        f_len = len(ft_hist["loss"])
        epochs = np.arange(1, w_len + f_len + 1)
        train_acc = np.concatenate([warmup_hist["accuracy"], ft_hist["accuracy"]]) * 100.0
        val_acc = np.concatenate([warmup_hist["val_accuracy"], ft_hist["val_accuracy"]]) * 100.0
        train_loss = np.concatenate([warmup_hist["loss"], ft_hist["loss"]])
        val_loss = np.concatenate([warmup_hist["val_loss"], ft_hist["val_loss"]])
        best_global_ep = w_len + shown_ft_ep
        split_ep = w_len + 0.5
    else:
        epochs = np.arange(1, len(ft_hist["loss"]) + 1)
        train_acc = np.asarray(ft_hist["accuracy"]) * 100.0
        val_acc = np.asarray(ft_hist["val_accuracy"]) * 100.0
        train_loss = np.asarray(ft_hist["loss"])
        val_loss = np.asarray(ft_hist["val_loss"])
        best_global_ep = shown_ft_ep
        split_ep = None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.6, 4.9))

    ax1.plot(epochs, train_acc, "o-", label="Train Accuracy (%)", linewidth=2.1, markersize=3.8)
    ax1.plot(epochs, val_acc, "s-", label="Val Accuracy (%)", linewidth=2.1, markersize=3.8)
    if split_ep is not None:
        ax1.axvline(
            split_ep,
            color="#2b5c8f",
            linestyle=":",
            linewidth=2,
            label="Початок Fine-Tuning (Unfreeze)",
        )
    ax1.axvline(
        best_global_ep,
        color="crimson",
        linestyle="--",
        alpha=0.82,
        label=ep_marker_label,
    )
    ax1.set_title("Динаміка точності (Accuracy, %)")
    ax1.set_xlabel("Епоха (Етап 1: Warm-up → Етап 2: Fine-Tuning)")
    ax1.set_ylabel("Accuracy (%)")
    ax1.legend(loc="lower right", fontsize=9)

    ax2.plot(epochs, train_loss, "o-", label="Train Loss", linewidth=2.1, markersize=3.8)
    ax2.plot(epochs, val_loss, "s-", label="Val Loss", linewidth=2.1, markersize=3.8)
    if split_ep is not None:
        ax2.axvline(
            split_ep,
            color="#2b5c8f",
            linestyle=":",
            linewidth=2,
            label="Початок Fine-Tuning (Unfreeze)",
        )
    ax2.axvline(
        best_global_ep,
        color="crimson",
        linestyle="--",
        alpha=0.82,
        label=ep_marker_label,
    )
    ax2.set_title("Функція втрат (Binary Cross-Entropy)")
    ax2.set_xlabel("Епоха (Етап 1: Warm-up → Етап 2: Fine-Tuning)")
    ax2.set_ylabel("Binary Cross-Entropy")
    ax2.legend(loc="upper right", fontsize=9)

    fig.suptitle(label_title, fontsize=13.2, y=1.03)
    plt.tight_layout()
    _save_fig(fig, save_name)
    plt.show()


def plot_multi_curves_comparison(
    records: list[dict[str, Any]],
    label_fn: Callable[[dict[str, Any]], str],
    suptitle: str,
    save_name: str | None = None,
) -> None:
    """Plot 2x2 comparative analysis (Val Loss, Val Acc, Overfitting Gap, Test Acc) across runs."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    palette = sns.color_palette("tab10", n_colors=len(records))

    labels: list[str] = []
    for color, rec in zip(palette, records, strict=True):
        lbl = label_fn(rec)
        labels.append(lbl)
        hist = rec["history"]
        epochs = np.arange(1, len(hist["loss"]) + 1)
        acc_gap = (np.asarray(hist["accuracy"]) - np.asarray(hist["val_accuracy"])) * 100.0

        axes[0, 0].plot(epochs, hist["val_loss"], "o-", color=color, linewidth=2, markersize=4, label=lbl)
        axes[0, 1].plot(
            epochs,
            np.asarray(hist["val_accuracy"]) * 100.0,
            "s-",
            color=color,
            linewidth=2,
            markersize=4,
            label=lbl,
        )
        axes[1, 0].plot(epochs, acc_gap, "^-", color=color, linewidth=2, markersize=4, label=lbl)

    all_val_losses = [v for r in records for v in r["history"]["val_loss"]]
    if max(all_val_losses) / max(min(all_val_losses), 1e-6) > 25.0:
        axes[0, 0].set_yscale("log")
        axes[0, 0].set_ylabel("Val Loss (лог. шкала, нижче = краще)")
    else:
        axes[0, 0].set_ylabel("Val Loss (нижче = краще)")
    axes[0, 0].set_title("Оціночна помилка (Validation Loss) за епохами")
    axes[0, 0].set_xlabel("Епоха Fine-Tuning")
    axes[0, 0].legend(fontsize=9)

    axes[0, 1].set_title("Оціночна точність (Validation Accuracy, %)")
    axes[0, 1].set_xlabel("Епоха Fine-Tuning")
    axes[0, 1].set_ylabel("Val Accuracy (%)")
    axes[0, 1].legend(fontsize=9)

    axes[1, 0].axhline(0.0, color="gray", linestyle="--", linewidth=1)
    axes[1, 0].set_title("Розрив перенавчання: Train Acc − Val Acc (%)")
    axes[1, 0].set_xlabel("Епоха Fine-Tuning")
    axes[1, 0].set_ylabel("Overfitting Gap (%)")
    axes[1, 0].legend(fontsize=9)

    x_pos = np.arange(len(records))
    test_accs = [r["test_metrics"]["accuracy"] * 100.0 for r in records]
    test_losses = [r["test_metrics"]["loss"] for r in records]

    ax_bar = axes[1, 1]
    bars = ax_bar.bar(x_pos - 0.18, test_accs, width=0.35, label="Test Accuracy (%)", color="#2b7bba")
    ax_bar2 = ax_bar.twinx()
    ax_bar2.bar(
        x_pos + 0.18,
        test_losses,
        width=0.35,
        label="Test Loss (BCE)",
        color="#e06666",
        alpha=0.85,
    )

    ax_bar.set_xticks(x_pos)
    ax_bar.set_xticklabels(labels, rotation=12)
    min_acc = min(test_accs)
    ax_bar.set_ylim(max(43.0, min_acc - 2.5), 101.2)
    ax_bar2.set_ylim(0.0, max(test_losses) * 1.18)
    ax_bar.set_ylabel("Test Accuracy (%)", color="#2b7bba")
    ax_bar2.set_ylabel("Test BCE Loss", color="#c0392b")
    ax_bar2.grid(False)
    ax_bar.set_title("Підсумок на тестовій вибірці: Accuracy vs Loss")

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
    ax_bar.legend(
        lines_1 + lines_2,
        labels_1 + labels_2,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=2,
        fontsize=8.5,
    )

    fig.suptitle(suptitle, fontsize=13.5, y=1.01)
    plt.tight_layout()
    _save_fig(fig, save_name)
    plt.show()


def plot_hyperparameter_bars(
    df_study: pd.DataFrame,
    x_col: str,
    title: str,
    save_name: str | None = None,
) -> None:
    """Plot clean 1x2 summary bars (Test Accuracy % and Test Loss) for a hyperparameter sweep."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
    palette = sns.color_palette("crest", n_colors=len(df_study))
    x_labels = [str(v) for v in df_study[x_col]]

    b1 = ax1.bar(x_labels, df_study["Test Accuracy (%)"], color=palette)
    ax1.set_title("Точність на тестовій вибірці (Test Accuracy, %)")
    ax1.set_xlabel(x_col)
    ax1.set_ylabel("Test Accuracy (%)")
    min_y = float(df_study["Test Accuracy (%)"].min())
    ax1.set_ylim(max(45.0, min_y - 0.8), min(100.15, float(df_study["Test Accuracy (%)"].max()) + 0.45))
    for bar in b1:
        ax1.annotate(
            f"{bar.get_height():.2f}%",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center",
            va="bottom",
            fontsize=9,
        )

    b2 = ax2.bar(x_labels, df_study["Test Loss"], color=sns.color_palette("flare", n_colors=len(df_study)))
    ax2.set_title("Помилка на тестовій вибірці (Test BCE Loss)")
    ax2.set_xlabel(x_col)
    ax2.set_ylabel("Binary Cross-Entropy (нижче = краще)")
    for bar in b2:
        ax2.annotate(
            f"{bar.get_height():.4f}",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.suptitle(title, fontsize=13, y=1.03)
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
        [[f"{cm_arr[i, j]}\n({cm_pct[i, j]:.2f}%)" for j in range(2)] for i in range(2)]
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
    """Visualize predictions and confidences on 10 external unseen cat & dog images."""
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
            f"Conf: {row['confidence']*100:.2f}%",
            color=color,
            fontsize=9.5,
            fontweight="bold",
        )
        ax.axis("off")

    for extra_ax in axes_flat[n:]:
        extra_ax.axis("off")

    acc = df_preds["correct"].mean() * 100.0
    mean_conf = df_preds["confidence"].mean() * 100.0
    fig.suptitle(
        f"{model_title} — Зовнішні фото: {acc:.1f}% ({int(df_preds['correct'].sum())}/{n}) | "
        f"Сер. впевненість: {mean_conf:.2f}%",
        fontsize=12.5,
        y=1.02,
    )
    plt.tight_layout()
    _save_fig(fig, save_name)
    plt.show()


def plot_lab8_vs_lab9_comparison(
    df_delta: pd.DataFrame,
    save_name: str | None = "05_lab8_vs_lab9_comparison.png",
) -> None:
    """Plot grouped Before (Lab 8 Frozen TL) vs After (Lab 9 Fine-Tuning) comparison across models."""
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2))
    x = np.arange(len(df_delta))
    w = 0.36
    models = list(df_delta["Model"])

    # 1. Test Accuracy (%)
    b8_acc = axes[0].bar(
        x - w / 2,
        df_delta["Lab 8 Acc (%)"],
        width=w,
        label="Лаб 8 (Frozen TL)",
        color="#74a9cf",
    )
    b9_acc = axes[0].bar(
        x + w / 2,
        df_delta["Lab 9 FT Acc (%)"],
        width=w,
        label="Лаб 9 (Fine-Tuning)",
        color="#045a8d",
    )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models, rotation=15)
    min_acc = float(min(df_delta["Lab 8 Acc (%)"].min(), df_delta["Lab 9 FT Acc (%)"].min()))
    max_acc = float(max(df_delta["Lab 8 Acc (%)"].max(), df_delta["Lab 9 FT Acc (%)"].max()))
    axes[0].set_ylim(max(97.5, min_acc - 0.5), min(100.0, max_acc + 0.35))
    axes[0].set_title("Точність на тесті (Test Accuracy, %)")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].legend(loc="lower right", fontsize=9)
    for b in list(b8_acc) + list(b9_acc):
        axes[0].annotate(
            f"{b.get_height():.2f}%",
            (b.get_x() + b.get_width() / 2, b.get_height()),
            ha="center",
            va="bottom",
            fontsize=8,
        )

    # 2. Test BCE Loss
    b8_loss = axes[1].bar(
        x - w / 2,
        df_delta["Lab 8 Loss"],
        width=w,
        label="Лаб 8 (Frozen TL)",
        color="#fdae6b",
    )
    b9_loss = axes[1].bar(
        x + w / 2,
        df_delta["Lab 9 FT Loss"],
        width=w,
        label="Лаб 9 (Fine-Tuning)",
        color="#d94801",
    )
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models, rotation=15)
    max_loss = float(max(df_delta["Lab 8 Loss"].max(), df_delta["Lab 9 FT Loss"].max()))
    axes[1].set_ylim(0.0, max_loss * 1.24)
    axes[1].set_title("Функція втрат на тесті (Test Loss)")
    axes[1].set_ylabel("Binary Cross-Entropy (нижче = краще)")
    axes[1].legend(loc="upper right", fontsize=9)
    for b in list(b8_loss) + list(b9_loss):
        axes[1].annotate(
            f"{b.get_height():.4f}",
            (b.get_x() + b.get_width() / 2, b.get_height()),
            ha="center",
            va="bottom",
            fontsize=8,
        )

    # 3. Misclassified test images out of 2,502
    b8_err = axes[2].bar(
        x - w / 2,
        df_delta["Lab 8 Errors"],
        width=w,
        label="Лаб 8 (Помилок з 2502)",
        color="#bcbddc",
    )
    b9_err = axes[2].bar(
        x + w / 2,
        df_delta["Lab 9 Errors"],
        width=w,
        label="Лаб 9 (Помилок з 2502)",
        color="#54278f",
    )
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(models, rotation=15)
    max_err = float(max(df_delta["Lab 8 Errors"].max(), df_delta["Lab 9 Errors"].max()))
    axes[2].set_ylim(0.0, max_err * 1.26)
    axes[2].set_title("Кількість помилок на тесті (з 2 502 фото)")
    axes[2].set_ylabel("Кількість помилкових прогнозів")
    axes[2].legend(loc="upper right", fontsize=9)
    for b in list(b8_err) + list(b9_err):
        axes[2].annotate(
            f"{int(round(b.get_height()))}",
            (b.get_x() + b.get_width() / 2, b.get_height()),
            ha="center",
            va="bottom",
            fontsize=8.5,
        )

    fig.suptitle(
        "Порівняльний аналіз: Перенесення навчання без донавчання (Лаб 8) vs Fine-Tuning (Лаб 9)",
        fontsize=13.5,
        y=1.03,
    )
    plt.tight_layout()
    _save_fig(fig, save_name)
    plt.show()

