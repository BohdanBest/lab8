"""Streaming tf.data.Dataset builders optimized for 8 GB unified memory on Apple Silicon."""

from __future__ import annotations

from pathlib import Path

import keras
import pandas as pd
import tensorflow as tf

from src.config import BATCH_SIZE, CLASS_NAMES, SEED, SPLIT_DATA_DIR


def create_image_datasets(
    image_size: tuple[int, int],
    split_dir: Path = SPLIT_DATA_DIR,
    batch_size: int = BATCH_SIZE,
    shuffle_train: bool = True,
    seed: int = SEED,
) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """Create streaming train, validation, and test tf.data.Datasets without in-RAM caching.

    Note:
        Images are yielded as float32 tensors in [0, 255] with shape (B, H, W, 3).
        Architecture-specific normalization (`preprocess_input`) is embedded inside each model.
    """
    common_kwargs = {
        "labels": "inferred",
        "label_mode": "binary",
        "class_names": list(CLASS_NAMES),
        "image_size": image_size,
        "batch_size": batch_size,
        "interpolation": "bilinear",
    }

    train_ds = keras.utils.image_dataset_from_directory(
        split_dir / "train",
        shuffle=shuffle_train,
        seed=seed if shuffle_train else None,
        **common_kwargs,
    )
    val_ds = keras.utils.image_dataset_from_directory(
        split_dir / "val",
        shuffle=False,
        **common_kwargs,
    )
    test_ds = keras.utils.image_dataset_from_directory(
        split_dir / "test",
        shuffle=False,
        **common_kwargs,
    )

    autotune = tf.data.AUTOTUNE
    return (
        train_ds.prefetch(autotune),
        val_ds.prefetch(autotune),
        test_ds.prefetch(autotune),
    )


def summarize_splits_dataframe(split_dir: Path = SPLIT_DATA_DIR) -> pd.DataFrame:
    """Return a clean DataFrame summarizing image counts per split and class."""
    rows: list[dict[str, int | str | float]] = []
    for split_name in ("train", "val", "test"):
        cats_count = sum(1 for _ in (split_dir / split_name / "cats").glob("*.jpg"))
        dogs_count = sum(1 for _ in (split_dir / split_name / "dogs").glob("*.jpg"))
        total = cats_count + dogs_count
        rows.append(
            {
                "Split": split_name,
                "Cats (0)": cats_count,
                "Dogs (1)": dogs_count,
                "Total": total,
                "Dog Ratio (%)": round(100.0 * dogs_count / max(total, 1), 2),
            }
        )
    df = pd.DataFrame(rows)
    totals_row = pd.DataFrame(
        [
            {
                "Split": "TOTAL",
                "Cats (0)": int(df["Cats (0)"].sum()),
                "Dogs (1)": int(df["Dogs (1)"].sum()),
                "Total": int(df["Total"].sum()),
                "Dog Ratio (%)": round(
                    100.0 * df["Dogs (1)"].sum() / max(int(df["Total"].sum()), 1), 2
                ),
            }
        ]
    )
    return pd.concat([df, totals_row], ignore_index=True)

