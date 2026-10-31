"""Streaming tf.data.Dataset builders and stratified subset loaders for 8 GB Apple Silicon."""

from __future__ import annotations

import random
from pathlib import Path

import keras
import numpy as np
import pandas as pd
import tensorflow as tf

from .config import CLASS_NAMES, EXTRACT_BATCH_SIZE, SEED, SPLIT_DATA_DIR


def create_image_datasets(
    image_size: tuple[int, int],
    split_dir: Path = SPLIT_DATA_DIR,
    batch_size: int = EXTRACT_BATCH_SIZE,
    shuffle_train: bool = True,
    seed: int = SEED,
) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """Create streaming train, validation, and test tf.data.Datasets without in-RAM caching."""
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


def build_subset_path_dataset(
    file_paths: list[str],
    labels: np.ndarray,
    image_size: tuple[int, int],
    batch_size: int = EXTRACT_BATCH_SIZE,
) -> tf.data.Dataset:
    """Create a deterministic, streaming tf.data.Dataset from an explicit list of JPEG paths."""
    target_h, target_w = image_size

    def _load_and_resize(path: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        raw = tf.io.read_file(path)
        img = tf.image.decode_jpeg(raw, channels=3)
        img_resized = tf.image.resize(img, [target_h, target_w], method="bilinear")
        return tf.cast(img_resized, tf.float32), tf.cast(label, tf.float32)

    ds = tf.data.Dataset.from_tensor_slices((file_paths, labels.astype(np.float32)))
    ds = ds.map(_load_and_resize, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def get_stratified_split_paths(
    split_name: str,
    max_samples: int | None = None,
    split_dir: Path = SPLIT_DATA_DIR,
    seed: int = SEED,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Return deterministic (file_paths, labels, original_indices) for full or stratified subset.

    Note:
        Matches the exact alphabetical file ordering of `image_dataset_from_directory(..., shuffle=False)`,
        so `original_indices` directly index into cached bottleneck feature arrays `train_x` / `val_x`!
    """
    all_paths: list[str] = []
    all_labels: list[int] = []
    class_indices: dict[int, list[int]] = {0: [], 1: []}

    idx_counter = 0
    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        cls_files = sorted((split_dir / split_name / cls_name).glob("*.jpg"), key=lambda p: p.name)
        for p in cls_files:
            all_paths.append(str(p))
            all_labels.append(cls_idx)
            class_indices[cls_idx].append(idx_counter)
            idx_counter += 1

    labels_arr = np.asarray(all_labels, dtype=np.float32)
    if max_samples is None or max_samples >= len(all_paths):
        full_idx = np.arange(len(all_paths), dtype=np.int64)
        return all_paths, labels_arr, full_idx

    per_class = max_samples // 2
    rng = random.Random(seed)
    chosen_0 = sorted(rng.sample(class_indices[0], per_class))
    chosen_1 = sorted(rng.sample(class_indices[1], max_samples - per_class))
    selected_idx = np.asarray(chosen_0 + chosen_1, dtype=np.int64)
    sub_paths = [all_paths[i] for i in selected_idx]
    sub_labels = labels_arr[selected_idx]
    return sub_paths, sub_labels, selected_idx


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

