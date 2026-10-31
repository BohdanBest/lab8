"""Two-stage Fine-Tuning engine, temporary frozen-prefix memmap caching, and End-to-End verification."""

from __future__ import annotations

import gc
import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import keras
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .config import (
    CLASS_NAMES,
    DEFAULT_BATCH_SIZE,
    DEFAULT_DENSE_UNITS,
    DEFAULT_DROPOUT,
    DEFAULT_FINETUNE_EPOCHS,
    DEFAULT_FINETUNE_LR,
    DEFAULT_WARMUP_EPOCHS,
    DEFAULT_WARMUP_LR,
    EXTERNAL_TEST_DIR,
    EXTRACT_BATCH_SIZE,
    FEATURES_CACHE_DIR,
    HISTORIES_DIR,
    MODEL_SPECS,
    MODELS_DIR,
    SEED,
    SPLIT_DATA_DIR,
    TEMP_PREFIX_CACHE_DIR,
    cleanup_temp_prefix_cache,
    set_global_seed,
)
from .dataset import (
    build_subset_path_dataset,
    create_image_datasets,
    get_stratified_split_paths,
)
from .models_factory import (
    assemble_finetuned_end_to_end_model,
    build_classifier_head,
    build_feature_extractor,
    build_prefix_extractor,
    build_trainable_suffix_model,
    count_model_parameters,
)


def release_memory() -> None:
    """Clear Keras global graph state, Python GC, and Apple Metal MPS cache (for 8 GB M1)."""
    keras.backend.clear_session()
    gc.collect()
    try:
        import torch

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except ImportError:
        pass


def _use_mps_nchw() -> bool:
    """Check whether Keras is running on PyTorch with Apple Silicon MPS GPU available."""
    if keras.backend.backend() != "torch":
        return False
    try:
        import torch

        return bool(torch.backends.mps.is_available())
    except ImportError:
        return False


class _BestWeightsTracker(keras.callbacks.Callback):
    """Track best validation-loss weights in memory across all epochs."""

    def __init__(self, restore_on_end: bool = True) -> None:
        super().__init__()
        self.restore_on_end = restore_on_end
        self.best_val_loss: float = float("inf")
        self.best_epoch: int = 1
        self.best_weights: list[np.ndarray] | None = None

    def on_epoch_end(self, epoch: int, logs: dict[str, float] | None = None) -> None:
        logs = logs or {}
        val_loss = float(logs.get("val_loss", float("inf")))
        model_ref: Any = self.model
        if (val_loss < self.best_val_loss or self.best_weights is None) and model_ref is not None:
            self.best_val_loss = val_loss
            self.best_epoch = epoch + 1
            self.best_weights = model_ref.get_weights()

    def on_train_end(self, logs: dict[str, float] | None = None) -> None:
        model_ref: Any = self.model
        if self.restore_on_end and self.best_weights is not None and model_ref is not None:
            model_ref.set_weights(self.best_weights)


def compute_binary_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, Any]:
    """Compute comprehensive binary classification metrics from predicted probabilities."""
    y_true_int = y_true.reshape(-1).astype(int)
    y_prob_sanitized = np.nan_to_num(
        y_prob.reshape(-1).astype(np.float64),
        nan=0.5,
        posinf=1.0,
        neginf=0.0,
    )
    y_prob_flat = np.clip(y_prob_sanitized, 1e-7, 1.0 - 1e-7)
    y_pred_int = (y_prob_flat >= 0.5).astype(int)

    bce_loss = float(
        -np.mean(y_true_int * np.log(y_prob_flat) + (1 - y_true_int) * np.log(1.0 - y_prob_flat))
    )
    cm = confusion_matrix(y_true_int, y_pred_int)
    errors_count = int(np.sum(y_true_int != y_pred_int))
    return {
        "accuracy": round(float(accuracy_score(y_true_int, y_pred_int)), 5),
        "loss": round(bce_loss, 5),
        "roc_auc": round(float(roc_auc_score(y_true_int, y_prob_flat)), 5),
        "f1": round(float(f1_score(y_true_int, y_pred_int, zero_division=0)), 5),
        "precision": round(float(precision_score(y_true_int, y_pred_int, zero_division=0)), 5),
        "recall": round(float(recall_score(y_true_int, y_pred_int, zero_division=0)), 5),
        "errors_count": errors_count,
        "total_samples": int(len(y_true_int)),
        "confusion_matrix": cm.tolist(),
    }


def _extract_full_bottleneck_arrays(
    extractor: keras.Model,
    dataset: Any,
    split_label: str,
    mps_fast: bool,
) -> tuple[np.ndarray, np.ndarray]:
    if mps_fast:
        import torch
        from tqdm.auto import tqdm

        extractor.eval()
        feat_batches: list[np.ndarray] = []
        label_batches: list[np.ndarray] = []
        total_batches = int(dataset.cardinality().numpy())
        with torch.inference_mode():
            for x_tf, y_tf in tqdm(dataset, total=total_batches, desc=f"MPS Bottleneck [{split_label}]"):
                t_nchw = (
                    torch.from_numpy(x_tf.numpy())
                    .to(device="mps", dtype=torch.float16)
                    .permute(0, 3, 1, 2)
                    .contiguous()
                )
                feats = extractor(t_nchw).float().cpu().numpy()
                feat_batches.append(feats)
                label_batches.append(y_tf.numpy().reshape(-1).astype(np.float32))
        torch.mps.synchronize()
        return np.concatenate(feat_batches, axis=0), np.concatenate(label_batches, axis=0)

    features = extractor.predict(dataset, verbose=1)
    labels = np.concatenate([y.numpy() for _, y in dataset], axis=0).reshape(-1).astype(np.float32)
    return features.astype(np.float32), labels


def extract_or_load_bottleneck_features(
    model_key: str,
    split_dir: Path = SPLIT_DATA_DIR,
    cache_dir: Path = FEATURES_CACHE_DIR,
    batch_size: int = EXTRACT_BATCH_SIZE,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], float]:
    """Load (or extract once) Stage 1 GlobalAveragePooling2D bottleneck features for all 24,997 images."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    npz_path = cache_dir / f"{model_key}_features.npz"
    meta_path = cache_dir / f"{model_key}_meta.json"

    if npz_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        with np.load(npz_path) as data:
            features_dict = {
                "train": (data["train_x"], data["train_y"]),
                "val": (data["val_x"], data["val_y"]),
                "test": (data["test_x"], data["test_y"]),
            }
        print(
            f"[trainer] Loaded Stage-1 bottleneck features for '{model_key}' "
            f"(dim={features_dict['train'][0].shape[1]}, train={len(features_dict['train'][1])})."
        )
        return features_dict, float(meta["extraction_seconds"])

    spec = MODEL_SPECS[model_key]
    train_ds, val_ds, test_ds = create_image_datasets(
        image_size=spec.image_size,
        split_dir=split_dir,
        batch_size=batch_size,
        shuffle_train=False,
    )
    mps_fast = _use_mps_nchw()
    try:
        if mps_fast:
            keras.config.set_image_data_format("channels_first")
            keras.mixed_precision.set_global_policy("mixed_float16")
            extractor, _ = build_feature_extractor(model_key, data_format="channels_first")
        else:
            extractor, _ = build_feature_extractor(model_key, data_format="channels_last")

        t0 = time.perf_counter()
        train_x, train_y = _extract_full_bottleneck_arrays(extractor, train_ds, "train", mps_fast)
        val_x, val_y = _extract_full_bottleneck_arrays(extractor, val_ds, "val", mps_fast)
        test_x, test_y = _extract_full_bottleneck_arrays(extractor, test_ds, "test", mps_fast)
        extraction_seconds = time.perf_counter() - t0
    finally:
        keras.config.set_image_data_format("channels_last")
        keras.mixed_precision.set_global_policy("float32")
        release_memory()

    np.savez(
        npz_path,
        train_x=train_x,
        train_y=train_y,
        val_x=val_x,
        val_y=val_y,
        test_x=test_x,
        test_y=test_y,
    )
    meta_path.write_text(
        json.dumps(
            {
                "model_key": model_key,
                "feature_dim": int(train_x.shape[1]),
                "extraction_seconds": round(extraction_seconds, 2),
                "train_samples": int(train_x.shape[0]),
                "val_samples": int(val_x.shape[0]),
                "test_samples": int(test_x.shape[0]),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return (
        {"train": (train_x, train_y), "val": (val_x, val_y), "test": (test_x, test_y)},
        extraction_seconds,
    )


@dataclass
class PrefixCacheHandle:
    """Memory-mapped frozen-prefix feature store keeping RAM usage < 250 MB."""

    model_key: str
    cut_layer_name: str
    data_format: str
    feature_shape: tuple[int, ...]
    cache_subdir: Path
    train_mmap_path: Path
    val_mmap_path: Path
    test_mmap_path: Path
    train_y: np.ndarray
    val_y: np.ndarray
    test_y: np.ndarray
    train_indices: np.ndarray
    val_indices: np.ndarray
    extraction_seconds: float

    def open_split(self, split: str) -> tuple[np.memmap, np.ndarray]:
        if split == "train":
            mm = np.memmap(
                self.train_mmap_path,
                dtype=np.float16,
                mode="r",
                shape=(len(self.train_y), *self.feature_shape),
            )
            return mm, self.train_y
        if split == "val":
            mm = np.memmap(
                self.val_mmap_path,
                dtype=np.float16,
                mode="r",
                shape=(len(self.val_y), *self.feature_shape),
            )
            return mm, self.val_y
        if split == "test":
            mm = np.memmap(
                self.test_mmap_path,
                dtype=np.float16,
                mode="r",
                shape=(len(self.test_y), *self.feature_shape),
            )
            return mm, self.test_y
        raise ValueError(f"Unknown split '{split}'")


def _stream_prefix_to_memmap(
    prefix_model: keras.Model,
    dataset: Any,
    mmap_path: Path,
    n_samples: int,
    feat_shape: tuple[int, ...],
    mps_fast: bool,
    desc: str,
) -> None:
    from tqdm.auto import tqdm

    mm = np.memmap(mmap_path, dtype=np.float16, mode="w+", shape=(n_samples, *feat_shape))
    cursor = 0
    total_batches = int(dataset.cardinality().numpy())

    if mps_fast:
        import torch

        prefix_model.eval()
        with torch.inference_mode():
            for x_tf, _ in tqdm(dataset, total=total_batches, desc=desc):
                t_nchw = (
                    torch.from_numpy(x_tf.numpy())
                    .to(device="mps", dtype=torch.float16)
                    .permute(0, 3, 1, 2)
                    .contiguous()
                )
                out_fp16 = prefix_model(t_nchw).to(dtype=torch.float16).cpu().numpy()
                bsz = out_fp16.shape[0]
                mm[cursor : cursor + bsz] = out_fp16
                cursor += bsz
        torch.mps.synchronize()
    else:
        for x_tf, _ in tqdm(dataset, total=total_batches, desc=desc):
            out = keras.ops.convert_to_numpy(prefix_model(x_tf.numpy(), training=False)).astype(
                np.float16
            )
            bsz = out.shape[0]
            mm[cursor : cursor + bsz] = out
            cursor += bsz

    mm.flush()
    del mm


def prepare_prefix_cache(
    model_key: str,
    unfreeze_blocks: int,
    train_samples: int | None = 2000,
    val_samples: int | None = 500,
    split_dir: Path = SPLIT_DATA_DIR,
    batch_size: int = EXTRACT_BATCH_SIZE,
    seed: int = SEED,
) -> PrefixCacheHandle:
    """Extract (or reuse) frozen-prefix spatial feature maps in a bounded temporary memmap cache.

    Automatically purges any stale temporary caches from other models/cuts first so peak SSD usage
    never exceeds ~4.8 GB on a 13 GB free drive.
    """
    spec = MODEL_SPECS[model_key]
    cut_spec = spec.unfreeze_options[unfreeze_blocks]
    cut_layer_name = cut_spec.cut_layer_name

    mps_fast = _use_mps_nchw()
    data_format = "channels_first" if mps_fast else "channels_last"

    train_paths, train_y, train_idx = get_stratified_split_paths("train", train_samples, split_dir, seed)
    val_paths, val_y, val_idx = get_stratified_split_paths("val", val_samples, split_dir, seed)
    test_paths, test_y, _ = get_stratified_split_paths("test", None, split_dir, seed)

    tag = f"{model_key}_{cut_layer_name}_{data_format}_tr{len(train_y)}_va{len(val_y)}"
    cache_subdir = TEMP_PREFIX_CACHE_DIR / tag
    meta_file = cache_subdir / "meta.json"

    # Purge any other temporary caches first to protect disk space
    if TEMP_PREFIX_CACHE_DIR.exists():
        for child in TEMP_PREFIX_CACHE_DIR.iterdir():
            if child != cache_subdir:
                shutil.rmtree(child, ignore_errors=True)

    train_mm_path = cache_subdir / "train.fp16.mmap"
    val_mm_path = cache_subdir / "val.fp16.mmap"
    test_mm_path = cache_subdir / "test.fp16.mmap"

    if (
        meta_file.exists()
        and train_mm_path.exists()
        and val_mm_path.exists()
        and test_mm_path.exists()
    ):
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        return PrefixCacheHandle(
            model_key=model_key,
            cut_layer_name=cut_layer_name,
            data_format=data_format,
            feature_shape=tuple(meta["feature_shape"]),
            cache_subdir=cache_subdir,
            train_mmap_path=train_mm_path,
            val_mmap_path=val_mm_path,
            test_mmap_path=test_mm_path,
            train_y=train_y,
            val_y=val_y,
            test_y=test_y,
            train_indices=train_idx,
            val_indices=val_idx,
            extraction_seconds=float(meta["extraction_seconds"]),
        )

    cache_subdir.mkdir(parents=True, exist_ok=True)
    train_ds = build_subset_path_dataset(train_paths, train_y, spec.image_size, batch_size)
    val_ds = build_subset_path_dataset(val_paths, val_y, spec.image_size, batch_size)
    test_ds = build_subset_path_dataset(test_paths, test_y, spec.image_size, batch_size)

    try:
        if mps_fast:
            keras.config.set_image_data_format("channels_first")
            keras.mixed_precision.set_global_policy("mixed_float16")
            prefix_model = build_prefix_extractor(model_key, cut_layer_name, data_format="channels_first")
        else:
            prefix_model = build_prefix_extractor(model_key, cut_layer_name, data_format="channels_last")

        feat_shape = tuple(int(d) for d in prefix_model.output.shape[1:])
        est_gb = (
            (len(train_y) + len(val_y) + len(test_y)) * int(np.prod(feat_shape)) * 2 / (1024**3)
        )
        print(
            f"[trainer] Caching frozen prefix '{model_key}:{cut_layer_name}' "
            f"shape={feat_shape} ({len(train_y)} tr / {len(val_y)} val / {len(test_y)} test, "
            f"temp disk={est_gb:.2f} GB) ..."
        )

        t0 = time.perf_counter()
        _stream_prefix_to_memmap(
            prefix_model, train_ds, train_mm_path, len(train_y), feat_shape, mps_fast, f"Prefix [{model_key}:train]"
        )
        _stream_prefix_to_memmap(
            prefix_model, val_ds, val_mm_path, len(val_y), feat_shape, mps_fast, f"Prefix [{model_key}:val]"
        )
        _stream_prefix_to_memmap(
            prefix_model, test_ds, test_mm_path, len(test_y), feat_shape, mps_fast, f"Prefix [{model_key}:test]"
        )
        extract_sec = time.perf_counter() - t0
    finally:
        keras.config.set_image_data_format("channels_last")
        keras.mixed_precision.set_global_policy("float32")
        release_memory()

    meta_file.write_text(
        json.dumps(
            {
                "model_key": model_key,
                "cut_layer_name": cut_layer_name,
                "data_format": data_format,
                "feature_shape": list(feat_shape),
                "extraction_seconds": round(extract_sec, 2),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return PrefixCacheHandle(
        model_key=model_key,
        cut_layer_name=cut_layer_name,
        data_format=data_format,
        feature_shape=feat_shape,
        cache_subdir=cache_subdir,
        train_mmap_path=train_mm_path,
        val_mmap_path=val_mm_path,
        test_mmap_path=test_mm_path,
        train_y=train_y,
        val_y=val_y,
        test_y=test_y,
        train_indices=train_idx,
        val_indices=val_idx,
        extraction_seconds=extract_sec,
    )


def _predict_suffix_on_memmap(
    suffix_model: keras.Model,
    mm: np.ndarray,
    batch_size: int,
    mps_fast: bool,
    use_autocast: bool = True,
) -> np.ndarray:
    n = mm.shape[0]
    preds: list[np.ndarray] = []
    if mps_fast:
        import torch

        with torch.no_grad():
            for start in range(0, n, batch_size):
                batch_np = np.array(mm[start : start + batch_size], dtype=np.float32, copy=True)
                bx = torch.from_numpy(batch_np).to("mps")
                if use_autocast:
                    with torch.autocast(device_type="mps", dtype=torch.float16):
                        out = suffix_model(bx, training=False)
                    if not bool(torch.isfinite(out).all().item()):
                        out = suffix_model(bx, training=False)
                else:
                    out = suffix_model(bx, training=False)
                preds.append(out.float().cpu().numpy().reshape(-1))
        torch.mps.synchronize()
    else:
        for start in range(0, n, batch_size):
            batch_np = np.array(mm[start : start + batch_size], dtype=np.float32, copy=True)
            out = keras.ops.convert_to_numpy(suffix_model(batch_np, training=False))
            preds.append(out.reshape(-1))
    return np.concatenate(preds, axis=0)


def _train_suffix_epochs(
    suffix_model: keras.Model,
    prefix_cache: PrefixCacheHandle,
    epochs: int,
    batch_size: int,
    optimizer_name: str,
    learning_rate: float,
    restore_best_weights: bool,
    seed: int,
    milestone_epochs: tuple[int, ...] = (),
    seed_from_initial_val: bool = False,
    cosine_decay: bool = False,
) -> tuple[dict[str, list[float]], int, float, list[dict[str, Any]]]:
    """Train `suffix_model` over memory-mapped frozen-prefix features and track metrics."""
    train_mm, train_y = prefix_cache.open_split("train")
    val_mm, val_y = prefix_cache.open_split("val")
    test_mm, test_y = prefix_cache.open_split("test")
    mps_fast = prefix_cache.data_format == "channels_first" and _use_mps_nchw()
    # Run pure FP32 when learning_rate >= 5e-4 so large activations never hit FP16 overflow (65504)
    use_autocast = learning_rate < 5e-4

    hist: dict[str, list[float]] = {
        "loss": [],
        "accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
    }
    milestones: list[dict[str, Any]] = []
    best_val_loss = float("inf")
    best_epoch = 1
    best_weights: list[np.ndarray] | None = None
    n_train = len(train_y)

    if seed_from_initial_val:
        val_probs_0 = _predict_suffix_on_memmap(
            suffix_model,
            val_mm,
            max(batch_size, 64),
            mps_fast=mps_fast,
            use_autocast=use_autocast,
        )
        best_val_loss = float(compute_binary_metrics(val_y, val_probs_0)["loss"])
        best_epoch = 0
        best_weights = suffix_model.get_weights()

    if mps_fast:
        import torch

        params: list[Any] = [w.value for w in suffix_model.trainable_weights]
        opt_key = optimizer_name.strip().lower()
        if opt_key == "adam":
            opt: torch.optim.Optimizer = torch.optim.Adam(params, lr=learning_rate, eps=1e-7)
        elif opt_key == "rmsprop":
            opt = torch.optim.RMSprop(params, lr=learning_rate, alpha=0.9, eps=1e-7)
        elif opt_key == "sgd":
            opt = torch.optim.SGD(params, lr=learning_rate, momentum=0.9)
        else:
            raise ValueError(f"Unsupported optimizer '{optimizer_name}'")

        scheduler = (
            torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=max(epochs, 1), eta_min=learning_rate * 0.1
            )
            if cosine_decay
            else None
        )

        for ep in range(1, epochs + 1):
            rng = np.random.default_rng(seed + ep)
            perm = rng.permutation(n_train)
            running_loss = 0.0
            running_correct = 0
            seen = 0

            for start in range(0, n_train, batch_size):
                batch_idx = perm[start : start + batch_size]
                # Read from memmap cleanly
                bx_np = np.array(train_mm[batch_idx], dtype=np.float32, copy=True)
                by_np = train_y[batch_idx].reshape(-1, 1).astype(np.float32)

                opt.zero_grad()
                bx = torch.from_numpy(bx_np).to("mps")
                by = torch.from_numpy(by_np).to("mps")
                if use_autocast:
                    with torch.autocast(device_type="mps", dtype=torch.float16):
                        probs = suffix_model(bx, training=True)
                else:
                    probs = suffix_model(bx, training=True)
                probs_f32 = probs.float().clamp(1e-6, 1.0 - 1e-6)
                loss = torch.nn.functional.binary_cross_entropy(probs_f32, by)
                if not bool(torch.isfinite(loss).item()):
                    continue
                loss.backward()
                opt.step()

                bsz = len(batch_idx)
                running_loss += float(loss.detach().cpu().item()) * bsz
                preds_bin = (probs_f32.detach().cpu().numpy().reshape(-1) >= 0.5).astype(int)
                running_correct += int(np.sum(preds_bin == by_np.reshape(-1).astype(int)))
                seen += bsz

            if scheduler is not None:
                scheduler.step()

            torch.mps.synchronize()
            tr_loss = running_loss / max(seen, 1)
            tr_acc = running_correct / max(seen, 1)

            val_probs = _predict_suffix_on_memmap(
                suffix_model,
                val_mm,
                max(batch_size, 64),
                mps_fast=True,
                use_autocast=use_autocast,
            )
            val_m = compute_binary_metrics(val_y, val_probs)
            v_loss = float(val_m["loss"])
            v_acc = float(val_m["accuracy"])

            hist["loss"].append(round(tr_loss, 6))
            hist["accuracy"].append(round(tr_acc, 6))
            hist["val_loss"].append(round(v_loss, 6))
            hist["val_accuracy"].append(round(v_acc, 6))

            if v_loss < best_val_loss or best_weights is None:
                best_val_loss = v_loss
                best_epoch = ep
                best_weights = suffix_model.get_weights()

            if ep in milestone_epochs:
                # Snapshot test metrics for the best weights seen up to epoch `ep` (and current epoch)
                cur_weights = suffix_model.get_weights()
                if restore_best_weights and best_weights is not None:
                    suffix_model.set_weights(best_weights)
                test_probs_ep = _predict_suffix_on_memmap(
                    suffix_model,
                    test_mm,
                    max(batch_size, 64),
                    mps_fast=True,
                    use_autocast=use_autocast,
                )
                tm_ep = compute_binary_metrics(test_y, test_probs_ep)
                milestones.append(
                    {
                        "epochs_budget": ep,
                        "best_epoch_so_far": best_epoch,
                        "train_accuracy": round(tr_acc, 5),
                        "val_accuracy": round(v_acc, 5),
                        "best_val_loss": round(best_val_loss, 5),
                        "overfitting_acc_gap": round(tr_acc - v_acc, 5),
                        "test_metrics": tm_ep,
                    }
                )
                suffix_model.set_weights(cur_weights)
    else:
        # Universal Keras backend fallback (e.g. Google Colab TensorFlow GPU)
        train_x_ram = np.array(train_mm, dtype=np.float32)
        val_x_ram = np.array(val_mm, dtype=np.float32)
        for ep in range(1, epochs + 1):
            h = suffix_model.fit(
                train_x_ram,
                train_y,
                validation_data=(val_x_ram, val_y),
                epochs=1,
                batch_size=batch_size,
                shuffle=True,
                verbose=0,
            )
            tr_loss = float(h.history["loss"][0])
            tr_acc = float(h.history["accuracy"][0])
            v_loss = float(h.history["val_loss"][0])
            v_acc = float(h.history["val_accuracy"][0])
            hist["loss"].append(round(tr_loss, 6))
            hist["accuracy"].append(round(tr_acc, 6))
            hist["val_loss"].append(round(v_loss, 6))
            hist["val_accuracy"].append(round(v_acc, 6))
            if v_loss < best_val_loss or best_weights is None:
                best_val_loss = v_loss
                best_epoch = ep
                best_weights = suffix_model.get_weights()
            if ep in milestone_epochs:
                cur_weights = suffix_model.get_weights()
                if restore_best_weights and best_weights is not None:
                    suffix_model.set_weights(best_weights)
                test_probs_ep = _predict_suffix_on_memmap(
                    suffix_model, test_mm, max(batch_size, 64), mps_fast=False
                )
                milestones.append(
                    {
                        "epochs_budget": ep,
                        "best_epoch_so_far": best_epoch,
                        "train_accuracy": round(tr_acc, 5),
                        "val_accuracy": round(v_acc, 5),
                        "best_val_loss": round(best_val_loss, 5),
                        "overfitting_acc_gap": round(tr_acc - v_acc, 5),
                        "test_metrics": compute_binary_metrics(test_y, test_probs_ep),
                    }
                )
                suffix_model.set_weights(cur_weights)

    if restore_best_weights and best_weights is not None:
        suffix_model.set_weights(best_weights)

    del train_mm, val_mm, test_mm
    return hist, best_epoch, best_val_loss, milestones


def run_finetuning_experiment(
    model_key: str,
    bottleneck_features: dict[str, tuple[np.ndarray, np.ndarray]],
    prefix_cache: PrefixCacheHandle,
    unfreeze_blocks: int = 1,
    dense_units: int = DEFAULT_DENSE_UNITS,
    dropout_rate: float = DEFAULT_DROPOUT,
    second_dense_units: int = 0,
    optimizer_name: str = "adam",
    learning_rate: float = DEFAULT_FINETUNE_LR,
    batch_size: int = DEFAULT_BATCH_SIZE,
    warmup_epochs: int = DEFAULT_WARMUP_EPOCHS,
    finetune_epochs: int = DEFAULT_FINETUNE_EPOCHS,
    warmup_on_full_train: bool = False,
    restore_best_weights: bool = True,
    milestone_epochs: tuple[int, ...] = (),
    experiment_name: str | None = None,
    keep_model_in_memory: bool = False,
    seed: int = SEED,
) -> tuple[keras.Model | None, dict[str, Any]]:
    """Execute complete 2-stage Fine-Tuning (Stage 1 Head Warm-up -> Stage 2 Conv Block Unfreezing)."""
    spec = MODEL_SPECS[model_key]
    cut_spec = spec.unfreeze_options[unfreeze_blocks]
    tag = experiment_name or (
        f"{model_key}_b{unfreeze_blocks}_d{dense_units}_d2_{second_dense_units}_"
        f"do{str(dropout_rate).replace('.', '_')}_{optimizer_name}_lr{learning_rate:.0e}_bs{batch_size}_ep{finetune_epochs}"
    )

    # Check if already computed in histories/ to support fast idempotent reruns
    record_path = HISTORIES_DIR / f"{tag}.json"
    if record_path.exists() and not keep_model_in_memory:
        cached_record = json.loads(record_path.read_text(encoding="utf-8"))
        print(
            f"[trainer] Loaded existing experiment '{tag}' -> "
            f"Test Acc: {cached_record['test_metrics']['accuracy']*100:.2f}% | "
            f"Loss: {cached_record['test_metrics']['loss']:.4f}"
        )
        return None, cached_record

    set_global_seed(seed)

    # ------------------------------------------------------------------
    # Stage 1: Warm-up classification head on frozen bottleneck features
    # ------------------------------------------------------------------
    full_tr_x, full_tr_y = bottleneck_features["train"]
    full_va_x, full_va_y = bottleneck_features["val"]

    if warmup_on_full_train:
        w_tr_x, w_tr_y = full_tr_x, full_tr_y
        w_va_x, w_va_y = full_va_x, full_va_y
    else:
        w_tr_x = full_tr_x[prefix_cache.train_indices]
        w_tr_y = full_tr_y[prefix_cache.train_indices]
        w_va_x = full_va_x[prefix_cache.val_indices]
        w_va_y = full_va_y[prefix_cache.val_indices]

    head = build_classifier_head(
        feature_dim=int(w_tr_x.shape[1]),
        dense_units=dense_units,
        dropout_rate=dropout_rate,
        second_dense_units=second_dense_units,
        optimizer_name="adam",
        learning_rate=DEFAULT_WARMUP_LR,
        name=f"{tag}_warmup_head",
    )
    warmup_tracker = _BestWeightsTracker(restore_on_end=True)
    t_w0 = time.perf_counter()
    warmup_hist_obj = head.fit(
        w_tr_x,
        w_tr_y,
        validation_data=(w_va_x, w_va_y),
        epochs=warmup_epochs,
        batch_size=64,
        shuffle=True,
        callbacks=[warmup_tracker],
        verbose=0,
    )
    warmup_seconds = time.perf_counter() - t_w0
    warmup_hist = {
        k: [round(float(v), 6) for v in vals] for k, vals in warmup_hist_obj.history.items()
    }

    # ------------------------------------------------------------------
    # Stage 2: Unfreeze top convolutional block(s) & Fine-Tune
    # ------------------------------------------------------------------
    mps_fast = prefix_cache.data_format == "channels_first" and _use_mps_nchw()
    try:
        if mps_fast:
            keras.config.set_image_data_format("channels_first")
        else:
            keras.config.set_image_data_format("channels_last")

        suffix_model, unfreeze_meta = build_trainable_suffix_model(
            model_key=model_key,
            cut_layer_name=cut_spec.cut_layer_name,
            trained_head=head,
            dense_units=dense_units,
            dropout_rate=dropout_rate,
            second_dense_units=second_dense_units,
            optimizer_name=optimizer_name,
            learning_rate=learning_rate,
            freeze_batch_norm=spec.has_batch_norm,
            data_format=prefix_cache.data_format,
        )
        suffix_trainable_params = int(
            sum(keras.ops.size(w) for w in suffix_model.trainable_weights)
        )

        t_ft0 = time.perf_counter()
        ft_hist, best_ep, best_v_loss, milestones = _train_suffix_epochs(
            suffix_model=suffix_model,
            prefix_cache=prefix_cache,
            epochs=finetune_epochs,
            batch_size=batch_size,
            optimizer_name=optimizer_name,
            learning_rate=learning_rate,
            restore_best_weights=restore_best_weights,
            seed=seed,
            milestone_epochs=milestone_epochs,
            seed_from_initial_val=warmup_on_full_train,
            cosine_decay=warmup_on_full_train,
        )
        finetune_seconds = time.perf_counter() - t_ft0

        test_mm, test_y = prefix_cache.open_split("test")
        test_probs = _predict_suffix_on_memmap(
            suffix_model,
            test_mm,
            max(batch_size, 64),
            mps_fast=mps_fast,
            use_autocast=(learning_rate < 5e-4),
        )
        del test_mm
        test_metrics = compute_binary_metrics(test_y, test_probs)
    finally:
        keras.config.set_image_data_format("channels_last")
        keras.mixed_precision.set_global_policy("float32")

    final_tr_acc = ft_hist["accuracy"][-1]
    final_va_acc = ft_hist["val_accuracy"][-1]
    final_tr_loss = ft_hist["loss"][-1]
    final_va_loss = ft_hist["val_loss"][-1]

    record: dict[str, Any] = {
        "experiment_name": tag,
        "model_key": model_key,
        "display_name": spec.display_name,
        "unfreeze_blocks": unfreeze_blocks,
        "cut_layer_name": cut_spec.cut_layer_name,
        "unfreeze_description": cut_spec.description,
        "dense_units": dense_units,
        "second_dense_units": second_dense_units,
        "dropout_rate": dropout_rate,
        "optimizer": optimizer_name.lower(),
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "warmup_epochs": warmup_epochs,
        "finetune_epochs": finetune_epochs,
        "train_samples_ft": int(len(prefix_cache.train_y)),
        "val_samples_ft": int(len(prefix_cache.val_y)),
        "best_epoch": best_ep,
        "best_val_loss": round(float(best_v_loss), 5),
        "best_val_accuracy": round(float(max(ft_hist["val_accuracy"])), 5),
        "final_train_accuracy": round(final_tr_acc, 5),
        "final_val_accuracy": round(final_va_acc, 5),
        "final_train_loss": round(final_tr_loss, 5),
        "final_val_loss": round(final_va_loss, 5),
        "overfitting_acc_gap": round(final_tr_acc - final_va_acc, 5),
        "overfitting_loss_gap": round(final_va_loss - final_tr_loss, 5),
        "trainable_params": suffix_trainable_params,
        **unfreeze_meta,
        "warmup_seconds": round(warmup_seconds, 2),
        "finetune_seconds": round(finetune_seconds, 2),
        "test_metrics": test_metrics,
        "warmup_history": warmup_hist,
        "history": ft_hist,
        "milestones": milestones,
    }

    HISTORIES_DIR.mkdir(parents=True, exist_ok=True)
    record_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"[trainer] {tag} | Best FT Ep {best_ep}/{finetune_epochs} | "
        f"Val Loss: {record['best_val_loss']:.4f} | "
        f"Test Acc: {test_metrics['accuracy']*100:.2f}% ({test_metrics['errors_count']} errs) | "
        f"Time: {finetune_seconds:.1f}s"
    )

    del head
    if not keep_model_in_memory:
        del suffix_model
        release_memory()
        return None, record
    return suffix_model, record


def finalize_and_save_finetuned_model(
    model_key: str,
    trained_suffix_model: keras.Model,
    experiment_record: dict[str, Any],
    prefix_cache: PrefixCacheHandle,
    split_dir: Path = SPLIT_DATA_DIR,
    models_dir: Path = MODELS_DIR,
) -> dict[str, Any]:
    """Stitch full NHWC FP32 End-to-End model, verify raw JPEG parity, and save <100 MB .keras file."""
    spec = MODEL_SPECS[model_key]
    keras.config.set_image_data_format("channels_last")
    keras.mixed_precision.set_global_policy("float32")

    full_model = assemble_finetuned_end_to_end_model(
        model_key=model_key,
        cut_layer_name=experiment_record["cut_layer_name"],
        trained_suffix_model=trained_suffix_model,
        dense_units=int(experiment_record["dense_units"]),
        dropout_rate=float(experiment_record["dropout_rate"]),
        second_dense_units=int(experiment_record["second_dense_units"]),
        optimizer_name=str(experiment_record["optimizer"]),
        learning_rate=float(experiment_record["learning_rate"]),
        freeze_batch_norm=spec.has_batch_norm,
    )
    param_stats = count_model_parameters(full_model)

    # Verify numerical parity between raw JPEG End-to-End model and split suffix model
    _, _, test_ds = create_image_datasets(
        image_size=spec.image_size,
        split_dir=split_dir,
        batch_size=EXTRACT_BATCH_SIZE,
        shuffle_train=False,
    )
    sample_x_tf, _ = next(iter(test_ds))
    e2e_probs = keras.ops.convert_to_numpy(
        full_model(sample_x_tf.numpy(), training=False)
    ).reshape(-1)

    mps_fast = prefix_cache.data_format == "channels_first" and _use_mps_nchw()
    try:
        if mps_fast:
            keras.config.set_image_data_format("channels_first")
        test_mm, _ = prefix_cache.open_split("test")
        suf_probs = _predict_suffix_on_memmap(
            trained_suffix_model, test_mm[: len(e2e_probs)], batch_size=len(e2e_probs), mps_fast=mps_fast
        )
        del test_mm
    finally:
        keras.config.set_image_data_format("channels_last")

    max_diff = float(np.max(np.abs(e2e_probs - suf_probs)))
    if max_diff > 2.5e-2:
        raise RuntimeError(
            f"End-to-End parity check failed for {spec.display_name}: max diff = {max_diff:.5f}"
        )
    print(
        f"[trainer] Verified End-to-End {spec.display_name} on raw test JPEGs "
        f"(max prob diff vs FP16 prefix split = {max_diff:.5f})."
    )

    models_dir.mkdir(parents=True, exist_ok=True)
    save_path = models_dir / spec.save_filename
    full_model.save(save_path)
    size_mb = round(save_path.stat().st_size / (1024 * 1024), 2)
    if size_mb >= 100.0:
        raise RuntimeError(
            f"Saved model {save_path.name} is {size_mb} MB (exceeds 100 MB repository limit)!"
        )

    final_summary: dict[str, Any] = {
        **experiment_record,
        "save_filename": spec.save_filename,
        "model_size_mb": size_mb,
        "image_size": list(spec.image_size),
        "preprocess_description": spec.preprocess_description,
        "prefix_extraction_seconds": round(prefix_cache.extraction_seconds, 2),
        "total_pipeline_seconds": round(
            prefix_cache.extraction_seconds
            + float(experiment_record["warmup_seconds"])
            + float(experiment_record["finetune_seconds"]),
            2,
        ),
        **param_stats,
    }

    (HISTORIES_DIR / f"{model_key}_final_summary.json").write_text(
        json.dumps(final_summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    tm = final_summary["test_metrics"]
    print(
        f"[trainer] Saved {save_path.name} ({size_mb} MB) | "
        f"Test Acc: {tm['accuracy']*100:.2f}% ({tm['errors_count']} errs / {tm['total_samples']}) | "
        f"Loss: {tm['loss']:.4f} | ROC-AUC: {tm['roc_auc']:.5f}"
    )

    del full_model
    del trained_suffix_model
    cleanup_temp_prefix_cache()
    release_memory()
    return final_summary


def predict_external_images(
    model_path: Path,
    external_dir: Path = EXTERNAL_TEST_DIR,
) -> pd.DataFrame:
    """Load a saved .keras model from disk and classify all external unseen cat/dog images."""
    keras.config.set_image_data_format("channels_last")
    keras.mixed_precision.set_global_policy("float32")
    model: Any = keras.models.load_model(model_path, compile=False)
    target_h, target_w = int(model.input_shape[1]), int(model.input_shape[2])

    rows: list[dict[str, Any]] = []
    batch_imgs: list[np.ndarray] = []
    meta_items: list[tuple[str, Path]] = []

    for cls_name in CLASS_NAMES:
        cls_folder = external_dir / cls_name
        for img_path in sorted(cls_folder.glob("*.jpg")):
            with Image.open(img_path) as img:
                resized = img.convert("RGB").resize((target_w, target_h), Image.Resampling.BILINEAR)
                arr = np.asarray(resized, dtype=np.float32)
            batch_imgs.append(arr)
            meta_items.append((cls_name, img_path))

    x_batch = np.stack(batch_imgs, axis=0)
    probs = keras.ops.convert_to_numpy(model(x_batch, training=False)).reshape(-1)

    for (true_cls, img_path), dog_prob in zip(meta_items, probs, strict=True):
        p_clip = float(np.clip(dog_prob, 1e-7, 1.0 - 1e-7))
        y_true = 1.0 if true_cls == "dogs" else 0.0
        sample_bce = -(y_true * np.log(p_clip) + (1.0 - y_true) * np.log(1.0 - p_clip))
        pred_cls = "dogs" if dog_prob >= 0.5 else "cats"
        confidence = float(dog_prob if pred_cls == "dogs" else (1.0 - dog_prob))
        rows.append(
            {
                "filename": img_path.name,
                "rel_path": f"{true_cls}/{img_path.name}",
                "true_class": true_cls,
                "pred_class": pred_cls,
                "dog_probability": round(float(dog_prob), 5),
                "confidence": round(confidence, 5),
                "bce_loss": round(float(sample_bce), 5),
                "correct": bool(true_cls == pred_cls),
            }
        )

    del model
    release_memory()
    return pd.DataFrame(rows)

