"""Training, feature extraction caching, end-to-end verification, and JSON logging."""

from __future__ import annotations

import gc
import json
import time
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

from src.config import (
    BATCH_SIZE,
    CLASS_NAMES,
    DEFAULT_EPOCHS,
    DEFAULT_LEARNING_RATE,
    EXTERNAL_TEST_DIR,
    FEATURES_CACHE_DIR,
    HEAD_BATCH_SIZE,
    HISTORIES_DIR,
    MODEL_SPECS,
    MODELS_DIR,
    SEED,
    SPLIT_DATA_DIR,
    set_global_seed,
)
from src.dataset import create_image_datasets
from src.models_factory import (
    assemble_end_to_end_model,
    build_classifier_head,
    build_feature_extractor,
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


class _BestWeightsTracker(keras.callbacks.Callback):
    """Track best validation-loss weights in memory without cutting epochs short."""

    def __init__(self, restore_on_end: bool = True) -> None:
        super().__init__()
        self.restore_on_end = restore_on_end
        self.best_val_loss: float = float("inf")
        self.best_epoch: int = 0
        self.best_weights: list[np.ndarray] | None = None

    def on_epoch_end(self, epoch: int, logs: dict[str, float] | None = None) -> None:
        logs = logs or {}
        val_loss = float(logs.get("val_loss", float("inf")))
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_epoch = epoch + 1
            self.best_weights = self.model.get_weights()

    def on_train_end(self, logs: dict[str, float] | None = None) -> None:
        if self.restore_on_end and self.best_weights is not None:
            self.model.set_weights(self.best_weights)


def _use_mps_nchw_fp16() -> bool:
    """Check whether Keras is running on PyTorch with Apple Silicon MPS GPU available."""
    if keras.backend.backend() != "torch":
        return False
    try:
        import torch

        return bool(torch.backends.mps.is_available())
    except ImportError:
        return False


def _extract_split_arrays(
    extractor: keras.Model,
    dataset: Any,
    split_label: str,
    mps_fast_path: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Run frozen backbone forward pass once over a dataset and collect labels."""
    print(f"[trainer] Extracting frozen features for '{split_label}' split ...")
    if mps_fast_path:
        import torch
        from tqdm.auto import tqdm

        extractor.eval()
        feat_batches: list[np.ndarray] = []
        label_batches: list[np.ndarray] = []
        total_batches = int(dataset.cardinality().numpy())
        with torch.inference_mode():
            for x_tf, y_tf in tqdm(dataset, total=total_batches, desc=f"MPS GPU [{split_label}]"):
                # Convert NHWC float32 batch -> NCHW float16 directly on Apple Metal GPU
                x_np = x_tf.numpy()
                t_nchw = (
                    torch.from_numpy(x_np)
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


def extract_or_load_features(
    model_key: str,
    split_dir: Path = SPLIT_DATA_DIR,
    cache_dir: Path = FEATURES_CACHE_DIR,
    batch_size: int = BATCH_SIZE,
    force: bool = False,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], float]:
    """Extract (or load cached) GlobalAveragePooling2D features for all 25,000 images."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    npz_path = cache_dir / f"{model_key}_features.npz"
    meta_path = cache_dir / f"{model_key}_meta.json"

    if npz_path.exists() and meta_path.exists() and not force:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        with np.load(npz_path) as data:
            features_dict = {
                "train": (data["train_x"], data["train_y"]),
                "val": (data["val_x"], data["val_y"]),
                "test": (data["test_x"], data["test_y"]),
            }
        print(
            f"[trainer] Loaded cached '{model_key}' features "
            f"(dim={features_dict['train'][0].shape[1]}, "
            f"original extraction={meta['extraction_seconds']:.1f}s)."
        )
        return features_dict, float(meta["extraction_seconds"])

    spec = MODEL_SPECS[model_key]
    train_ds, val_ds, test_ds = create_image_datasets(
        image_size=spec.image_size,
        split_dir=split_dir,
        batch_size=batch_size,
        shuffle_train=False,
    )

    mps_fast = _use_mps_nchw_fp16()
    try:
        if mps_fast:
            keras.config.set_image_data_format("channels_first")
            keras.mixed_precision.set_global_policy("mixed_float16")
            extractor, _ = build_feature_extractor(model_key, data_format="channels_first")
        else:
            extractor, _ = build_feature_extractor(model_key, data_format="channels_last")

        t0 = time.perf_counter()
        train_x, train_y = _extract_split_arrays(extractor, train_ds, "train", mps_fast_path=mps_fast)
        val_x, val_y = _extract_split_arrays(extractor, val_ds, "val", mps_fast_path=mps_fast)
        test_x, test_y = _extract_split_arrays(extractor, test_ds, "test", mps_fast_path=mps_fast)
        extraction_seconds = time.perf_counter() - t0
    finally:
        # Always restore standard channels_last FP32 so heads & saved .keras models are universal
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
        {
            "train": (train_x, train_y),
            "val": (val_x, val_y),
            "test": (test_x, test_y),
        },
        extraction_seconds,
    )


def compute_binary_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, Any]:
    """Compute comprehensive binary classification metrics from probabilities."""
    y_true_int = y_true.reshape(-1).astype(int)
    y_prob_flat = np.clip(y_prob.reshape(-1).astype(np.float64), 1e-7, 1.0 - 1e-7)
    y_pred_int = (y_prob_flat >= 0.5).astype(int)

    bce_loss = float(
        -np.mean(y_true_int * np.log(y_prob_flat) + (1 - y_true_int) * np.log(1.0 - y_prob_flat))
    )
    cm = confusion_matrix(y_true_int, y_pred_int)
    return {
        "accuracy": round(float(accuracy_score(y_true_int, y_pred_int)), 5),
        "loss": round(bce_loss, 5),
        "roc_auc": round(float(roc_auc_score(y_true_int, y_prob_flat)), 5),
        "f1": round(float(f1_score(y_true_int, y_pred_int)), 5),
        "precision": round(float(precision_score(y_true_int, y_pred_int)), 5),
        "recall": round(float(recall_score(y_true_int, y_pred_int)), 5),
        "confusion_matrix": cm.tolist(),
    }


def train_head_experiment(
    model_key: str,
    features: dict[str, tuple[np.ndarray, np.ndarray]],
    dropout_rate: float,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = HEAD_BATCH_SIZE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    restore_best_weights: bool = True,
    experiment_name: str | None = None,
    seed: int = SEED,
) -> tuple[keras.Model, dict[str, Any]]:
    """Train a classification head for `epochs` and record full learning & overfitting metrics."""
    set_global_seed(seed)
    train_x, train_y = features["train"]
    val_x, val_y = features["val"]
    test_x, test_y = features["test"]

    tag = experiment_name or f"{model_key}_dropout_{str(dropout_rate).replace('.', '_')}"
    head = build_classifier_head(
        feature_dim=int(train_x.shape[1]),
        dropout_rate=dropout_rate,
        learning_rate=learning_rate,
        name=f"{tag}_head",
    )

    tracker = _BestWeightsTracker(restore_on_end=restore_best_weights)
    t0 = time.perf_counter()
    history_obj = head.fit(
        train_x,
        train_y,
        validation_data=(val_x, val_y),
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        callbacks=[tracker],
        verbose=0,
    )
    head_train_seconds = time.perf_counter() - t0

    hist = {k: [round(float(v), 6) for v in vals] for k, vals in history_obj.history.items()}
    final_train_acc = hist["accuracy"][-1]
    final_val_acc = hist["val_accuracy"][-1]
    final_train_loss = hist["loss"][-1]
    final_val_loss = hist["val_loss"][-1]

    test_probs = head.predict(test_x, batch_size=batch_size, verbose=0)
    test_metrics = compute_binary_metrics(test_y, test_probs)

    record: dict[str, Any] = {
        "experiment_name": tag,
        "model_key": model_key,
        "display_name": MODEL_SPECS[model_key].display_name,
        "dropout_rate": dropout_rate,
        "epochs": epochs,
        "best_epoch": tracker.best_epoch,
        "best_val_loss": round(float(min(hist["val_loss"])), 5),
        "best_val_accuracy": round(float(max(hist["val_accuracy"])), 5),
        "final_train_accuracy": round(final_train_acc, 5),
        "final_val_accuracy": round(final_val_acc, 5),
        "final_train_loss": round(final_train_loss, 5),
        "final_val_loss": round(final_val_loss, 5),
        "overfitting_acc_gap": round(final_train_acc - final_val_acc, 5),
        "overfitting_loss_gap": round(final_val_loss - final_train_loss, 5),
        "head_train_seconds": round(head_train_seconds, 2),
        "test_metrics": test_metrics,
        "history": hist,
    }

    HISTORIES_DIR.mkdir(parents=True, exist_ok=True)
    (HISTORIES_DIR / f"{tag}.json").write_text(
        json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return head, record


def finalize_and_save_model(
    model_key: str,
    trained_head: keras.Model,
    experiment_record: dict[str, Any],
    extraction_seconds: float,
    split_dir: Path = SPLIT_DATA_DIR,
    models_dir: Path = MODELS_DIR,
) -> dict[str, Any]:
    """Assemble full end-to-end model, verify on raw test images, and save .keras file."""
    spec = MODEL_SPECS[model_key]
    dropout_rate = float(experiment_record["dropout_rate"])

    full_model = assemble_end_to_end_model(
        model_key=model_key,
        trained_head=trained_head,
        dropout_rate=dropout_rate,
    )
    param_stats = count_model_parameters(full_model)

    # Verify end-to-end equivalence directly on a raw JPEG test batch from disk
    _, _, test_ds = create_image_datasets(
        image_size=spec.image_size,
        split_dir=split_dir,
        batch_size=BATCH_SIZE,
        shuffle_train=False,
    )
    sample_x_tf, _ = next(iter(test_ds))
    e2e_sample_probs = full_model.predict(sample_x_tf.numpy(), batch_size=BATCH_SIZE, verbose=0).reshape(-1)

    with np.load(FEATURES_CACHE_DIR / f"{model_key}_features.npz") as cached:
        head_sample_probs = trained_head.predict(
            cached["test_x"][: len(e2e_sample_probs)], verbose=0
        ).reshape(-1)
    max_diff = float(np.max(np.abs(e2e_sample_probs - head_sample_probs)))
    if max_diff > 2e-2:
        raise RuntimeError(
            f"End-to-end parity check failed for {spec.display_name}: max prob diff = {max_diff:.5f}"
        )
    print(
        f"[trainer] Verified end-to-end {spec.display_name} on raw test JPEGs "
        f"(max prob diff vs cached features = {max_diff:.5f})."
    )
    e2e_test_metrics = experiment_record["test_metrics"]

    models_dir.mkdir(parents=True, exist_ok=True)
    save_path = models_dir / spec.save_filename
    full_model.save(save_path)
    size_mb = round(save_path.stat().st_size / (1024 * 1024), 2)

    final_summary: dict[str, Any] = {
        **experiment_record,
        "save_filename": spec.save_filename,
        "model_size_mb": size_mb,
        "image_size": list(spec.image_size),
        "preprocess_description": spec.preprocess_description,
        "extraction_seconds": round(extraction_seconds, 2),
        "total_pipeline_seconds": round(
            extraction_seconds + float(experiment_record["head_train_seconds"]), 2
        ),
        **param_stats,
        "test_metrics": e2e_test_metrics,
    }

    (HISTORIES_DIR / f"{model_key}_final_summary.json").write_text(
        json.dumps(final_summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        f"[trainer] Saved {save_path.name} ({size_mb} MB) | "
        f"End-to-End Test Acc: {e2e_test_metrics['accuracy']*100:.2f}% | "
        f"ROC-AUC: {e2e_test_metrics['roc_auc']:.4f}"
    )

    del full_model
    release_memory()
    return final_summary


def predict_external_images(
    model_path: Path,
    external_dir: Path = EXTERNAL_TEST_DIR,
) -> pd.DataFrame:
    """Load a saved .keras model from disk and classify all external unseen cat/dog images."""
    model = keras.models.load_model(model_path, compile=False)
    target_h, target_w = int(model.input_shape[1]), int(model.input_shape[2])

    rows: list[dict[str, Any]] = []
    batch_imgs: list[np.ndarray] = []
    meta_items: list[tuple[str, Path]] = []

    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        cls_folder = external_dir / cls_name
        for img_path in sorted(cls_folder.glob("*.jpg")):
            with Image.open(img_path) as img:
                resized = img.convert("RGB").resize((target_w, target_h), Image.Resampling.BILINEAR)
                arr = np.asarray(resized, dtype=np.float32)
            batch_imgs.append(arr)
            meta_items.append((cls_name, img_path))

    x_batch = np.stack(batch_imgs, axis=0)
    probs = model.predict(x_batch, verbose=0).reshape(-1)

    for (true_cls, img_path), dog_prob in zip(meta_items, probs, strict=True):
        pred_cls = "dogs" if dog_prob >= 0.5 else "cats"
        confidence = float(dog_prob if pred_cls == "dogs" else (1.0 - dog_prob))
        rows.append(
            {
                "filename": img_path.name,
                "rel_path": f"{true_cls}/{img_path.name}",
                "true_class": true_cls,
                "pred_class": pred_cls,
                "dog_probability": round(float(dog_prob), 4),
                "confidence": round(confidence, 4),
                "correct": bool(true_cls == pred_cls),
            }
        )

    del model
    release_memory()
    return pd.DataFrame(rows)

