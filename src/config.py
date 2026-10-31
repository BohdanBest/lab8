"""Centralized configuration, paths, reproducibility seeds, and model specifications."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import sys

# Project directory layout
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
SPLIT_DATA_DIR: Path = DATA_DIR / "split"
EXTERNAL_TEST_DIR: Path = DATA_DIR / "external_test"
FEATURES_CACHE_DIR: Path = DATA_DIR / "features_cache"
KERAS_CACHE_DIR: Path = DATA_DIR / "keras_cache"
MPL_CACHE_DIR: Path = DATA_DIR / "mpl_cache"

# Ensure local writable caches for Keras weights and Matplotlib fonts (outside Google Colab)
if "google.colab" not in sys.modules:
    KERAS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("KERAS_HOME", str(KERAS_CACHE_DIR))
    os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))

MODELS_DIR: Path = PROJECT_ROOT / "models"
RESULTS_DIR: Path = PROJECT_ROOT / "results"
HISTORIES_DIR: Path = RESULTS_DIR / "histories"
FIGURES_DIR: Path = RESULTS_DIR / "figures"

# Dataset & training constants
DATASET_URL: str = (
    "https://download.microsoft.com/download/3/E/1/"
    "3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip"
)
CLASS_NAMES: tuple[str, str] = ("cats", "dogs")  # 0 -> cats, 1 -> dogs

SEED: int = 42
TRAIN_RATIO: float = 0.80
VAL_RATIO: float = 0.10
TEST_RATIO: float = 0.10

BATCH_SIZE: int = 32
HEAD_BATCH_SIZE: int = 64
DEFAULT_EPOCHS: int = 20
DEFAULT_LEARNING_RATE: float = 1e-3
DENSE_UNITS: int = 256
DROPOUT_EXPERIMENTS: tuple[float, ...] = (0.0, 0.2, 0.5, 0.7)


@dataclass(frozen=True)
class ModelSpec:
    """Specification of a pretrained Keras Applications backbone."""

    key: str
    display_name: str
    image_size: tuple[int, int]
    preprocess_description: str
    save_filename: str


MODEL_SPECS: dict[str, ModelSpec] = {
    "vgg16": ModelSpec(
        key="vgg16",
        display_name="VGG16",
        image_size=(224, 224),
        preprocess_description="Caffe-style (RGB -> BGR, zero-centered ImageNet mean)",
        save_filename="vgg16_best.keras",
    ),
    "resnet50": ModelSpec(
        key="resnet50",
        display_name="ResNet50",
        image_size=(224, 224),
        preprocess_description="Caffe-style (RGB -> BGR, zero-centered ImageNet mean)",
        save_filename="resnet50.keras",
    ),
    "inception_v3": ModelSpec(
        key="inception_v3",
        display_name="InceptionV3",
        image_size=(299, 299),
        preprocess_description="TF-style (pixels scaled to [-1, 1])",
        save_filename="inception_v3.keras",
    ),
    "mobilenet_v2": ModelSpec(
        key="mobilenet_v2",
        display_name="MobileNetV2",
        image_size=(224, 224),
        preprocess_description="TF-style (pixels scaled to [-1, 1])",
        save_filename="mobilenet_v2.keras",
    ),
}


def ensure_directories() -> None:
    """Create all project working directories if they do not exist."""
    for directory in (
        RAW_DATA_DIR,
        SPLIT_DATA_DIR,
        EXTERNAL_TEST_DIR,
        FEATURES_CACHE_DIR,
        MODELS_DIR,
        HISTORIES_DIR,
        FIGURES_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)


def set_global_seed(seed: int = SEED) -> None:
    """Fix random seeds across Python, NumPy, and Keras/TensorFlow for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import keras

        keras.utils.set_random_seed(seed)
    except ImportError:
        pass
