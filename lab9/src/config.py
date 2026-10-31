"""Centralized configuration, paths, reproducibility seeds, and Fine-Tuning block specifications."""

from __future__ import annotations

import os
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Project directory layout (Lab 9 lives in lab8/lab9 and reuses shared ../data locally)
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
PARENT_LAB8_ROOT: Path = PROJECT_ROOT.parent
_USE_SHARED_DATA: bool = (PARENT_LAB8_ROOT / "data" / "split").exists()

DATA_DIR: Path = (PARENT_LAB8_ROOT / "data") if _USE_SHARED_DATA else (PROJECT_ROOT / "data")
RAW_DATA_DIR: Path = DATA_DIR / "raw"
SPLIT_DATA_DIR: Path = DATA_DIR / "split"
EXTERNAL_TEST_DIR: Path = DATA_DIR / "external_test"
FEATURES_CACHE_DIR: Path = DATA_DIR / "features_cache"
KERAS_CACHE_DIR: Path = DATA_DIR / "keras_cache"
MPL_CACHE_DIR: Path = DATA_DIR / "mpl_cache"

# Temporary on-disk memmap directory for frozen-prefix tensors (auto-purged after each backbone)
TEMP_PREFIX_CACHE_DIR: Path = PROJECT_ROOT / ".tmp_prefix_cache"

# Lab 8 reference artifacts for direct "Before vs After Fine-Tuning" comparison
LAB8_MODELS_DIR: Path = PARENT_LAB8_ROOT / "models"
LAB8_RESULTS_DIR: Path = PARENT_LAB8_ROOT / "results"
LAB8_SUMMARY_CSV: Path = LAB8_RESULTS_DIR / "lab8_summary_for_lab9.csv"
LAB8_SUMMARY_JSON: Path = LAB8_RESULTS_DIR / "lab8_summary_for_lab9.json"

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

# Dataset & reproducibility constants (identical split to Lab 8 for 1-to-1 comparison)
DATASET_URL: str = (
    "https://download.microsoft.com/download/3/E/1/"
    "3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip"
)
CLASS_NAMES: tuple[str, str] = ("cats", "dogs")  # 0 -> cats, 1 -> dogs

SEED: int = 42
TRAIN_RATIO: float = 0.80
VAL_RATIO: float = 0.10
TEST_RATIO: float = 0.10

# Stratified hyperparameter-search subset size for fast, controlled multi-factor ablations
TUNE_SUBSET_TRAIN: int = 4000
TUNE_SUBSET_VAL: int = 1000

# Default training hyperparameters
EXTRACT_BATCH_SIZE: int = 32
DEFAULT_WARMUP_EPOCHS: int = 12
DEFAULT_WARMUP_LR: float = 1e-3
DEFAULT_FINETUNE_EPOCHS: int = 15
DEFAULT_FINETUNE_LR: float = 1e-5
DEFAULT_BATCH_SIZE: int = 50
DEFAULT_DENSE_UNITS: int = 256
DEFAULT_DROPOUT: float = 0.5

# Task 4 Exercise hyperparameter grids (VGG16)
EX_DENSE_UNITS: tuple[int, ...] = (32, 128, 256, 512, 1024)
EX_DROPOUT_RATES: tuple[float, ...] = (0.25, 0.5, 0.6, 0.8)
EX_SECOND_DENSE_UNITS: tuple[int, ...] = (0, 32, 128, 256)
EX_EPOCH_CHECKPOINTS: tuple[int, ...] = (5, 10, 15, 20, 25)
EX_BATCH_SIZES: tuple[int, ...] = (25, 50, 100, 125)
EX_OPTIMIZERS: tuple[str, ...] = ("sgd", "adam", "rmsprop")
EX_ADAM_LRS: tuple[float, ...] = (1e-3, 1e-4, 1e-5, 1e-6)
EX_UNFREEZE_BLOCKS: tuple[int, ...] = (1, 2, 3)


@dataclass(frozen=True)
class UnfreezeBlockSpec:
    """Clean single-tensor graph cut point before the unfrozen top blocks of a backbone.

    Attributes:
        num_blocks: Logical number of top architectural blocks unfrozen (1, 2, or 3).
        cut_layer_name: Output layer name of the frozen prefix (all layers after it are in suffix).
        description: Human-readable explanation of which convolutional blocks become trainable.
    """

    num_blocks: int
    cut_layer_name: str
    description: str


@dataclass(frozen=True)
class ModelSpec:
    """Specification of an ImageNet-pretrained Keras Applications backbone for Fine-Tuning."""

    key: str
    display_name: str
    image_size: tuple[int, int]
    preprocess_description: str
    save_filename: str
    lab8_save_filename: str
    has_batch_norm: bool
    unfreeze_options: dict[int, UnfreezeBlockSpec]


MODEL_SPECS: dict[str, ModelSpec] = {
    "vgg16": ModelSpec(
        key="vgg16",
        display_name="VGG16",
        image_size=(224, 224),
        preprocess_description="Caffe-style (RGB -> BGR, zero-centered ImageNet mean)",
        save_filename="vgg16_finetuned.keras",
        lab8_save_filename="vgg16_best.keras",
        has_batch_norm=False,
        unfreeze_options={
            1: UnfreezeBlockSpec(
                num_blocks=1,
                cut_layer_name="block4_pool",
                description="1 блок: block5_conv1..block5_conv3 (3 згорткові шари)",
            ),
            2: UnfreezeBlockSpec(
                num_blocks=2,
                cut_layer_name="block3_pool",
                description="2 блоки: block4 + block5 (6 згорткових шарів)",
            ),
            3: UnfreezeBlockSpec(
                num_blocks=3,
                cut_layer_name="block2_pool",
                description="3 блоки: block3 + block4 + block5 (9 згорткових шарів)",
            ),
        },
    ),
    "resnet50": ModelSpec(
        key="resnet50",
        display_name="ResNet50",
        image_size=(224, 224),
        preprocess_description="Caffe-style (RGB -> BGR, zero-centered ImageNet mean)",
        save_filename="resnet50_finetuned.keras",
        lab8_save_filename="resnet50.keras",
        has_batch_norm=True,
        unfreeze_options={
            1: UnfreezeBlockSpec(
                num_blocks=1,
                cut_layer_name="conv4_block6_out",
                description="1 етап: conv5_block1..conv5_block3 (BatchNorm заморожено)",
            ),
            2: UnfreezeBlockSpec(
                num_blocks=2,
                cut_layer_name="conv3_block4_out",
                description="2 етапи: conv4_block1..6 + conv5_block1..3 (BatchNorm заморожено)",
            ),
        },
    ),
    "inception_v3": ModelSpec(
        key="inception_v3",
        display_name="InceptionV3",
        image_size=(299, 299),
        preprocess_description="TF-style (pixels scaled to [-1, 1])",
        save_filename="inception_v3_finetuned.keras",
        lab8_save_filename="inception_v3.keras",
        has_batch_norm=True,
        unfreeze_options={
            1: UnfreezeBlockSpec(
                num_blocks=1,
                cut_layer_name="mixed9",
                description="1 блок: mixed10 (BatchNorm заморожено)",
            ),
            2: UnfreezeBlockSpec(
                num_blocks=2,
                cut_layer_name="mixed8",
                description="2 блоки: mixed9 + mixed10 (BatchNorm заморожено)",
            ),
            3: UnfreezeBlockSpec(
                num_blocks=3,
                cut_layer_name="mixed7",
                description="3 блоки: mixed8 + mixed9 + mixed10 (BatchNorm заморожено)",
            ),
        },
    ),
    "mobilenet_v2": ModelSpec(
        key="mobilenet_v2",
        display_name="MobileNetV2",
        image_size=(224, 224),
        preprocess_description="TF-style (pixels scaled to [-1, 1])",
        save_filename="mobilenet_v2_finetuned.keras",
        lab8_save_filename="mobilenet_v2.keras",
        has_batch_norm=True,
        unfreeze_options={
            1: UnfreezeBlockSpec(
                num_blocks=1,
                cut_layer_name="block_15_add",
                description="1 блок: block_16 + Conv_1 1280ch (BatchNorm заморожено)",
            ),
            2: UnfreezeBlockSpec(
                num_blocks=2,
                cut_layer_name="block_14_add",
                description="2 блоки: block_15 + block_16 + Conv_1 (BatchNorm заморожено)",
            ),
            3: UnfreezeBlockSpec(
                num_blocks=3,
                cut_layer_name="block_12_add",
                description="4 блоки: block_13..block_16 + Conv_1 (BatchNorm заморожено)",
            ),
        },
    ),
}


def ensure_directories() -> None:
    """Create all working directories for Lab 9."""
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


def cleanup_temp_prefix_cache() -> None:
    """Remove temporary frozen-prefix memmap files to keep SSD usage minimal."""
    if TEMP_PREFIX_CACHE_DIR.exists():
        shutil.rmtree(TEMP_PREFIX_CACHE_DIR, ignore_errors=True)


def set_global_seed(seed: int = SEED) -> None:
    """Fix random seeds across Python, NumPy, Keras, and PyTorch for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import keras

        keras.utils.set_random_seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass

