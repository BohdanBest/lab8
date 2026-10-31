"""Dataset download, strict JPEG/PIL validation, stratified splitting, and external test set setup."""

from __future__ import annotations

import hashlib
import io
import json
import random
import shutil
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import requests
from PIL import Image, ImageFile
from tqdm.auto import tqdm

from src.config import (
    CLASS_NAMES,
    DATASET_URL,
    EXTERNAL_TEST_DIR,
    RAW_DATA_DIR,
    SEED,
    SPLIT_DATA_DIR,
    TEST_RATIO,
    TRAIN_RATIO,
    VAL_RATIO,
)

# Ensure truncated images raise an error instead of silently filling zeros
ImageFile.LOAD_TRUNCATED_IMAGES = False

HTTP_HEADERS: dict[str, str] = {
    "User-Agent": (
        "UniversityDeepLearningLab8/1.0 "
        "(TransferLearningResearch; Python-Requests)"
    )
}

# Curated external real-world cat & dog photographs (Wikimedia Commons / open repositories)
# Primary + fallback URLs per sample to guarantee resilient automated setup.
EXTERNAL_SAMPLES: dict[str, list[tuple[str, list[str]]]] = {
    "cats": [
        (
            "ext_cat_01_tabby.jpg",
            [
                "https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba?w=640&q=85",
                "https://upload.wikimedia.org/wikipedia/commons/3/3a/Cat03.jpg",
            ],
        ),
        (
            "ext_cat_02_orange.jpg",
            [
                "https://images.unsplash.com/photo-1573865526739-10659fec78a5?w=640&q=85",
                "https://upload.wikimedia.org/wikipedia/commons/4/4d/Cat_November_2010-1a.jpg",
            ],
        ),
        (
            "ext_cat_03_white_cat.jpg",
            [
                "https://images.unsplash.com/photo-1513245543132-31f507417b26?w=640&q=85",
                "https://upload.wikimedia.org/wikipedia/commons/2/25/Siam_lilacpoint.jpg",
            ],
        ),
        (
            "ext_cat_04_kitten.jpg",
            [
                "https://images.unsplash.com/photo-1595433707802-6b2626ef1c91?w=640&q=85",
                "https://upload.wikimedia.org/wikipedia/commons/b/bc/Juvenile_Ragdoll.jpg",
            ],
        ),
        (
            "ext_cat_05_portrait.jpg",
            [
                "https://images.unsplash.com/photo-1495360010541-f48722b34f7d?w=640&q=85",
                "https://upload.wikimedia.org/wikipedia/commons/b/bb/Kittyply_edit1.jpg",
            ],
        ),
    ],
    "dogs": [
        (
            "ext_dog_01_golden_retriever.jpg",
            [
                "https://images.unsplash.com/photo-1552053831-71594a27632d?w=640&q=85",
                "https://upload.wikimedia.org/wikipedia/commons/b/bd/Golden_Retriever_Dukedestiny01_drvd.jpg",
            ],
        ),
        (
            "ext_dog_02_samoyed.jpg",
            [
                "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg",
                "https://images.unsplash.com/photo-1583511655857-d19b40a7a54e?w=640&q=85",
            ],
        ),
        (
            "ext_dog_03_beagle.jpg",
            [
                "https://images.unsplash.com/photo-1543466835-00a7907e9de1?w=640&q=85",
                "https://upload.wikimedia.org/wikipedia/commons/5/55/Beagle_600.jpg",
            ],
        ),
        (
            "ext_dog_04_german_shepherd.jpg",
            [
                "https://images.unsplash.com/photo-1589941013453-ec89f33b5e95?w=640&q=85",
                "https://upload.wikimedia.org/wikipedia/commons/d/d0/German_Shepherd_-_DSC_0346_%2810096362833%29.jpg",
            ],
        ),
        (
            "ext_dog_05_pug.jpg",
            [
                "https://images.unsplash.com/photo-1517849845537-4d257902454a?w=640&q=85",
                "https://upload.wikimedia.org/wikipedia/commons/f/f0/Mops_oct09_cropped2.jpg",
            ],
        ),
    ],
}


def download_dogs_vs_cats(raw_dir: Path = RAW_DATA_DIR, url: str = DATASET_URL) -> Path:
    """Download and extract the 25,000-image Dogs vs. Cats archive if not already present."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    pet_images_dir = raw_dir / "PetImages"
    cat_dir = pet_images_dir / "Cat"
    dog_dir = pet_images_dir / "Dog"

    if cat_dir.is_dir() and dog_dir.is_dir():
        n_cats = sum(1 for _ in cat_dir.glob("*.jpg"))
        n_dogs = sum(1 for _ in dog_dir.glob("*.jpg"))
        if n_cats >= 12000 and n_dogs >= 12000:
            print(f"[data_prep] Raw PetImages already extracted ({n_cats} cats, {n_dogs} dogs).")
            return pet_images_dir

    zip_path = raw_dir / "kagglecatsanddogs_5340.zip"
    if not zip_path.exists():
        tmp_path = raw_dir / "kagglecatsanddogs_5340.zip.part"
        print(f"[data_prep] Downloading Dogs vs. Cats archive (~786 MB) from:\n  {url}")
        with requests.get(url, headers=HTTP_HEADERS, stream=True, timeout=60) as resp:
            resp.raise_for_status()
            total_bytes = int(resp.headers.get("content-length", 0))
            with (
                open(tmp_path, "wb") as f_out,
                tqdm(
                    total=total_bytes,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    desc="Downloading archive",
                ) as pbar,
            ):
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f_out.write(chunk)
                        pbar.update(len(chunk))
        tmp_path.rename(zip_path)

    print(f"[data_prep] Extracting {zip_path.name} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = [m for m in zf.namelist() if m.startswith("PetImages/")]
        for member in tqdm(members, desc="Extracting PetImages"):
            zf.extract(member, path=raw_dir)

    return pet_images_dir


def _inspect_image(src_path: Path) -> tuple[Path, str, str]:
    """Validate an image file for TensorFlow JPEG compatibility.

    Returns:
        (src_path, status, detail) where status is one of:
        - 'copy': strict 3-channel RGB JPEG with valid SOI + JFIF/Exif marker
        - 'reencode': valid decodable image (e.g. CMYK, Grayscale, PNG-in-disguise) needing RGB JPEG save
        - 'corrupt': unreadable, zero-byte, truncated, or too small (<32x32)
    """
    try:
        raw_bytes = src_path.read_bytes()
        if len(raw_bytes) < 128:
            return (src_path, "corrupt", "empty_or_tiny_file")

        with Image.open(io.BytesIO(raw_bytes)) as img:
            img.load()  # Force full pixel decompression to catch truncated streams
            width, height = img.size
            if width < 32 or height < 32:
                return (src_path, "corrupt", f"dimensions_too_small_{width}x{height}")

            img_format = img.format
            img_mode = img.mode

        has_jpeg_soi = raw_bytes[:2] == b"\xff\xd8"
        has_std_header = (b"JFIF" in raw_bytes[:64]) or (b"Exif" in raw_bytes[:64])

        if has_jpeg_soi and img_format == "JPEG" and img_mode == "RGB" and has_std_header:
            return (src_path, "copy", "strict_rgb_jpeg")
        return (src_path, "reencode", f"{img_format}_{img_mode}")
    except Exception as exc:  # noqa: BLE001
        return (src_path, "corrupt", type(exc).__name__)


def _materialize_image(item: tuple[Path, Path, str]) -> None:
    """Copy a strict RGB JPEG or re-encode a non-standard image into clean 3-channel RGB JPEG."""
    src_path, dst_path, mode = item
    if mode == "copy":
        shutil.copy2(src_path, dst_path)
    else:
        with Image.open(src_path) as img:
            rgb_img = img.convert("RGB")
            rgb_img.save(dst_path, format="JPEG", quality=95)


def prepare_dataset_splits(
    raw_dir: Path = RAW_DATA_DIR,
    split_dir: Path = SPLIT_DATA_DIR,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
    seed: int = SEED,
    force: bool = False,
    max_workers: int = 8,
) -> dict[str, Any]:
    """Validate all 25,000 raw images and build stratified train/val/test folders."""
    manifest_path = split_dir / "split_manifest.json"
    if manifest_path.exists() and not force:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        print(
            f"[data_prep] Existing split found ({manifest['total_valid']} valid images). "
            "Skipping re-split."
        )
        return manifest

    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must sum to 1.0")

    pet_images_dir = download_dogs_vs_cats(raw_dir=raw_dir)
    if split_dir.exists():
        shutil.rmtree(split_dir)

    for split_name in ("train", "val", "test"):
        for cls_name in CLASS_NAMES:
            (split_dir / split_name / cls_name).mkdir(parents=True, exist_ok=True)

    raw_class_map = {
        "cats": pet_images_dir / "Cat",
        "dogs": pet_images_dir / "Dog",
    }

    rng = random.Random(seed)
    stats: dict[str, Any] = {
        "seed": seed,
        "ratios": {"train": train_ratio, "val": val_ratio, "test": test_ratio},
        "splits": {"train": {}, "val": {}, "test": {}},
        "corrupted_files": [],
        "reencoded_count": 0,
        "total_valid": 0,
    }

    copy_jobs: list[tuple[Path, Path, str]] = []

    for cls_name, src_cls_dir in raw_class_map.items():
        candidates = sorted(
            [p for p in src_cls_dir.iterdir() if p.is_file() and p.suffix.lower() == ".jpg"],
            key=lambda p: p.name,
        )
        valid_records: list[tuple[Path, str]] = []

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            results = list(
                tqdm(
                    pool.map(_inspect_image, candidates),
                    total=len(candidates),
                    desc=f"Validating raw {cls_name}",
                )
            )

        for src_path, status, detail in results:
            if status == "corrupt":
                stats["corrupted_files"].append(
                    {"file": f"{src_cls_dir.name}/{src_path.name}", "reason": detail}
                )
            else:
                if status == "reencode":
                    stats["reencoded_count"] += 1
                valid_records.append((src_path, status))

        rng.shuffle(valid_records)
        n_total = len(valid_records)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)

        split_slices = {
            "train": valid_records[:n_train],
            "val": valid_records[n_train : n_train + n_val],
            "test": valid_records[n_train + n_val :],
        }

        prefix = cls_name[:-1]  # 'cat' or 'dog'
        for split_name, records in split_slices.items():
            stats["splits"][split_name][cls_name] = len(records)
            stats["total_valid"] += len(records)
            for src_path, status in records:
                dst_name = f"{prefix}_{src_path.stem}.jpg"
                dst_path = split_dir / split_name / cls_name / dst_name
                copy_jobs.append((src_path, dst_path, status))

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        list(
            tqdm(
                pool.map(_materialize_image, copy_jobs),
                total=len(copy_jobs),
                desc="Materializing train/val/test splits",
            )
        )

    manifest_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"[data_prep] Split complete: {stats['total_valid']} valid images "
        f"({len(stats['corrupted_files'])} corrupted skipped, "
        f"{stats['reencoded_count']} standardized to RGB JPEG)."
    )
    return stats


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def prepare_external_test_images(
    external_dir: Path = EXTERNAL_TEST_DIR,
    split_dir: Path = SPLIT_DATA_DIR,
    force: bool = False,
) -> list[dict[str, str]]:
    """Download 10 external cat/dog photos and verify SHA-256 non-overlap with Kaggle splits."""
    manifest_path = external_dir / "external_manifest.json"
    if manifest_path.exists() and not force:
        records = json.loads(manifest_path.read_text(encoding="utf-8"))
        if len(records) >= 10 and all((external_dir / r["rel_path"]).exists() for r in records):
            print(f"[data_prep] External test set ready ({len(records)} images).")
            return records

    for cls_name in CLASS_NAMES:
        (external_dir / cls_name).mkdir(parents=True, exist_ok=True)

    # Collect SHA-256 hashes of dataset splits if present to guarantee zero data leakage
    dataset_hashes: set[str] = set()
    if split_dir.exists():
        for img_p in split_dir.rglob("*.jpg"):
            dataset_hashes.add(_sha256_file(img_p))

    downloaded_records: list[dict[str, str]] = []
    for cls_name, items in EXTERNAL_SAMPLES.items():
        for filename, urls in items:
            dst_path = external_dir / cls_name / filename
            saved = False
            last_err = ""
            for url in urls:
                try:
                    resp = requests.get(url, headers=HTTP_HEADERS, timeout=30)
                    resp.raise_for_status()
                    with Image.open(io.BytesIO(resp.content)) as img:
                        rgb = img.convert("RGB")
                        rgb.save(dst_path, format="JPEG", quality=95)
                    saved = True
                    source_url = url
                    break
                except Exception as exc:  # noqa: BLE001
                    last_err = str(exc)

            if not saved:
                raise RuntimeError(f"Failed to download external sample {filename}: {last_err}")

            file_hash = _sha256_file(dst_path)
            if file_hash in dataset_hashes:
                raise RuntimeError(f"External sample {filename} collided with Kaggle dataset!")

            downloaded_records.append(
                {
                    "class_name": cls_name,
                    "filename": filename,
                    "rel_path": f"{cls_name}/{filename}",
                    "sha256": file_hash,
                    "source_url": source_url,
                }
            )

    manifest_path.write_text(
        json.dumps(downloaded_records, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        f"[data_prep] Downloaded and verified {len(downloaded_records)} external test images "
        "(0 SHA-256 collisions with Kaggle dataset)."
    )
    return downloaded_records
