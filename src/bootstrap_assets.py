"""One-shot network asset downloader (dataset archive, external test photos, ImageNet weights).

Separating network fetches from local data processing allows all subsequent data validation,
splitting, feature extraction, model training, and notebook execution to run offline/sandboxed.
"""

from __future__ import annotations

import requests
from tqdm.auto import tqdm

# Import src first so KERAS_HOME and MPLCONFIGDIR point to ./data/*_cache
import src  # noqa: F401
from src.config import DATASET_URL, MODEL_SPECS, RAW_DATA_DIR, ensure_directories
from src.data_prep import HTTP_HEADERS, prepare_external_test_images


def download_archive_only() -> None:
    """Download kagglecatsanddogs_5340.zip without extracting yet."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = RAW_DATA_DIR / "kagglecatsanddogs_5340.zip"
    if zip_path.exists() and zip_path.stat().st_size > 700_000_000:
        print(f"[bootstrap] Archive already present: {zip_path} ({zip_path.stat().st_size / 1e6:.1f} MB)")
        return

    tmp_path = RAW_DATA_DIR / "kagglecatsanddogs_5340.zip.part"
    print(f"[bootstrap] Downloading {DATASET_URL} ...")
    with requests.get(DATASET_URL, headers=HTTP_HEADERS, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        total_bytes = int(resp.headers.get("content-length", 0))
        with (
            open(tmp_path, "wb") as f_out,
            tqdm(total=total_bytes, unit="B", unit_scale=True, unit_divisor=1024, desc="DogsVsCats.zip") as pbar,
        ):
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f_out.write(chunk)
                    pbar.update(len(chunk))
    tmp_path.rename(zip_path)
    print(f"[bootstrap] Saved {zip_path} ({zip_path.stat().st_size / 1e6:.1f} MB)")


def prefetch_imagenet_weights() -> None:
    """Download and cache ImageNet weights for all 4 backbones into KERAS_HOME."""
    import keras

    cls_map = {
        "vgg16": keras.applications.VGG16,
        "resnet50": keras.applications.ResNet50,
        "inception_v3": keras.applications.InceptionV3,
        "mobilenet_v2": keras.applications.MobileNetV2,
    }
    for key, spec in MODEL_SPECS.items():
        print(f"[bootstrap] Caching ImageNet weights for {spec.display_name} {spec.image_size} ...")
        _ = cls_map[key](weights="imagenet", include_top=False, input_shape=(*spec.image_size, 3))
        keras.backend.clear_session()
    print("[bootstrap] All 4 ImageNet weight files cached successfully.")


def main() -> None:
    ensure_directories()
    download_archive_only()
    prepare_external_test_images(force=False)
    prefetch_imagenet_weights()


if __name__ == "__main__":
    main()

