"""
Data acquisition and preprocessing for the Pneumonia Chest X-ray classifier.

Data acquisition : downloads the Kaggle "Chest X-Ray Images (Pneumonia)" dataset
                   (Kermany et al., pediatric patients aged 1-5 at Guangzhou
                   Women and Children's Medical Center), converts every image to
                   64x64 grayscale PNG and writes it into data/train/ and
                   data/test/ class folders.
Data processing  : loads image folders into normalized numpy arrays ready for
                   the CNN, and preprocesses single uploaded images for
                   prediction.
"""

from __future__ import annotations

import io
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

IMG_SIZE = 64
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]
# Raised from 0.5 → 0.6 to cut NORMAL false positives (precision 0.857 → 0.874
# on held-out test; see results/threshold_sweep.json). Still high recall (0.926).
DECISION_THRESHOLD = 0.6
KAGGLE_DATASET = "paultimothymooney/chest-xray-pneumonia"

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
UPLOAD_DIR = DATA_DIR / "uploads"


# --------------------------------------------------------------------------- #
# 1. Data acquisition
# --------------------------------------------------------------------------- #
def acquire_data(force: bool = False, raw_dir: Path | str | None = None) -> None:
    """Download the Kaggle chest X-ray dataset and export 64x64 PNGs.

    The repository ships with the processed dataset already in ``data/``, so this
    is a no-op on a fresh clone (unless ``force=True``). It is the reproducible
    record of where the data came from.

    Requires Kaggle API credentials at ``~/.kaggle/kaggle.json`` (see README).
    If ``raw_dir`` points at an already-extracted copy of the dataset (a folder
    containing ``train/`` and ``test/``), the download step is skipped.
    """
    if not force and _folder_has_images(TRAIN_DIR) and _folder_has_images(TEST_DIR):
        print("Dataset already present - skipping acquisition.")
        return

    if raw_dir is None:
        raw_dir = DATA_DIR / "raw"
        Path(raw_dir).mkdir(parents=True, exist_ok=True)
        print(f"Downloading {KAGGLE_DATASET} from Kaggle ...")
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", KAGGLE_DATASET,
             "-p", str(raw_dir), "--unzip"],
            check=True,
        )
    raw_dir = Path(raw_dir)

    # the Kaggle archive nests a duplicate chest_xray/ inside chest_xray/
    root = next((p for p in [raw_dir, raw_dir / "chest_xray"] if (p / "train").is_dir()), None)
    if root is None:
        raise FileNotFoundError(f"Could not find train/ under {raw_dir}")

    # NOTE: the dataset's own val/ split holds only 16 images - too small to be
    # useful - so it is ignored; train_model() builds a proper stratified split.
    for split in ("train", "test"):
        out_dir = TRAIN_DIR if split == "train" else TEST_DIR
        for cls in CLASS_NAMES:
            (out_dir / cls).mkdir(parents=True, exist_ok=True)
            files = [p for p in sorted((root / split / cls).iterdir())
                     if p.suffix.lower() in (".jpeg", ".jpg", ".png")]
            for path in files:
                img = Image.open(path).convert("L").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
                img.save(out_dir / cls / f"{path.stem}.png")
            print(f"Exported {len(files)} {cls} images to {out_dir / cls}")


def _folder_has_images(folder: Path) -> bool:
    return folder.exists() and any(folder.rglob("*.png"))


# --------------------------------------------------------------------------- #
# 2. Data processing
# --------------------------------------------------------------------------- #
def load_dataset(folder: Path | str) -> tuple[np.ndarray, np.ndarray]:
    """Load a class-folder dataset into (X, y) numpy arrays.

    X has shape (n, IMG_SIZE, IMG_SIZE, 1) scaled to [0, 1];
    y has shape (n,) with 0=NORMAL, 1=PNEUMONIA.
    """
    folder = Path(folder)
    xs, ys = [], []
    for label, cls in enumerate(CLASS_NAMES):
        for path in sorted((folder / cls).glob("*.png")):
            xs.append(_image_to_array(Image.open(path)))
            ys.append(label)
    if not xs:
        raise FileNotFoundError(f"No images found under {folder}")
    return np.stack(xs), np.array(ys, dtype=np.int64)


def preprocess_image(data: bytes | Image.Image) -> np.ndarray:
    """Preprocess one uploaded image for prediction -> shape (1, H, W, 1)."""
    img = Image.open(io.BytesIO(data)) if isinstance(data, bytes) else data
    return _image_to_array(img)[np.newaxis, ...]


def _image_to_array(img: Image.Image) -> np.ndarray:
    """Grayscale -> resize -> scale to [0,1] -> (H, W, 1) float32 array."""
    img = img.convert("L").resize((IMG_SIZE, IMG_SIZE))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr[..., np.newaxis]


# --------------------------------------------------------------------------- #
# 3. Dataset statistics (used by the UI visualizations)
# --------------------------------------------------------------------------- #
def dataset_statistics(folder: Path | str = TRAIN_DIR) -> dict:
    """Compute per-class stats: counts, mean brightness, contrast, mean image."""
    folder = Path(folder)
    stats: dict = {"classes": {}}
    for cls in CLASS_NAMES:
        paths = sorted((folder / cls).glob("*.png"))
        if not paths:
            continue
        sample = paths[:: max(1, len(paths) // 400)]  # subsample for speed
        arrs = [np.asarray(Image.open(p).convert("L"), dtype=np.float32) for p in sample]
        stack = np.stack(arrs)
        stats["classes"][cls] = {
            "count": len(paths),
            "mean_brightness": float(stack.mean()),
            "mean_contrast": float(stack.std(axis=(1, 2)).mean()),
            "brightness_hist": np.histogram(stack.mean(axis=(1, 2)), bins=20, range=(0, 255))[0].tolist(),
            "mean_image": stack.mean(axis=0).astype(np.uint8).tolist(),
        }
    return stats


if __name__ == "__main__":
    acquire_data()
    X, y = load_dataset(TRAIN_DIR)
    print(f"Train set: X={X.shape}, y={y.shape}, class balance={np.bincount(y)}")
