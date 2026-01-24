from __future__ import annotations

from typing import Callable
from pathlib import Path
import zipfile
import subprocess
import gdown
import dataset_tools as dtools

from utils.logger import Logger


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def is_nonempty_dir(path: str | Path) -> bool:
    p = Path(path)
    return p.exists() and p.is_dir() and any(p.iterdir())


def file_exists(path: str | Path) -> bool:
    p = Path(path)
    return p.exists() and p.is_file() and p.stat().st_size > 0


def log(details: Logger, msg: str) -> None:
    details.info(msg) if details else print(msg)


def safe_unlink(path: str | Path) -> None:
    p = Path(path)
    p.unlink(missing_ok=True)


def extract_zip(
    zip_dir: str | Path,
    dst_dir: str | Path,
    details: Logger
) -> None:
    zip_dir = Path(zip_dir)
    dst_dir = Path(dst_dir)

    ensure_dir(dst_dir)
    if is_nonempty_dir(dst_dir):
        safe_unlink(zip_dir)
        details.info(f"Skipping extraction, target exists: {dst_dir}")
        return

    log(details, f"Extracting {zip_dir.name} -> {dst_dir}")
    with zipfile.ZipFile(str(zip_dir), "r") as zf:
        zf.extractall(str(dst_dir))
    safe_unlink(zip_dir)


def download_asset(
    dst_dir: str | Path,
    details: Logger, name: str,
    check_exists: Callable[[Path], bool],
    download_fn: Callable[[Path], None],
    force: bool = False
) -> Path:
    dst_dir = Path(dst_dir)

    if (not force) and check_exists(dst_dir):
        log(details, f"{name}: exists, skipping -> {dst_dir}")
        return dst_dir

    if dst_dir.suffix:
        ensure_dir(dst_dir.parent)
    else:
        ensure_dir(dst_dir)

    log(details, f"{name}: downloading -> {dst_dir}")
    download_fn(dst_dir)

    if not check_exists(dst_dir):
        raise RuntimeError(f"{name}: download finished but target is missing/empty -> {dst_dir}")

    log(details, f"{name}: downloaded -> {dst_dir}")
    return dst_dir


def download_gdrive(
    dst_dir: str | Path,
    details: Logger,
    file_id: str,
    name: str = "gdrive",
    force: bool = False,
    quiet: bool = False
) -> Path:
    dst_dir = Path(dst_dir)

    def exists(p: Path) -> bool:
        return file_exists(p)

    def download_fn(p: Path) -> None:
        url = f"https://drive.google.com/uc?id={file_id}"
        saved = gdown.download(url, str(p), quiet=quiet, fuzzy=True)
        if saved is None:
            raise RuntimeError("gdown returned None (download failed)")

    return download_asset(
        name=name, dst_dir=dst_dir, force=force,
        check_exists=exists, download_fn=download_fn, details=details)


def download_dataset_supervisely(
    dst_dir: str | Path,
    dataset_name: str,
    details: Logger,
    force: bool = False
) -> Path:
    dst_dir = Path(dst_dir)

    def exists(p: Path) -> bool:
        return is_nonempty_dir(p)

    def download_fn(p: Path) -> None:
        dtools.download(dataset=dataset_name, dst_dir=str(p))

    return download_asset(
        name=f"dataset:{dataset_name}", dst_dir=dst_dir, force=force,
        check_exists=exists, download_fn=download_fn, details=details)


def download_voc(
    dst_dir: str | Path,
    details: Logger,
    years: tuple[str, ...],
    force: bool = False,
) -> None:
    """
    Download Pascal VOC from Kaggle directly to correct path structure.
    After extraction, the structure is:
    VOCdevkit/
        VOC2007/
            JPEGImages/
            Annotations/
        VOC2012/
            JPEGImages/
            Annotations/
    """
    VOC_KAGGLE_DATASETS: dict[str, str] = {
        "2007": "zaraks/pascal-voc-2007",                   # Pascal VOC 2007 - usually used for testing
        "2012": "gopalbhattrai/pascal-voc-2012-dataset",    # Pascal VOC 2012 - usually used for training
    }    

    final_devkit = Path(dst_dir) / "VOCdevkit"
    final_devkit.mkdir(parents=True, exist_ok=True)

    if not force and all((final_devkit / f"VOC{year}" / "JPEGImages").exists() for year in years):
        if details:
            details.info("VOC dataset exists, skipping download")
        return

    for year in years:
        year = str(year)
        target_dir = final_devkit / f"VOC{year}"

        if target_dir.exists() and (target_dir / "JPEGImages").exists():
            if details:
                details.info(f"VOC{year} exists, skipping download")
            continue
        if year not in VOC_KAGGLE_DATASETS:
            raise RuntimeError(f"No Kaggle dataset for VOC{year}")

        kaggle_slug = VOC_KAGGLE_DATASETS[year]
        if details:
            details.info(f"Downloading VOC{year}: {kaggle_slug}")

        cmd = ["kaggle", "datasets", "download", "-d", kaggle_slug, "-p", str(target_dir), "--unzip"]

        try:
            subprocess.run(cmd, check=True)
        except FileNotFoundError:
            raise RuntimeError("Kaggle CLI not found. Install with: pip install kaggle")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Kaggle download failed: {kaggle_slug} (code: {e.returncode})")

        if details:
            details.info(f"VOC{year} downloaded to: {target_dir}")


def download_uavdt(dst_dir: str | Path, details: Logger, force: bool = True) -> Path:
    """
    Download UAVDT dataset from Supervisely.
    After extraction, the structure is:
    uavdt/
        images/
        annotations/
    """
    return download_dataset_supervisely(dst_dir, "uavdt", details, force)


def download_visdrone(dst_dir: str | Path, details: Logger, force: bool = True) -> Path:
    """
    Download VisDrone dataset from Supervisely.
    After extraction, the structure is:
    visdrone2019-det/
        train/
        val/
        test/
    """
    return download_dataset_supervisely(dst_dir, "visdrone2019-det", details, force)


def download_auair(dst_dir: str | Path, details: Logger, force: bool = True, quiet: bool = False) -> Path:
    """
    Download AU-AIR dataset from Google Drive.
    After extraction, the structure is:
    AU_AIR/
        images/
        annotations/
    """
    AUAIR_IMAGES_ID = "1pJ3xfKtHiTdysX5G3dxqKTdGESOBYCxJ"
    AUAIR_ANN_ID = "1boGF0L6olGe_Nu7rd1R8N7YmQErCb0xA"

    dst_dir = Path(dst_dir)
    ensure_dir(dst_dir)

    images_dir = dst_dir / "images"
    ann_dir = dst_dir / "annotations"

    def exists(_: Path) -> bool:
        return is_nonempty_dir(images_dir) and is_nonempty_dir(ann_dir)

    def download_fn(_: Path) -> None:
        img_zip = dst_dir / "auair_images.zip"
        ann_zip = dst_dir / "auair_annotations.zip"

        download_gdrive(img_zip, details, AUAIR_IMAGES_ID, "AU-AIR Images", force, quiet)
        download_gdrive(ann_zip, details, AUAIR_ANN_ID, "AU-AIR Annotations", force, quiet)
    
        extract_zip(img_zip, images_dir, details)
        extract_zip(ann_zip, ann_dir, details)

    return download_asset(
        name="dataset:AU-AIR", dst_dir=dst_dir, force=force,
        check_exists=exists, download_fn=download_fn, details=details)


def download_all_datasets(
    details: Logger,
    root: str | Path = "./datasets",
    voc: str = "",  # already includes "VOCdevkit"
    visdrone: str = "",  # already includes "visdrone2019-det"
    uavdt: str = "",  # already includes "uavdt"
    auair: str = "AU_AIR",  # does not include "AU_AIR"
    force: bool = True, quiet: bool = False
) -> None:
    download_voc(Path(root) / voc, details, ("2007", "2012"), force)
    download_uavdt(Path(root) / uavdt, details, force)
    download_visdrone(Path(root) / visdrone, details, force)
    download_auair(Path(root) / auair, details, force, quiet)
