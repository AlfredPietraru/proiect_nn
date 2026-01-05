from __future__ import annotations

from pathlib import Path
from typing import Callable
import zipfile
import subprocess

import gdown
import dataset_tools as dtools

from data.datasets.config import VOC_KAGGLE_DATASETS
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


def extract_zip(
    zip_dir: str | Path,
    dst_dir: str | Path, *,
    details: Logger
) -> None:
    zip_dir = Path(zip_dir)
    dst_dir = Path(dst_dir)

    ensure_dir(dst_dir)
    if is_nonempty_dir(dst_dir):
        return

    log(details, f"Extracting {zip_dir.name} -> {dst_dir}")
    with zipfile.ZipFile(str(zip_dir), "r") as zf:
        zf.extractall(str(dst_dir))


def download_asset(
    name: str,
    dst: str | Path, *,
    force: bool = False,
    check_exists: Callable[[Path], bool],
    download_fn: Callable[[Path], None],
    details: Logger,
) -> Path:
    dst_path = Path(dst)

    if (not force) and check_exists(dst_path):
        log(details, f"{name}: exists, skipping -> {dst_path}")
        return dst_path

    if dst_path.suffix:
        ensure_dir(dst_path.parent)
    else:
        ensure_dir(dst_path)

    log(details, f"{name}: downloading -> {dst_path}")
    download_fn(dst_path)

    if not check_exists(dst_path):
        raise RuntimeError(f"{name}: download finished but target is missing/empty -> {dst_path}")

    log(details, f"{name}: downloaded -> {dst_path}")
    return dst_path


def download_gdrive(
    file_id: str,
    out_path: str | Path, *,
    name: str = "gdrive",
    force: bool = False,
    quiet: bool = False,
    details: Logger
) -> Path:
    out_path = Path(out_path)

    def exists(p: Path) -> bool:
        return file_exists(p)

    def download_fn(p: Path) -> None:
        url = f"https://drive.google.com/uc?id={file_id}"
        saved = gdown.download(url, str(p), quiet=quiet, fuzzy=True)
        if saved is None:
            raise RuntimeError("gdown returned None (download failed)")

    return download_asset(
        name=name, dst=out_path, force=force,
        check_exists=exists, download_fn=download_fn, details=details)


def download_dataset_supervisely(
    dataset_name: str,
    dst_dir: str | Path, *,
    force: bool = False,
    details: Logger,
) -> Path:
    dst_dir = Path(dst_dir)

    def exists(p: Path) -> bool:
        return is_nonempty_dir(p)

    def download_fn(p: Path) -> None:
        dtools.download(dataset=dataset_name, dst_dir=str(p))

    return download_asset(
        name=f"dataset:{dataset_name}", dst=dst_dir, force=force,
        check_exists=exists, download_fn=download_fn, details=details)


def download_voc(
    dst_dir: str | Path,
    details: Logger,
    force: bool = False,
    years: tuple[str, ...] = ("2007", "2012"),
) -> Path:
    dst_dir = Path(dst_dir)
    devkit = dst_dir / "VOCdevkit"
    ensure_dir(devkit)

    def exists(_: Path) -> bool:
        for y in years:
            if not (devkit / f"VOC{y}" / "JPEGImages").exists():
                return False
        return True

    def download_fn(_: Path) -> None:
        for y in years:
            voc_dir = devkit / f"VOC{y}"
            if (not force) and (voc_dir / "JPEGImages").exists():
                log(details, f"VOC{y} exists, skipping download")
                continue

            slug = VOC_KAGGLE_DATASETS.get(str(y))
            if slug is None:
                raise RuntimeError(f"No Kaggle dataset configured for VOC{y}")

            log(details, f"Downloading VOC{y}: {slug}")
            cmd = ["kaggle", "datasets", "download", "-d", slug, "-p", str(dst_dir), "--unzip"]

            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"Kaggle download failed: {slug} (code: {e.returncode})") from e

            log(details, f"VOC{y} downloaded to: {voc_dir}")

    return download_asset(
        name="dataset:VOC", dst=devkit, force=force,
        check_exists=exists, download_fn=download_fn, details=details)


def download_uavdt(dst_dir: str | Path, details: Logger, force: bool = False) -> Path:
    return download_dataset_supervisely("UAVDT", dst_dir, force=force, details=details)


def download_visdrone(dst_dir: str | Path, details: Logger, force: bool = False) -> Path:
    return download_dataset_supervisely("VisDrone2019-DET", dst_dir, force=force, details=details)


def download_auair(dst_dir: str | Path, details: Logger, force: bool = False, quiet: bool = False) -> Path:
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

        download_gdrive(AUAIR_IMAGES_ID, img_zip, name="AU-AIR:images", force=force, quiet=quiet, details=details)
        download_gdrive(AUAIR_ANN_ID, ann_zip, name="AU-AIR:annotations", force=force, quiet=quiet, details=details)

        extract_zip(img_zip, images_dir, details=details)
        extract_zip(ann_zip, ann_dir, details=details)

    return download_asset(
        name="dataset:AU-AIR", dst=dst_dir, force=force,
        check_exists=exists, download_fn=download_fn, details=details)


def download_all_datasets(
    details: Logger,
    voc_dir: str | Path = "datasets/VOCdevkit",
    uavdt_dir: str | Path = "datasets/UAVDT_SUPERVISELY",
    visdrone_dir: str | Path = "datasets/VISDRONE_SUPERVISELY",
    auair_dir: str | Path = "datasets/AU_AIR",
    force: bool = False, quiet: bool = False
) -> None:
    download_voc(voc_dir, details=details, force=force, years=("2007", "2012"))
    download_uavdt(uavdt_dir, details=details, force=force)
    download_visdrone(visdrone_dir, details=details, force=force)
    download_auair(auair_dir, details=details, force=force, quiet=quiet)


if __name__ == "__main__":
    download_all_datasets(force=False, quiet=False, details=Logger(app="DATASET_DOWNLOADER"))
