from __future__ import annotations

from pathlib import Path
from typing import Callable, List
import zipfile
import subprocess
import shutil

import gdown
import dataset_tools as dtools

from .config import VOC_KAGGLE_DATASETS
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


def voc_is_valid(v: Path) -> bool:
    return (
        v.is_dir()
        and (v / "JPEGImages").is_dir()
        and (v / "Annotations").is_dir()
        and (v / "ImageSets" / "Main").is_dir()
    )


def voc_remove_segmentation(v: Path) -> None:
    shutil.rmtree(v / "SegmentationClass", ignore_errors=True)
    shutil.rmtree(v / "SegmentationObject", ignore_errors=True)


def voc_cleanup_wrappers(devkit: Path, details: Logger) -> None:
    for p in devkit.iterdir():
        if not p.is_dir():
            continue
        if p.name.startswith("VOC20"):
            continue
        if (
            p.name.startswith("VOCtrainval_")
            or p.name.startswith("VOCtest_")
            or p.name == "VOCdevkit"
            or p.name == "PASCAL_VOC"
        ):
            log(details, f"VOC: removing wrapper -> {p}")
            shutil.rmtree(p, ignore_errors=True)


def voc_find_sources(devkit: Path, year: str) -> List[Path]:
    sources: List[Path] = []

    for p in devkit.rglob(f"VOCdevkit/VOC{year}"):
        if voc_is_valid(p):
            sources.append(p)

    for p in devkit.glob(f"VOC{year}_*"):
        if voc_is_valid(p):
            sources.append(p)

    for p in devkit.rglob(f"VOC{year}"):
        if p == devkit / f"VOC{year}":
            continue
        if voc_is_valid(p):
            sources.append(p)

    seen = set()
    uniq: List[Path] = []
    for s in sources:
        r = str(s.resolve())
        if r not in seen:
            uniq.append(s)
            seen.add(r)
    return uniq


def voc_pick_trainval_and_test(sources: List[Path]) -> tuple[Path, Path | None]:
    trainval = None
    test = None

    for s in sources:
        s_str = str(s).lower()
        if ("trainval" in s_str) or ("train_val" in s_str) or ("train-val" in s_str):
            trainval = s
        if ("/voctest_" in s_str) or ("_test" in s_str) or ("test" in s_str and "train" not in s_str):
            test = s

    base = trainval or sources[0]
    return base, test


def copy_missing_dir(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.rglob("*"):
        rel = item.relative_to(src)
        out = dst / rel
        if item.is_dir():
            out.mkdir(parents=True, exist_ok=True)
        else:
            out.parent.mkdir(parents=True, exist_ok=True)
            if not out.exists():
                shutil.copy2(item, out)


def voc_ensure_trainval_txt(voc_dir: Path) -> None:
    main = voc_dir / "ImageSets" / "Main"
    if not main.exists():
        return

    trainval = main / "trainval.txt"
    train = main / "train.txt"
    val = main / "val.txt"

    if trainval.exists():
        return

    if train.exists() and val.exists():
        ids: List[str] = []
        for f in (train, val):
            ids.extend([x.strip() for x in f.read_text(encoding="utf-8").splitlines() if x.strip()])
        trainval.write_text("\n".join(ids) + "\n", encoding="utf-8")


def download_voc(
    dst_dir: str | Path,
    details: Logger, force: bool = False,
    years: tuple[str, ...] = ("2007", "2012"),
) -> Path:
    devkit = ensure_dir(dst_dir)

    def exists(_: Path) -> bool:
        return all((devkit / f"VOC{y}").exists() and voc_is_valid(devkit / f"VOC{y}") for y in years)

    def merge_test_into_target(test_src: Path, target: Path) -> None:
        copy_missing_dir(test_src / "JPEGImages", target / "JPEGImages")

        test_txt = test_src / "ImageSets" / "Main" / "test.txt"
        if test_txt.exists():
            dst = target / "ImageSets" / "Main" / "test.txt"
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(test_txt, dst)

    def download_fn(_: Path) -> None:
        for y in years:
            target = devkit / f"VOC{y}"

            if target.exists() and voc_is_valid(target) and not force:
                voc_remove_segmentation(target)
                voc_ensure_trainval_txt(target)
                continue

            slug = VOC_KAGGLE_DATASETS.get(str(y))
            if slug is None:
                raise RuntimeError(f"No Kaggle dataset configured for VOC{y}")

            log(details, f"Downloading VOC{y}: {slug}")
            subprocess.run(["kaggle", "datasets", "download", "-d", slug, "-p", str(devkit), "--unzip"], check=True)

            sources = voc_find_sources(devkit, y)
            if not sources:
                raise RuntimeError(f"VOC{y}: not found after unzip in {devkit}")

            base, test = voc_pick_trainval_and_test(sources)

            if target.exists():
                shutil.rmtree(target, ignore_errors=True)

            log(details, f"VOC{y}: moving base -> {target}")
            shutil.move(str(base), str(target))

            voc_remove_segmentation(target)
            voc_ensure_trainval_txt(target)

            if test is not None and test.exists() and voc_is_valid(test):
                log(details, f"VOC{y}: merging test -> {target}")
                merge_test_into_target(test, target)
                shutil.rmtree(test, ignore_errors=True)

            for s in sources:
                if s == base or s == test:
                    continue
                shutil.rmtree(s, ignore_errors=True)

            voc_cleanup_wrappers(devkit, details)

            if not voc_is_valid(target):
                raise RuntimeError(f"VOC{y}: final structure invalid -> {target}")

            log(details, f"VOC{y} ready -> {target}")

        voc_cleanup_wrappers(devkit, details)

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
