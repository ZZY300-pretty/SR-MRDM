import argparse
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

import cv2
import numpy as np
from PIL import Image


DOTA_CLASSES = (
    "plane",
    "baseball-diamond",
    "bridge",
    "ground-track-field",
    "small-vehicle",
    "large-vehicle",
    "ship",
    "tennis-court",
    "basketball-court",
    "storage-tank",
    "soccer-ball-field",
    "roundabout",
    "harbor",
    "swimming-pool",
    "helicopter",
)

NORMALIZED_CLASS_MAP = {
    "plane": "plane",
    "baseball diamond": "baseball-diamond",
    "bridge": "bridge",
    "ground track field": "ground-track-field",
    "small vehicle": "small-vehicle",
    "large vehicle": "large-vehicle",
    "ship": "ship",
    "tennis court": "tennis-court",
    "basketball court": "basketball-court",
    "storage tank": "storage-tank",
    "soccer ball field": "soccer-ball-field",
    "roundabout": "roundabout",
    "harbor": "harbor",
    "swimming pool": "swimming-pool",
    "helicopter": "helicopter",
}

SPLIT_CONFIG = {
    "train": {
        "annotation": Path("TrainData/train/Annotations/iSAID_train.json"),
        "images": Path("TrainData/train/images_cropped"),
    },
    "val": {
        "annotation": Path("ValidationData/val/Annotations/iSAID_val.json"),
        "images": Path("ValidationData/val/images_cropped"),
    },
    "test": {
        "annotation": Path("TestData/TestData/Annotations/test_info.json"),
        "images": Path("TestData/TestData/images_cropped"),
    },
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert cropped iSAID-style patch datasets into MMRotate/DOTA "
            "annfiles layout."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("datasets/archive_random_10pct_experiment_crops"),
        help="Input cropped dataset root.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("datasets/archive_random_10pct_experiment_crops_dota"),
        help="Output directory in DOTA annfiles layout.",
    )
    parser.add_argument(
        "--image-mode",
        choices=("hardlink", "copy"),
        default="hardlink",
        help="How to materialize .png images in the output dataset.",
    )
    parser.add_argument(
        "--emit-trainval",
        action="store_true",
        help="Also emit a merged trainval split for compatibility with stock DOTA configs.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output directory if it already exists.",
    )
    return parser


def normalize_name(name: str) -> str:
    return name.strip().lower().replace("_", " ").replace("-", " ")


def ensure_clean_dir(path: Path, overwrite: bool) -> None:
    if not path.exists():
        return
    if not overwrite:
        raise FileExistsError(
            f"Output directory already exists: {path}. Use --overwrite to replace it."
        )
    shutil.rmtree(path)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def order_points_clockwise(points: np.ndarray) -> np.ndarray:
    center = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    ordered = points[np.argsort(angles)]
    start = np.argmin(ordered[:, 0] + ordered[:, 1])
    return np.roll(ordered, -start, axis=0)


def bbox_to_polygon(bbox: List[float]) -> np.ndarray:
    x, y, w, h = [float(v) for v in bbox]
    return np.array(
        [[x, y], [x + w, y], [x + w, y + h], [x, y + h]],
        dtype=np.float32,
    )


def segmentation_to_dota_polygon(
    segmentation: List[List[float]],
    bbox: List[float] | None,
) -> np.ndarray | None:
    points = []
    for segment in segmentation or []:
        if len(segment) < 6 or len(segment) % 2 != 0:
            continue
        points.append(np.array(segment, dtype=np.float32).reshape(-1, 2))

    if not points:
        if bbox and len(bbox) == 4:
            points = [bbox_to_polygon(bbox)]
        else:
            return None

    merged = np.concatenate(points, axis=0)
    rect = cv2.minAreaRect(merged)
    box = cv2.boxPoints(rect)
    return order_points_clockwise(box)


def format_polygon_line(polygon: np.ndarray, class_name: str, difficulty: int = 0) -> str:
    coords = [f"{float(v):.2f}" for v in polygon.reshape(-1)]
    return " ".join([*coords, class_name, str(difficulty)])


def materialize_png(src_path: Path, dst_path: Path, image_mode: str) -> None:
    ensure_dir(dst_path.parent)
    if dst_path.exists():
        return

    if src_path.suffix.lower() == ".png":
        if image_mode == "hardlink":
            try:
                os.link(src_path, dst_path)
                return
            except OSError:
                pass
        shutil.copy2(src_path, dst_path)
        return

    with Image.open(src_path) as image_obj:
        image_obj.convert("RGB").save(dst_path)


def build_category_mapping(categories: Iterable[Dict]) -> Dict[int, str]:
    mapping = {}
    for category in categories:
        category_id = int(category["id"])
        normalized = normalize_name(category["name"])
        if normalized not in NORMALIZED_CLASS_MAP:
            raise KeyError(
                f"Unsupported category '{category['name']}'. "
                f"Expected one of: {sorted(NORMALIZED_CLASS_MAP)}"
            )
        mapping[category_id] = NORMALIZED_CLASS_MAP[normalized]
    return mapping


def build_output_name(image_name: str) -> str:
    return f"{Path(image_name).stem}.png"


def write_split(
    dataset_root: Path,
    output_root: Path,
    split: str,
    image_mode: str,
) -> Dict:
    cfg = SPLIT_CONFIG[split]
    annotation_path = dataset_root / cfg["annotation"]
    image_root = dataset_root / cfg["images"]

    data = json.loads(annotation_path.read_text(encoding="utf-8"))
    images = data.get("images", [])
    categories = data.get("categories", [])
    category_mapping = build_category_mapping(categories)

    split_root = output_root / split
    ann_dir = split_root / "annfiles"
    img_dir = split_root / "images"
    ensure_dir(ann_dir)
    ensure_dir(img_dir)

    anns_by_image = defaultdict(list)
    for ann in data.get("annotations", []):
        if ann.get("iscrowd", 0):
            continue
        polygon = segmentation_to_dota_polygon(
            ann.get("segmentation", []),
            ann.get("bbox"),
        )
        if polygon is None:
            continue
        class_name = category_mapping[int(ann["category_id"])]
        anns_by_image[int(ann["image_id"])].append(
            format_polygon_line(polygon, class_name, difficulty=0)
        )

    missing_images: List[str] = []
    empty_ann_count = 0
    written_ann_count = 0
    written_image_count = 0

    for image_info in images:
        image_id = int(image_info["id"])
        src_name = image_info["file_name"]
        src_path = image_root / src_name
        if not src_path.exists():
            missing_images.append(src_name)
            continue
        dst_name = build_output_name(src_name)
        dst_path = img_dir / dst_name
        materialize_png(src_path, dst_path, image_mode=image_mode)
        written_image_count += 1

        ann_lines = anns_by_image.get(image_id, [])
        if not ann_lines:
            empty_ann_count += 1
        else:
            written_ann_count += len(ann_lines)

        ann_path = ann_dir / f"{Path(dst_name).stem}.txt"
        ann_path.write_text(
            "\n".join(ann_lines) + ("\n" if ann_lines else ""),
            encoding="utf-8",
        )

    return {
        "split": split,
        "image_count": len(images),
        "written_image_count": written_image_count,
        "annotation_count": written_ann_count,
        "empty_annotation_files": empty_ann_count,
        "missing_image_count": len(missing_images),
        "missing_image_preview": missing_images[:10],
        "image_dir": str(img_dir),
        "ann_dir": str(ann_dir),
    }


def write_test_split(dataset_root: Path, output_root: Path, image_mode: str) -> Dict:
    cfg = SPLIT_CONFIG["test"]
    annotation_path = dataset_root / cfg["annotation"]
    image_root = dataset_root / cfg["images"]

    data = json.loads(annotation_path.read_text(encoding="utf-8"))
    images = data.get("images", [])

    split_root = output_root / "test"
    img_dir = split_root / "images"
    ensure_dir(img_dir)

    missing_images: List[str] = []
    written_image_count = 0
    for image_info in images:
        src_name = image_info["file_name"]
        src_path = image_root / src_name
        if not src_path.exists():
            missing_images.append(src_name)
            continue
        dst_name = build_output_name(src_name)
        materialize_png(src_path, img_dir / dst_name, image_mode=image_mode)
        written_image_count += 1

    return {
        "split": "test",
        "image_count": len(images),
        "written_image_count": written_image_count,
        "annotation_count": 0,
        "missing_image_count": len(missing_images),
        "missing_image_preview": missing_images[:10],
        "image_dir": str(img_dir),
    }


def emit_trainval(output_root: Path, image_mode: str) -> Dict:
    trainval_root = output_root / "trainval"
    ann_dir = trainval_root / "annfiles"
    img_dir = trainval_root / "images"
    ensure_dir(ann_dir)
    ensure_dir(img_dir)

    image_count = 0
    ann_file_count = 0

    for split in ("train", "val"):
        split_img_dir = output_root / split / "images"
        split_ann_dir = output_root / split / "annfiles"

        for src_path in sorted(split_img_dir.glob("*.png")):
            materialize_png(src_path, img_dir / src_path.name, image_mode=image_mode)
            image_count += 1

        for src_path in sorted(split_ann_dir.glob("*.txt")):
            dst_path = ann_dir / src_path.name
            if not dst_path.exists():
                shutil.copy2(src_path, dst_path)
                ann_file_count += 1

    return {
        "split": "trainval",
        "image_count": image_count,
        "annfile_count": ann_file_count,
        "image_dir": str(img_dir),
        "ann_dir": str(ann_dir),
    }


def main() -> None:
    args = build_parser().parse_args()
    dataset_root = args.dataset_root.resolve()
    output_root = args.output_root.resolve()

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    ensure_clean_dir(output_root, overwrite=args.overwrite)
    ensure_dir(output_root)

    split_summaries = [
        write_split(dataset_root, output_root, "train", args.image_mode),
        write_split(dataset_root, output_root, "val", args.image_mode),
        write_test_split(dataset_root, output_root, args.image_mode),
    ]

    if args.emit_trainval:
        split_summaries.append(emit_trainval(output_root, args.image_mode))

    (output_root / "classes.txt").write_text(
        "\n".join(DOTA_CLASSES) + "\n",
        encoding="utf-8",
    )
    (output_root / "conversion_summary.json").write_text(
        json.dumps(
            {
                "input_root": str(dataset_root),
                "output_root": str(output_root),
                "image_mode": args.image_mode,
                "splits": split_summaries,
                "classes": list(DOTA_CLASSES),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    for summary in split_summaries:
        print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
