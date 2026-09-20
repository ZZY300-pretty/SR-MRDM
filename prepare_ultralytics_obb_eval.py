import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


MODEL_NAMES = [
    "plane",
    "ship",
    "storage tank",
    "baseball diamond",
    "tennis court",
    "basketball court",
    "ground track field",
    "harbor",
    "bridge",
    "large vehicle",
    "small vehicle",
    "helicopter",
    "roundabout",
    "soccer ball field",
    "swimming pool",
]

SPLITS = {
    "train": {
        "annotation": Path("TrainData/train/Annotations/iSAID_train.json"),
        "images": Path("train/cloud"),
    },
    "val": {
        "annotation": Path("ValidationData/val/Annotations/iSAID_val.json"),
        "images": Path("val/cloud"),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert iSAID-style instance polygons to Ultralytics YOLO-OBB labels for evaluation."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("datasets/archive_random_10pct_experiment_crops_heavy_cloud"),
        help="Cloudy dataset root directory.",
    )
    parser.add_argument(
        "--annotation-root",
        type=Path,
        default=Path("datasets/archive_random_10pct_experiment_crops"),
        help="Clean source dataset root directory containing the original annotations.",
    )
    return parser.parse_args()


def normalize_name(name: str) -> str:
    return name.strip().lower().replace("_", " ")


def polygon_to_obb(segmentation: list[list[float]], width: int, height: int) -> list[float] | None:
    points = []
    for segment in segmentation or []:
        if len(segment) < 6:
            continue
        pts = np.array(segment, dtype=np.float32).reshape(-1, 2)
        points.append(pts)

    if not points:
        return None

    merged = np.concatenate(points, axis=0)
    rect = cv2.minAreaRect(merged)
    box = cv2.boxPoints(rect)
    box = order_points_clockwise(box)

    normalized = []
    for x, y in box:
        x = min(max(float(x) / width, 0.0), 1.0)
        y = min(max(float(y) / height, 0.0), 1.0)
        normalized.extend([x, y])
    return normalized


def order_points_clockwise(points: np.ndarray) -> np.ndarray:
    center = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    ordered = points[np.argsort(angles)]
    start = np.argmin(ordered[:, 0] + ordered[:, 1])
    return np.roll(ordered, -start, axis=0)


def build_class_mapping(categories: list[dict]) -> dict[int, int]:
    model_index = {normalize_name(name): i for i, name in enumerate(MODEL_NAMES)}
    mapping = {}
    for category in categories:
        key = normalize_name(category["name"])
        if key not in model_index:
            raise KeyError(f"Category '{category['name']}' is not present in MODEL_NAMES.")
        mapping[int(category["id"])] = model_index[key]
    return mapping


def convert_split(annotation_root: Path, dataset_root: Path, split_name: str, split_cfg: dict) -> None:
    annotation_path = annotation_root / split_cfg["annotation"]
    images_dir = dataset_root / split_cfg["images"]
    data = json.loads(annotation_path.read_text(encoding="utf-8"))

    class_mapping = build_class_mapping(data["categories"])
    image_info = {int(image["id"]): image for image in data["images"]}
    labels_by_image = defaultdict(list)

    for ann in data["annotations"]:
        if ann.get("iscrowd", 0):
            continue
        image = image_info[int(ann["image_id"])]
        obb = polygon_to_obb(ann.get("segmentation", []), int(image["width"]), int(image["height"]))
        if obb is None:
            continue
        cls = class_mapping[int(ann["category_id"])]
        label_line = " ".join([str(cls), *[f"{v:.6f}" for v in obb]])
        labels_by_image[image["file_name"]].append(label_line)

    for image in data["images"]:
        image_name = image["file_name"]
        label_path = images_dir / f"{Path(image_name).stem}.txt"
        lines = labels_by_image.get(image_name, [])
        label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    print(
        f"{split_name}: wrote {len(data['images'])} label files and {sum(len(v) for v in labels_by_image.values())} objects"
    )


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    annotation_root = args.annotation_root.resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")
    if not annotation_root.exists():
        raise FileNotFoundError(f"Annotation root not found: {annotation_root}")

    for split_name, split_cfg in SPLITS.items():
        convert_split(annotation_root, dataset_root, split_name, split_cfg)


if __name__ == "__main__":
    main()
