import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from PIL import Image


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


@dataclass(frozen=True)
class CropBox:
    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Crop the archive dataset for experiments: train/val use 1024 patches "
            "with 256 overlap, test uses 4000 patches with 1024 overlap."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("datasets/archive"),
        help="Root directory of the archive dataset.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("datasets/archive_experiment_crops"),
        help="Output directory for the cropped dataset.",
    )
    parser.add_argument(
        "--train-patch-size",
        type=int,
        default=1024,
        help="Patch size for train and val.",
    )
    parser.add_argument(
        "--train-overlap",
        type=int,
        default=256,
        help="Overlap for train and val.",
    )
    parser.add_argument(
        "--test-patch-size",
        type=int,
        default=4000,
        help="Patch size for test.",
    )
    parser.add_argument(
        "--test-overlap",
        type=int,
        default=1024,
        help="Overlap for test.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output files if they already exist.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only estimate outputs without writing files.",
    )
    return parser


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def validate_args(args: argparse.Namespace) -> None:
    for patch_size, overlap, label in [
        (args.train_patch_size, args.train_overlap, "train/val"),
        (args.test_patch_size, args.test_overlap, "test"),
    ]:
        if patch_size <= 0:
            raise ValueError(f"{label} patch size must be positive.")
        if overlap < 0:
            raise ValueError(f"{label} overlap must be non-negative.")
        if overlap >= patch_size:
            raise ValueError(f"{label} overlap must be smaller than patch size.")


def generate_starts(length: int, patch_size: int, stride: int) -> List[int]:
    if length <= patch_size:
        return [0]
    starts = list(range(0, length - patch_size + 1, stride))
    last = length - patch_size
    if starts[-1] != last:
        starts.append(last)
    return starts


def compute_crop_boxes(width: int, height: int, patch_size: int, stride: int) -> List[CropBox]:
    xs = generate_starts(width, patch_size, stride)
    ys = generate_starts(height, patch_size, stride)
    return [
        CropBox(x0=x0, y0=y0, x1=x0 + patch_size, y1=y0 + patch_size)
        for y0 in ys
        for x0 in xs
    ]


def pad_image_to_patch(image: Image.Image, patch_size: int, fill_value) -> Image.Image:
    width, height = image.size
    if width >= patch_size and height >= patch_size:
        return image
    canvas = Image.new(
        image.mode,
        (max(width, patch_size), max(height, patch_size)),
        fill_value,
    )
    canvas.paste(image, (0, 0))
    return canvas


def infer_fill_value(mode: str):
    if mode == "RGB":
        return (0, 0, 0)
    if mode == "RGBA":
        return (0, 0, 0, 0)
    return 0


def discover_image_files(root: Path) -> List[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def boxes_intersect(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> bool:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    return ax0 < bx1 and bx0 < ax1 and ay0 < by1 and by0 < ay1


def shift_and_clip_polygon(
    polygon: Sequence[float],
    crop_box: CropBox,
    patch_size: int,
) -> List[float]:
    shifted = []
    for idx in range(0, len(polygon), 2):
        x = float(polygon[idx]) - crop_box.x0
        y = float(polygon[idx + 1]) - crop_box.y0
        shifted.extend(
            [
                float(np.clip(x, 0, patch_size)),
                float(np.clip(y, 0, patch_size)),
            ]
        )
    return shifted


def clip_bbox_to_crop(bbox: Sequence[float], crop_box: CropBox) -> List[float]:
    x, y, w, h = [float(v) for v in bbox]
    x0 = max(x, crop_box.x0)
    y0 = max(y, crop_box.y0)
    x1 = min(x + w, crop_box.x1)
    y1 = min(y + h, crop_box.y1)
    if x1 <= x0 or y1 <= y0:
        return []
    return [x0 - crop_box.x0, y0 - crop_box.y0, x1 - x0, y1 - y0]


def crop_annotations_for_patch(
    annotations: Sequence[Dict],
    crop_box: CropBox,
    patch_size: int,
    patch_image_id: int,
    next_annotation_id: int,
) -> Tuple[List[Dict], int]:
    patch_rect = (crop_box.x0, crop_box.y0, crop_box.x1, crop_box.y1)
    out = []

    for ann in annotations:
        bbox = ann.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        ann_rect = (
            float(bbox[0]),
            float(bbox[1]),
            float(bbox[0] + bbox[2]),
            float(bbox[1] + bbox[3]),
        )
        if not boxes_intersect(patch_rect, ann_rect):
            continue

        clipped_bbox = clip_bbox_to_crop(bbox, crop_box)
        if not clipped_bbox:
            continue

        patch_ann = dict(ann)
        patch_ann["id"] = next_annotation_id
        patch_ann["image_id"] = patch_image_id
        patch_ann["bbox"] = [round(v, 2) for v in clipped_bbox]
        patch_ann["area"] = round(clipped_bbox[2] * clipped_bbox[3], 2)

        segmentation = ann.get("segmentation")
        if isinstance(segmentation, list):
            patch_ann["segmentation"] = [
                shift_and_clip_polygon(poly, crop_box, patch_size)
                for poly in segmentation
                if isinstance(poly, list) and len(poly) >= 6 and len(poly) % 2 == 0
            ]

        out.append(patch_ann)
        next_annotation_id += 1

    return out, next_annotation_id


def crop_and_save_image(
    src_path: Path,
    dst_dir: Path,
    crop_box: CropBox,
    overwrite: bool,
) -> Path:
    ensure_dir(dst_dir)
    out_name = f"{src_path.stem}_{crop_box.y0}_{crop_box.y1}_{crop_box.x0}_{crop_box.x1}{src_path.suffix}"
    out_path = dst_dir / out_name
    if out_path.exists() and not overwrite:
        return out_path

    with Image.open(src_path) as image_obj:
        image = image_obj.copy()
    image = pad_image_to_patch(image, crop_box.width, infer_fill_value(image.mode))
    patch = image.crop((crop_box.x0, crop_box.y0, crop_box.x1, crop_box.y1))
    patch.save(out_path)
    return out_path


def crop_and_save_mask(
    src_path: Path,
    dst_path: Path,
    crop_box: CropBox,
    overwrite: bool,
) -> None:
    ensure_dir(dst_path.parent)
    if dst_path.exists() and not overwrite:
        return
    with Image.open(src_path) as mask_obj:
        mask = mask_obj.copy()
    mask = pad_image_to_patch(mask, crop_box.width, infer_fill_value(mask.mode))
    patch = mask.crop((crop_box.x0, crop_box.y0, crop_box.x1, crop_box.y1))
    patch.save(dst_path)


def build_image_lookup(files: Iterable[Path]) -> Dict[str, Path]:
    return {path.stem: path for path in files}


def build_mask_lookup(files: Iterable[Path], suffix: str) -> Dict[str, Path]:
    lookup = {}
    for path in files:
        stem = path.stem
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
        lookup[stem] = path
    return lookup


def process_train_or_val(
    split: str,
    split_root: Path,
    output_root: Path,
    patch_size: int,
    overlap: int,
    overwrite: bool,
    verify_only: bool,
) -> Dict:
    stride = patch_size - overlap
    annotation_path = split_root / "Annotations" / f"iSAID_{split}.json"
    annotation_data = json.loads(annotation_path.read_text(encoding="utf-8"))

    image_files = discover_image_files(split_root / "images")
    image_lookup = build_image_lookup(image_files)
    image_meta_by_id = {int(item["id"]): item for item in annotation_data.get("images", [])}
    annotations_by_image_id: Dict[int, List[Dict]] = {}
    for ann in annotation_data.get("annotations", []):
        annotations_by_image_id.setdefault(int(ann["image_id"]), []).append(ann)

    instance_lookup = build_mask_lookup(
        (split_root / "Instance_masks" / "images" / "images").glob("*.png"),
        "_instance_id_RGB",
    )
    semantic_lookup = build_mask_lookup(
        (split_root / "Semantic_masks" / "images" / "images").glob("*.png"),
        "_instance_color_RGB",
    )

    output_split_root = output_root / ("TrainData" if split == "train" else "ValidationData")
    output_split_root = output_split_root / ("train" if split == "train" else "val")
    out_images = output_split_root / "images_cropped"
    out_instance = output_split_root / "Instance_masks_cropped"
    out_semantic = output_split_root / "Semantic_masks_cropped"

    patch_images = []
    patch_annotations = []
    patch_image_id = 0
    patch_annotation_id = 0
    source_image_count = 0

    for image_id in sorted(image_meta_by_id):
        image_meta = image_meta_by_id[image_id]
        image_stem = Path(image_meta["file_name"]).stem
        image_path = image_lookup.get(image_stem)
        if image_path is None:
            continue

        source_image_count += 1
        with Image.open(image_path) as image_obj:
            width, height = image_obj.size

        padded_width = max(width, patch_size)
        padded_height = max(height, patch_size)
        crop_boxes = compute_crop_boxes(padded_width, padded_height, patch_size, stride)

        for crop_box in crop_boxes:
            out_image_name = (
                f"{image_stem}_{crop_box.y0}_{crop_box.y1}_{crop_box.x0}_{crop_box.x1}.png"
            )
            patch_images.append(
                {
                    "id": patch_image_id,
                    "width": patch_size,
                    "height": patch_size,
                    "file_name": out_image_name,
                    "ins_file_name": out_image_name,
                    "seg_file_name": out_image_name,
                    "source_image_name": image_meta["file_name"],
                }
            )

            cropped_annotations, patch_annotation_id = crop_annotations_for_patch(
                annotations=annotations_by_image_id.get(image_id, []),
                crop_box=crop_box,
                patch_size=patch_size,
                patch_image_id=patch_image_id,
                next_annotation_id=patch_annotation_id,
            )
            patch_annotations.extend(cropped_annotations)

            if not verify_only:
                crop_and_save_image(
                    src_path=image_path,
                    dst_dir=out_images,
                    crop_box=crop_box,
                    overwrite=overwrite,
                )

                instance_src = instance_lookup.get(image_stem)
                if instance_src is not None:
                    crop_and_save_mask(
                        src_path=instance_src,
                        dst_path=out_instance / out_image_name,
                        crop_box=crop_box,
                        overwrite=overwrite,
                    )

                semantic_src = semantic_lookup.get(image_stem)
                if semantic_src is not None:
                    crop_and_save_mask(
                        src_path=semantic_src,
                        dst_path=out_semantic / out_image_name,
                        crop_box=crop_box,
                        overwrite=overwrite,
                    )

            patch_image_id += 1

    out_json = {
        "images": patch_images,
        "categories": annotation_data.get("categories", []),
        "annotations": patch_annotations,
    }

    summary = {
        "split": split,
        "patch_size": patch_size,
        "overlap": overlap,
        "stride": stride,
        "source_image_count": source_image_count,
        "patch_image_count": len(patch_images),
        "patch_annotation_count": len(patch_annotations),
    }

    if not verify_only:
        ensure_dir(output_split_root / "Annotations")
        (output_split_root / "Annotations" / f"iSAID_{split}.json").write_text(
            json.dumps(out_json, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (output_split_root / "manifest.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    return summary


def process_test(
    split_root: Path,
    output_root: Path,
    patch_size: int,
    overlap: int,
    overwrite: bool,
    verify_only: bool,
) -> Dict:
    stride = patch_size - overlap
    image_files = discover_image_files(split_root / "images")
    out_split_root = output_root / "TestData" / "TestData"
    out_images = out_split_root / "images_cropped"

    original_test_info = json.loads(
        (split_root / "Annotations" / "test_info.json").read_text(encoding="utf-8")
    )
    categories = original_test_info.get("categories", [])

    patch_images = []
    patch_image_id = 0
    for image_path in image_files:
        with Image.open(image_path) as image_obj:
            width, height = image_obj.size

        padded_width = max(width, patch_size)
        padded_height = max(height, patch_size)
        crop_boxes = compute_crop_boxes(padded_width, padded_height, patch_size, stride)

        for crop_box in crop_boxes:
            out_image_name = (
                f"{image_path.stem}_{crop_box.y0}_{crop_box.y1}_{crop_box.x0}_{crop_box.x1}.png"
            )
            patch_images.append(
                {
                    "id": patch_image_id,
                    "width": patch_size,
                    "height": patch_size,
                    "file_name": out_image_name,
                    "ins_file_name": f"{Path(out_image_name).stem}_instance_id_RGB.png",
                    "seg_file_name": f"{Path(out_image_name).stem}_instance_color_RGB.png",
                    "source_image_name": image_path.name,
                }
            )

            if not verify_only:
                crop_and_save_image(
                    src_path=image_path,
                    dst_dir=out_images,
                    crop_box=crop_box,
                    overwrite=overwrite,
                )

            patch_image_id += 1

    out_json = {"images": patch_images, "categories": categories}
    summary = {
        "split": "test",
        "patch_size": patch_size,
        "overlap": overlap,
        "stride": stride,
        "source_image_count": len(image_files),
        "patch_image_count": len(patch_images),
        "patch_annotation_count": 0,
    }

    if not verify_only:
        ensure_dir(out_split_root / "Annotations")
        (out_split_root / "Annotations" / "test_info.json").write_text(
            json.dumps(out_json, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (out_split_root / "manifest.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    return summary


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)

    dataset_root = args.dataset_root
    output_root = args.output_root

    train_summary = process_train_or_val(
        split="train",
        split_root=dataset_root / "TrainData" / "train",
        output_root=output_root,
        patch_size=args.train_patch_size,
        overlap=args.train_overlap,
        overwrite=args.overwrite,
        verify_only=args.verify_only,
    )
    print(json.dumps(train_summary, ensure_ascii=False, indent=2))

    val_summary = process_train_or_val(
        split="val",
        split_root=dataset_root / "ValidationData" / "val",
        output_root=output_root,
        patch_size=args.train_patch_size,
        overlap=args.train_overlap,
        overwrite=args.overwrite,
        verify_only=args.verify_only,
    )
    print(json.dumps(val_summary, ensure_ascii=False, indent=2))

    test_summary = process_test(
        split_root=dataset_root / "TestData" / "TestData",
        output_root=output_root,
        patch_size=args.test_patch_size,
        overlap=args.test_overlap,
        overwrite=args.overwrite,
        verify_only=args.verify_only,
    )
    print(json.dumps(test_summary, ensure_ascii=False, indent=2))

    if not args.verify_only:
        ensure_dir(output_root)
        (output_root / "summary.json").write_text(
            json.dumps(
                {
                    "train": train_summary,
                    "val": val_summary,
                    "test": test_summary,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
