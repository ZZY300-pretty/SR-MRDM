import argparse
import json
import math
import random
import shutil
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from PIL import Image


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Sample a small random subset from archive train/val/test and copy "
            "aligned images, masks, and annotation files where available."
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
        default=Path("datasets/archive_random_10pct"),
        help="Destination directory for the sampled subset.",
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=0.10,
        help="Sampling ratio per split.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=3407,
        help="Random seed for reproducible sampling.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the destination directory if it already exists.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only compute counts without writing files.",
    )
    return parser


def ensure_clean_dir(path: Path, overwrite: bool) -> None:
    if not path.exists():
        return
    if not overwrite:
        raise FileExistsError(
            f"Output directory already exists: {path}. Use --overwrite to replace it."
        )
    shutil.rmtree(path)


def list_image_files(root: Path) -> List[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def choose_subset(items: Sequence, ratio: float, rng: random.Random) -> List:
    if not items:
        return []
    sample_size = max(1, int(len(items) * ratio))
    sample_size = min(sample_size, len(items))
    indexed = list(items)
    chosen = rng.sample(indexed, sample_size)
    return sorted(chosen)


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_stem_lookup(files: Sequence[Path], suffix_to_strip: str = "") -> Dict[str, Path]:
    lookup = {}
    for path in files:
        stem = path.stem
        if suffix_to_strip and stem.endswith(suffix_to_strip):
            stem = stem[: -len(suffix_to_strip)]
        lookup[stem] = path
    return lookup


def resolve_split_root(dataset_root: Path, split: str) -> Path:
    mapping = {
        "train": dataset_root / "TrainData" / "train",
        "val": dataset_root / "ValidationData" / "val",
        "test": dataset_root / "TestData" / "TestData",
    }
    return mapping[split]


def copy_selected_images(
    selected_images: Sequence[Path],
    source_root: Path,
    target_root: Path,
    verify_only: bool,
) -> List[str]:
    names = []
    for src in selected_images:
        rel = src.relative_to(source_root)
        names.append(src.name)
        if verify_only:
            continue
        copy_file(src, target_root / rel)
    return names


def subset_train_or_val(
    split: str,
    split_root: Path,
    output_root: Path,
    ratio: float,
    rng: random.Random,
    verify_only: bool,
) -> Dict:
    annotation_name = f"iSAID_{split}.json"
    annotation_path = split_root / "Annotations" / annotation_name
    data = json.loads(annotation_path.read_text(encoding="utf-8"))

    source_images_root = split_root / "images"
    all_image_files = list_image_files(source_images_root)
    selected_image_files = choose_subset(all_image_files, ratio, rng)
    selected_stems = {path.stem for path in selected_image_files}

    selected_images_meta = [
        item for item in data.get("images", []) if Path(item["file_name"]).stem in selected_stems
    ]
    selected_ids = {int(item["id"]) for item in selected_images_meta}
    selected_annotations = [
        item for item in data.get("annotations", []) if int(item["image_id"]) in selected_ids
    ]

    instance_root = split_root / "Instance_masks" / "images" / "images"
    semantic_root = split_root / "Semantic_masks" / "images" / "images"
    instance_lookup = build_stem_lookup(
        list(instance_root.glob("*.png")), suffix_to_strip="_instance_id_RGB"
    )
    semantic_lookup = build_stem_lookup(
        list(semantic_root.glob("*.png")), suffix_to_strip="_instance_color_RGB"
    )

    copied_instance = 0
    copied_semantic = 0

    output_split_root = output_root / ("TrainData" if split == "train" else "ValidationData")
    output_split_root = output_split_root / ("train" if split == "train" else "val")

    selected_names = copy_selected_images(
        selected_images=selected_image_files,
        source_root=source_images_root,
        target_root=output_split_root / "images",
        verify_only=verify_only,
    )

    for stem in sorted(selected_stems):
        instance_src = instance_lookup.get(stem)
        semantic_src = semantic_lookup.get(stem)
        if instance_src is not None:
            copied_instance += 1
            if not verify_only:
                copy_file(
                    instance_src,
                    output_split_root / "Instance_masks" / "images" / "images" / instance_src.name,
                )
        if semantic_src is not None:
            copied_semantic += 1
            if not verify_only:
                copy_file(
                    semantic_src,
                    output_split_root / "Semantic_masks" / "images" / "images" / semantic_src.name,
                )

    subset_json = {
        "images": selected_images_meta,
        "categories": data.get("categories", []),
        "annotations": selected_annotations,
    }

    summary = {
        "split": split,
        "source_image_count": len(all_image_files),
        "selected_image_count": len(selected_image_files),
        "selected_annotation_count": len(selected_annotations),
        "copied_instance_mask_count": copied_instance,
        "copied_semantic_mask_count": copied_semantic,
        "selected_image_names_preview": selected_names[:10],
    }

    if not verify_only:
        write_json(output_split_root / "Annotations" / annotation_name, subset_json)
        write_json(output_split_root / "manifest.json", summary)

    return summary


def build_test_subset_annotation(
    selected_images: Sequence[Path],
    categories: Sequence[Dict],
) -> Dict:
    images = []
    for idx, path in enumerate(selected_images):
        with Image.open(path) as image_obj:
            width, height = image_obj.size
        images.append(
            {
                "id": idx,
                "width": width,
                "height": height,
                "file_name": path.name,
                "ins_file_name": f"{path.stem}_instance_id_RGB.png",
                "seg_file_name": f"{path.stem}_instance_color_RGB.png",
            }
        )
    return {"images": images, "categories": list(categories)}


def subset_test(
    split_root: Path,
    output_root: Path,
    ratio: float,
    rng: random.Random,
    verify_only: bool,
) -> Dict:
    source_images_root = split_root / "images"
    all_image_files = list_image_files(source_images_root)
    selected_image_files = choose_subset(all_image_files, ratio, rng)

    annotation_path = split_root / "Annotations" / "test_info.json"
    annotation_data = json.loads(annotation_path.read_text(encoding="utf-8"))
    subset_annotation = build_test_subset_annotation(
        selected_images=selected_image_files,
        categories=annotation_data.get("categories", []),
    )

    output_split_root = output_root / "TestData" / "TestData"
    selected_names = copy_selected_images(
        selected_images=selected_image_files,
        source_root=source_images_root,
        target_root=output_split_root / "images",
        verify_only=verify_only,
    )

    summary = {
        "split": "test",
        "source_image_count": len(all_image_files),
        "selected_image_count": len(selected_image_files),
        "selected_annotation_count": len(subset_annotation.get("images", [])),
        "copied_instance_mask_count": 0,
        "copied_semantic_mask_count": 0,
        "selected_image_names_preview": selected_names[:10],
        "note": "TestData has no local Instance_masks/Semantic_masks under archive, so only images and a subset annotation file are copied.",
    }

    if not verify_only:
        write_json(output_split_root / "Annotations" / "test_info.json", subset_annotation)
        write_json(output_split_root / "manifest.json", summary)

    return summary


def main() -> None:
    args = build_parser().parse_args()
    rng = random.Random(args.seed)

    if not args.verify_only:
        ensure_clean_dir(args.output_root, args.overwrite)
        args.output_root.mkdir(parents=True, exist_ok=True)

    train_summary = subset_train_or_val(
        split="train",
        split_root=resolve_split_root(args.dataset_root, "train"),
        output_root=args.output_root,
        ratio=args.ratio,
        rng=rng,
        verify_only=args.verify_only,
    )
    print(json.dumps(train_summary, ensure_ascii=False, indent=2))

    val_summary = subset_train_or_val(
        split="val",
        split_root=resolve_split_root(args.dataset_root, "val"),
        output_root=args.output_root,
        ratio=args.ratio,
        rng=rng,
        verify_only=args.verify_only,
    )
    print(json.dumps(val_summary, ensure_ascii=False, indent=2))

    test_summary = subset_test(
        split_root=resolve_split_root(args.dataset_root, "test"),
        output_root=args.output_root,
        ratio=args.ratio,
        rng=rng,
        verify_only=args.verify_only,
    )
    print(json.dumps(test_summary, ensure_ascii=False, indent=2))

    summary = {
        "dataset_root": str(args.dataset_root),
        "output_root": str(args.output_root),
        "ratio": args.ratio,
        "seed": args.seed,
        "train": train_summary,
        "val": val_summary,
        "test": test_summary,
    }
    if not args.verify_only:
        write_json(args.output_root / "summary.json", summary)


if __name__ == "__main__":
    main()
