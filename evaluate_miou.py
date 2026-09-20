"""Evaluate pixel-wise mean Intersection over Union (mIoU).

Important:
    mIoU must compare label masks against predicted label masks. RGB restored
    images in a directory such as ``cloud`` are not segmentation predictions.
    For the iSAID-derived masks in this project, ``mode="binary"`` treats all
    non-zero pixels in ``semantic_masks`` as foreground.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
PROJECT_ROOT = Path(__file__).resolve().parent

# These are safe defaults for the local Windows workspace.
DEFAULT_GT_DIR = (
    PROJECT_ROOT
    / "datasets"
    / "archive_random_10pct_experiment_crops_heavy_cloud"
    / "test"
    / "semantic_masks"
)

# Fill this with directories containing predicted masks, not restored RGB images.
# Example:
# PRED_DIRS = {
#     "my_model": PROJECT_ROOT / "path/to/predicted_semantic_masks",
# }
PRED_DIRS: Dict[str, Path] = {}

# "binary" is correct for the RGB instance-color masks currently present in
# datasets/.../semantic_masks when the desired metric is foreground/background.
# Use "class" only when the files contain class IDs or a complete color mapping.
EVAL_MODE = "binary"
NUM_CLASSES = 2
CLASS_NAMES = ("background", "foreground")
LABEL_MAPPING: Optional[Mapping[int, int]] = None
COLOR_MAPPING: Optional[Mapping[Tuple[int, int, int], int]] = None


def _as_path(path: str | Path) -> Path:
    return path if isinstance(path, Path) else Path(path)


def discover_image_files(root: Path) -> list[Path]:
    """Return supported image files below root in deterministic order."""
    return sorted(
        (
            path
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        ),
        key=lambda path: path.relative_to(root).as_posix().lower(),
    )


def _build_name_lookup(paths: Iterable[Path]) -> Dict[str, list[Path]]:
    lookup: Dict[str, list[Path]] = {}
    for path in paths:
        lookup.setdefault(path.name.lower(), []).append(path)
    return lookup


def find_matching_file(
    source_path: Path,
    source_root: Path,
    target_root: Path,
    target_name_lookup: Optional[Dict[str, list[Path]]] = None,
) -> Optional[Path]:
    """Match a prediction or valid mask by relative path, then by unique name."""
    relative_path = source_path.relative_to(source_root)
    direct_path = target_root / relative_path
    if direct_path.exists():
        return direct_path

    for suffix in IMAGE_SUFFIXES:
        candidate = (target_root / relative_path).with_suffix(suffix)
        if candidate.exists():
            return candidate

    if target_name_lookup is None:
        target_name_lookup = _build_name_lookup(discover_image_files(target_root))
    candidates = target_name_lookup.get(source_path.name.lower(), [])
    if len(candidates) == 1:
        return candidates[0]
    return None


def read_label_image(path: Path) -> np.ndarray:
    """Read a label image and convert BGR/BGRA data to RGB channel order."""
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Unable to read image: {path}")

    if image.ndim == 3:
        if image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        elif image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image = image[..., 0]
    return image


def validate_prediction_source(
    pred_root: Path,
    pred_paths: Sequence[Path],
    mode: str,
    allow_likely_rgb_images: bool,
) -> None:
    """Reject likely natural RGB images in binary mask mode."""
    if mode != "binary" or not pred_paths:
        return

    sample = read_label_image(pred_paths[0])
    if sample.ndim != 3 or sample.shape[2] != 3:
        return

    pixels = sample.reshape(-1, 3)
    if len(pixels) > 100_000:
        stride = int(np.ceil(len(pixels) / 100_000))
        pixels = pixels[::stride]
    unique_color_count = int(np.unique(pixels, axis=0).shape[0])
    if unique_color_count <= 512:
        return

    message = (
        f"{pred_root} looks like a natural RGB image directory "
        f"({unique_color_count} sampled colors in {pred_paths[0].name}), "
        "not a predicted mask directory. Binary mIoU would treat almost every "
        "non-black pixel as foreground and produce a meaningless score. "
        "Point --pred to predicted semantic masks instead."
    )
    if allow_likely_rgb_images:
        print(f"[warning] {message}")
    else:
        raise ValueError(
            message
            + " If this is intentional, pass "
            "--allow-likely-rgb-images explicitly."
        )


def _apply_scalar_mapping(
    labels: np.ndarray,
    mapping: Mapping[int, int],
    path: Path,
    ignore_index: Optional[int],
) -> np.ndarray:
    mapped = np.full(labels.shape, -1, dtype=np.int64)
    for source_value, class_id in mapping.items():
        mapped[labels == int(source_value)] = int(class_id)

    unknown = mapped == -1
    if np.any(unknown):
        unknown_values = np.unique(labels[unknown]).tolist()
        if ignore_index is not None and ignore_index in unknown_values:
            mapped[labels == ignore_index] = ignore_index
            unknown_values = [value for value in unknown_values if value != ignore_index]
        if unknown_values:
            preview = unknown_values[:10]
            raise ValueError(
                f"{path} contains unmapped scalar values {preview}. "
                "Complete label_mapping or use mode='binary'."
            )
    return mapped


def _apply_color_mapping(
    image: np.ndarray,
    mapping: Mapping[Tuple[int, int, int], int],
    path: Path,
    ignore_index: Optional[int],
) -> np.ndarray:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Color mapping requires an RGB image: {path}")

    mapped = np.full(image.shape[:2], -1, dtype=np.int64)
    for color, class_id in mapping.items():
        if len(color) != 3:
            raise ValueError(f"Invalid RGB color key {color!r} in mapping.")
        color_array = np.asarray(color, dtype=image.dtype)
        mapped[np.all(image == color_array, axis=2)] = int(class_id)

    unknown = mapped == -1
    if np.any(unknown):
        unknown_colors = np.unique(image[unknown], axis=0).tolist()
        if ignore_index is not None:
            ignore_color = np.all(
                image == np.asarray((ignore_index,) * 3, dtype=image.dtype),
                axis=2,
            )
            mapped[ignore_color] = ignore_index
            unknown_colors = [
                color
                for color in unknown_colors
                if tuple(color) != (ignore_index,) * 3
            ]
        if unknown_colors:
            raise ValueError(
                f"{path} contains unmapped RGB colors {unknown_colors[:10]}. "
                "Complete color_mapping or use mode='binary'."
            )
    return mapped


def decode_label_image(
    path: Path,
    mode: str,
    num_classes: int,
    label_mapping: Optional[Mapping[int, int]] = None,
    color_mapping: Optional[Mapping[Tuple[int, int, int], int]] = None,
    ignore_index: Optional[int] = None,
) -> np.ndarray:
    """Decode a mask into an integer class-ID array."""
    image = read_label_image(path)

    if mode == "binary":
        if label_mapping is not None or color_mapping is not None:
            raise ValueError(
                "Do not combine label/color mappings with mode='binary'; "
                "binary mode maps zero to background and non-zero to foreground."
            )
        labels = (image != 0).any(axis=2) if image.ndim == 3 else image != 0
        labels = labels.astype(np.int64)
    elif mode == "class":
        if color_mapping is not None:
            labels = _apply_color_mapping(image, color_mapping, path, ignore_index)
        else:
            if image.ndim == 3:
                channels_equal = np.all(image == image[..., :1], axis=2)
                if not np.all(channels_equal):
                    raise ValueError(
                        f"{path} is RGB but no color_mapping was provided. "
                        "Use mode='binary' for instance-color foreground masks."
                    )
                image = image[..., 0]
            labels = image.astype(np.int64)

        if label_mapping is not None:
            labels = _apply_scalar_mapping(
                labels,
                label_mapping,
                path,
                ignore_index,
            )
    else:
        raise ValueError(f"Unsupported evaluation mode: {mode!r}")

    labels = labels.astype(np.int64, copy=False)
    valid_labels = labels != ignore_index if ignore_index is not None else np.ones(
        labels.shape, dtype=bool
    )
    invalid = valid_labels & ((labels < 0) | (labels >= num_classes))
    if np.any(invalid):
        values = np.unique(labels[invalid]).tolist()
        raise ValueError(
            f"{path} contains class IDs {values[:10]}, but num_classes={num_classes}."
        )
    return labels


def resize_labels(labels: np.ndarray, shape_hw: Tuple[int, int]) -> np.ndarray:
    """Resize integer labels without introducing interpolated class IDs."""
    if labels.shape == shape_hw:
        return labels
    return cv2.resize(
        labels.astype(np.int32),
        (shape_hw[1], shape_hw[0]),
        interpolation=cv2.INTER_NEAREST,
    ).astype(np.int64)


class Evaluator:
    """Accumulate a confusion matrix and compute per-class IoU and mIoU."""

    def __init__(self, num_classes: int, ignore_index: Optional[int] = None):
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2.")
        self.num_classes = int(num_classes)
        self.ignore_index = ignore_index
        self.reset()

    def reset(self) -> None:
        self.hist = np.zeros(
            (self.num_classes, self.num_classes),
            dtype=np.int64,
        )

    def _fast_hist(
        self,
        label_pred: np.ndarray,
        label_true: np.ndarray,
    ) -> np.ndarray:
        pred = np.asarray(label_pred, dtype=np.int64).reshape(-1)
        true = np.asarray(label_true, dtype=np.int64).reshape(-1)
        if pred.shape != true.shape:
            raise ValueError(
                f"Prediction and ground-truth sizes differ: {pred.shape} vs {true.shape}"
            )

        mask = (
            (true >= 0)
            & (true < self.num_classes)
            & (pred >= 0)
            & (pred < self.num_classes)
        )
        if self.ignore_index is not None:
            mask &= (true != self.ignore_index) & (pred != self.ignore_index)

        encoded = self.num_classes * true[mask] + pred[mask]
        return np.bincount(
            encoded,
            minlength=self.num_classes**2,
        ).reshape(self.num_classes, self.num_classes)

    def add_batch(
        self,
        label_pred: np.ndarray,
        label_true: np.ndarray,
    ) -> None:
        self.hist += self._fast_hist(label_pred, label_true)

    def evaluate(
        self,
        include_absent_classes: bool = False,
    ) -> Tuple[np.ndarray, float, float]:
        intersection = np.diag(self.hist).astype(np.float64)
        union = (
            self.hist.sum(axis=1)
            + self.hist.sum(axis=0)
            - intersection
        ).astype(np.float64)

        iou = np.full(self.num_classes, np.nan, dtype=np.float64)
        present = union > 0
        iou[present] = intersection[present] / union[present]

        if include_absent_classes:
            miou = float(np.nan_to_num(iou, nan=0.0).mean())
        elif np.any(present):
            miou = float(np.nanmean(iou))
        else:
            miou = float("nan")

        total = float(self.hist.sum())
        pixel_accuracy = (
            float(intersection.sum() / total) if total > 0 else float("nan")
        )
        return iou, miou, pixel_accuracy


def _format_score(value: float) -> str:
    return "nan" if np.isnan(value) else f"{value:.6f}"


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, (np.floating, float)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def run_evaluation(
    gt_dir: str | Path,
    pred_dirs: Mapping[str, str | Path],
    num_classes: int = NUM_CLASSES,
    label_mapping: Optional[Mapping[int, int]] = LABEL_MAPPING,
    color_mapping: Optional[Mapping[Tuple[int, int, int], int]] = COLOR_MAPPING,
    mode: str = EVAL_MODE,
    class_names: Optional[Sequence[str]] = CLASS_NAMES,
    ignore_index: Optional[int] = None,
    include_absent_classes: bool = False,
    allow_missing: bool = False,
    allow_likely_rgb_images: bool = False,
    max_images: Optional[int] = None,
    save_json: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """Evaluate one or more predicted mask directories."""
    gt_root = _as_path(gt_dir)
    if not gt_root.exists():
        raise FileNotFoundError(f"Ground-truth directory not found: {gt_root}")

    gt_paths = discover_image_files(gt_root)
    if max_images is not None:
        gt_paths = gt_paths[:max_images]
    if not gt_paths:
        raise RuntimeError(f"No supported mask files found under {gt_root}")

    names = list(class_names or [f"class_{idx}" for idx in range(num_classes)])
    if len(names) != num_classes:
        raise ValueError(
            f"class_names has {len(names)} names, expected {num_classes}."
        )

    results: Dict[str, Any] = {
        "gt_dir": gt_root,
        "mode": mode,
        "num_classes": num_classes,
        "class_names": names,
        "images_in_gt": len(gt_paths),
        "methods": {},
    }

    print(f"Ground-truth masks: {len(gt_paths)}")
    print(f"Ground-truth directory: {gt_root}")

    for method_name, pred_dir_value in pred_dirs.items():
        pred_root = _as_path(pred_dir_value)
        if not pred_root.exists():
            raise FileNotFoundError(
                f"Prediction directory for {method_name!r} not found: {pred_root}"
            )

        pred_paths = discover_image_files(pred_root)
        if not pred_paths:
            raise RuntimeError(
                f"No supported prediction mask files found under {pred_root}"
            )
        validate_prediction_source(
            pred_root,
            pred_paths,
            mode=mode,
            allow_likely_rgb_images=allow_likely_rgb_images,
        )
        pred_lookup = _build_name_lookup(pred_paths)
        evaluator = Evaluator(num_classes=num_classes, ignore_index=ignore_index)
        processed = 0
        missing: list[str] = []
        resized = 0

        for gt_path in gt_paths:
            pred_path = find_matching_file(
                gt_path,
                gt_root,
                pred_root,
                target_name_lookup=pred_lookup,
            )
            if pred_path is None:
                missing.append(gt_path.name)
                continue

            gt_labels = decode_label_image(
                gt_path,
                mode=mode,
                num_classes=num_classes,
                label_mapping=label_mapping,
                color_mapping=color_mapping,
                ignore_index=ignore_index,
            )
            pred_labels = decode_label_image(
                pred_path,
                mode=mode,
                num_classes=num_classes,
                label_mapping=label_mapping,
                color_mapping=color_mapping,
                ignore_index=ignore_index,
            )

            if pred_labels.shape != gt_labels.shape:
                pred_labels = resize_labels(pred_labels, gt_labels.shape)
                resized += 1
            evaluator.add_batch(pred_labels, gt_labels)
            processed += 1

        if missing and not allow_missing:
            preview = ", ".join(missing[:10])
            suffix = " ..." if len(missing) > 10 else ""
            raise FileNotFoundError(
                f"{method_name!r} is missing {len(missing)} prediction masks "
                f"(for example: {preview}{suffix}). Use allow_missing=True "
                "only when missing files should be skipped."
            )

        ious, miou, pixel_accuracy = evaluator.evaluate(
            include_absent_classes=include_absent_classes
        )
        method_result = {
            "prediction_dir": pred_root,
            "processed_images": processed,
            "missing_images": len(missing),
            "resized_predictions": resized,
            "per_class_iou": ious,
            "miou": miou,
            "pixel_accuracy": pixel_accuracy,
            "confusion_matrix": evaluator.hist,
        }
        results["methods"][method_name] = method_result

        print("\n" + "-" * 60)
        print(f"Method: {method_name}")
        print(f"Prediction directory: {pred_root}")
        print(
            f"Processed: {processed}/{len(gt_paths)}; "
            f"missing: {len(missing)}; resized: {resized}"
        )
        for class_id, iou in enumerate(ious):
            print(
                f"  class {class_id} ({names[class_id]}): "
                f"IoU={_format_score(float(iou))}"
            )
        print(f"  mIoU: {_format_score(miou)}")
        print(f"  pixel accuracy: {_format_score(pixel_accuracy)}")
        print("-" * 60)

    if save_json is not None:
        output_path = _as_path(save_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(_json_safe(results), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\nSaved metrics to: {output_path}")

    return results


def parse_prediction_spec(value: str) -> Tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "Prediction must use NAME=DIRECTORY, for example EMRDM=pred_masks"
        )
    name, directory = value.split("=", 1)
    name = name.strip()
    directory = directory.strip()
    if not name or not directory:
        raise argparse.ArgumentTypeError(
            "Prediction must use NAME=DIRECTORY with both parts non-empty."
        )
    return name, Path(directory)


def parse_mapping_json(value: str) -> Dict[Any, int]:
    """Parse scalar mappings or RGB mappings from a JSON object."""
    try:
        raw = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"Invalid JSON mapping: {exc}") from exc
    if not isinstance(raw, dict):
        raise argparse.ArgumentTypeError("Mapping must be a JSON object.")

    mapping: Dict[Any, int] = {}
    for key, class_id in raw.items():
        if isinstance(key, str) and "," in key:
            parts = tuple(int(part.strip()) for part in key.split(","))
            if len(parts) != 3:
                raise argparse.ArgumentTypeError(
                    f"RGB key must look like 'R,G,B', got {key!r}."
                )
            normalized_key: Any = parts
        else:
            try:
                normalized_key = int(key)
            except (TypeError, ValueError) as exc:
                raise argparse.ArgumentTypeError(
                    f"Scalar mapping key must be an integer, got {key!r}."
                ) from exc
        mapping[normalized_key] = int(class_id)
    return mapping


def split_mapping(
    mapping: Optional[Mapping[Any, int]],
) -> Tuple[Optional[Mapping[int, int]], Optional[Mapping[Tuple[int, int, int], int]]]:
    if mapping is None:
        return None, None
    scalar: Dict[int, int] = {}
    colors: Dict[Tuple[int, int, int], int] = {}
    for key, class_id in mapping.items():
        if isinstance(key, tuple):
            colors[key] = int(class_id)
        else:
            scalar[int(key)] = int(class_id)
    if scalar and colors:
        raise ValueError("Use either scalar mapping keys or RGB mapping keys, not both.")
    return (scalar or None), (colors or None)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute pixel-wise IoU and mean IoU for segmentation masks."
    )
    parser.add_argument("--gt-dir", type=Path, default=DEFAULT_GT_DIR)
    parser.add_argument(
        "--valid-dir",
        type=Path,
        default=DEFAULT_VALID_MASK_DIR,
        help="Valid-region masks; use --no-valid-dir to disable.",
    )
    parser.add_argument(
        "--no-valid-dir",
        action="store_true",
        help="Evaluate all pixels instead of excluding invalid padded borders.",
    )
    parser.add_argument(
        "--pred",
        action="append",
        type=parse_prediction_spec,
        metavar="NAME=DIRECTORY",
        help="Prediction mask directory; repeat for multiple methods.",
    )
    parser.add_argument(
        "--mode",
        choices=("binary", "class"),
        default=EVAL_MODE,
        help="binary: zero/non-zero masks; class: class IDs or mapped RGB colors.",
    )
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES)
    parser.add_argument(
        "--class-names",
        nargs="+",
        default=list(CLASS_NAMES),
        help="Names in class-ID order.",
    )
    parser.add_argument(
        "--mapping",
        type=parse_mapping_json,
        default=None,
        help=(
            "JSON mapping for class mode, e.g. "
            '\'{"0": 0, "127": 1}\' or '
            '\'{"0,0,0": 0, "0,127,127": 1}\'.'
        ),
    )
    parser.add_argument("--ignore-index", type=int, default=None)
    parser.add_argument(
        "--include-absent-classes",
        action="store_true",
        help="Include classes with union=0 in the mean as IoU=0.",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Skip missing prediction files instead of failing.",
    )
    parser.add_argument(
        "--allow-likely-rgb-images",
        action="store_true",
        help=(
            "Allow directories that look like natural RGB images in binary mode. "
            "This is not recommended because the resulting mIoU is usually invalid."
        ),
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Evaluate only the first N ground-truth masks for a smoke test.",
    )
    parser.add_argument("--save-json", type=Path, default=None)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    configured_predictions = dict(PRED_DIRS)
    if args.pred:
        configured_predictions = dict(args.pred)
    if not configured_predictions:
        parser.error(
            "No prediction directories configured. Add PRED_DIRS in the script "
            "or pass one or more --pred NAME=DIRECTORY arguments. "
            "The directory must contain predicted masks, not restored RGB images."
        )

    scalar_mapping, color_mapping = split_mapping(args.mapping)
    run_evaluation(
        gt_dir=args.gt_dir,
        pred_dirs=configured_predictions,
        num_classes=args.num_classes,
        label_mapping=scalar_mapping,
        color_mapping=color_mapping,
        mode=args.mode,
        class_names=args.class_names,
        ignore_index=args.ignore_index,
        include_absent_classes=args.include_absent_classes,
        allow_missing=args.allow_missing,
        allow_likely_rgb_images=args.allow_likely_rgb_images,
        max_images=args.max_images,
        save_json=args.save_json,
    )


if __name__ == "__main__":
    main()
