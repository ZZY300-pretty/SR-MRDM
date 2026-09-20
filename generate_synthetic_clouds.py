import argparse
import json
import math
import shutil
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageFilter


DEFAULT_INPUT_DIR = Path("datasets/archive_random_10pct_experiment_crops")
DEFAULT_OUTPUT_DIR = Path("datasets/archive_random_10pct_experiment_crops_heavy_cloud")
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
EXCLUDE_DIR_NAMES = {
    "annotations",
    "instance_masks",
    "semantic_masks",
    "mask",
    "masks",
    "meta",
    "cloud",
    "clouds",
    "label",
    "labels",
}
EXCLUDE_FILE_KEYWORDS = (
    "_instance_",
    "_semantic_",
    "_mask",
    "_label",
    "_alpha",
    "_cloud",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate semi-transparent synthetic clouds for remote sensing images. "
            "The default settings are tuned for cloud removal plus small-object "
            "detection, so object contours remain faintly visible."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=(
            "Directory containing the clean source images. By default this points "
            "to datasets/archive_random_10pct_experiment_crops."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=(
            "Output dataset directory. The script creates cloud/ label/ mask/ meta "
            "subfolders, and semantic_masks when --semantic-mask-dir is provided. "
            "By default it writes to datasets/archive_random_10pct_experiment_crops_thin_cloud."
        ),
    )
    parser.add_argument(
        "--extensions",
        nargs="*",
        default=sorted(IMAGE_SUFFIXES),
        help="Image suffixes to scan, for example: .png .jpg .tif",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Only process the first N discovered images.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=3407,
        help="Global random seed.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output files if they already exist.",
    )
    parser.add_argument(
        "--copy-label",
        action="store_true",
        help="Also copy the clean image into output_dir/label for paired training.",
    )
    parser.add_argument(
        "--semantic-mask-dir",
        type=Path,
        default=None,
        help=(
            "Optional directory containing semantic masks with the same relative "
            "paths as --input-dir. Matching files are copied into output_dir/semantic_masks."
        ),
    )
    parser.add_argument(
        "--preserve-relative-paths",
        action="store_true",
        help="Mirror the input subdirectory structure inside cloud/ label/ mask/ meta.",
    )
    parser.add_argument(
        "--work-size",
        type=int,
        default=1024,
        help="Maximum side length used to synthesize the cloud field before resizing back.",
    )
    parser.add_argument(
        "--min-coverage",
        type=float,
        default=0.10,
        help="Minimum cloud coverage ratio.",
    )
    parser.add_argument(
        "--max-coverage",
        type=float,
        default=0.32,
        help="Maximum cloud coverage ratio.",
    )
    parser.add_argument(
        "--min-optical-depth",
        type=float,
        default=0.45,
        help="Minimum cloud optical depth scaling for thin-cloud transmittance.",
    )
    parser.add_argument(
        "--max-optical-depth",
        type=float,
        default=1.10,
        help="Maximum cloud optical depth scaling for thin-cloud transmittance.",
    )
    parser.add_argument(
        "--haze-strength",
        type=float,
        default=0.30, #全局薄雾强度 原来是0.08
        help="Atmospheric veiling strength around clouds.",
    )
    parser.add_argument(
        "--blur-radius",
        type=float,
        default=2.0,
        help="Final alpha blur radius on the low-resolution cloud map.",
    )
    parser.add_argument(
        "--max-cloud-alpha",
        type=float,
        default=1.0, # 【修改这里】放开alpha上限，让云保持白色高光
        help="Upper bound of cloud alpha, keeping clouds semi-transparent.",
    )
    parser.add_argument(
        "--transmittance-floor",
        type=float,
        default=0.15, # 【修改这里】0.15、0.20、0.45 足够保留目标了，0.58 可能会太透
        help="Minimum amount of clean signal preserved under dense cloud regions.",
    )
    parser.add_argument(
        "--detail-preservation",
        type=float,
        default=0.30,
        help="How strongly local image structure is preserved through the cloud veil.",
    )
    parser.add_argument(
        "--output-image-suffix",
        type=str,
        default="",
        help="Optional override suffix for synthetic cloudy images, for example .png",
    )
    return parser


@dataclass
class CloudSampleParams:
    coverage_target: float
    optical_depth_scale: float
    alpha_scale: float
    threshold_quantile: float
    threshold_softness: float
    warp_strength: float
    haze_strength: float
    blur_radius: float
    max_cloud_alpha: float
    cloud_color: Tuple[float, float, float]
    gradient_angle_deg: float
    gradient_bias: float
    gradient_gain: float
    blob_count: int
    base_res: Tuple[int, int]
    detail_res: Tuple[int, int]
    work_shape: Tuple[int, int]


def clamp01(arr: np.ndarray) -> np.ndarray:
    return np.clip(arr, 0.0, 1.0)


def smoothstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    denom = max(edge1 - edge0, 1e-6)
    t = clamp01((x - edge0) / denom)
    return t * t * (3.0 - 2.0 * t)


def normalize_map(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    min_v = float(arr.min())
    max_v = float(arr.max())
    if max_v - min_v < 1e-6:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - min_v) / (max_v - min_v)


def blur_map(arr: np.ndarray, radius: float) -> np.ndarray:
    if radius <= 0:
        return arr.astype(np.float32)
    img = Image.fromarray(np.clip(arr * 255.0, 0, 255).astype(np.uint8), mode="L")
    img = img.filter(ImageFilter.GaussianBlur(radius=radius))
    return np.asarray(img, dtype=np.float32) / 255.0


def resize_map(arr: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    img = Image.fromarray(np.clip(arr * 255.0, 0, 255).astype(np.uint8), mode="L")
    img = img.resize(size, resample=Image.Resampling.BICUBIC)
    return np.asarray(img, dtype=np.float32) / 255.0


def bilinear_sample(arr: np.ndarray, yy: np.ndarray, xx: np.ndarray) -> np.ndarray:
    h, w = arr.shape
    yy = np.clip(yy, 0.0, h - 1.0001)
    xx = np.clip(xx, 0.0, w - 1.0001)

    y0 = np.floor(yy).astype(np.int32)
    x0 = np.floor(xx).astype(np.int32)
    y1 = np.minimum(y0 + 1, h - 1)
    x1 = np.minimum(x0 + 1, w - 1)

    wy = yy - y0
    wx = xx - x0

    top = arr[y0, x0] * (1.0 - wx) + arr[y0, x1] * wx
    bottom = arr[y1, x0] * (1.0 - wx) + arr[y1, x1] * wx
    return top * (1.0 - wy) + bottom * wy


def random_gradients(rng: np.random.Generator, h: int, w: int) -> np.ndarray:
    angles = rng.uniform(0.0, 2.0 * math.pi, size=(h, w)).astype(np.float32)
    return np.stack((np.cos(angles), np.sin(angles)), axis=-1)


def perlin_noise_2d(
    shape: Tuple[int, int],
    res: Tuple[int, int],
    rng: np.random.Generator,
) -> np.ndarray:
    h, w = shape
    ry = max(1, int(res[0]))
    rx = max(1, int(res[1]))

    gradients = random_gradients(rng, ry + 1, rx + 1)

    y = np.linspace(0.0, ry, h, endpoint=False, dtype=np.float32)
    x = np.linspace(0.0, rx, w, endpoint=False, dtype=np.float32)
    yy, xx = np.meshgrid(y, x, indexing="ij")

    y0 = np.floor(yy).astype(np.int32)
    x0 = np.floor(xx).astype(np.int32)
    y1 = y0 + 1
    x1 = x0 + 1

    yf = yy - y0
    xf = xx - x0

    g00 = gradients[y0, x0]
    g10 = gradients[y1, x0]
    g01 = gradients[y0, x1]
    g11 = gradients[y1, x1]

    d00 = np.stack((xf, yf), axis=-1)
    d10 = np.stack((xf, yf - 1.0), axis=-1)
    d01 = np.stack((xf - 1.0, yf), axis=-1)
    d11 = np.stack((xf - 1.0, yf - 1.0), axis=-1)

    n00 = np.sum(g00 * d00, axis=-1)
    n10 = np.sum(g10 * d10, axis=-1)
    n01 = np.sum(g01 * d01, axis=-1)
    n11 = np.sum(g11 * d11, axis=-1)

    u = xf * xf * xf * (xf * (xf * 6.0 - 15.0) + 10.0)
    v = yf * yf * yf * (yf * (yf * 6.0 - 15.0) + 10.0)

    nx0 = n00 * (1.0 - u) + n01 * u
    nx1 = n10 * (1.0 - u) + n11 * u
    return nx0 * (1.0 - v) + nx1 * v


def fbm_noise(
    shape: Tuple[int, int],
    base_res: Tuple[int, int],
    octaves: int,
    persistence: float,
    lacunarity: float,
    rng: np.random.Generator,
) -> np.ndarray:
    total = np.zeros(shape, dtype=np.float32)
    amplitude = 1.0
    frequency = 1.0
    amplitude_sum = 0.0

    for _ in range(octaves):
        res = (
            max(1, int(round(base_res[0] * frequency))),
            max(1, int(round(base_res[1] * frequency))),
        )
        total += amplitude * perlin_noise_2d(shape, res, rng)
        amplitude_sum += amplitude
        amplitude *= persistence
        frequency *= lacunarity

    total /= max(amplitude_sum, 1e-6)
    return normalize_map(total)


def generate_blob_field(
    shape: Tuple[int, int], rng: np.random.Generator
) -> Tuple[np.ndarray, int]:
    h, w = shape
    yy = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
    xx = np.linspace(0.0, 1.0, w, dtype=np.float32)[None, :]
    field = np.zeros(shape, dtype=np.float32)

    blob_count = int(rng.integers(5, 15))
    for _ in range(blob_count):
        cy = float(rng.uniform(0.05, 0.95))
        cx = float(rng.uniform(0.05, 0.95))
        ry = float(rng.uniform(0.06, 0.22))
        rx = float(rng.uniform(0.08, 0.28))
        theta = float(rng.uniform(0.0, math.pi))
        strength = float(rng.uniform(0.45, 1.00))
        softness = float(rng.uniform(1.20, 2.80))

        y_shift = yy - cy
        x_shift = xx - cx
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        xr = (x_shift * cos_t + y_shift * sin_t) / max(rx, 1e-4)
        yr = (-x_shift * sin_t + y_shift * cos_t) / max(ry, 1e-4)
        dist = xr * xr + yr * yr
        field += strength * np.exp(-softness * dist).astype(np.float32)

    return normalize_map(field), blob_count


def directional_gradient(
    shape: Tuple[int, int], rng: np.random.Generator
) -> Tuple[np.ndarray, float, float, float]:
    h, w = shape
    yy = np.linspace(-1.0, 1.0, h, dtype=np.float32)[:, None]
    xx = np.linspace(-1.0, 1.0, w, dtype=np.float32)[None, :]
    angle = float(rng.uniform(0.0, 2.0 * math.pi))
    bias = float(rng.uniform(-0.35, 0.35))
    gain = float(rng.uniform(0.8, 1.5))
    gradient = (math.cos(angle) * xx + math.sin(angle) * yy + bias) * gain
    return normalize_map(gradient), math.degrees(angle), bias, gain


def compute_work_shape(size: Tuple[int, int], max_side: int) -> Tuple[int, int]:
    width, height = size
    longest = max(width, height)
    if longest <= max_side:
        return height, width
    scale = max_side / float(longest)
    work_w = max(128, int(round(width * scale)))
    work_h = max(128, int(round(height * scale)))
    return work_h, work_w


def compute_detail_strength(clear_rgb: np.ndarray) -> np.ndarray:
    # Keep thin-cloud synthesis friendly to small-object detection by preserving
    # part of the high-frequency structure where the clean image already has edges.
    luminance = (
        0.299 * clear_rgb[..., 0]
        + 0.587 * clear_rgb[..., 1]
        + 0.114 * clear_rgb[..., 2]
    ).astype(np.float32)
    low_freq = blur_map(luminance, radius=1.2)
    detail = np.abs(luminance - low_freq)
    return normalize_map(detail)


def synthesize_cloud_layers(
    image_size: Tuple[int, int],
    rng: np.random.Generator,
    work_size: int,
    min_coverage: float,
    max_coverage: float,
    min_optical_depth: float,
    max_optical_depth: float,
    haze_strength: float,
    blur_radius: float,
    max_cloud_alpha: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, CloudSampleParams]:
    width, height = image_size
    work_h, work_w = compute_work_shape(image_size, work_size)
    work_shape = (work_h, work_w)

    coverage_target = float(rng.uniform(min_coverage, max_coverage))
    optical_depth_scale = float(rng.uniform(min_optical_depth, max_optical_depth))
    warp_strength = float(rng.uniform(6.0, 18.0))
    threshold_softness = float(rng.uniform(0.08, 0.18))

    base_res = (
        int(rng.integers(2, 5)),
        int(rng.integers(2, 5)),
    )
    detail_res = (
        int(rng.integers(6, 10)),
        int(rng.integers(6, 10)),
    )

    coarse = fbm_noise(work_shape, base_res, octaves=5, persistence=0.55, lacunarity=1.9, rng=rng)
    detail = fbm_noise(work_shape, detail_res, octaves=4, persistence=0.50, lacunarity=2.1, rng=rng)
    warp_x = fbm_noise(work_shape, (2, 2), octaves=3, persistence=0.55, lacunarity=2.0, rng=rng)
    warp_y = fbm_noise(work_shape, (2, 2), octaves=3, persistence=0.55, lacunarity=2.0, rng=rng)
    blob_field, blob_count = generate_blob_field(work_shape, rng)
    grad_map, angle_deg, grad_bias, grad_gain = directional_gradient(work_shape, rng)

    yy, xx = np.meshgrid(
        np.arange(work_h, dtype=np.float32),
        np.arange(work_w, dtype=np.float32),
        indexing="ij",
    )
    warp_scale = min(work_h, work_w) / 256.0
    warped = bilinear_sample(
        coarse,
        yy + (warp_y - 0.5) * warp_strength * warp_scale,
        xx + (warp_x - 0.5) * warp_strength * warp_scale,
    )
    warped = normalize_map(warped)

    structure = 0.50 * warped + 0.24 * blob_field + 0.16 * detail + 0.10 * grad_map
    structure = normalize_map(structure)

    threshold_quantile = float(1.0 - coverage_target)
    threshold = float(np.quantile(structure, threshold_quantile))
    alpha = smoothstep(threshold - threshold_softness, threshold + threshold_softness, structure)
    # Avoid re-normalizing to 1.0 here; keeping the dynamic range compressed makes
    # the cloud field look more like thin, semi-transparent veil instead of thick fog.
    alpha = alpha * (0.42 + 0.33 * grad_map + 0.25 * detail)
    alpha = clamp01(alpha)
    alpha = blur_map(alpha, blur_radius)
    alpha = np.power(alpha, float(rng.uniform(1.15, 1.65))).astype(np.float32)

    thickness_detail = 0.55 * detail + 0.25 * grad_map + 0.20 * blob_field
    thickness_detail = normalize_map(thickness_detail)
    alpha = clamp01(alpha * (0.70 + 0.30 * thickness_detail))
    alpha_scale = float(rng.uniform(max_cloud_alpha * 0.72, max_cloud_alpha))
    alpha = np.minimum(alpha * alpha_scale, max_cloud_alpha).astype(np.float32)

    # 替换为更平滑的衰减，而不是硬截断：
    alpha_scale = float(rng.uniform(0.75, 1.0))
    alpha = (alpha * alpha_scale).astype(np.float32)
    if max_cloud_alpha < 1.0:
        alpha = np.minimum(alpha, max_cloud_alpha) # 如果确实传了小于1的值，才做截断

    optical_depth = alpha * (0.35 + 0.65 * thickness_detail) * optical_depth_scale
    haze = clamp01(np.power(alpha / max(max_cloud_alpha, 1e-6), 0.90) * haze_strength)

    alpha_full = resize_map(alpha, (width, height))
    optical_depth_full = resize_map(optical_depth, (width, height))
    haze_full = resize_map(haze, (width, height))

    cloud_color = (
        float(rng.uniform(0.92, 0.97)),
        float(rng.uniform(0.93, 0.98)),
        float(rng.uniform(0.94, 0.99)),
    )

    params = CloudSampleParams(
        coverage_target=coverage_target,
        optical_depth_scale=optical_depth_scale,
        alpha_scale=alpha_scale,
        threshold_quantile=threshold_quantile,
        threshold_softness=threshold_softness,
        warp_strength=warp_strength,
        haze_strength=haze_strength,
        blur_radius=blur_radius,
        max_cloud_alpha=max_cloud_alpha,
        cloud_color=cloud_color,
        gradient_angle_deg=angle_deg,
        gradient_bias=grad_bias,
        gradient_gain=grad_gain,
        blob_count=blob_count,
        base_res=base_res,
        detail_res=detail_res,
        work_shape=work_shape,
    )
    return alpha_full, optical_depth_full, haze_full, params


def compose_cloudy_image(
    clear_rgb: np.ndarray,
    alpha: np.ndarray,
    optical_depth: np.ndarray,
    haze: np.ndarray,
    cloud_color: Sequence[float],
    transmittance_floor: float,
    detail_preservation: float,
) -> np.ndarray:
    cloud_color_arr = np.asarray(cloud_color, dtype=np.float32).reshape(1, 1, 3)

    # 提取高频细节前，先避开 DOTA 黑色 Padding 产生的假边缘
    # 如果像素值之和小于 0.01，判定为黑色 Padding 区域
    padding_mask = (clear_rgb.sum(axis=-1, keepdims=True) <= 0.01)

    # 计算高频细节
    detail_strength = compute_detail_strength(clear_rgb)[..., None]

    transmittance = np.exp(-optical_depth[..., None]).astype(np.float32)
    local_floor = 1.0 - alpha[..., None] * (1.0 - transmittance_floor)
    transmittance = np.maximum(transmittance, local_floor)
    cloudy = clear_rgb * transmittance + cloud_color_arr * (1.0 - transmittance)

    haze_rgb = (cloud_color_arr * 0.985).astype(np.float32)
    cloudy = cloudy * (1.0 - haze[..., None]) + haze_rgb * haze[..., None]

    # Reinstate part of the clean-image structure under cloud so tiny targets keep
    # a faint contour, which is more useful for cloud removal + downstream detection.
    detail_weight = clamp01(alpha[..., None] * detail_strength * detail_preservation)
    cloudy = cloudy * (1.0 - detail_weight) + clear_rgb * detail_weight

    edge_whitening = np.power(alpha, 1.45, dtype=np.float32)[..., None] * 0.04
    cloudy = cloudy * (1.0 - edge_whitening) + edge_whitening

    cloudy_final = clamp01(cloudy)
    
    # 【新增核心修复】：把原本纯黑色的 Padding 区域强行还原回纯黑
    # 这样既不会浪费模型算力去识别虚空，也不会在边界产生假边缘
    cloudy_final = np.where(padding_mask, clear_rgb, cloudy_final)
    
    return cloudy_final


def load_rgb_image(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        rgb = img.convert("RGB")
        return np.asarray(rgb, dtype=np.float32) / 255.0


def save_rgb_image(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.fromarray(np.clip(arr * 255.0, 0, 255).astype(np.uint8), mode="RGB")
    img.save(path)


def save_gray_image(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.fromarray(np.clip(arr * 255.0, 0, 255).astype(np.uint8), mode="L")
    img.save(path)


def should_skip_file(path: Path) -> bool:
    parts_lower = {part.lower() for part in path.parts}
    if any(name in parts_lower for name in EXCLUDE_DIR_NAMES):
        return True
    name_lower = path.name.lower()
    return any(keyword in name_lower for keyword in EXCLUDE_FILE_KEYWORDS)


def discover_images(input_dir: Path, extensions: Iterable[str], output_dir: Path) -> List[Path]:
    suffixes = {ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in extensions}
    output_dir_resolved = output_dir.resolve()
    candidates: List[Path] = []

    for path in input_dir.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in suffixes:
            continue
        try:
            resolved = path.resolve()
            if output_dir_resolved == resolved or output_dir_resolved in resolved.parents:
                continue
        except OSError:
            pass
        if should_skip_file(path):
            continue
        candidates.append(path)

    return sorted(candidates)


def build_output_stem(
    image_path: Path,
    input_dir: Path,
    preserve_relative_paths: bool,
) -> Path:
    if preserve_relative_paths:
        return image_path.relative_to(input_dir)
    return Path(image_path.name)


def maybe_copy_label(src: Path, dst: Path, overwrite: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and not overwrite:
        return
    shutil.copy2(src, dst)


def maybe_copy_semantic_mask(
    semantic_mask_dir: Path,
    image_path: Path,
    input_dir: Path,
    output_dir: Path,
    rel_path: Path,
    overwrite: bool,
) -> None:
    semantic_src = semantic_mask_dir / image_path.relative_to(input_dir)
    if not semantic_src.exists():
        print(f"[warn] semantic mask not found for {image_path.name}: {semantic_src}")
        return

    semantic_dst = (output_dir / "semantic_masks" / rel_path).with_suffix(
        semantic_src.suffix
    )
    semantic_dst.parent.mkdir(parents=True, exist_ok=True)
    if semantic_dst.exists() and not overwrite:
        return
    shutil.copy2(semantic_src, semantic_dst)


def process_one_image(
    image_path: Path,
    input_dir: Path,
    output_dir: Path,
    rng: np.random.Generator,
    args: argparse.Namespace,
) -> None:
    clear_rgb = load_rgb_image(image_path)
    height, width = clear_rgb.shape[:2]

    alpha, optical_depth, haze, params = synthesize_cloud_layers(
        image_size=(width, height),
        rng=rng,
        work_size=args.work_size,
        min_coverage=args.min_coverage,
        max_coverage=args.max_coverage,
        min_optical_depth=args.min_optical_depth,
        max_optical_depth=args.max_optical_depth,
        haze_strength=args.haze_strength,
        blur_radius=args.blur_radius,
        max_cloud_alpha=args.max_cloud_alpha,
    )

    cloudy = compose_cloudy_image(
        clear_rgb=clear_rgb,
        alpha=alpha,
        optical_depth=optical_depth,
        haze=haze,
        cloud_color=params.cloud_color,
        transmittance_floor=args.transmittance_floor,
        detail_preservation=args.detail_preservation,
    )

    rel_path = build_output_stem(
        image_path=image_path,
        input_dir=input_dir,
        preserve_relative_paths=args.preserve_relative_paths,
    )

    cloud_suffix = args.output_image_suffix or image_path.suffix
    if not cloud_suffix.startswith("."):
        cloud_suffix = f".{cloud_suffix}"

    cloud_path = (output_dir / "cloud" / rel_path).with_suffix(cloud_suffix)
    label_path = output_dir / "label" / rel_path
    mask_path = (output_dir / "mask" / rel_path).with_suffix(".png")
    meta_path = (output_dir / "meta" / rel_path).with_suffix(".json")

    if cloud_path.exists() and not args.overwrite:
        print(f"[skip] {image_path}")
        return

    save_rgb_image(cloud_path, cloudy)
    save_gray_image(mask_path, alpha)

    if args.copy_label:
        maybe_copy_label(image_path, label_path, args.overwrite)
    if args.semantic_mask_dir is not None:
        maybe_copy_semantic_mask(
            semantic_mask_dir=args.semantic_mask_dir,
            image_path=image_path,
            input_dir=input_dir,
            output_dir=output_dir,
            rel_path=rel_path,
            overwrite=args.overwrite,
        )

    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "source_image": str(image_path),
                "cloud_image": str(cloud_path),
                "alpha_mask": str(mask_path),
                "image_size": [width, height],
                "actual_coverage": float((alpha > 0.15).mean()),
                "mean_alpha": float(alpha.mean()),
                "mean_optical_depth": float(optical_depth.mean()),
                "transmittance_floor": float(args.transmittance_floor),
                "detail_preservation": float(args.detail_preservation),
                "params": asdict(params),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"[done] {image_path.name} -> {cloud_path}")


def validate_args(args: argparse.Namespace) -> None:
    if not args.input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {args.input_dir}")
    if not args.input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {args.input_dir}")
    if args.semantic_mask_dir is not None:
        if not args.semantic_mask_dir.exists():
            raise FileNotFoundError(
                f"Semantic mask directory does not exist: {args.semantic_mask_dir}"
            )
        if not args.semantic_mask_dir.is_dir():
            raise NotADirectoryError(
                f"Semantic mask path is not a directory: {args.semantic_mask_dir}"
            )
    if not (0.0 < args.min_coverage < 1.0 and 0.0 < args.max_coverage < 1.0):
        raise ValueError("Cloud coverage must be in (0, 1).")
    if args.min_coverage > args.max_coverage:
        raise ValueError("min_coverage must be <= max_coverage.")
    if args.min_optical_depth <= 0 or args.max_optical_depth <= 0:
        raise ValueError("Optical depth must be positive.")
    if args.min_optical_depth > args.max_optical_depth:
        raise ValueError("min_optical_depth must be <= max_optical_depth.")
    if not (0.0 < args.max_cloud_alpha <= 1.0):
        raise ValueError("max_cloud_alpha must be in (0, 1].")
    if not (0.0 < args.transmittance_floor <= 1.0):
        raise ValueError("transmittance_floor must be in (0, 1].")
    if not (0.0 <= args.detail_preservation <= 1.0):
        raise ValueError("detail_preservation must be in [0, 1].")
    if args.work_size < 128:
        raise ValueError("work_size should be at least 128.")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args)

    images = discover_images(args.input_dir, args.extensions, args.output_dir)

    if args.max_images is not None:
        images = images[: args.max_images]

    if not images:
        raise RuntimeError(
            "No valid images were found. Please check --input-dir, --extensions, "
            "or whether the folder only contains masks/annotations."
        )

    print(f"Found {len(images)} image(s) in {args.input_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    base_rng = np.random.default_rng(args.seed)
    for idx, image_path in enumerate(images, start=1):
        image_seed = int(base_rng.integers(0, 2**31 - 1))
        image_rng = np.random.default_rng(image_seed)
        print(f"[{idx}/{len(images)}] Processing {image_path}")
        process_one_image(image_path, args.input_dir, args.output_dir, image_rng, args)

    print(f"Finished. Outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
