import os

import cv2
import numpy as np
import skimage.io as io
from torch.utils import data


def _find_first_existing(parent_dir, candidates):
    for name in candidates:
        candidate = os.path.join(parent_dir, name)
        if os.path.isdir(candidate):
            return candidate
    return None


def _find_first_with_files(parent_dir, candidates):
    fallback = None
    for name in candidates:
        candidate = os.path.join(parent_dir, name)
        if not os.path.isdir(candidate):
            continue
        if fallback is None:
            fallback = candidate
        for item in os.listdir(candidate):
            if os.path.isfile(os.path.join(candidate, item)):
                return candidate
    return fallback


def _read_image(path):
    image = io.imread(path).astype(np.float32)
    if image.ndim == 2:
        image = image[:, :, np.newaxis]
    return image


def _ensure_rgb(image):
    if image.shape[2] == 1:
        return np.repeat(image, 3, axis=2)
    if image.shape[2] >= 3:
        return image[:, :, :3]
    raise ValueError(f"Unsupported channel count: {image.shape}")


def _resize_image(image, width, height):
    if image.shape[1] == width and image.shape[0] == height:
        return image

    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC)
    if resized.ndim == 2:
        resized = resized[:, :, np.newaxis]
    return resized.astype(np.float32)


class TrainDataset(data.Dataset):
    def __init__(self, config, split="train"):
        super().__init__()
        self.datasets_dir = os.path.join(config.datasets_dir, split)
        if not os.path.isdir(self.datasets_dir):
            raise FileNotFoundError(f"Dataset split not found: {self.datasets_dir}")

        self.label_dir = _find_first_with_files(self.datasets_dir, ["label", "GT", "ground_truth", "gt"])
        self.cloud_dir = _find_first_with_files(
            self.datasets_dir, ["cloudy", "cloud", "cloud_image", "cloudy_image", "input"]
        )
        if self.label_dir is None or self.cloud_dir is None:
            raise FileNotFoundError(
                f"Expected label/GT and cloudy/cloud folders under {self.datasets_dir}"
            )

        nir_root = _find_first_existing(self.datasets_dir, ["nir", "NIR"])
        self.nir_label_dir = None
        self.nir_cloud_dir = None
        if nir_root is not None:
            self.nir_label_dir = _find_first_with_files(nir_root, ["label", "GT", "ground_truth", "gt"])
            self.nir_cloud_dir = _find_first_with_files(
                nir_root, ["cloudy", "cloud", "cloud_image", "cloudy_image", "input"]
            )
            if self.nir_label_dir is None or self.nir_cloud_dir is None:
                self.nir_label_dir = None
                self.nir_cloud_dir = None

        self.image_names = sorted(
            [name for name in os.listdir(self.label_dir) if os.path.isfile(os.path.join(self.label_dir, name))]
        )
        if not self.image_names:
            raise FileNotFoundError(f"No images found in {self.label_dir}")

        self.width = int(getattr(config, "width", 0) or 0)
        self.height = int(getattr(config, "height", 0) or 0)

    def __getitem__(self, index):
        filename = self.image_names[index]
        target = _ensure_rgb(_read_image(os.path.join(self.label_dir, filename)))
        cloudy = _ensure_rgb(_read_image(os.path.join(self.cloud_dir, filename)))

        if self.nir_label_dir and self.nir_cloud_dir:
            target_nir = _read_image(os.path.join(self.nir_label_dir, filename))[:, :, 0]
            cloudy_nir = _read_image(os.path.join(self.nir_cloud_dir, filename))[:, :, 0]
            target = np.concatenate([target, target_nir[:, :, np.newaxis]], axis=2)
            cloudy = np.concatenate([cloudy, cloudy_nir[:, :, np.newaxis]], axis=2)

        if self.width > 0 and self.height > 0:
            target = _resize_image(target, self.width, self.height)
            cloudy = _resize_image(cloudy, self.width, self.height)

        mask = np.clip((target - cloudy).sum(axis=2), 0, 1).astype(np.float32)
        target = (target / 255.0).transpose(2, 0, 1)
        cloudy = (cloudy / 255.0).transpose(2, 0, 1)

        return cloudy, target, mask, os.path.splitext(filename)[0]

    def __len__(self):
        return len(self.image_names)
