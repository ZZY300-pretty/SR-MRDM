import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import skimage.io as io
from torch.utils import data


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


class CloudRemovalPatchDataset(data.Dataset):
    def __init__(
        self,
        dataset_root: str,
        split: Optional[str] = None,
        cloudy_dirname: str = "cloud",
        label_dirname: str = "label",
        mask_dirname: str = "mask",
        meta_dirname: str = "meta",
        semantic_mask_dirname: str = "semantic_masks",
        return_meta: bool = False,
        require_mask: bool = False,
        return_semantic_mask: bool = False,
        require_semantic_mask: bool = False,
        semantic_mask_key: str = "semantic_mask",
        semantic_mask_available_key: str = "semantic_mask_available",
        semantic_positive_labels: Optional[List[int]] = None,
        semantic_nonzero_is_foreground: bool = True,
        repeat_to_three_channels: bool = False,
        cloud_channels: int = 3,
        label_channels: int = 3,
    ):
        super().__init__()
        self.dataset_root = Path(dataset_root)
        self.split = split
        self.return_meta = return_meta
        self.require_mask = require_mask
        self.repeat_to_three_channels = repeat_to_three_channels
        self.cloud_channels = cloud_channels
        self.label_channels = label_channels

        self.root = self._resolve_root(self.dataset_root, split)
        self.cloud_dir = self.root / cloudy_dirname
        self.label_dir = self.root / label_dirname
        self.mask_dir = self.root / mask_dirname
        self.meta_dir = self.root / meta_dirname
        self.semantic_mask_dir = self.root / semantic_mask_dirname
        self.return_semantic_mask = return_semantic_mask
        self.require_semantic_mask = require_semantic_mask
        self.semantic_mask_key = semantic_mask_key
        self.semantic_mask_available_key = semantic_mask_available_key
        self.semantic_positive_labels = semantic_positive_labels
        self.semantic_nonzero_is_foreground = semantic_nonzero_is_foreground

        if not self.cloud_dir.exists():
            raise FileNotFoundError(f"Cloud directory not found: {self.cloud_dir}")
        if not self.label_dir.exists():
            raise FileNotFoundError(f"Label directory not found: {self.label_dir}")
        if self.require_mask and not self.mask_dir.exists():
            raise FileNotFoundError(f"Mask directory not found: {self.mask_dir}")
        if self.require_semantic_mask and not self.semantic_mask_dir.exists():
            raise FileNotFoundError(
                f"Semantic mask directory not found: {self.semantic_mask_dir}"
            )

        self.samples = self._build_samples()
        if not self.samples:
            raise RuntimeError(f"No paired patch samples found under {self.root}")

    def _resolve_root(self, dataset_root: Path, split: Optional[str]) -> Path:
        if split is None:
            return dataset_root
        split_root = dataset_root / split
        return split_root if split_root.exists() else dataset_root

    def _discover_image_files(self, root: Path) -> List[Path]:
        return sorted(
            path
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )

    def _resolve_optional_image_file(self, root: Path, rel_path: Path) -> Optional[Path]:
        candidates = [root / rel_path]
        candidates.extend((root / rel_path).with_suffix(suffix) for suffix in IMAGE_SUFFIXES)
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    def _build_samples(self) -> List[Dict[str, Optional[Path]]]:
        samples = []
        cloud_files = self._discover_image_files(self.cloud_dir)

        for cloud_path in cloud_files:
            rel_path = cloud_path.relative_to(self.cloud_dir)
            label_path = self.label_dir / rel_path
            if not label_path.exists():
                continue

            mask_path = self._resolve_optional_image_file(self.mask_dir, rel_path)
            if self.require_mask and mask_path is None:
                continue

            meta_path = (self.meta_dir / rel_path).with_suffix(".json")
            if not meta_path.exists():
                meta_path = None

            semantic_mask_path = self._resolve_optional_image_file(
                self.semantic_mask_dir, rel_path
            )
            if self.require_semantic_mask and semantic_mask_path is None:
                continue

            samples.append(
                {
                    "cloud": cloud_path,
                    "label": label_path,
                    "mask": mask_path,
                    "semantic_mask": semantic_mask_path,
                    "meta": meta_path,
                    "relative_path": rel_path,
                }
            )
        return samples

    def _ensure_channel_last(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            image = image[:, :, None]
        if image.ndim != 3:
            raise ValueError(f"Unsupported image shape: {image.shape}")
        return image

    def _coerce_channels(
        self, image: np.ndarray, num_channels: Optional[int], image_path: Path
    ) -> np.ndarray:
        image = self._ensure_channel_last(image)
        if num_channels is None or image.shape[2] == num_channels:
            return image

        if image.shape[2] > num_channels:
            # RGB training should ignore alpha or any extra channels by default.
            return image[:, :, :num_channels]

        if image.shape[2] == 1 and num_channels == 3:
            return np.repeat(image, 3, axis=2)

        raise ValueError(
            f"Cannot coerce {image_path} from {image.shape[2]} channels to {num_channels}."
        )

    def _normalize_image(self, image: np.ndarray, image_path: Path) -> np.ndarray:
        image = self._ensure_channel_last(image).astype(np.float32)
        if image.max() <= 1.0:
            image = image * 255.0
        if self.repeat_to_three_channels and image.shape[2] == 1:
            image = np.repeat(image, 3, axis=2)
        image = (image / 255.0) * 2.0 - 1.0
        return image.transpose(2, 0, 1)

    def _load_mask(self, path: Optional[Path], shape_hw) -> np.ndarray:
        if path is None:
            return np.zeros(shape_hw, dtype=np.float32)

        mask = io.imread(path).astype(np.float32)
        if mask.ndim == 3:
            mask = mask[..., 0]
        if mask.max() > 1.0:
            mask = mask / 255.0
        return np.clip(mask, 0.0, 1.0).astype(np.float32)

    def _load_semantic_mask(self, path: Optional[Path], shape_hw) -> np.ndarray:
        if path is None:
            return np.zeros(shape_hw, dtype=np.float32)

        semantic_mask = io.imread(path)
        if semantic_mask.ndim == 3:
            if self.semantic_positive_labels is not None:
                semantic_mask = semantic_mask[..., 0]
            else:
                semantic_mask = np.any(semantic_mask != 0, axis=2).astype(np.float32)

        semantic_mask = semantic_mask.astype(np.float32)
        if self.semantic_positive_labels is not None:
            semantic_mask = np.isin(
                semantic_mask.astype(np.int64), self.semantic_positive_labels
            ).astype(np.float32)
        elif self.semantic_nonzero_is_foreground:
            semantic_mask = (semantic_mask != 0).astype(np.float32)
        elif semantic_mask.max() > 1.0:
            semantic_mask = semantic_mask / 255.0

        return np.clip(semantic_mask, 0.0, 1.0).astype(np.float32)

    def __getitem__(self, index: int) -> Dict:
        sample = self.samples[index]
        cloudy = io.imread(sample["cloud"])
        label = io.imread(sample["label"])

        cloudy = self._coerce_channels(cloudy, self.cloud_channels, sample["cloud"])
        label = self._coerce_channels(label, self.label_channels, sample["label"])
        if cloudy.shape[:2] != label.shape[:2]:
            raise ValueError(
                f"Cloud/label size mismatch: {sample['cloud']} vs {sample['label']}"
            )

        cloudy_tensor = self._normalize_image(cloudy, sample["cloud"])
        label_tensor = self._normalize_image(label, sample["label"])
        mask = self._load_mask(sample["mask"], cloudy.shape[:2])

        out = {
            "cloudy": cloudy_tensor[:3, ...],
            "cond_image": cloudy_tensor,
            "label": label_tensor,
            "M": mask,
            "image_path": sample["relative_path"].as_posix(),
        }

        if self.return_semantic_mask or self.require_semantic_mask:
            semantic_available = sample["semantic_mask"] is not None
            out[self.semantic_mask_key] = self._load_semantic_mask(
                sample["semantic_mask"], cloudy.shape[:2]
            )
            out[self.semantic_mask_available_key] = np.float32(semantic_available)

        if self.return_meta:
            meta = {}
            if sample["meta"] is not None:
                with open(sample["meta"], "r", encoding="utf-8") as f:
                    meta = json.load(f)
            out["meta"] = meta

        return out

    def __len__(self) -> int:
        return len(self.samples)
