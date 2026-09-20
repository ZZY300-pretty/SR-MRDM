# code from https://github.com/littlebeen/DDPM-Enhancement-for-Cloud-Removal
import numpy as np
import torch
import lpips
from skimage.metrics import peak_signal_noise_ratio as PSNR
from skimage.metrics import structural_similarity as SSIM


loss_fn = lpips.LPIPS(net="alex", version=0.1, verbose=False)
_lpips_device = None


def _to_float01(img: torch.Tensor) -> torch.Tensor:
    return img.detach().float().clamp(0.0, 1.0)


def _to_numpy_hwc(img: torch.Tensor) -> np.ndarray:
    return img.detach().cpu().numpy().transpose(1, 2, 0)


def _to_luma(img: torch.Tensor) -> np.ndarray:
    img_hwc = _to_numpy_hwc(img)
    if img_hwc.shape[2] == 1:
        return img_hwc[..., 0]
    rgb = img_hwc[..., :3]
    return np.tensordot(rgb, [0.298912, 0.586611, 0.114478], axes=1)


def _to_lpips_rgb(img: torch.Tensor) -> torch.Tensor:
    if img.shape[0] == 1:
        img = img.repeat(3, 1, 1)
    elif img.shape[0] > 3:
        img = img[:3]
    return img.unsqueeze(0) * 2.0 - 1.0


def caculate_ssim(imgA, imgB):
    imgA1 = _to_luma(imgA)
    imgB1 = _to_luma(imgB)
    return float(SSIM(imgA1, imgB1, data_range=1.0))


def caculate_psnr(imgA, imgB):
    imgA1 = _to_numpy_hwc(imgA)
    imgB1 = _to_numpy_hwc(imgB)
    return float(PSNR(imgA1, imgB1, data_range=1.0))


def caculate_lpips(img0, img1):
    global _lpips_device

    im1 = _to_lpips_rgb(img0)
    im2 = _to_lpips_rgb(img1)

    if _lpips_device != im1.device:
        loss_fn.to(im1.device)
        _lpips_device = im1.device

    with torch.no_grad():
        current_lpips_distance = loss_fn(im1, im2)
    return float(current_lpips_distance.item())


def img_metrics(target, pred):
    target = _to_float01(target)
    pred = _to_float01(pred)

    rmse = torch.sqrt(torch.mean((target - pred) ** 2)).item()
    imgA = pred.squeeze(0)
    imgB = target.squeeze(0)

    psnr = caculate_psnr(imgA, imgB)

    if imgA.shape[0] == 4:
        ssim = 0.0
        lpips_value = 0.0
        for i in range(imgA.shape[0]):
            imA = imgA[i : i + 1]
            imB = imgB[i : i + 1]
            ssim += caculate_ssim(imA, imB)
            lpips_value += caculate_lpips(imA, imB)
        ssim /= imgA.shape[0]
        lpips_value /= imgA.shape[0]
    else:
        ssim = caculate_ssim(imgA, imgB)
        lpips_value = caculate_lpips(imgA, imgB)

    return {
        "PSNR": psnr,
        "SSIM": ssim,
        "LPIPS": lpips_value,
        "RMSE": rmse,
    }


class avg_img_metrics():
    def __init__(self):
        self.metrics = ["PSNR", "SSIM", "LPIPS", "RMSE"]
        self.running_img_metrics = {}
        self.running_nonan_count = {}
        self.reset()

    def reset(self):
        for metric in self.metrics:
            self.running_nonan_count[metric] = 0
            self.running_img_metrics[metric] = np.nan

    def add(self, metrics_dict):
        for key, val in metrics_dict.items():
            if key not in self.metrics:
                continue
            if torch.is_tensor(val):
                continue
            if isinstance(val, tuple):
                val = val[0]
            if np.isnan(val):
                continue
            if not self.running_nonan_count[key]:
                self.running_nonan_count[key] = 1
                self.running_img_metrics[key] = val
            else:
                self.running_nonan_count[key] += 1
                self.running_img_metrics[key] = (
                    (self.running_nonan_count[key] - 1)
                    / self.running_nonan_count[key]
                    * self.running_img_metrics[key]
                    + 1 / self.running_nonan_count[key] * val
                )

    def value(self):
        return self.running_img_metrics
