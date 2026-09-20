import os
import cv2
import numpy as np
import torch
import lpips
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import mean_squared_error as compare_mse
from tqdm import tqdm
import argparse

def img2tensor(img_bgr, device):
    """将 OpenCV 读取的 BGR 图像转换为 LPIPS 需要的 Tensor 格式 (RGB, 归一化到 [-1, 1])"""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    # 归一化到 [-1, 1]
    img_tensor = torch.from_numpy(img_rgb).float() / 127.5 - 1.0
    # 调整维度从 HWC 到 BCHW (1, C, H, W)
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)
    return img_tensor.to(device)

def main(args):
    degraded_dir = args.degraded
    gt_dir = args.gt

    # 初始化 LPIPS 模型 (默认使用 alexnet 作为特征提取，速度快且学术界通用)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"正在加载 LPIPS 模型，使用设备: {device}...")
    loss_fn_vgg = lpips.LPIPS(net='alex').to(device)

    img_names = [f for f in os.listdir(degraded_dir) if f.endswith(('.png', '.jpg', '.tif'))]
    print(f"找到 {len(img_names)} 张测试图像，准备计算全维度基线指标...")

    psnr_list, ssim_list, rmse_list, lpips_list = [], [], [], []

    # 禁用梯度计算以加速推理
    with torch.no_grad():
        for img_name in tqdm(img_names):
            deg_path = os.path.join(degraded_dir, img_name)
            gt_path = os.path.join(gt_dir, img_name)

            if not os.path.exists(gt_path):
                print(f"警告: 找不到对应的真值图 {gt_path}，跳过该图。")
                continue

            # 1. 读取图像 (BGR)
            img_deg = cv2.imread(deg_path)
            img_gt = cv2.imread(gt_path)

            # 确保尺寸一致
            if img_deg.shape != img_gt.shape:
                img_deg = cv2.resize(img_deg, (img_gt.shape[1], img_gt.shape[0]))

            # 2. 计算传统像素/结构级指标 (CPU 上进行)
            psnr = compare_psnr(img_gt, img_deg, data_range=255)
            ssim = compare_ssim(img_gt, img_deg, data_range=255, channel_axis=2)
            mse = compare_mse(img_gt, img_deg)
            rmse = np.sqrt(mse) / 255.0  # 归一化到 0-1 范围，与扩散模型输出对齐

            # 3. 计算感知相似度 LPIPS (GPU 上进行)
            tensor_deg = img2tensor(img_deg, device)
            tensor_gt = img2tensor(img_gt, device)
            lpips_val = loss_fn_vgg(tensor_deg, tensor_gt).item()

            psnr_list.append(psnr)
            ssim_list.append(ssim)
            rmse_list.append(rmse)
            lpips_list.append(lpips_val)

    # 统计平均值
    avg_psnr = np.mean(psnr_list)
    avg_ssim = np.mean(ssim_list)
    avg_rmse = np.mean(rmse_list)
    avg_lpips = np.mean(lpips_list)

    print("\n" + "="*50)
    print("🎯 测试集图像质量基线结果 (Raw Baseline):")
    print(f"  * PSNR  (越大越好): {avg_psnr:.4f} dB")
    print(f"  * SSIM  (越大越好): {avg_ssim:.4f}")
    print(f"  * RMSE  (越小越好): {avg_rmse:.4f}")
    print(f"  * LPIPS (越小越好): {avg_lpips:.4f}")
    print("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="计算全维度图像基线指标 (PSNR, SSIM, RMSE, LPIPS)")
    parser.add_argument("--degraded", type=str, required=True, help="/mnt/mydisk/zwj/zzy/EMRDM-main/datasets/restored_target_cloud")
    parser.add_argument("--gt", type=str, required=True, help="/mnt/mydisk/zwj/zzy/EMRDM-main/datasets/archive_random_10pct_experiment_crops_dota/test/images")
    args = parser.parse_args()
    main(args)