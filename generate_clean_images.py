import os
import torch
from omegaconf import OmegaConf
from PIL import Image
import numpy as np
import torchvision.transforms as T
from tqdm import tqdm

# ==========================================
# ⚙️ 需要你根据实际情况修改的配置项
# ==========================================

# 1. 你的 Baseline 配置文件路径
CONFIG_PATH = "configs/example_training/patch_synthetic_clouds.yaml"

# 2. 你的最佳 Checkpoint 权重路径 (修改为你认为最好的一轮，比如 epoch 40)
CKPT_PATH = "logs2/2026-06-22T19-10-57_baseline_heavy_cloud/checkpoints/epoch=000040.ckpt"

# 3. 存放“带云原图”的文件夹路径 (你需要去云的测试集或验证集)
# 例如，你想对 heavy_cloud 的测试集去云：
INPUT_DIR = "datasets/archive_random_10pct_experiment_crops_heavy_cloud/test/cloud"

# 4. 去云结果保存的目标文件夹路径
OUTPUT_DIR = "datasets/archive_random_10pct_experiment_crops_heavy_cloud_cleaned/test/cloud"

# ==========================================

def load_model_from_config(config_path, ckpt_path):
    print(f"Loading configuration from: {config_path}")
    config = OmegaConf.load(config_path)
    
    # 根据 SGM 框架动态实例化模型
    # 注意: 这里假设你的项目里有一个 instantiate_from_config 的工具函数
    # 通常在 sgm.util 或者 main.py 中定义
    from sgm.util import instantiate_from_config
    
    model = instantiate_from_config(config.model)
    print(f"Loading weights from: {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location="cpu")
    
    # 兼容 PyTorch Lightning 的 state_dict (通常保存在 'state_dict' 键下)
    if "state_dict" in state_dict:
        sd = state_dict["state_dict"]
        # 去掉 'model.' 前缀 (如果存在)
        sd = {k.replace("model.", ""): v for k, v in sd.items()}
    else:
        sd = state_dict
        
    model.load_state_dict(sd, strict=False)
    model.eval()
    return model

def process_image(img_path, model, device):
    # 1. 加载图像并进行预处理
    img = Image.open(img_path).convert("RGB")
    
    # 假设你的模型输入需要归一化到 [-1, 1] 之间
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]) 
    ])
    
    img_tensor = transform(img).unsqueeze(0).to(device) # Shape: (1, C, H, W)
    
    # 2. 构建 batch (模拟 Dataloader 的输出格式)
    # 这一步非常关键，必须和你 yaml 里的 input_key 和 mean_key 对应
    # 假设模型推理只需要 cond_image (带云图) 
    batch = {
        "cond_image": img_tensor,
        # 如果推理时还需要其它输入（比如空的 mask），在这里添加
    }
    
    # 3. 前向传播 (采样)
    with torch.no_grad():
        with torch.autocast(device_type='cuda', dtype=torch.float16): # 使用混合精度加速
            # 调用 diffusion 模型的采样函数 (通常是 _run_reverse_diffusion_sampling 或类似方法)
            # 你可能需要检查 sgm/models/diffusion.py 中 test_step 是如何调用采样的
            samples, _ = model._run_reverse_diffusion_sampling(batch)
            
    # 4. 后处理 (从 [-1, 1] 还原到 [0, 255])
    # 假设 samples shape 是 (1, C, H, W)
    result_tensor = samples[0].cpu().float()
    result_tensor = (result_tensor + 1.0) / 2.0  # 映射到 [0, 1]
    result_tensor = result_tensor.clamp(0, 1)
    
    result_img = T.ToPILImage()(result_tensor)
    return result_img

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = load_model_from_config(CONFIG_PATH, CKPT_PATH)
    model = model.to(device)
    
    image_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"Found {len(image_files)} images to process in {INPUT_DIR}.")
    
    for filename in tqdm(image_files, desc="Cleaning clouds"):
        input_path = os.path.join(INPUT_DIR, filename)
        output_path = os.path.join(OUTPUT_DIR, filename)
        
        # 如果已经生成过，则跳过 (方便断点续传)
        if os.path.exists(output_path):
            continue
            
        try:
            clean_img = process_image(input_path, model, device)
            clean_img.save(output_path)
        except Exception as e:
            print(f"\nError processing {filename}: {e}")

if __name__ == "__main__":
    main()