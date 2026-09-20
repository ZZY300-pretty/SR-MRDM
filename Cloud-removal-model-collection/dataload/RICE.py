import numpy as np
import os
from torch.utils import data
from utils.imgproc import imresize
import skimage.io as io
import blobfile as bf

class TrainDataset(data.Dataset):
    def __init__(self, config, isTrain=True):
        super().__init__()
        self.config = config
        base_dir = config.datasets_dir
        
        # 1. 明确定义路径前缀
        if isTrain:
            self.datasets_dir = os.path.join(base_dir, 'train')
        else:
            self.datasets_dir = os.path.join(base_dir, 'test')
            
        # 2. 明确定义 label_dir (防止未赋值)
        label_dir = os.path.join(self.datasets_dir, 'label')
        
        # 3. 路径校验
        if not os.path.exists(label_dir):
            raise FileNotFoundError(f"找不到路径: {label_dir}")
            
        # 4. 路径绝对化
        abs_label_dir = os.path.abspath(label_dir)
            
        # 5. 读取文件列表
        self.imlist = sorted([f for f in os.listdir(abs_label_dir) if f.endswith('.png')])

    def __getitem__(self, index):
        # 1. 明确构建文件路径
        filename = self.imlist[index]
        label_path = os.path.join(self.datasets_dir, 'label', filename)
        cloud_path = os.path.join(self.datasets_dir, 'cloudy', filename)
        
        # 2. 读取图片
        t = io.imread(label_path, as_gray=False)
        x = io.imread(cloud_path, as_gray=False)

        # 3. 维度对齐处理
        if t.ndim == 2: t = np.stack([t]*3, axis=-1)
        if x.ndim == 2: x = np.stack([x]*3, axis=-1)
        if t.shape[-1] == 4: t = t[..., :3]
        if x.shape[-1] == 4: x = x[..., :3]

        # 4. 统一尺寸
        t = imresize(t, 1/2)
        x = imresize(x, 1/2)

        # 5. 计算 M 并标准化
        M = np.clip((t - x).sum(axis=2), 0, 10).astype(np.float32)
        x = x / 255.0
        t = t / 255.0
        
        # 6. 转置为 PyTorch 需要的 (C, H, W)
        x = x.transpose(2, 0, 1)
        t = t.transpose(2, 0, 1)
        
        name = filename.split('.')[0]
        return x, t, M, name

    def __len__(self):
        return len(self.imlist)