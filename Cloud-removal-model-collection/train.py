import collections
import collections.abc
collections.Mapping = collections.abc.Mapping
collections.MutableMapping = collections.abc.MutableMapping
collections.Sequence = collections.abc.Sequence

import os
import shutil
import torch  # <--- 1. 这里导入 torch
from model.init import init
import utils.utils as utils
from utils.config import load_config

if __name__ == '__main__':
    config = load_config("config.yml")

    # --- 2. 在这里注入补丁，确保 config 拥有必须的 device 属性 ---
    config.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 也可以顺便补充可能缺失的其他属性，预防后续报错
    if not hasattr(config, 'gpu_ids'):
        config.gpu_ids = [0]

    utils.make_manager()
    os.makedirs(config.out_dir, exist_ok=True)

    # 保存本次训练时的配置
    shutil.copyfile('config.yml', os.path.join(config.out_dir, 'config.yml'))

    # 3. 开始训练
    init(config.model, config)