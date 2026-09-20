_base_ = ['./oriented_rcnn_r50_fpn_1x_dota_le90.py']

# 建议在 data_root 最后加一个 '/'，方便下面直接拼接相对路径
data_root = '/mnt/mydisk/zwj/zzy/EMRDM-main/datasets/archive_random_10pct_experiment_crops_dota/'

# 加载预训练权重用于微调
load_from = '/mnt/mydisk/zwj/zzy/EMRDM-main/checkpoints/oriented_rcnn_r50_fpn_1x_dota_le90-6d2b2ce0.pth'

# 利用 MMCV 的字典合并机制，只修改需要覆盖的字段（路径和 batch_size）
data = dict(
    samples_per_gpu=2,  # 显卡 batch_size
    workers_per_gpu=2,  # 数据加载线程数
    train=dict(
        ann_file=data_root + 'train/annfiles/',
        img_prefix=data_root + 'train/images/'),
    val=dict(
        ann_file=data_root + 'val/annfiles/',
        img_prefix=data_root + 'val/images/'),
    test=dict(
        # DOTA 测试集通常没有标注文件，MMRotate 默认读取 img_prefix 下的图片进行预测
        ann_file=data_root + 'test/annfiles/', 
        img_prefix='/mnt/mydisk/zwj/zzy/EMRDM-main/datasets/restored_target_cloud') #调整为清晰图像，带云图像，去云图像，得到三组mAP
)

# 验证频率与指标
evaluation = dict(interval=1, metric='mAP')

# 把原版的学习率（通常是 0.005）缩小 10 倍，用于微调
optimizer = dict(lr=0.0005)