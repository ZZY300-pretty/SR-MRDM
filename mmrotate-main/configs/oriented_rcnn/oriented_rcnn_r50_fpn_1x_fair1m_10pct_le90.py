_base_ = ['./oriented_rcnn_r50_fpn_1x_dota_le90.py']

data_root = 'D:/EMRDM-main/datasets/fair1m_10pct_1024_dota/'
load_from = 'D:/EMRDM-main/checkpoints/oriented_rcnn_r50_fpn_1x_dota_le90-6d2b2ce0.pth'

classes = (
    'a220',
    'a321',
    'a330',
    'a350',
    'arj21',
    'baseball-field',
    'basketball-court',
    'boeing737',
    'boeing747',
    'boeing777',
    'boeing787',
    'bridge',
    'bus',
    'c919',
    'cargo-truck',
    'dry-cargo-ship',
    'dump-truck',
    'engineering-ship',
    'excavator',
    'fishing-boat',
    'football-field',
    'intersection',
    'liquid-cargo-ship',
    'motorboat',
    'passenger-ship',
    'roundabout',
    'small-car',
    'tennis-court',
    'tractor',
    'trailer',
    'truck-tractor',
    'tugboat',
    'van',
    'warship',
    'other-airplane',
    'other-ship',
    'other-vehicle',
)

model = dict(
    roi_head=dict(
        bbox_head=dict(
            num_classes=len(classes),
        )
    )
)

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        classes=classes,
        ann_file=data_root + 'train/annfiles/',
        img_prefix=data_root + 'train/images/',
    ),
    val=dict(
        classes=classes,
        ann_file=data_root + 'val/annfiles/',
        img_prefix=data_root + 'val/images/',
    ),
    test=dict(
        classes=classes,
        ann_file=data_root + 'test/annfiles/',
        img_prefix=data_root + 'test/images/',
    ),
)

evaluation = dict(interval=1, metric='mAP')
optimizer = dict(lr=0.0005)
