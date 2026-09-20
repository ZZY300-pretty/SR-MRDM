_base_ = [
    'mmdet::faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py',
]

# Replace these with your own dataset paths and categories.
data_root = '/path/to/restored_test_dataset/'
classes = (
    'class_1',
    'class_2',
    'class_3',
)
num_classes = len(classes)
backend_args = None

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'),
    ),
]

model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=num_classes),
    ),
    # Keep the detector frozen at test time. The checkpoint you load should
    # already be trained on the same category set as "classes".
    train_cfg=None,
)

test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        metainfo=dict(classes=classes),
        ann_file='annotations/test.json',
        data_prefix=dict(img='images/'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args,
    ),
)

test_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/test.json',
    metric='bbox',
    classwise=True,
    backend_args=backend_args,
)

# Optional visualization directory for debugging.
default_hooks = dict(
    visualization=dict(type='DetVisualizationHook', draw=False),
)

work_dir = './work_dirs/faster_rcnn_r50_fpn_restored_eval'
