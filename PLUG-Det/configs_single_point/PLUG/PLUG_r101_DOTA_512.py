PLUG_test_index = True
norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)  

dataset_type = 'CocoCPDataset'
data_root = '/media/h/H/DOTA10_512_128/'

img_norm_cfg = dict( mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(512, 512), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_bboxes_ignore', 'gt_pseudo_bboxes']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),  # add
    dict(
        type='MultiScaleFlipAug',
        scale_factor=[1.0],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip', flip_ratio = 0.5),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'), 
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_bboxes_ignore', 'gt_anns_id', 'gt_pseudo_bboxes']),
        ])
]
data = dict(
    samples_per_gpu=8, workers_per_gpu=0,
    train=dict(
        type=dataset_type,
        ann_file=data_root + '/annotations/DOTA_train_512_coarse_seg.json',
        img_prefix=data_root + '/train/images',
        pipeline=train_pipeline
    ),
    val=dict(
        type=dataset_type,
        ann_file=data_root + '/annotations/DOTA_train_512_coarse_seg.json',
        img_prefix=data_root + '/train/images',
        pipeline=test_pipeline,
    ),
    test=dict(
        type=dataset_type,
        ann_file=data_root + '/annotations/DOTA_train_512_coarse_seg.json',
        img_prefix=data_root + '/train/images',
        pipeline=test_pipeline)
)
model=dict(
    type='PLUG',
    pretrained='torchvision://resnet101',
    backbone=dict(
        type='ResNet',
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1, 
        add_extra_convs='on_input',
        num_outs=1,
        norm_cfg=norm_cfg,  
    ),
    bbox_head=dict(
        type='PLUGHead',
        sfg_flag = True,
        num_classes=15,
        embed_dims=256,
        strides= [8], #[4,8,16,32],  #  [8, 16, 32, 64, 128],  # [4, 8, 16, 32, 64] # [8, 16, 32, 64, 128]
        loss_cfg=dict(
            with_neg_loss=True,
            neg_loss_weight=1,
            with_gt_loss=True,
            gt_loss_weight=1,
            with_color_loss = True,
            color_loss_weight = 1,
        ),
        train_cfg = None,
        pred_cfg=dict(
            pred_diff = True,
            boundary_diff = True,
            boundary_diff_weight = 0.5,
            bg_threshold = 0.5,
        ),
        )
)
 
find_unused_parameters = True
# optimizer
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict( policy='step', warmup='linear', warmup_iters=500, warmup_ratio=0.001, step=[8, 11])
work_dir = '/media/h/M/P2B/1dota/PLUG-res50-fuse/'

log_config = dict( interval=50, hooks=[ dict(type='TextLoggerHook'), dict(type='TensorboardLoggerHook') ])
custom_hooks = [dict(type = 'SetEpochInfoHook'), dict(type = 'SetIterInfoHook')]
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None 
workflow = [('train', 1)]
# disable opencv multithreading to avoid system being overloaded
opencv_num_threads = 0
# set multi-process start method as `fork` to speed up the training
mp_start_method = 'fork'
runner = dict(type='EpochBasedRunner', max_epochs=12)
checkpoint_config = dict(interval=1)
evaluation = dict(interval = 100, metric=['bbox'])