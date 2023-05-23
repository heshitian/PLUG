'''
Author: error: git config user.name && git config user.email & please set dead value or install git
Date: 2022-05-06 09:04:16
LastEditors: error: git config user.name && git config user.email & please set dead value or install git
LastEditTime: 2022-06-07 14:39:15
FilePath: /mmdetection-2.22.0/configs/_base_/datasets/dota_detection.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
# dataset settings
dataset_type = 'CocoCPDataset'
data_root = '/media/h/H/DOTA10_512_128/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(512, 512), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_pseudo_bboxes']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=0,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/DOTA_train_512_coarse.json',
        img_prefix=data_root + 'train/images/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/DOTA_train_512_coarse.json',
        img_prefix=data_root + 'train/images/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/DOTA_train_512_coarse.json',
        img_prefix=data_root + 'train/images/',
        pipeline=test_pipeline))
evaluation = dict(metric=['bbox'])
