_base_ = './base.py'
# model settings
dataset_type = 'CocoCPDataset'
data_root = '/media/h/H/DOTA10_512_128/'
img_norm_cfg = dict( mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadWeakAnnotations'),
    dict(type='LoadProposals'),
    dict(type='Resize', img_scale=(512, 512)),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_labels', 'proposals']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadProposals'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
        #img_scale=[(500, 2000), (600, 2000), (700, 2000), (800, 2000), (900, 2000)],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img', 'proposals']),
        ])
]
data = dict(
    samples_per_gpu=8, workers_per_gpu=0,
    train=dict(
        # min_gt_size=2,  # add
        type=dataset_type,
        ann_file=data_root + '/annotations/DOTA_train_512_coarse.json',
        # ann_file=data_root + '/annotations/DOTA_train_512_center.json',
        img_prefix=data_root + '/train/images',
        proposal_file= data_root + '/train/SSW/selective_search.pkl',
        pipeline=train_pipeline
    ),
    val=dict(
        # min_gt_size=2,  # add
        type=dataset_type,
        ann_file=data_root + '/annotations/DOTA_train_512_coarse.json',
        img_prefix=data_root + '/train/images',
        proposal_file= data_root + '/train/SSW/selective_search.pkl',
        pipeline=test_pipeline,
        # test_mode=False  # modified
    ),
    test=dict(
        samples_per_gpu=1,
        type=dataset_type,
        ann_file=data_root + '/annotations/DOTA_train_512_coarse.json',
        # ann_file=data_root + '/annotations/DOTA_train_512_center.json',
        img_prefix=data_root + '/train/images',
        pipeline=test_pipeline
        )
)
model = dict(
    type='WeakRCNN',
    pretrained='/home/h/checkpoints/vgg16_caffe-292e1171.pth',
    backbone=dict(type='VGG16'),
    # backbone=dict(type='VGG16', init_cfg=dict(type='Pretrained', checkpoint='/home/h/checkpoints/vgg16_caffe-292e1171.pth')),
    neck=None,
    roi_head=dict(
        type='WSDDNRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIPool', output_size=7),
            out_channels=512,
            featmap_strides=[8]),
        bbox_head=dict(
            type='WSDDNHead',
            in_channels=512,
            hidden_channels=4096,
            roi_feat_size=7,
            num_classes=15)),
)
work_dir = '/media/h/M/P2B/1dota/WSOD/wsddn_vgg16temp/'
