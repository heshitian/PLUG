debug = False
norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)  # add

dataset_type = 'CocoCPDataset'
data_root = '/media/h/H/DOTA10_512_128/'
# data_root = '/media/h/H/ISAID_512_128/'
img_norm_cfg = dict( mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(512, 512), keep_ratio=True),
    # dict(type='Resize', scale_factor=[1.0], keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5) if not debug else dict(type='RandomFlip', flip_ratio=0.),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_bboxes_ignore', 'gt_pseudo_bboxes']),  # gt_true_bboxes use for debug
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
            dict(type='RandomFlip', flip_ratio = 0.00000000000000000000001),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),  # add
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_bboxes_ignore', 'gt_anns_id', 'gt_pseudo_bboxes']),  # gt_true_bboxes use for debug
        ])
]
data = dict(
    samples_per_gpu=8, workers_per_gpu=0,
    train=dict(
        # min_gt_size=2,  # add
        type=dataset_type,
        ann_file=data_root + '/annotations/DOTA_train_512_add3.json',
        # ann_file=data_root + '/annotations/DOTA_train_512_center.json',
        img_prefix=data_root + '/train/images',
        pipeline=train_pipeline
    ),
    val=dict(
        # min_gt_size=2,  # add
        type=dataset_type,
        ann_file=data_root + '/annotations/DOTA_train_512_coarse.json',
        img_prefix=data_root + '/train/images',
        pipeline=test_pipeline,
        # test_mode=False  # modified
    ),
    test=dict(
        samples_per_gpu=1,
        type=dataset_type,
        ann_file=data_root + '/annotations/DOTA_train_512_coarse.json',
        # ann_file=data_root + '/annotations/DOTA_train_512_center.json',
        img_prefix=data_root + '/train/images',
        pipeline=test_pipeline)
)

num_stages = 1
# r=[7, 5] #28
r = [0]
model=dict(
    type='BasicLocator',
    pretrained='torchvision://resnet50',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch'),
    # neck=dict(
    #     type='ChannelMapper',
    #     in_channels=[256, 512, 1024, 2048],
    #     kernel_size=1,
    #     out_channels=256,
    #     act_cfg=None,
    #     norm_cfg=dict(type='GN', num_groups=32),
    #     num_outs=4),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,  # 1
        add_extra_convs='on_input',
        num_outs=1,  # 5
        # conv_cfg=dict(type='DCNv2'),
        norm_cfg=norm_cfg,  # add
    ),
    bbox_head=dict(
        type='CascadeFPNCPRHeadTRM',
        support_flag = True,
        bn_flag = True,
        norm_cfg=norm_cfg,
        num_classes=15,
        num_query = 15,
        embed_dims=256,
        num_stages = num_stages,
        in_channels=256,
        feat_channels=256,
        stacked_convs=0,
        num_cls_fcs=0,
        strides= [8], #[4,8,16,32],  #  [8, 16, 32, 64, 128],  # [4, 8, 16, 32, 64] # [8, 16, 32, 64, 128]
        radius=r,
        loss_mil=dict(
            type='MILLoss',
            binary_ins=False,
            loss_weight=0.25),  # weight
        loss_type=0,
        loss_cfg=dict(
            with_neg_loss=True,
            neg_loss_weight=0.75,
            refine_bag_policy='only_refine_bag',
            random_remove_rate=0.4,
            with_gt_loss=True,
            gt_loss_weight=0.125,
            with_gt_low_loss = False,
            with_color_loss = True,
            with_mil_loss=False,
            with_pseudo_gt_loss = False,
            pseudo_gt_loss_weight = 0.25,
        ),
        normal_cfg=dict(
            prob_cls_type='sigmoid',
            out_bg_cls=False,
        ),
        # train_pts_extractor=dict(
        #     pos_generator=dict(type='CirclePtFeatGenerator', radius=r),
        #     neg_generator=dict(type='OutCirclePtFeatGenerator', radius=r, class_wise=True),
        # ),
        # refine_pts_extractor=dict(
        #     pos_generator=dict(type='CirclePtFeatGenerator', radius=r),
        #     neg_generator=dict(type='OutCirclePtFeatGenerator', radius=r, keep_wh=True, class_wise=True),
        # ),
        # point_refiner=dict(
        #     merge_th=0.1,
        #     refine_th=0.1,
        #     classify_filter=True,
        #     nearest_filter=True,
        # ),
    ),
    test_cfg=dict(
        nms_pre=2000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=1000)
)
 
find_unused_parameters = True
# optimizer
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)
# lr_config = dict( policy='step', warmup='linear', warmup_iters=500, warmup_ratio=0.001, step=[100])

optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict( policy='step', warmup='linear', warmup_iters=500, warmup_ratio=0.001, step=[8, 11])
work_dir = '/media/h/M/P2B/1dota/fuse_r0_res18/'
# runner = dict(type='EpochBasedRunner', max_epochs=12)

log_config = dict( interval=50, hooks=[ dict(type='TextLoggerHook'), dict(type='TensorboardLoggerHook') ])
custom_hooks = [dict(type='NumClassCheckHook'), dict(type = 'SetEpochInfoHook'), dict(type = 'SetIterInfoHook')]

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
# load_from ='/media/h/M/P2B/0isaid/fuse3temp/epoch_5.pth'
# resume_from = None
resume_from = None #'/media/h/M/P2B/1dota/fuse_r0_onlygt/epoch_6.pth'
workflow = [('train', 1)]
# disable opencv multithreading to avoid system being overloaded
opencv_num_threads = 0
# set multi-process start method as `fork` to speed up the training
mp_start_method = 'fork'

runner = dict(type='EpochBasedRunner', max_epochs=12)
# checkpoint_config = dict(interval=100, by_epoch = False)
checkpoint_config = dict(interval=1)
evaluation = dict(interval = 100, metric=['bbox'])
# Runner type
# runner = dict(type='IterBasedRunner', max_iters=90000)
# checkpoint_config = dict(interval=10000)
# evaluation = dict(interval=10000, metric='bbox')
