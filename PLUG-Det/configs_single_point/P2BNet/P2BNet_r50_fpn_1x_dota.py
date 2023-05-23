_base_ = [
    # '../base/faster_rcnn_r50_fpn_1x_tinycoco.py',
    # '../../_base_/datasets/TinyCOCO/TinyCOCO_detection.py',
    # '../../../configs/_base_/schedules/schedule_1x.py',
    '../../configs/_base_/default_runtime.py'
]
norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)  # add
debug = False
# model settings

num_stages = 1
model = dict(
    type='P2BNet',
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
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=0,
        add_extra_convs='on_input',
        num_outs=4,  # 5
        norm_cfg=norm_cfg
    ),
    roi_head=dict(
        type='P2BHead',
        num_stages=num_stages,
        top_k=7,
        with_atten=False,
        # stage_loss_weights=[1] * num_stages,
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='Shared2FCInstanceMILHead',
            num_stages=num_stages,
            with_loss_pseudo=False,
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=15,
            num_ref_fcs=0,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=True,
            loss_type='MIL',
            loss_mil1=dict(
                type='MILLoss',
                binary_ins=False,
                loss_weight=0.25,
                loss_type='gfocal_loss'),  # weight binary_cross_entropy
            loss_mil2=dict(
                type='MILLoss',
                binary_ins=False,
                loss_weight=0.25,
                loss_type='gfocal_loss'),),# weight
            # loss_bbox=dict(type='SmoothL1Loss', loss_weight=1.0)),
    ),
    # model training and testing settings
    train_cfg=dict(
        base_proposal=dict(
            base_scales=[4, 8, 16, 32, 64, 128],
            base_ratios=[1 / 3, 1 / 2, 1 / 1.5, 1.0, 1.5, 2.0, 3.0],
            # base_scales=[4, 16, 32], #1
            # base_ratios=[1 / 2, 1.0, 2.0], #1
            shake_ratio=None,
            cut_mode='symmetry',  # 'clamp',
            gen_num_neg=0),
        fine_proposal=dict(
            gen_proposal_mode='fix_gen',
            cut_mode=None,
            shake_ratio=[0.1],
            base_ratios=[1, 1.2, 1.3, 0.8, 0.7],
            # gen_num_per_box=10,
            iou_thr=0.3,
            gen_num_neg=500,
        ),
        rcnn=None
    ),
    test_cfg=dict(
        rpn=None,
        rcnn=None,
    ))

# dataset settings
dataset_type = 'CocoCPDataset'
data_root = '/media/h/H/DOTA10_512_128/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    # dict(type='Resize', img_scale=[(2000, 480), (2000, 576), (2000, 688), (2000, 864), (2000, 1000), (2000, 1200)],
    #      multiscale_mode='value',
    #      keep_ratio=True),
    dict(type='Resize', img_scale=(512, 512), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5) if not debug else dict(type='RandomFlip', flip_ratio=0.),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_bboxes_ignore', 'gt_pseudo_bboxes']),
]

test_scale = 1200
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='MultiScaleFlipAug',
        # img_scale=(2000, test_scale) if test_scale else (1333, 800),
        scale_factor=[1.0],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect',
                 keys=['img', 'gt_bboxes', 'gt_labels', 'gt_bboxes_ignore', 'gt_anns_id', 'gt_pseudo_bboxes']),
        ])
]
data = dict(
    samples_per_gpu=8,  # 2
    workers_per_gpu=0,  # didi-debug 2
    shuffle=False if debug else None,
    train=dict(
        type=dataset_type,
        ann_file=data_root + '/annotations/DOTA_train_512_coarse.json',
        img_prefix=data_root + '/train/images',
        pipeline=train_pipeline,
    ),
    val=dict(
        # samples_per_gpu=2,
        type=dataset_type,
        ann_file=data_root +'/annotations/DOTA_train_512_coarse.json',
        img_prefix=data_root + '/train/images',
        pipeline=test_pipeline,
        test_mode=False,  # modified
    ),
    test=dict(
        # samples_per_gpu=2,
        type=dataset_type,
        ann_file=data_root +'/annotations/DOTA_train_512_coarse.json',
        img_prefix=data_root + '/train/images',
        pipeline=test_pipeline))

check = dict(stop_while_nan=False)  # add by hui

# optimizer
optimizer = dict(type='SGD', lr=0.0005, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11])
runner = dict(type='EpochBasedRunner', max_epochs=12)
work_dir =  '/media/h/M/P2B/1123P2B/P2B_DOTA_1024_0.0005_stage1_basescales0_bs2_shake'
evaluation = dict(interval = 12, metric=['bbox'])
# evaluation = dict(
#     interval=12, metric='bbox',
#     save_result_file=work_dir + '_' + str(test_scale) + '_latest_result.json',
#     do_first_eval=False,  # test
#     do_final_eval=True,
# )


