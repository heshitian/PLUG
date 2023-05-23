_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/dota_detection.py',
    '../_base_/schedules/schedule_1x.py', 
    '../_base_/default_runtime.py'
]
checkpoint_config = dict(interval=12)
evaluation = dict(interval=12, metric='bbox',classwise = True)
work_dir = '/media/h/M/P2B/1dota/P2B_work_dirs/work_dirs/P2B_DOTA_1024_0.0005_stage2_basescales0_bs2/faster-rcnn'
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11])
runner = dict(type='EpochBasedRunner', max_epochs=12)