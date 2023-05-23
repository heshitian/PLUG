# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import time
import warnings
from natsort import natsorted
import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
import gc
from mmdet.apis import multi_gpu_test, single_gpu_test
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.models import build_detector
from mmdet.utils import setup_multi_processes

out_meta0 = '/media/h/M/P2B/1dota/fuse_r0/'
out_meta = out_meta0 + '/num/'
def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('--config', default = \
        out_meta0 + '/cascade_CPR_r50_FPN_1x_DOTA_512_trm.py',
        # out_meta0 + '/faster_rcnn_r50_fpn_1x_dota.py',
        help='test config file path')
    parser.add_argument('--checkpoint',default=\
        out_meta0 + '/epoch_12.pth', 
        help='checkpoint file')
    parser.add_argument(
        '--work-dir',
        default=out_meta ,
        help='the directory to save the file containing evaluation metrics')
    parser.add_argument('--out', default = out_meta +'/result.pkl', help='output result file in pickle format')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='(Deprecated, please use --gpu-id) ids of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use '
        '(only applicable to non-distributed testing)')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        default=['bbox','segm'],
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument('--show', action='store_true', default = False, help='show results')
    parser.add_argument(
        '--show-dir', 
        default=out_meta +'/images',
        # default=None,
        help='directory where painted images will be saved')
    parser.add_argument(
        '--show-score-thr',
        type=float,
        default=0.0,
        help='score threshold (default: 0.3)')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--options',
        nargs='+',
        default={
        'jsonfile_prefix': out_meta + '/results',
        'iou_thrs':[0.5],
        'classwise':True,
        # 'iou_thrs':[0.5],
        },
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function (deprecate), '
        'change to --eval-options instead.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        # default={
        # 'iou_thrs':[0.5],
        # 'jsonfile_prefix': out_meta ,
        # 'proposal_nums':(20, 20, 20),
        # 'classwise':True},
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            '--options and --eval-options cannot be both '
            'specified, --options is deprecated in favor of --eval-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --eval-options')
        args.eval_options = args.options
    return args


def main():
    
    num_meta = '/media/h/H/DOTA10_512_128/train/images_num/'
    num_all = os.listdir(num_meta)
    num_all.pop(os.listdir(num_meta).index('annotations'))
    num_all.pop(os.listdir(num_meta).index('0'))
    num_all = natsorted(num_all)#[8:]
    # num_all = ['1', ]
    for num in num_all:
        args = parse_args()
        args.out =  args.out.replace('/result.pkl', str(num)+'/result.pkl')
        args.show_dir = args.show_dir.replace('/images', str(num)+'/images')
        args.work_dir = args.work_dir + str(num) + '/'
        if not os.path.isdir(args.work_dir): os.mkdir(args.work_dir) 
        args.options['jsonfile_prefix'] = args.options['jsonfile_prefix'].replace('/results', str(num)+'/results')
        assert args.out or args.eval or args.format_only or args.show \
            or args.show_dir, \
            ('Please specify at least one operation (save/eval/format/show the '
            'results / save the results) with the argument "--out", "--eval"'
            ', "--format-only", "--show" or "--show-dir"')

        if args.eval and args.format_only:
            raise ValueError('--eval and --format_only cannot be both specified')

        if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
            raise ValueError('The output file must be a pkl file.')

        cfg = Config.fromfile(args.config)
        cfg['data']['test']['img_prefix'] = cfg['data']['test']['img_prefix'].replace('/images/','/images_num/'+ str(num) +'/' )
        cfg['data']['test']['ann_file'] = cfg['data']['test']['ann_file'].\
            replace('/annotations/DOTA_train_512_coarse.json','/train/images_num/annotations/train_'+str(num)+'.json')
        cfg['work_dir'] = cfg['work_dir'] + 'num/'
        if args.cfg_options is not None:
            cfg.merge_from_dict(args.cfg_options)

        # set multi-process settings
        setup_multi_processes(cfg)

        # set cudnn_benchmark
        if cfg.get('cudnn_benchmark', False):
            torch.backends.cudnn.benchmark = True

        if 'pretrained' in cfg.model:
            cfg.model.pretrained = None
        elif 'init_cfg' in cfg.model.backbone:
            cfg.model.backbone.init_cfg = None

        if cfg.model.get('neck'):
            if isinstance(cfg.model.neck, list):
                for neck_cfg in cfg.model.neck:
                    if neck_cfg.get('rfp_backbone'):
                        if neck_cfg.rfp_backbone.get('pretrained'):
                            neck_cfg.rfp_backbone.pretrained = None
            elif cfg.model.neck.get('rfp_backbone'):
                if cfg.model.neck.rfp_backbone.get('pretrained'):
                    cfg.model.neck.rfp_backbone.pretrained = None

        # in case the test dataset is concatenated
        samples_per_gpu = 100
        if isinstance(cfg.data.test, dict):
            # cfg.data.test.test_mode = True # change by H
            samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1) # 1
            if samples_per_gpu > 1:
                # Replace 'ImageToTensor' to 'DefaultFormatBundle'
                cfg.data.test.pipeline = replace_ImageToTensor(
                    cfg.data.test.pipeline)
        elif isinstance(cfg.data.test, list):
            for ds_cfg in cfg.data.test:
                ds_cfg.test_mode = True
            samples_per_gpu = max(
                [ds_cfg.pop('samples_per_gpu', 1) for ds_cfg in cfg.data.test])
            if samples_per_gpu > 1:
                for ds_cfg in cfg.data.test:
                    ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

        if args.gpu_ids is not None:
            cfg.gpu_ids = args.gpu_ids[0:1]
            warnings.warn('`--gpu-ids` is deprecated, please use `--gpu-id`. '
                        'Because we only support single GPU mode in '
                        'non-distributed testing. Use the first GPU '
                        'in `gpu_ids` now.')
        else:
            cfg.gpu_ids = [args.gpu_id]

        # init distributed env first, since logger depends on the dist info.
        if args.launcher == 'none':
            distributed = False
        else:
            distributed = True
            init_dist(args.launcher, **cfg.dist_params)

        rank, _ = get_dist_info()
        # allows not to create
        if args.work_dir is not None and rank == 0:
            mmcv.mkdir_or_exist(osp.abspath(args.work_dir))
            timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
            json_file = osp.join(args.work_dir, f'eval_{timestamp}.json')

        # build the dataloader
        dataset = build_dataset(cfg.data.test)
        data_loader = build_dataloader(
            dataset,
            samples_per_gpu=samples_per_gpu,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=distributed,
            shuffle=False)

        # build the model and load checkpoint
        # cfg.model.train_cfg = None  # change by H
        model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
        fp16_cfg = cfg.get('fp16', None)
        if fp16_cfg is not None:
            wrap_fp16_model(model)
        checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
        if args.fuse_conv_bn:
            model = fuse_conv_bn(model)
        # old versions did not save class info in checkpoints, this walkaround is
        # for backward compatibility
        if 'CLASSES' in checkpoint.get('meta', {}):
            model.CLASSES = checkpoint['meta']['CLASSES']
        else:
            model.CLASSES = dataset.CLASSES

        if not distributed:
            model = MMDataParallel(model, device_ids=cfg.gpu_ids)
            outputs = single_gpu_test(model, data_loader, args.show, args.show_dir,
                                    args.show_score_thr)
        else:
            model = MMDistributedDataParallel(
                model.cuda(),
                device_ids=[torch.cuda.current_device()],
                broadcast_buffers=False)
            outputs = multi_gpu_test(model, data_loader, args.tmpdir,
                                    args.gpu_collect)

        rank, _ = get_dist_info()
        if rank == 0:
            if args.out:
                print(f'\nwriting results to {args.out}')
                mmcv.dump(outputs, args.out)
            kwargs = {} if args.eval_options is None else args.eval_options
            if args.format_only:
                dataset.format_results(outputs, **kwargs)
            if args.eval:
                eval_kwargs = cfg.get('evaluation', {}).copy()
                # hard-code way to remove EvalHook args
                for key in [
                        'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
                        'rule', 'dynamic_intervals'
                ]:
                    eval_kwargs.pop(key, None)
                eval_kwargs.update(dict(metric=args.eval, **kwargs))
                metric = dataset.evaluate(outputs, **eval_kwargs)
                print(metric)
                metric_dict = dict(config=args.config, metric=metric)
                if args.work_dir is not None and rank == 0:
                    mmcv.dump(metric_dict, json_file)
    gc.collect()


if __name__ == '__main__':
    main()
