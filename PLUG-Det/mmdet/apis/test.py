# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import pickle
import shutil
import tempfile
import time

import mmcv
import torch
import torch.distributed as dist
from mmcv.image import tensor2imgs
from mmcv.runner import get_dist_info
import torch.nn.functional as F
from mmdet.core import encode_mask_results
import numpy as np
import matplotlib.pyplot as plt
import os
            
def single_gpu_test(model, 
                    data_loader,
                    show=False,
                    out_dir=None,
                    show_score_thr=0.3):
    model.eval()
    results = []
    dataset = data_loader.dataset
    PALETTE = getattr(dataset, 'PALETTE', None)
    prog_bar = mmcv.ProgressBar(len(dataset))
    # timess = []
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
            # timess.append(np.append(result[0][1],result[1]))
            # result = result[0][0]
        batch_size = len(result)
        if show or out_dir:
            if batch_size == 1 and isinstance(data['img'][0], torch.Tensor):
                img_tensor = data['img'][0]
            else:
                img_tensor = data['img'][0].data[0]
            img_metas = data['img_metas'][0].data[0]
            imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
            assert len(imgs) == len(img_metas)

            for i, (img, img_meta) in enumerate(zip(imgs, img_metas)):
                img_meta['ori_filename'] = img_meta['filename'].split('/')[-1] #hhh
                h, w, _ = img_meta['img_shape']
                img_show = img[:h, :w, :]

                ori_h, ori_w = img_meta['ori_shape'][:-1]
                img_show = mmcv.imresize(img_show, (ori_w, ori_h))

                if out_dir:
                    out_file = osp.join(out_dir, img_meta['ori_filename'])
                else:
                    out_file = None

                model.module.show_result(
                    img_show,
                    result[i],
                    bbox_color=PALETTE,
                    text_color=PALETTE,
                    mask_color=PALETTE,
                    show=show,
                    out_file=out_file,
                    score_thr=show_score_thr)

        # encode mask results
        if isinstance(result[0], tuple):
            result = [(bbox_results, encode_mask_results(mask_results))
                      for bbox_results, mask_results in result]
        results.extend(result)

        for _ in range(batch_size):
            prog_bar.update()
    return results

def single_gpu_test_twomodel(model, model_b, 
                    data_loader,
                    show=False,
                    out_dir=None,
                    show_score_thr=0.3):
    model.eval()
    model_b.eval()
    results = []
    dataset = data_loader.dataset
    PALETTE = getattr(dataset, 'PALETTE', None)
    prog_bar = mmcv.ProgressBar(len(dataset))
    # timess = []
    for i, data in enumerate(data_loader):
        data_b = data.copy()
        with torch.no_grad():
            result_a = model(return_loss=False, rescale=True, two_model = True,  **data)
            result_a = result_a[1].permute(0,3,2,1)
            result_a = F.interpolate(result_a, data['img_metas'][0]._data[0][0]['pad_shape'][:2], mode='bilinear', align_corners=False)
            result_b = model_b(return_loss=False, rescale=True, two_model = True, **data)
            result_b = result_b[1].permute(0,3,2,1)
            result_b = F.interpolate(result_b, data['img_metas'][0]._data[0][0]['pad_shape'][:2], mode='bilinear', align_corners=False)
            bs = result_a.shape[0]
            gt_mask = data['gt_masks'][0].data[0][0].masks
            gt_mask = torch.tensor(gt_mask)
            # gt_mask = F.interpolate(gt_mask[None].float(), result_a[1].shape[1:3], mode='bilinear', align_corners=False)[0]
            # plt.figure()
            # plt.imshow(gt_mask[0])
            # plt.show()
            gt_mask_ind = torch.nonzero(gt_mask)
            gt_labels =  data['gt_labels'][0].data[0][0]
            gt_distribution = torch.zeros_like( result_a, device = 'cpu')
            gt_distribution[0,  gt_labels[gt_mask_ind[:,0]], gt_mask_ind[:,1], gt_mask_ind[:,2]] = 1.0
            gt_distribution = gt_distribution.permute(0,2,3,1)
            probs_a, probs_b = result_a.permute(0, 2, 3, 1).cpu(), result_b.permute(0, 2, 3, 1).cpu()
            save_dir = '/media/h/M/P2B/1dota/fuse_r0/num/'
            save_path0 = os.path.join(save_dir, data['img_metas'][0].data[0][0]['filename'].split('/')[-2], 'dist')
            os.mkdir(save_path0)  if not os.path.isdir(save_path0) else None
            
            save_path = os.path.join(save_path0, data['img_metas'][0].data[0][0]['ori_filename'][:-4]+'_fuse.png')
            fig, ax = plt.subplots()
            ax.imshow(((gt_distribution - probs_a)**2).mean(-1).sqrt()[0].cpu())
            ax.set_axis_off()
            fig.savefig(save_path, dpi = 300, bbox_inches = 'tight', pad_inches = 0)
            
            save_path = os.path.join(save_path0, data['img_metas'][0].data[0][0]['ori_filename'][:-4]+'_nofuse.png')
            fig, ax = plt.subplots()
            ax.imshow(((gt_distribution - probs_b)**2).mean(-1).sqrt()[0].cpu())
            ax.set_axis_off()
            fig.savefig(save_path, dpi = 300, bbox_inches = 'tight', pad_inches = 0)        
            

            # save_path = os.path.join(save_path, img_metas[bs_ind]['ori_filename'].replace('.png', '_'+ str(lvl_stride) +'.png'))
            # feature_visual = cls_prob_out.permute(0, 2, 1, 3)
            # num_grid = 5
            # feaU = []
            # for u in range(3):                
            #     feaV = []
            #     for v in range(num_grid):                    
            #         k = u*num_grid + v
            #         temp =feature_visual[bs_ind, :, :, k]   # H x W
            #         # feaV.append(torch.abs(temp))
            #         feaV.append(temp)
            #     feaV = torch.cat(feaV, dim=1)
            #     feaU.append(feaV)
            # feaU = torch.cat(feaU, dim=0)
            # plt.imshow(feaU.data.cpu())
            # plt.savefig(save_path, format='png')
            # timess.append(np.append(result[0][1],result[1]))
            # result = result[0][0]
        result = result_a[0]
        batch_size = len(result)
        if show or out_dir:
            if batch_size == 1 and isinstance(data['img'][0], torch.Tensor):
                img_tensor = data['img'][0]
            else:
                img_tensor = data['img'][0].data[0]
            img_metas = data['img_metas'][0].data[0]
            imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
            assert len(imgs) == len(img_metas)

            for i, (img, img_meta) in enumerate(zip(imgs, img_metas)):
                img_meta['ori_filename'] = img_meta['filename'].split('/')[-1] #hhh
                h, w, _ = img_meta['img_shape']
                img_show = img[:h, :w, :]

                ori_h, ori_w = img_meta['ori_shape'][:-1]
                img_show = mmcv.imresize(img_show, (ori_w, ori_h))

                if out_dir:
                    out_file = osp.join(out_dir, img_meta['ori_filename'])
                else:
                    out_file = None

                model.module.show_result(
                    img_show,
                    result[i],
                    bbox_color=PALETTE,
                    text_color=PALETTE,
                    mask_color=PALETTE,
                    show=show,
                    out_file=out_file,
                    score_thr=show_score_thr)

        # encode mask results
        if isinstance(result[0], tuple):
            result = [(bbox_results, encode_mask_results(mask_results))
                      for bbox_results, mask_results in result]
        results.extend(result)

        for _ in range(batch_size):
            prog_bar.update()
    return results

def multi_gpu_test(model, data_loader, tmpdir=None, gpu_collect=False):
    """Test model with multiple gpus.

    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.

    Returns:
        list: The prediction results.
    """
    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    time.sleep(2)  # This line can prevent deadlock problem in some cases.
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
            # encode mask results
            if isinstance(result[0], tuple):
                result = [(bbox_results, encode_mask_results(mask_results))
                          for bbox_results, mask_results in result]
        results.extend(result)

        if rank == 0:
            batch_size = len(result)
            for _ in range(batch_size * world_size):
                prog_bar.update()

    # collect results from all ranks
    if gpu_collect:
        results = collect_results_gpu(results, len(dataset))
    else:
        results = collect_results_cpu(results, len(dataset), tmpdir)
    return results


def collect_results_cpu(result_part, size, tmpdir=None):
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            mmcv.mkdir_or_exist('.dist_test')
            tmpdir = tempfile.mkdtemp(dir='.dist_test')
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, f'part_{rank}.pkl'))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, f'part_{i}.pkl')
            part_list.append(mmcv.load(part_file))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results


def collect_results_gpu(result_part, size):
    rank, world_size = get_dist_info()
    # dump result part to tensor with pickle
    part_tensor = torch.tensor(
        bytearray(pickle.dumps(result_part)), dtype=torch.uint8, device='cuda')
    # gather all result part tensor shape
    shape_tensor = torch.tensor(part_tensor.shape, device='cuda')
    shape_list = [shape_tensor.clone() for _ in range(world_size)]
    dist.all_gather(shape_list, shape_tensor)
    # padding result part tensor to max length
    shape_max = torch.tensor(shape_list).max()
    part_send = torch.zeros(shape_max, dtype=torch.uint8, device='cuda')
    part_send[:shape_tensor[0]] = part_tensor
    part_recv_list = [
        part_tensor.new_zeros(shape_max) for _ in range(world_size)
    ]
    # gather all result part
    dist.all_gather(part_recv_list, part_send)

    if rank == 0:
        part_list = []
        for recv, shape in zip(part_recv_list, shape_list):
            part_list.append(
                pickle.loads(recv[:shape[0]].cpu().numpy().tobytes()))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        return ordered_results
