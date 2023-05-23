import os
import mmcv
import json
import numpy as np
meta_dir = '/media/h/M/P2B/1dota/fuse_r0/seg/1.5/'
bbox_json_file = os.path.join(meta_dir, 'results.bbox.json')
ref_json_file = '/media/h/H/DOTA10_512_128/annotations/DOTA_train_512_coarse_seg.json'
ref_json = mmcv.load(ref_json_file)
bbox_all = mmcv.load(bbox_json_file)
IoU_list,IoU_list_s, IoU_list_m, IoU_list_l  = [],[],[],[]
area_range_list = [0 ** 2, 32 ** 2, 96 ** 2, 1e5 ** 2]
for bbox_info, ref_info in zip(bbox_all, ref_json['annotations']):
        IoU_list.append(bbox_info['score'])
        area = ref_info['area']
        if area > area_range_list[0] and area<= area_range_list[1]:
                IoU_list_s.append(bbox_info['score'])
        if area > area_range_list[1] and area<= area_range_list[2]:
                IoU_list_m.append(bbox_info['score'])
        if area > area_range_list[2] and area<= area_range_list[3]:
                IoU_list_l.append(bbox_info['score'])
mIoU = np.array(IoU_list).mean()
mIoU_s = np.array(IoU_list_s).mean()
mIoU_m = np.array(IoU_list_m).mean()
mIoU_l = np.array(IoU_list_l).mean()
print(mIoU, mIoU_s, mIoU_m, mIoU_l)