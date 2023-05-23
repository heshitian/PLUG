import os
import mmcv
from tqdm import tqdm

meta_dir = '/media/h/M/P2B/1dota/PLUG-res50-fuse/'
bbox_json_file = os.path.join(meta_dir, 'results.segm.json')
ref_json_file = '/media/h/H/DOTA10_512_128/annotations/DOTA_train_512_coarse_seg.json'
result_json_file = os.path.join(meta_dir, 'DOTA_train_512_seg.json')
ref_json = mmcv.load(ref_json_file)
bbox_all = mmcv.load(bbox_json_file)
for id, bbox_info in tqdm(enumerate(bbox_all)):
        bbox_info['area'] =  bbox_info['bbox'][2]*bbox_info['bbox'][3]
        bbox_info['iscrowd'] = 0
        bbox_info['id'] = id+1
ref_json['annotations'] = bbox_all
mmcv.dump(ref_json, result_json_file)