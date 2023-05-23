import os
import mmcv
import json
from pycocotools.coco import COCO
import glob
from tqdm import tqdm
import shutil
import numpy as np
images_dir = '/media/h/H/ISAID_512_128/train/images/'
img_path_list = glob.glob(images_dir +'/*.png')
ref_json_file = '/media/h/H/ISAID_512_128/annotations/DOTA_train_512_coarse.json'
out_dir = '/media/h/H/ISAID_512_128/train/images_num/'
out_json_dir = os.path.join(out_dir,'annotations')
if not os.path.isdir(out_json_dir): os.mkdir(out_json_dir) 
ref_json = mmcv.load(ref_json_file)
ref_coco = COCO(ref_json_file)
ref_coco.getImgIds()
img_ids = ref_coco.getImgIds()
ins_nums = []
new_images = dict()
new_categories = ref_json['categories'].copy()
new_annotations = dict()
for imgId in tqdm(img_ids):
        img = ref_coco.loadImgs(imgId)[0]
        img_name = img['file_name']
        ori_img_path = os.path.join(images_dir, img_name)
        ins_num = len(ref_coco.imgToAnns[imgId])
        save_dir = os.path.join(out_dir, str(ins_num))
        if not os.path.isdir(save_dir): os.mkdir(save_dir) 
        save_img_path = os.path.join(save_dir, img_name)
        shutil.copyfile(ori_img_path, save_img_path)
        ins_nums.append(ins_num)
        if not ins_num in new_images.keys(): new_images.update({ins_num: list()})
        img_modify = img.copy()
        img_modify['id']=len(new_images[ins_num])+1
        new_images[ins_num].append(img_modify)
        if not ins_num in new_annotations.keys(): new_annotations.update({ins_num: list()})
        for ins_id in range(ins_num):
                ann_modify = ref_coco.imgToAnns[imgId][ins_id]
                ann_modify['id'] = len(new_annotations[ins_num])+1
                ann_modify['image_id'] = img_modify['id']
                new_annotations[ins_num].append(ann_modify)
ins_nums = np.unique(ins_nums)
for ins_num in ins_nums:
        ins_json = dict()
        ins_json['images'] = new_images[ins_num]
        ins_json['categories'] = new_categories
        ins_json['annotations'] = new_annotations[ins_num]
        out_json_path = os.path.join(out_json_dir, 'train_'+str(ins_num)+'.json')
        mmcv.dump(ins_json, out_json_path)