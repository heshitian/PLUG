import copy
import os
from argparse import ArgumentParser
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json

def main():
    ann_file ='/media/h/M/dataset/AITOD/annotations/aitod_train_v1.json'
    res_dir ='/media/h/M/dataset/AITOD/annotations/single/'

    data = open(ann_file, "r", encoding="utf-8")
    ann_dict = json.load(data)    # 字典
    data.close()
    dicts_list = []
    ann_image_ids = []
    for class_idx, cat in enumerate(ann_dict['categories']):
        dict =  {'annotations':[],'categories':[],'images':[]}
        dicts_list.append(dict)
        dicts_list[class_idx]['categories'].append(cat)
        dicts_list[class_idx]['categories'][0]['id']=1
        ann_image_ids.append([])
    for idx, ann in enumerate(ann_dict['annotations']):
        dicts_list[ann['category_id']-1]['annotations'].append(ann)
        ann_image_ids[ann['category_id']-1].append( ann['image_id'])
    for idx in range(8):
        ann_image_ids[idx] = np.unique(ann_image_ids[idx] )
        ann_image_ids[idx] .sort()   
        for i in ann_image_ids[idx]:
            for image in ann_dict['images']:
                if image['id']==i:
                    dicts_list[idx]['images'].append(image)    
        res_file = res_dir + 'aitod_train_{}.json'.format(ann_dict['categories'][idx]['name'])
        with open (res_file,'w') as f:
            json.dump(dicts_list[idx] ,f)


if __name__ == "__main__":
    main()