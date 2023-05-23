import json
import os
import xml.dom.minidom
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import random
import os
import matplotlib.pyplot as plt
import numpy as np

def main():
    json_file = '/media/h/H/ISAID_512_128/annotations/ISAID_train_512.json'
    fr = open(json_file)
    data = json.load(fr)
    new_data = data.copy()
    fake_wh = 8
    for data_image in new_data['images']:
        data_image['file_name'] = data_image['file_name'].replace('instance_id_RGB_','')
    for data_ann in new_data['annotations']:
        # 1 center point
        # center_temp = np.array(data_ann['bbox']).reshape(-1,2)
        # center_temp = (center_temp[0]+center_temp[1]/2).astype(int)
        # mask_points =  np.array(data_ann['segmentation'][0]).reshape(-1,2)
        # flag_center = (center_temp==mask_points)[:,0]&(center_temp==mask_points)[:,1]
        # if flag_center.sum() > 0:
        #     center_point = center_temp
        # else:
        #     center_index = ((mask_points-center_temp)**2).sum(-1).argmin()
        #     center_point = mask_points[center_index]
        #  data_ann['point'] = center_point.tolist()
        # 2 coarse point
        data_ann['point'] = random.sample(list(np.array(data_ann['segmentation'][0]).reshape(-1,2)),1)[0].tolist()        
        
        data_ann['pseudo_bbox'] = [data_ann['point'][0]-fake_wh/2, data_ann['point'][1]-fake_wh/2,fake_wh , fake_wh]
    out_file = open('/media/h/H/ISAID_512_128/annotations/train_512_coarse.json', "w") 
    json.dump(new_data,out_file)
    
if __name__ == '__main__':
    main()
