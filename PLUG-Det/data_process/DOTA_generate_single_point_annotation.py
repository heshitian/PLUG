import json
import numpy as np
from pycocotools.coco import COCO
from tqdm import tqdm
import numpy as np

def main():
    dota_json_file = '/media/h/H/DOTA10_512_128/annotations/DOTA_train_512.json'
    isaid_json_file = '/media/h/H/ISAID_512_128/annotations/iSAID_train_512.json'
    dump_file = '/media/h/H/DOTA10_512_128/annotations/DOTA_train_512_coarse_seg.json'
    dota_json_info = json.load(open(dota_json_file,'r'))
    new_annotations = []
    isaid_coco = COCO(isaid_json_file)
    dota_coco = COCO(dota_json_file)
    id_temp = 0
    for img_id in tqdm(dota_coco.getImgIds()):
        dota_anns = dota_coco.imgToAnns[img_id]
        file_name = dota_coco.imgs[img_id]['file_name']
        for img_id_ in isaid_coco.imgs:
            if isaid_coco.imgs[img_id_]['file_name']==file_name:
                isaid_img_id = isaid_coco.imgs[img_id_]['id']
        isaid_anns = isaid_coco.imgToAnns[isaid_img_id]
        assert file_name == isaid_coco.imgs[isaid_img_id]['file_name']
        for dota_ann in dota_anns:
            dota_box = dota_ann['bbox']
            isaid_bboxes = [isaid_anns[i]['bbox'] for i in range(len(isaid_anns))]
            ious = []
            if len(isaid_bboxes)>0:
                for isaid_box in isaid_bboxes:
                    iou = calculate_iou(dota_box, isaid_box)
                    ious.append(iou)
            else:
                ious = [0]
            if max(ious) >= 0.1:
                fuse_index = np.argmax(ious)
                isaid_ann = isaid_anns[fuse_index]
                #dota_ann['point'] = isaid_ann['point']
                #dota_ann['pseudo_box'] = isaid_ann['pseudo_box']
                dota_ann['point'].update('point':isaid_ann['point'])
                dota_ann['pseudo_box'].update('pseudo_box':isaid_ann['pseudo_box'])
                dota_ann['bbox'] = isaid_ann['bbox']
                dota_ann['area'] = isaid_ann['area']
                dota_ann['segmentation'] = isaid_ann['segmentation']
                id_temp+=1
                dota_ann['id'] = id_temp
                new_annotations.append(dota_ann)
    dota_json_info['annotations'] = new_annotations
    json.dump(dota_json_info,  open(dump_file, 'w'), indent=4)
def calculate_iou(box1, box2):
    box1_area = (box1[-2]+1)*(box1[-1]+1)
    box2_area = (box2[-2]+1)*(box2[-1]+1)
    intersection_area = get_intersection_area(box1, box2)
    union_area = box1_area + box2_area - intersection_area
    iou = float(intersection_area) / float(union_area)
    return iou
def get_intersection_area(box1, box2):
    """
    Calculates the intersection area of two bounding boxes where (x1,y1) indicates the top left corner and (x2,y2)
    indicates the bottom right corner
    :param box1: List of coordinates(x1,y1,x2,y2) of box1
    :param box2: List of coordinates(x1,y1,x2,y2) of box2
    :return: float: area of intersection of the two boxes
    """
    x1 = max(box1[0], box2[0])
    x2 = min(box1[0] + box1[2], box2[0] + box2[2])
    y1 = max(box1[1], box2[1])
    y2 = min(box1[1] + box1[3], box2[1] + box2[3])
    # Check for the condition if there is no overlap between the bounding boxes (either height or width
    # of intersection box are negative)
    if (x2 - x1 < 0) or (y2 - y1 < 0):
        return 0.0
    else:
        return (x2 - x1 + 1) * (y2 - y1 + 1)
if __name__ == '__main__':
    main()
