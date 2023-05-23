import cv2
import argparse
import xml.etree.ElementTree as ET
import os
import cv2
import numpy as np
from tqdm import tqdm
import glob
import mmcv
import json
# python selective_search.py ./JPEGImages/000480.jpg ./Annotations/000480.xml color

def get_intersection_area(box1, box2):
    """
    Calculates the intersection area of two bounding boxes where (x1,y1) indicates the top left corner and (x2,y2)
    indicates the bottom right corner
    :param box1: List of coordinates(x1,y1,x2,y2) of box1
    :param box2: List of coordinates(x1,y1,x2,y2) of box2
    :return: float: area of intersection of the two boxes
    """
    x1 = max(box1[0], box2[0])
    x2 = min(box1[2], box2[2])
    y1 = max(box1[1], box2[1])
    y2 = min(box1[3], box2[3])
    # Check for the condition if there is no overlap between the bounding boxes (either height or width
    # of intersection box are negative)
    if (x2 - x1 < 0) or (y2 - y1 < 0):
        return 0.0
    else:
        return (x2 - x1 + 1) * (y2 - y1 + 1)


def calculate_iou(proposal_boxes, gt_boxes):
    """
    Returns the bounding boxes that have Intersection over Union (IOU) > 0.5 with the ground truth boxes
    :param proposal_boxes: List of proposed bounding boxes(x1,y1,x2,y2) where (x1,y1) indicates the top left corner
    and (x2,y2) indicates the bottom right corner of the proposed bounding box
    :param gt_boxes: List of ground truth boxes(x1,y1,x2,y2) where (x1,y1) indicates the top left corner and (x2,y2)
    indicates the bottom right corner of the ground truth box
    :return iou_qualified_boxes: List of all proposed bounding boxes that have IOU > 0.5 with any of the ground
    truth boxes
    :return final_boxes: List of the best proposed bounding box with each of the ground truth box (if available)
    """
    iou_qualified_boxes = []
    final_boxes = []
    for gt_box in gt_boxes:
        best_box_iou = 0
        best_box = 0
        area_gt_box = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])
        for prop_box in proposal_boxes:
            area_prop_box = (prop_box[2] - prop_box[0] + 1) * (prop_box[3] - prop_box[1] + 1)
            intersection_area = get_intersection_area(prop_box, gt_box)
            union_area = area_prop_box + area_gt_box - intersection_area
            iou = float(intersection_area) / float(union_area)
            if iou > 0.5:
                iou_qualified_boxes.append(prop_box)
                if iou > best_box_iou:
                    best_box_iou = iou
                    best_box = prop_box
        if best_box_iou != 0:
            final_boxes.append(best_box)
    return iou_qualified_boxes, final_boxes


def get_groundtruth_boxes(annoted_img_path):
    """
    Parses the xml file of the annotated image to obtain the ground truth boxes
    :param annoted_img_path: String: File path of the annotated image containing the ground truth
    :return gt_boxes: List of ground truth boxes(x1,y1,x2,y2) where (x1,y1) indicates the top left corner and (x2,y2)
    indicates the bottom right corner of the ground truth box
    """
    gt_boxes = []
    tree = ET.parse(annoted_img_path)
    root = tree.getroot()
    for items in root.findall('object/bndbox'):
        xmin = items.find('xmin')
        ymin = items.find('ymin')
        xmax = items.find('xmax')
        ymax = items.find('ymax')
        gt_boxes.append([int(xmin.text), int(ymin.text), int(xmax.text), int(ymax.text)])
    return gt_boxes

def main():
    strategy = ['color', 'all'][0]
    img_dir = '/media/h/H/DOTA10_512_128/val/images/'
    save_path = '/media/h/H/DOTA10_512_128/train/SSW/selective_search_val.pkl'
    ref_json_file = '/media/h/H/DOTA10_512_128/annotations/DOTA_val_512.json'
    fr = open(ref_json_file)
    ref_data = json.load(fr)
    img_list = ref_data['images']
    pkl_infoes = []
    for img_dict in tqdm(img_list):
        img_name = img_dict['file_name']
        img_path = os.path.join(img_dir, img_name)
        img_ori=cv2.imread(img_path)
        ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
        # Convert image from BGR (default color in OpenCV) to RGB
        rgb_im = cv2.cvtColor(img_ori, cv2.COLOR_BGR2RGB)
        ss.addImage(rgb_im)
        gs = cv2.ximgproc.segmentation.createGraphSegmentation()
        # gs.setK(150)
        # gs.setSigma(0.8)
        ss.addGraphSegmentation(gs)
        # Creating strategy using color similarity
        if strategy == "color":
            strategy_color = cv2.ximgproc.segmentation.createSelectiveSearchSegmentationStrategyColor()
            ss.addStrategy(strategy_color)
        # Creating strategy using all similarities (size,color,fill,texture)
        elif strategy == "all":
            strategy_color = cv2.ximgproc.segmentation.createSelectiveSearchSegmentationStrategyColor()
            strategy_fill = cv2.ximgproc.segmentation.createSelectiveSearchSegmentationStrategyFill()
            strategy_size = cv2.ximgproc.segmentation.createSelectiveSearchSegmentationStrategySize()
            strategy_texture = cv2.ximgproc.segmentation.createSelectiveSearchSegmentationStrategyTexture()
            strategy_multiple = cv2.ximgproc.segmentation.createSelectiveSearchSegmentationStrategyMultiple(
                strategy_color, strategy_fill, strategy_size, strategy_texture)
            ss.addStrategy(strategy_multiple)
        get_boxes = ss.process()
        print("Total proposals = ", len(get_boxes))
        boxes = [[box[0], box[1], box[0] + box[2], box[1] + box[3]] for box in get_boxes]
        proposal_box_limit = 1000
        boxes = [[box[0], box[1], box[0] + box[2], box[1] + box[3]] for box in get_boxes[0:proposal_box_limit]]
        pkl_info =dict()
        pkl_info['file_name'] = img_name
        pkl_info['id'] = img_name
        pkl_info['width'] = img_name 
        pkl_info['height'] = img_name
        pkl_info['selective_search'] = np.array(boxes)
        pkl_infoes.append(pkl_info)
    mmcv.dump(pkl_infoes, save_path)
if __name__ == "__main__":
    main()