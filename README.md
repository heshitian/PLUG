## Learning Remote Sensing Object Detection with Single Point Supervision
<br>

This is the PyTorch implementation of the method in our paper "*Learning Remote Sensing Object Detection with Single Point Supervision*".
<!-- [[project](https://yingqianwang.github.io/LF-DAnet/)], [[paper](https://arxiv.org/pdf/2206.06214.pdf)]. -->
<br>

<br>

## Preparation:

#### 1. Requirement:
* [[mmdetection 2.22.0](https://github.com/open-mmlab/mmdetection)]
* [[DOTA_devkit](https://github.com/CAPTAIN-WHU/DOTA_devkit)]
* [[iSAID_devkit](https://github.com/CAPTAIN-WHU/iSAID_Devkit)]
* [[dijkstra](https://github.com/BraveGroup/PSPS)]

#### 2. Generating data with single point labels:
* First, we use the DOTA_devkit and iSAID_devkit toolbox to generate cropped images with json annotations.
* Second, we add single point annotations in the json file of iSAID dataset 
 `data_process/iSAID_generate_single_point_annotation.py`
* Third, we utilize the generated of iSAID json file to add single point information in the DOTA json file.
 `data_process/DOTA_generate_single_point_annotation.py`
## Model training and validation:
#### 1. Training PLUG:
* Run `train.py` with config `configs_single_point/PLUG_r50_DOTA_512.py` to train PLUG.
#### 2. Referencing and validating PLUG:
* Run `test_PLUG.py` to reference PLUG and generate pseudo boxes of training data.
* Run `data_process/calculate_mIoU.py` to output the mIoU results of PLUG.
#### 3. Training Faster-RCNN or Mask-RCNN:
* Run `data_process/bbox2json.py` to generate json of training data with pseudo boxes.
* Run `train.py` with config `configs/faster_rcnn/faster_rcnn_r50_fpn_1x_dota.py` to train Faster-RCNN.
* Run `data_process/segm2json.py` to generate json of training data with pseudo boxes and pseudo masks.
* Run `train.py` with config `configs/mask_rcnn/mask_rcnn_r50_fpn_1x_dota.py` to train Mask-RCNN.
#### 4. Testing Faster-RCNN or Mask-RCNN:
* Run `test.py` to reference and evaluate Faster-RCNN and Mask-RCNN.
* Run `DetVisGUI\DetVisGUI.py` to visualize the detection results of different detectors conveniently. ( [[DetVisGUI](https://github.com/Chien-Hung/DetVisGUI)])
#### 5. Other methods:
* We retrain P2BNet, WSDDN and OICR in our code based on [[P2BNet](https://github.com/ucas-vg/P2BNet)] and [[WSOD2](https://github.com/researchmm/WSOD2)]. 
* Run `train.py` with different configs to train the above methods.
#### 6. Other codes:
* We split the training dataset according the object numbers in images to evaluate the effects of dense obejects.
 `data_process/split_DOTA_image_and_json.py`
* Run `test_num.py` to generate the pseudo boxes of sub dataset with different object numbers cyclically. 
* Run `data_process/bar_chart.py` to generate the mIoU distribution of images with different object numbers.
## Our model and data annotations:
* Please download from the [[checkpoints(提取码：eh62)](https://pan.baidu.com/s/1yonTazs25aTLnwIkU_mOMw?pwd=eh62)].
<!-- * [Baidu Drive](https:) (key:). -->

## Citiation
**If you find this work helpful, please consider citing:**
```
@Article{PLUG-Det,
    author    = {He, Shitian and Zou, Huanxin and Wang, Yingqian and Li, Boyang and Cao, Xu and Jing, Ning},
    title     = {Learning Remote Sensing Object Detection with Single Point Supervision},
    journal   = {IEEE TGRS}, 
    year      = {2023},   
}
```
<br>

## Contact
**Welcome to raise issues or email to heshitian19@nudt.edu.cn for any question regarding this work.**
<!-- 
<details> 
<summary>statistics</summary>

![visitors](https://visitor-badge.glitch.me/badge?page_id=YingqianWang/LF-DAnet)

</details>  -->
