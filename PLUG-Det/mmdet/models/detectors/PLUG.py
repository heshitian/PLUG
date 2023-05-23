from mmdet.models.builder import DETECTORS
from mmdet.models.detectors.single_stage import SingleStageDetector
from mmdet.core import bbox2result
import torch.nn.functional as F
import torch.nn as nn
import torch 

@DETECTORS.register_module()
class PLUG(SingleStageDetector):
    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(PLUG,
              self).__init__(backbone, neck, bbox_head, train_cfg, test_cfg,
                             pretrained, init_cfg)

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_pseudo_bboxes=None,
                      ):
        super(SingleStageDetector, self).forward_train(img, img_metas)
        x = self.extract_feat(img) 
        losses = self.bbox_head.forward_train(x, img, img_metas, gt_pseudo_bboxes,
                                              gt_labels)
        return losses
    
    def set_epoch(self, epoch):
        self.bbox_head.epoch = epoch
    def set_iter(self, iter):
        self.bbox_head.iter = iter
    def set_inner_iter(self, inner_iter):
        self.bbox_head.inner_iter = inner_iter
    
    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x
    
    def simple_test(self, img, img_metas, rescale=False, gt_pseudo_bboxes=None, gt_labels=None,\
        gt_bboxes_ignore=None, gt_anns_id=None,gt_bboxes=None, gt_masks= None, two_model = None):
        """Test function without test-time augmentation.

        Args:
            img (torch.Tensor): Images with shape (N, C, H, W).
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        feat = self.extract_feat(img)
        results_list = self.bbox_head.simple_test(
            feat, img, img_metas, rescale=rescale, gt_pseudo_bboxes=gt_pseudo_bboxes, gt_labels=gt_labels,\
            gt_bboxes_ignore=gt_bboxes_ignore, gt_anns_id=gt_anns_id,gt_bboxes=gt_bboxes)
        final_results_list = results_list[0]
        bbox_results = [
            bbox2result(det_bbox, det_label, self.bbox_head.num_classes)
            for det_bbox, det_label in zip(final_results_list[0][0], gt_labels[0])
        ]
        mask_results = [
            self.mask2result(det_mask, det_label, self.bbox_head.num_classes)
            for det_mask, det_label in zip(final_results_list[1], gt_labels[0])
        ]
        return list(zip(bbox_results, mask_results))
    def mask2result(self, maskes, labels, num_classes):
        maskes = torch.stack(maskes,0)
        out_mask = [maskes[labels == i] for i in range(num_classes)]
        out_mask = [list(out_mask[i]) for i in range(num_classes)]
        return out_mask