from operator import index
from turtle import pos
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, Linear
import matplotlib.pyplot as plt
import skimage.morphology as mpg
from mmdet.core import multi_apply
from mmdet.models.builder import HEADS, build_loss
from mmdet.models.dense_heads.anchor_free_head import AnchorFreeHead, BaseDenseHead
import os
import cv2
from mmdet.models.losses.utils import weight_reduce_loss
from PIL import Image
from pydijkstra import dijkstra_image
import pydensecrf.densecrf as dcrf

@HEADS.register_module()
class PLUGHead(BaseDenseHead):
    """
    PLUG Head
    """ 
    def __init__(self,
                num_classes,
                sfg_flag = False,
                embed_dims = None,
                strides = [8],
                loss_cfg=dict(
                    with_neg_loss=True,
                    neg_loss_weight=1.0,
                    with_gt_loss=True,
                    gt_loss_weight=1.0,
                    with_color_loss = True,
                    color_loss_weight = 1.0,
                ),
                pred_cfg = dict(
                    pred_diff = True,
                    boundary_diff = True,
                    boundary_diff_weight = 0.5,
                    bg_threshold = 0.5,
                ),
                **kwargs
                ):
        super(PLUGHead, self).__init__()
        self.num_classes = num_classes# + 1 if normal_cfg["out_bg_cls"] else num_classes
        self.loss_cfg = loss_cfg
        self.embed_dims = embed_dims
        self.sfg_flag = sfg_flag
        self.sparse_avgpool = nn.AdaptiveAvgPool2d(1)
        self.strides = strides        
        self.pred_cfg = pred_cfg
        if self.sfg_flag == True:
            self.meta_querys = nn.Embedding(self.num_classes, self.embed_dims)
            self.meta_querys.weight.requires_grad = False
            self.meta_prototypes = [[] for cls_ in range(self.num_classes)]
            
        self._init_layers()
            
    def _init_layers(self):
        """Initialize layers of the head."""
        # predictor
        self.support_fc= Linear(self.embed_dims, self.num_classes)
        if self.sfg_flag == True:
            self.fcrelu1 =  nn.Sequential(
                    nn.Conv2d(self.embed_dims, self.embed_dims,kernel_size=1),
                    nn.BatchNorm2d(self.embed_dims),
                    nn.ReLU())
            self.fcrelu2 =  nn.Sequential(
                nn.Conv2d(self.embed_dims, self.embed_dims,kernel_size=1),
                nn.BatchNorm2d(self.embed_dims),
                nn.ReLU())
            self.fcrelu3 =  nn.Sequential(
                nn.Conv2d(self.embed_dims*3, self.embed_dims,kernel_size=1),
                nn.BatchNorm2d(self.embed_dims),
                nn.ReLU())
            self.aggretator_fc = Linear(self.embed_dims, self.num_classes)

    def forward_train(self, x, img, img_metas, gt_pseudo_bboxes, gt_labels=None):
        gt_pseudo_points = torch.cat(gt_pseudo_bboxes).reshape(-1,2,2).mean(dim=1)
        num_samples = [len(gt_labels_) for gt_labels_ in gt_labels]
        gt_pseudo_points = torch.split(gt_pseudo_points,num_samples,dim=0)
        losses = dict()
        if gt_labels is None:
            loss_inputs = tuple([x]) + (img, gt_pseudo_points, img_metas )
        else:
            loss_inputs = tuple([x]) + (img, gt_pseudo_points, gt_labels,img_metas)
        losses = self.loss(*loss_inputs)
        return losses

    def gfocal_loss(self, p, q, w=1.0,eps = 1e-6):
        l1 = (p - q) ** 2
        l2 = q * (p + eps).log() + (1 - q) * (1 - p + eps).log()
        return -(l1 * l2 * w).sum(dim=-1)
    
    def loss(self, cls_feat, img, gt_pseudo_points, gt_labels, img_metas):
        # 01 将points和bs_ind信息和标签合并在一起,方便后续计算
        gt_labels_all = torch.cat(gt_labels, dim=0)
        pseudo_gt_sampleid = []
        for sam_id in range(len(gt_pseudo_points)):
            for gt_pseudo_point in range(len(gt_pseudo_points[sam_id])):
                pseudo_gt_sampleid.append(sam_id)
        pseudo_gt_sampleid = torch.tensor(pseudo_gt_sampleid, device =gt_pseudo_points[0].device)
        pseudo_gt_points = torch.hstack((torch.vstack(gt_pseudo_points),pseudo_gt_sampleid.reshape(-1,1), gt_labels_all.reshape(-1,1)))
        #02 寻找稀疏图像和稀疏图像
        # # a 图像中只有一个目标的图像，和图像中的目标
        # if self.sfg_flag == True:
        #     sparse_bs_id = [ temp_i   for temp_i, gt_labels_ in enumerate(gt_labels) if int(gt_labels_.shape[0])  ==1]
        #     sparse_bs = len(sparse_bs_id)
        #     sparse_points = []
        #     for sparse_bs_id_ in sparse_bs_id:
        #         sparse_points.append(pseudo_gt_points[pseudo_gt_points[:,2]==sparse_bs_id_])
        #     if len(sparse_points)>0:
        #         sparse_points = torch.cat(sparse_points)
        #         sparse_bs = sparse_points.shape[0]
        # b 图像中某一类目标只有一个的图像，和该类的目标
        sparse_bs = 0
        if self.sfg_flag == True:
            sparse_points = []
            label_unique = torch.unique(pseudo_gt_points[:,3])
            for label_uni in label_unique:
                label_points = pseudo_gt_points[pseudo_gt_points[:,3]==label_uni]
                sparse_points_index = torch.bincount(label_points[:,2].int())==1
                sparse_sampleid = torch.nonzero(sparse_points_index==1)
                if sparse_sampleid.shape[0]>0:
                    sparse_points_index_ = [label_points[:,2]==i for i in sparse_sampleid.squeeze(1)]
                    sparse_points_index_ = torch.cat( sparse_points_index_).reshape(sparse_sampleid.shape[0],-1).sum(0).bool()
                    sparse_points.append(label_points[sparse_points_index_])
            if len(sparse_points)>0:
                sparse_points = torch.cat(sparse_points)
                sparse_bs = sparse_points.shape[0]
        # 03 分层计算损失函数
        losses = {}
        for i, lvl_stride in enumerate(self.strides):      
            cls_feat_lvl = cls_feat[i] #BS, fea, H, W
            num_bs = cls_feat_lvl.shape[0]   
            fea_shape = cls_feat_lvl.shape[-2:]
            cls_out_lvl = self.support_fc(cls_feat_lvl.permute(0,3,2,1)) # N W H C
            cls_prob_lvl = cls_out_lvl.sigmoid()
            # meta feature encoding 
            if sparse_bs>0:
                # predictor
                sparse_points_small = sparse_points.clone()
                sparse_points_small[:,:2] /= lvl_stride
                sparse_img = img[sparse_points_small[:,2].long()]
                sparse_feat = cls_feat_lvl[sparse_points_small[:,2].long()].detach()
                sparse_prob = cls_prob_lvl[sparse_points_small[:,2].long()].detach()
                # neighbor cost
                if self.pred_cfg['boundary_diff']:
                    sparse_img_small = F.interpolate(sparse_img, fea_shape, mode='bilinear', align_corners=False)
                    sparse_img_edge = self.image_to_boundary(sparse_img_small)
                    diff_bond = self.neighbour_diff(sparse_img_edge[:, None], 'max')
                else:
                    diff_bond = 0
                if self.pred_cfg['pred_diff']:
                    cls_probs_lvl_pred= sparse_prob.permute(0,3,2,1) # NCHW
                    diff_pred = self.neighbour_diff(cls_probs_lvl_pred, 'l1')
                else:
                    diff_pred = 0
                diff_all = diff_pred + diff_bond *  self.pred_cfg['boundary_diff_weight']
                diff_np = diff_all.permute(0, 2, 3, 1).data.cpu().numpy()  # [bs, 128, 128, 8]
                # instance label generation (ILG)
                sparse_label_news = []
                for bs_ind in range(sparse_bs):
                    sparse_points_small_bs = sparse_points_small[bs_ind][:2][None].flip(1)
                    mindist = dijkstra_image(diff_np[bs_ind], sparse_points_small_bs.detach().cpu()) # diff_np [bs,h,w,8] pnt_coords_np [num_points,2] mindist [3, h,w]
                    mindist = torch.as_tensor(mindist, dtype=torch.float32, device=cls_probs_lvl_pred.device)
                    mindist /= mindist.max() + 1e-5
                    dist_likeli = 1 - mindist # [n, H, W]
                    sparse_labels_ind = F.one_hot(sparse_points[bs_ind][3].long(), self.num_classes).float()[None]
                    clas_likeli = torch.einsum('nc,chw->nhw', sparse_labels_ind, cls_probs_lvl_pred[bs_ind])
                    likeli = dist_likeli*clas_likeli
                    likeli_out = likeli
                    if likeli_out[0, sparse_points_small_bs[0,0].cpu().long(), sparse_points_small_bs[0,1].cpu().long()] < self.pred_cfg['bg_threshold']:
                        likeli_out[0, sparse_points_small_bs[0,0].cpu().long(), sparse_points_small_bs[0,1].cpu().long()] = self.pred_cfg['bg_threshold']
                    mask_ind =  likeli_out
                    bg = mask_ind.max(0)[0][None]< self.pred_cfg['bg_threshold']
                    mask_ind = torch.vstack((mask_ind,bg))
                    mask_ind= mask_ind.argmax(0)
                    instance_diff = self.neighbour_diff(mask_ind[None, None].float(), 'l1')
                    instance_diff = instance_diff.permute(0, 2, 3, 1).data.cpu().numpy() 
                    instance_dist = dijkstra_image(instance_diff[0], sparse_points_small_bs.cpu().numpy())
                    instance_dist = torch.tensor(instance_dist) 
                    label_news = ((instance_dist[0]==0)*1)[None]
                    sparse_label_news.append(label_news)
                sparse_label_news = torch.cat(sparse_label_news).to(sparse_prob.device)
                sparse_label_news = sparse_label_news[:,None]
                # masked average pooling
                sparse_masked_feat =  sparse_feat*sparse_label_news
                sparse_masked_factor = sparse_label_news.shape[2]*sparse_label_news.shape[3]/sparse_label_news.sum(2).sum(2)
                sparse_prototype = self.sparse_avgpool(sparse_masked_feat).squeeze(-1).squeeze(-1)
                sparse_prototype *= sparse_masked_factor
                sparse_prototype = sparse_prototype
                if self.inner_iter == 0:
                    for ijk in range(self.num_classes):
                        if len(self.meta_prototypes[ijk])>0:
                            meta_queryss = torch.cat(self.meta_prototypes[ijk]).mean(0)
                            self.meta_querys.weight[ijk] = meta_queryss
                    self.meta_prototypes = [[] for cls_ in range(self.num_classes)]
                for sup_lab_ind in range(sparse_prototype.shape[0]):
                    sup_lab = sparse_points[sup_lab_ind, 3]
                    self.meta_prototypes[sup_lab.long()].append(sparse_prototype[sup_lab_ind][None])
            # feature aggregation
            if self.sfg_flag == True:
                cls_prob_lvl_new = cls_feat_lvl.new_full((cls_feat_lvl.shape[0],*fea_shape, self.num_classes),0)
                query_feat = cls_feat_lvl  #BS, fea, H, W
                query_code = self.meta_querys.weight[:,None,:,None,None].clone()
                aggregate_feat1 = torch.mul(query_feat,query_code) # C,BS,fea, H, W
                aggregate_feat2 = query_feat-query_code
                for cls_ in range(self.num_classes):
                    aggregate_feat1_cls = self.fcrelu1(aggregate_feat1[cls_])
                    aggregate_feat2_cls = self.fcrelu2(aggregate_feat2[cls_])
                    aggregate_feat_cls = torch.cat((aggregate_feat1_cls, aggregate_feat2_cls, query_feat), 1)
                    aggregate_feat_cls = self.fcrelu3(aggregate_feat_cls)
                    cls_out = aggregate_feat_cls.permute(0, 3, 2, 1)#BSWHC
                    cls_out = self.aggretator_fc(cls_out)
                    cls_out = cls_out.sigmoid()
                    cls_prob_lvl_new[..., cls_] = cls_out[..., cls_]
                cls_prob_lvl = cls_prob_lvl_new
            else:
                cls_prob_lvl = cls_prob_lvl
            # losses
            pseudo_gt_points_lvl = pseudo_gt_points.clone()
            pseudo_gt_points_lvl[:,:2] = (pseudo_gt_points_lvl[:,:2]/lvl_stride).int()
            # 提取正负样本位置的语义预测结果
            gt_cls_label = []
            gt_valid = torch.zeros((num_bs, fea_shape[1], fea_shape[0]), device= cls_prob_lvl.device).bool()
            gt_cls_prob = []
            for img_idx in range(num_bs):
                pseudo_gt_points_lvl_img = pseudo_gt_points_lvl[pseudo_gt_points_lvl[...,2]==img_idx]
                gt_cls_prob.append(cls_prob_lvl[img_idx, pseudo_gt_points_lvl_img[:,0].long(), pseudo_gt_points_lvl_img[:,1].long()])
                gt_cls_label.append(pseudo_gt_points_lvl_img[:,-1])
                gt_valid[img_idx, pseudo_gt_points_lvl_img[..., 0].long(), pseudo_gt_points_lvl_img[..., 1].long()] = 1.
            gt_valid = gt_valid.reshape(-1)
            lvl_cls_prob = cls_prob_lvl.reshape(-1,self.num_classes)
            gt_cls_prob =  torch.vstack(gt_cls_prob)
            gt_cls_labels =  torch.cat(gt_cls_label)
            neg_cls_prob = lvl_cls_prob[~ gt_valid, :]
            # 损失函数1 
            if self.loss_cfg.get('with_gt_loss', True):
                gt_label_onehot = torch.full((gt_cls_prob.shape[0], self.num_classes), 0., dtype=torch.float32).to(gt_cls_prob.device)
                gt_label_onehot[torch.arange(gt_cls_prob.shape[0]), gt_cls_labels.long()] = 1
                gt_weights = torch.FloatTensor([1.0] * len(gt_cls_labels)).to(gt_cls_labels.device)
                num_pos = max((gt_weights > 0).sum(), 1)
                gt_loss = self.gfocal_loss(gt_cls_prob, gt_label_onehot, gt_weights.unsqueeze(-1)) # change by H
                gt_loss = self.loss_cfg['gt_loss_weight'] * weight_reduce_loss(gt_loss, None, avg_factor=num_pos)
                losses.update({ f'lvl_{lvl_stride}_gt_loss': gt_loss})
            # 损失函数2
            if self.loss_cfg.get('with_color_loss', True):
                cls_probs_lvl_color = cls_prob_lvl.permute(0,3,2,1) # NCHW
                img_small = F.interpolate(img, (fea_shape[1], fea_shape[0]), mode='bilinear', align_corners=False)
                img_color_prior_loss = self.color_prior_loss(cls_probs_lvl_color, img_small, kernel_size=5)
                img_color_prior_loss = self.loss_cfg["color_loss_weight"] * img_color_prior_loss
                losses.update({ f'lvl_{lvl_stride}_color_loss': img_color_prior_loss})
            # 损失函数3
            if self.loss_cfg.get("with_neg_loss", True):
                num_neg, num_class = neg_cls_prob.shape
                neg_label_onehot = torch.full((num_neg, num_class), 0., dtype=torch.float32).to(neg_cls_prob.device)
                neg_loss = self.gfocal_loss(neg_cls_prob, neg_label_onehot)
                neg_loss = self.loss_cfg["neg_loss_weight"] * weight_reduce_loss(neg_loss, None, avg_factor=num_neg) 
                losses.update({ f'lvl_{lvl_stride}_neg_loss': neg_loss})
        return losses

    @torch.no_grad()
    def image_to_boundary(self, images):
        """ images: [bsz, 3, h, w]
        output: [bsz, h, w]
        """
        images = self._normalized_rgb_to_lab(images)

        #filter_g = _get_gaussian_kernel(5, 1).to(images).view(1, 1, 5, 5).expand(3, -1, -1, -1)
        #images = F.conv2d(images, filter_g, padding=2, groups=3)

        weight = torch.as_tensor([[
            1, 0, -1,
            2, 0, -2,
            1, 0, -1
            ], [
            1, 2, 1,
            0, 0, 0,
            -1, -2, -1]], dtype=images.dtype, device=images.device)
        weight = weight.view(2, 1, 3, 3).repeat(3, 1, 1, 1)
        edge = F.conv2d(images, weight, padding=1, groups=3)
        edge = (edge**2).mean(1) # [bsz, h, w]
        edge = edge / edge[:, 5:-5, 5:-5].max(2)[0].max(1)[0].view(-1, 1, 1).clip(min=1e-5)
        edge = edge.clip(min=0, max=1)
        #edge = edge / edge.max(1,keepdim=True)[0].max(2, keepdim=True)[0].clip(min=1e-5)
        #mag = mag / mag.max(1, keepdim=True)[0].max(2, keepdim=True)[0].clip(min=1e-5)
        return edge

    @torch.no_grad()
    def _normalized_rgb_to_lab(self, images, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        """
        images: [bsz, 3, h, w]
        """
        assert images.ndim == 4, images.shape
        assert images.shape[1] == 3, images.shape

        device = images.device
        mean = torch.as_tensor(mean, device=device).view(1, 3, 1, 1)
        std = torch.as_tensor(std, device=device).view(1, 3, 1, 1)
        images = (images * std + mean).clip(min=0, max=1)
        rgb = images

        mask = (rgb > .04045).float()
        rgb = (((rgb+.055)/1.055)**2.4)*mask + rgb/12.92*(1-mask)
        xyz_const = torch.as_tensor([
            .412453,.357580,.180423,
            .212671,.715160,.072169,
            .019334,.119193,.950227], device=device).view(3, 3)
        xyz = torch.einsum('mc,bchw->bmhw', xyz_const, rgb)

        sc = torch.as_tensor([0.95047, 1., 1.08883], device=device).view(1, 3, 1, 1)
        xyz_scale = xyz / sc
        mask = (xyz_scale > .008856).float()
        xyz_int = xyz_scale**(1/3.)*mask + (7.787*xyz_scale + 16./116.)*(1-mask)
        lab_const = torch.as_tensor([
            0., 116., 0.,
            500., -500., 0.,
            0., 200., -200.], device=device).view(3, 3)
        lab = torch.einsum('mc,bchw->bmhw', lab_const, xyz_int)
        lab[:, 0] -= 16
        return lab.float()

    def neighbour_diff(self, data, dist=None):
        assert data.ndim == 4
        bsz, c, h, w = data.shape
        neighbour =self._unfold_wo_center(data, 3, 1) # [bsz, c, 8, h, w]

        if dist is None:
            return neighbour

        if dist == 'l1':
            diff = (data[:, :, None] - neighbour).abs().sum(1) # [b, 8, h, w]
            return diff
        if dist == 'l2':
            diff = ((data[:, :, None] - neighbour)**2).sum(1) # [b, 8, h, w]
            return diff
        if dist == 'dot':
            diff = 1 - torch.einsum('bchw,bcnhw->bnhw', data, neighbour)
            return diff
        if dist == 'max':
            diff = neighbour.abs().max(1)[0] # [b, 8, h, w]
            return diff
        raise RuntimeError(dist)
        
    def color_prior_loss(self, data, images, masks=None, kernel_size=3, dilation=2, avg_factor=None):
        """
        data:   [bsz, classes, h, w] or [bsz, h, w]
        images: [bsz, 3, h, w]
        masks:  [bsz, h, w], (opt.), valid regions
        """
        if data.ndim == 4:
            log_prob = F.log_softmax(data, 1)
        elif data.ndim == 3:
            log_prob = torch.cat([F.logsigmoid(-data[:, None]), F.logsigmoid(data[:, None])], 1)
        else:
            raise ValueError(data.shape)
        B, C, H, W = data.shape
        assert images.shape == (B, 3, H, W), (images.shape, data.shape)
        if masks is not None:
            assert masks.shape == (B, H, W), (masks.shape, data.shape)

        log_prob_unfold = self._unfold_wo_center(log_prob, kernel_size, dilation) # [bsz, classes, k**2-1, h, w] 获得邻域
        log_same_prob = log_prob[:, :, None] + log_prob_unfold
        max_ = log_same_prob.max(1, keepdim=True)[0]
        log_same_prob = (log_same_prob - max_).exp().sum(1).log() + max_.squeeze(1) # [bsz, k**2-1, h, w]

        images = self._normalized_rgb_to_lab(images)
        images_unfold = self._unfold_wo_center(images, kernel_size, dilation)
        images_diff = images[:, :, None] - images_unfold
        images_sim = (-torch.norm(images_diff, dim=1) * 0.5).exp() # [bsz, k**2-1, h, w]

        loss_weight = (images_sim >= 0.3).float()
        if masks is not None:
            loss_weight = loss_weight * masks[:, None]

        #print(data.shape, log_prob.shape, log_same_prob.shape, loss_weight.shape, images_sim.shape)
        loss = -(log_same_prob * loss_weight).sum((1, 2, 3)) / loss_weight.sum((1, 2, 3)).clip(min=1)
        loss = loss.sum() / (len(loss) if avg_factor is None else avg_factor)
        return loss
    def _unfold_wo_center(self, x, kernel_size, dilation, with_center=False):
        """
        x: [bsz, c, h, w]
        kernel_size: k
        dilation: d
        return: [bsz, c, k**2-1, h, w]
        """
        assert x.ndim == 4, x.shape
        assert kernel_size % 2 == 1, kernel_size

        padding = (kernel_size + (dilation - 1) * (kernel_size - 1)) // 2
        unfolded_x = F.unfold(x, kernel_size=kernel_size, dilation=dilation, padding=padding)

        n, c, h, w = x.shape
        unfolded_x = unfolded_x.reshape(n, c, -1, h, w)

        if with_center:
            return unfolded_x

        # remove the center pixel
        size = kernel_size**2
        unfolded_x = torch.cat((unfolded_x[:, :, :size // 2], unfolded_x[:, :, size // 2 + 1:]), 2)
        return unfolded_x
    @torch.no_grad()
    def _normalized_rgb_to_lab(self, images, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        """
        images: [bsz, 3, h, w]
        """
        assert images.ndim == 4, images.shape
        assert images.shape[1] == 3, images.shape

        device = images.device
        mean = torch.as_tensor(mean, device=device).view(1, 3, 1, 1)
        std = torch.as_tensor(std, device=device).view(1, 3, 1, 1)
        images = (images * std + mean).clip(min=0, max=1)
        rgb = images

        mask = (rgb > .04045).float()
        rgb = (((rgb+.055)/1.055)**2.4)*mask + rgb/12.92*(1-mask)
        xyz_const = torch.as_tensor([
            .412453,.357580,.180423,
            .212671,.715160,.072169,
            .019334,.119193,.950227], device=device).view(3, 3)
        xyz = torch.einsum('mc,bchw->bmhw', xyz_const, rgb)

        sc = torch.as_tensor([0.95047, 1., 1.08883], device=device).view(1, 3, 1, 1)
        xyz_scale = xyz / sc
        mask = (xyz_scale > .008856).float()
        xyz_int = xyz_scale**(1/3.)*mask + (7.787*xyz_scale + 16./116.)*(1-mask)
        lab_const = torch.as_tensor([
            0., 116., 0.,
            500., -500., 0.,
            0., 200., -200.], device=device).view(3, 3)
        lab = torch.einsum('mc,bchw->bmhw', lab_const, xyz_int)
        lab[:, 0] -= 16
        return lab.float()

    def simple_test(self, feats, img, img_metas, rescale=False, gt_pseudo_bboxes=None, gt_labels=None,\
            gt_bboxes_ignore=None, gt_anns_id=None,gt_bboxes=None):
        return self.simple_test_bboxes(feats, img, img_metas, rescale=rescale, gt_pseudo_bboxes=gt_pseudo_bboxes, gt_labels=gt_labels,\
            gt_bboxes_ignore=gt_bboxes_ignore, gt_anns_id=gt_anns_id,gt_bboxes=gt_bboxes)
    def simple_test_bboxes(self, feats, img, img_metas, rescale, gt_pseudo_bboxes=None, gt_labels=None,\
            gt_bboxes_ignore=None, gt_anns_id=None,gt_bboxes=None):
        gt_pseudo_points = torch.cat(gt_pseudo_bboxes[0]).reshape(-1,2,2).mean(dim=1)
        num_samples = [len(gt_labels_) for gt_labels_ in gt_labels[0]]
        gt_pseudo_points = torch.split(gt_pseudo_points,num_samples,dim=0)
        # outs = self.forward(feats)
        results = self.get_bboxes(
            feats, img, img_metas=img_metas, gt_pseudo_points = gt_pseudo_points, gt_labels=gt_labels[0],\
            gt_bboxes_ignore=gt_bboxes_ignore[0], gt_bboxes=gt_bboxes[0],gt_anns_id=gt_anns_id[0] )
        return results
    def get_bboxes(self, cls_feat, img, img_metas, gt_pseudo_points = None, gt_labels=None, gt_bboxes_ignore=None, gt_bboxes=None, gt_anns_id=None):
        """
        Args:
            gt_bboxes: [num_img, (num_gts*num_refine, 4)]
            gt_labels: [num_img, (num_gts,)]
            gt_bboxes_ignore:
            gt_true_bboxes: [num_img, (num_gts, 4)] or None
        """
        # # 画图1:相似度矩阵
        # import matplotlib.pyplot as plt
        # matrix = []
        # for i in range(15):
        #     cosine_s = F.cosine_similarity(self.meta_querys.weight[i][None], self.meta_querys.weight)
        #     matrix.append(cosine_s[None])
        # matrix = torch.cat(matrix)
        # save_dir = '/media/h/M/P2B/1dota/fuse_r0/'
        # save_path = os.path.join(save_dir, 'prototype.eps')
        # fig, ax = plt.subplots(figsize=(8,8)) #, constrained_layout = True)
        # cax = ax.imshow(matrix.detach().cpu(), cmap='GnBu')
        # ax.tick_params(bottom = False, top = False, left = False, right = False)
        # ax.set_xticks(np.arange(matrix.shape[0]))
        # ax.set_xticklabels (['PL','BD', 'BR', 'GTF', 'SV', 'LV', 'SH', 'TC', 'BC', 'ST', 'SBF', 'RA', 'HB', 'SP', 'HC'], fontsize = 18)
        # ax.set_yticks(np.arange(matrix.shape[0]))
        # ax.set_yticklabels(['PL','BD', 'BR', 'GTF', 'SV', 'LV', 'SH', 'TC', 'BC', 'ST', 'SBF', 'RA', 'HB', 'SP', 'HC'], fontsize = 18)
        # manager = plt.get_current_fig_manager()
        # manager.resize(*manager.window.maxsize()) # TKAgg backend
        # cb = fig.colorbar(cax, ticks = [], extend='both') #  extend='both',, fraction=0.05
        # cb.set_label(label='Distinct                                                         Similar',loc='center', fontsize = 18) #loc参数
        # plt.show()
        # fig.savefig(save_path, bbox_inches = 'tight', pad_inches = 0)
        assert len(gt_labels) > 0
        gt_labels_all = torch.cat(gt_labels, dim=0)    
        pseudo_gt_sampleid = []
        for sam_id in range(len(gt_pseudo_points)):
            for gt_pseudo_point in range(len(gt_pseudo_points[sam_id])):
                pseudo_gt_sampleid.append(sam_id)
        pseudo_gt_sampleid = torch.tensor(pseudo_gt_sampleid, device =gt_pseudo_points[0].device )
        pseudo_gt_points = torch.hstack((torch.vstack(gt_pseudo_points),pseudo_gt_sampleid.reshape(-1,1), gt_labels_all.reshape(-1,1)))

        diffusion_bbox_results = []
        diffusion_mask_results = []
        cls_prob_outs = []
        for i, lvl_stride in enumerate(self.strides):        
            cls_feat_lvl = cls_feat[i] #BS, fea, H, W
            fea_shape = cls_feat_lvl.shape[-2:]
            # predictor
            cls_out_lvl = self.support_fc(cls_feat_lvl.permute(0,3,2,1)) # N W H C
            cls_prob_lvl = cls_out_lvl.sigmoid()
            # feature aggregation
            if self.sfg_flag == True:
                cls_prob_lvl_new = cls_feat_lvl.new_full((cls_feat_lvl.shape[0],*fea_shape, self.num_classes),0)
                query_feat = cls_feat_lvl  #BS, fea, H, W
                query_code = self.meta_querys.weight[:,None,:,None,None].clone()
                aggregate_feat1 = torch.mul(query_feat,query_code) # C,BS,fea, H, W
                aggregate_feat2 = query_feat-query_code
                cls_out_all = []
                for cls_ in range(self.num_classes):
                    aggregate_feat1_cls = self.fcrelu1(aggregate_feat1[cls_])
                    aggregate_feat2_cls = self.fcrelu2(aggregate_feat2[cls_])
                    aggregate_feat_cls = torch.cat((aggregate_feat1_cls, aggregate_feat2_cls, query_feat), 1)
                    aggregate_feat_cls = self.fcrelu3(aggregate_feat_cls)
                    cls_out = aggregate_feat_cls.permute(0, 3, 2, 1)#BSWHC
                    cls_out = self.aggretator_fc(cls_out)
                    cls_out = cls_out.sigmoid()
                    cls_out_all.append(cls_out)
                    cls_prob_lvl_new[..., cls_] = cls_out[..., cls_]
                cls_prob_lvl = cls_prob_lvl_new
            else:
                cls_prob_lvl = cls_prob_lvl
            cls_prob_outs.append(cls_prob_lvl)
            cls_prob_out = cls_prob_lvl
            # edge cost
            if self.pred_cfg['boundary_diff']:
                img_new = img.clone()
                img_edge = self.image_to_boundary(img_new)
                diff_bond = self.neighbour_diff(img_edge[:, None], 'max')
                # 画图：edge map
                # save_path_ori = '/media/h/KINGSTON/paper/network/edge/'
                # os.mkdir(save_path_ori)  if not os.path.isdir(save_path_ori) else None
                # for u in gt_labels[0]:
                #     save_path0 = os.path.join(save_path_ori, str(int(u)))
                #     os.mkdir(save_path0)  if not os.path.isdir(save_path0) else None
                #     save_path = os.path.join(save_path0, img_metas[0]['ori_filename'].replace('.png', '_prob.png'))
                #     fig, ax = plt.subplots(figsize=(8,8)) #, constrained_layout = True)
                #     ax.imshow(img_edge[0].data.cpu())
                #     ax.set_axis_off()
                #     fig.savefig(save_path, bbox_inches = 'tight', pad_inches = 0)
                #     plt.close()
            else:
                diff_bond = 0
            # pred_cost
            if self.pred_cfg['pred_diff']:
                cls_probs_lvl_pred= cls_prob_out.permute(0,3,2,1) # NCHW
                cls_probs_lvl_pred_new = cls_probs_lvl_pred.clone()
                for bs_ind in range(img.shape[0]):
                    gt_labels_bs = gt_labels[bs_ind].unique().tolist()
                    for gt_labels_bs_ in gt_labels_bs:
                        cls_probs_lvl_pred_new[bs_ind][gt_labels_bs_,:,:] /=  cls_probs_lvl_pred_new[bs_ind][gt_labels_bs_,:,:].max()
                cls_probs_lvl_pred_big =  F.interpolate(cls_probs_lvl_pred_new, img_metas[0]['pad_shape'][:2], mode='bilinear', align_corners=False)
                diff_pred = self.neighbour_diff(cls_probs_lvl_pred_big, 'l1')
                # 消融实验：生成合成的密集目标的特征
                # cls_probs_lvl_pred_muse_all = []
                # cls_probs_lvl_pred_muse = cls_probs_lvl_pred_big.clone()
                # cls_probs_lvl_pred_muse_all.append(cls_probs_lvl_pred_muse)
                # gt_bboxes_small = gt_bboxes[0].clone()
                # # gt_bboxes_small = gt_bboxes_small/lvl_stride
                # ori_gt_box = gt_bboxes_small[0]
                # for gt_bboxes_small_single in gt_bboxes_small[1:]:
                #     shift = ori_gt_box - gt_bboxes_small_single
                #     shift = shift.reshape(-1,2).mean(0)
                #     padd = nn.ZeroPad2d((0, int(shift[0]), 0, int(shift[1])))  # lrtb
                #     cls_probs_lvl_pred_muse_temp = padd(cls_probs_lvl_pred_muse)
                #     cls_probs_lvl_pred_muse_all.append(cls_probs_lvl_pred_muse_temp[:,:,-512:, -512:])
                # cls_probs_lvl_pred_muse_all = torch.cat(cls_probs_lvl_pred_muse_all).max(0)[0][None]
                # cls_probs_lvl_pred_big = cls_probs_lvl_pred_muse_all
                # diff_pred = self.neighbour_diff(cls_probs_lvl_pred_big, 'l1')
            else:
                diff_pred = 0
            diff_all = diff_pred + diff_bond * self.pred_cfg['boundary_diff_weight']
            
            diff_np = diff_all.permute(0, 2, 3, 1).data.cpu().numpy()  # [bs, 128, 128, 8]
            diffusion_mask_results_lvl = []
            diffusion_bbox_results_lvl = []
            for bs_ind in range(img.shape[0]):
                # # Excel：语义预测模块折线图所使用的excel
                # feature_visual = cls_prob_out.permute(0, 2, 1, 3)[bs_ind]
                # mask_dir = '/media/h/M/P2B/1dota/paper/dsp/masks/'
                # mask_path = os.path.join(mask_dir,img_metas[0]['ori_filename'] )
                # mask_np = np.array(Image.open(mask_path).convert('L'))
                # mask_np [mask_np != 0] =1
                # lab = gt_labels[bs_ind]
                # mask_np = torch.tensor(mask_np)
                # mask_np = F.interpolate(mask_np[None,None], feature_visual.shape[:2]).squeeze(0).squeeze(0)
                # feature_visual[:,:,lab] = feature_visual[:,:,lab]/feature_visual[:,:,lab].max()
                # feature_visual_out = feature_visual.cpu() * mask_np[:,:,None]
                # out_num = feature_visual_out.mean(0).mean(0)*mask_np.shape[0]*mask_np.shape[1]/(mask_np!=0).sum()
                # out_num = torch.cat((lab.cpu(), out_num))
                # out_dir = '/media/h/M/P2B/1dota/paper/dsp/' 
                # out_path = os.path.join(out_dir, img_metas[0]['ori_filename'].replace('.png', '.xls'))
                # data = pd.DataFrame(out_num)
                # writer = pd.ExcelWriter(out_path)
                # data.to_excel(writer, 'sheet_1', float_format='%.6f')
                # writer.save()
                # writer.close()
                # #画图：语义预测模块可视化
                # fig, ax = plt.subplots(figsize=(10,7))
                # x_list = np.arange(16)[1:]
                # ax.plot(x_list, out_num, c='red')#, linestyle='--')#, label=name_list[0])
                # font_dict = {'family': 'Calibri',
                # 'size': 18,
                # }
                # plt.scatter(x_list, out_num, c='green')
                # # plt.legend(loc='upper right')
                # plt.xlabel("Categories", fontdict=font_dict)
                # plt.ylabel("Masked response", fontdict=font_dict)
                # ax.set_xticks(x_list)
                # ax.set_xticklabels (['PL','BD', 'BR', 'GTF', 'SV', 'LV', 'SH', 'TC', 'BC', 'ST', 'SBF', 'RA', 'HB', 'SP', 'HC'], fontsize = 18)
                # index = np.around(np.arange(((out_num.max()*10)+2).int())*0.1, decimals=1)
                # ax.set_yticks(index)
                # ax.set_yticklabels(index, fontsize = 18)
                # # plt.show()
                # out_dir = '/media/h/M/P2B/1dota/paper/dsp/' 
                # out_path = os.path.join(out_dir, img_metas[0]['ori_filename'].replace('.png', '.svg'))
                # # plt.show()
                # fig.savefig(out_path, dpi = fig.dpi, bbox_inches = 'tight', pad_inches = 0.05)
                # plt.close()

                # # 画图：展示sfg热力图                
                # save_dir = '/media/h/M/P2B/1dota/visual/intra_class_num/nofuse_r0_visual/out/'
                # save_path_ori = os.path.join(save_dir, 'prob')
                # os.mkdir(save_path_ori)  if not os.path.isdir(save_path_ori) else None
                # feature_visual = cls_probs_lvl_pred_big                  
                # for u in gt_labels[bs_ind]:
                #     save_path0 = os.path.join(save_path_ori, str(int(u)))
                #     os.mkdir(save_path0)  if not os.path.isdir(save_path0) else None
                #     save_path = os.path.join(save_path0, img_metas[bs_ind]['ori_filename'].replace('.png', '_prob.png'))
                #     fig, ax = plt.subplots(figsize=(8,8)) #, constrained_layout = True)
                #     feature_visual_show = feature_visual[bs_ind,u,:,:].data.cpu()
                #     feature_visual_show = (feature_visual_show-feature_visual_show.min())/(feature_visual_show.max()-feature_visual_show.min())
                #     im = ax.imshow(feature_visual_show)
                #     fig.colorbar(mappable=im, ax = ax, ticks = [0, 0.2, 0.4, 0.6, 0.8, 1.0 ], fraction = 0.01)
                #     ax.set_axis_off()
                #     fig.savefig(save_path, bbox_inches = 'tight', pad_inches = 0)
                #     plt.close()
                #     save_path = os.path.join(save_path0, img_metas[bs_ind]['ori_filename'].replace('.png', '_probimg.png'))
                #     fig, ax = plt.subplots(figsize=(8,8)) #, constrained_layout = True)
                #     rgb_img = self.Normalize_back(img[bs_ind], img_metas[bs_ind])
                #     rgb_img = np.float32(rgb_img) / 255
                #     heatmap = F.interpolate(feature_visual[bs_ind,:,:,u][None][None], img_metas[0]['pad_shape'][:2], mode='bilinear', align_corners=True).squeeze(0).squeeze(0)[...,None]
                #     heatmap = (heatmap-heatmap.min())/(heatmap.max()-heatmap.min())
                #     heatmap_ = cv2.applyColorMap(np.uint8(255 * heatmap.cpu()), cv2.COLORMAP_JET)
                #     heatmap_ = cv2.cvtColor(heatmap_,cv2.COLOR_BGR2RGB)
                #     heatmap_ = np.float32(heatmap_) / 255
                #     cam = heatmap_ + np.float32(rgb_img)
                #     cam = cam / np.max(cam)
                #     cam = np.uint8(255 * cam)
                #     im = ax.imshow(cam)
                #     fig.colorbar(mappable=im, ax = ax, ticks = [0, 0.2, 0.4, 0.6, 0.8, 1.0 ], fraction = 0.01)
                #     ax.set_axis_off()
                #     fig.savefig(save_path, bbox_inches = 'tight', pad_inches = 0)
                #     plt.close()
                pseudo_gt_points_lvl_bs = pseudo_gt_points[pseudo_gt_points[...,2]==bs_ind][:,:2]
                pseudo_gt_points_lvl_bs = torch.flip(pseudo_gt_points_lvl_bs, [1])
                mindist = dijkstra_image(diff_np[bs_ind], pseudo_gt_points_lvl_bs.detach().cpu()) # diff_np [bs,h,w,8] pnt_coords_np [num_points,2] mindist [3, h,w]
                mindist = torch.as_tensor(mindist, dtype=torch.float32, device=pseudo_gt_points.device)
                mindist /= mindist.max() + 1e-5
                dist_likeli = 1 - mindist # [n, H, W]
                gt_labels_ind = F.one_hot(gt_labels[bs_ind], self.num_classes).float() # [n, c]
                clas_likeli = torch.einsum('nc,chw->nhw', gt_labels_ind, cls_probs_lvl_pred_big[bs_ind])
                likeli = dist_likeli*clas_likeli
                likeli_out = likeli
                if likeli.shape[0] > 1:
                    likeli_out =  likeli.softmax(0)
                    likeli_min_map = likeli_out.min(0)[0]
                    likeli_out = likeli_out - likeli_min_map[None]
                likeli_layer_max = likeli_out.view(likeli_out.shape[0],-1).max(1)[0][:,None,None]
                likeli_out = likeli_out/(likeli_layer_max + 1e-5)
                mask_ind =  likeli_out

                # 画图：展示 likeli map
                # save_path0 = '/media/h/KINGSTON/paper/network/likeli/'
                # os.mkdir(save_path0)  if not os.path.isdir(save_path0) else None
                # for mmm in range(len(gt_labels[bs_ind])):
                #     save_path = os.path.join(save_path0, img_metas[bs_ind]['ori_filename'][:-4] + '_likeli_' + str(mmm) + '.png')
                #     fig, ax = plt.subplots(figsize=(8,8))
                #     ax.imshow(likeli_out.cpu()[mmm])
                #     ax.set_axis_off()
                #     fig.savefig(save_path, dpi = 300, bbox_inches = 'tight', pad_inches = 0)        
                #     plt.close()
                
                # 确定背景
                bg = mask_ind.max(0)[0][None]< self.pred_cfg['bg_threshold']
                mask_ind = torch.vstack((mask_ind,bg))            
                mask_ind= mask_ind.argmax(0)
                # 输出box和mask
                instance_diff = self.neighbour_diff(mask_ind[None, None].float(), 'l1')
                instance_diff = instance_diff.permute(0, 2, 3, 1).data.cpu().numpy() 
                instance_dist = dijkstra_image(instance_diff[0], pseudo_gt_points_lvl_bs.cpu().numpy())
                instance_dist = torch.tensor(instance_dist)
                instance_bboxes = []
                instance_masks = []
                for kk, mask_ind_single in enumerate(instance_dist):
                    mask_ind_single[mask_ind_single!=0]=1
                    mask_ind_single_out = mask_ind_single
                    mask_coords = torch.nonzero(mask_ind_single_out==0)
                    mask_ins = mask_ind_single_out==0
                    ymin = mask_coords[:,0].min()
                    ymax = mask_coords[:,0].max() if mask_coords[:,0].max()!=mask_coords[:,0].min() else mask_coords[:,0].max()+5
                    xmin = mask_coords[:,1].min()
                    xmax = mask_coords[:,1].max() if mask_coords[:,1].max()!=mask_coords[:,1].min() else mask_coords[:,1].max()+5
                    bboxes = torch.tensor((xmin,ymin,xmax,ymax), device=pseudo_gt_points_lvl_bs.device)
                    area_gt_box = (gt_bboxes[bs_ind][kk][2] - gt_bboxes[bs_ind][kk][0]) * (gt_bboxes[bs_ind][kk][3] - gt_bboxes[bs_ind][kk][1])
                    area_pred_box = (bboxes[2] - bboxes[0]) * (bboxes[3] - bboxes[1])
                    intersection_area = self.get_intersection_area(bboxes, gt_bboxes[bs_ind][kk])
                    union_area = area_pred_box + area_gt_box - intersection_area
                    iou = float(intersection_area) / float(union_area)
                    bboxes = torch.cat((bboxes, torch.tensor(iou, device=bboxes.device)[None]))
                    instance_bboxes.append(bboxes)
                    instance_masks.append(mask_ins)
                instance_bboxes = torch.vstack(instance_bboxes)
                diffusion_mask_results_lvl.append(instance_masks)
                diffusion_bbox_results_lvl.append(instance_bboxes)
                diffusion_bbox_results.append(diffusion_bbox_results_lvl)
                diffusion_mask_results.append(diffusion_mask_results_lvl)
        final_det_bboxes = diffusion_bbox_results
        return list(zip([final_det_bboxes], diffusion_mask_results))
    
    def Normalize_back(self, img, img_meta):
        img = img.permute(1,2,0).cpu().numpy()
        means, std =img_meta['img_norm_cfg']['mean'], img_meta['img_norm_cfg']['std'] 
        img = img* std + means
        return img
    
    def get_intersection_area(self, box1, box2):
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
    def get_targets(self, ):
        pass
