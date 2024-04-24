import mmcv
import torch
import time
from mmcv.parallel import DataContainer as DC
from mmcv.runner import force_fp32
from os import path as osp
from torch import nn as nn
from torch.nn import functional as F
from mmcv.cnn import build_conv_layer
from mmdet3d.core import (Box3DMode, Coord3DMode, bbox3d2result,
                          merge_aug_bboxes_3d, show_result)
from mmdet3d.ops import Voxelization
from mmdet.core import multi_apply
# from mmdet3d.ops import spconv as spconv
import spconv.pytorch as spconv
from mmdet.models import DETECTORS
from .. import builder
from .mvx_two_stage import MVXTwoStageDetector
from torch.cuda.amp import autocast
from numba import jit
import numpy as np

from .tools import rotate_box, get_feats_in_rectangle
import os


@jit(nopython=True)
def type_assign(batch_3D_val, batch_2D_val, batch_3Dtype_idf, batch_2Dtype_idf):

    N, M = batch_3D_val.shape[-1], batch_2D_val.shape[-1]
    ii, jj = 0, 0

    # 0, 1 within batch_3D/2Dtype_idf to indicate only 3D/2D and 2D+3D mixed
    while (ii < N) and (jj < M):
        if batch_3D_val[ii] < batch_2D_val[jj]:
            ii += 1
        elif batch_3D_val[ii] == batch_2D_val[jj]:
            batch_3Dtype_idf[ii] = 1
            batch_2Dtype_idf[jj] = 1
            ii += 1
            jj += 1
        else:
            jj += 1

    return batch_3Dtype_idf, batch_2Dtype_idf

class SPPModule(nn.Module):
    def __init__(self, in_channels=336):
        super().__init__()

        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(256, eps=0.001, momentum=0.01, affine=True, track_running_stats=True),
            nn.ReLU(),
        )
        self.conv3x3 = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256, eps=0.001, momentum=0.01, affine=True, track_running_stats=True),
            nn.ReLU()
        )
        self.dilated_conv3x3_rate6 = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, stride=1, padding=6, dilation=6, bias=False),
            nn.BatchNorm2d(256, eps=0.001, momentum=0.01, affine=True, track_running_stats=True),
            nn.ReLU()
        )
        self.dilated_conv3x3_rate12 = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, stride=1, padding=12, dilation=12, bias=False),
            nn.BatchNorm2d(256, eps=0.001, momentum=0.01, affine=True, track_running_stats=True),
            nn.ReLU()
        )
        # self.dilated_conv3x3_rate18 = nn.Sequential(
        #     nn.Conv2d(512+256, 256, kernel_size=3, stride=1, padding=12, dilation=12, bias=False),
        #     nn.BatchNorm2d(256, eps=0.001, momentum=0.01, affine=True, track_running_stats=True),
        #     nn.ReLU()
        # )
        self.fuse = nn.Sequential(
            nn.Conv2d(256*4, 256, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(256, eps=0.001, momentum=0.01, affine=True, track_running_stats=True),
            nn.ReLU(),
        )

    def forward(self, x):
        x1 = self.conv1x1(x)
        x2 = self.conv3x3(x)
        # x2 = self.dilated_conv3x3_rate18(x)
        x3 = self.dilated_conv3x3_rate6(x)
        x4 = self.dilated_conv3x3_rate12(x)
        ret = self.fuse(torch.cat([x1, x2, x3, x4], dim=1))
        
        return ret

@DETECTORS.register_module()
class CSDSFusionDetector(MVXTwoStageDetector):
    """Base class of Multi-modality VoxelNet."""

    def __init__(self, **kwargs):
        super(CSDSFusionDetector, self).__init__(**kwargs)
        
        self.freeze_img = kwargs.get('freeze_img', True)
        self.spatial_shapes = kwargs.get('spatial_shapes')
        self.downscale_factors = kwargs.get('downscale_factors')
        self.fps_num_list = kwargs.get('fps_num_list')
        self.radius_list = kwargs.get('radius_list')
        self.max_cluster_samples_list = kwargs.get('max_cluster_samples_list')
        self.dist_thresh_list = kwargs.get('dist_thresh_list') 
        self.get_loss = 1
         # channel compression for ResNet50
        self.conv1x1_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(256+1, 49, kernel_size=5, stride=1, padding=2, bias=False),
                nn.BatchNorm2d(49, eps=0.001, momentum=0.01, affine=True, track_running_stats=True),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.Conv2d(256+1, 49, kernel_size=5, stride=1, padding=2, bias=False),
                nn.BatchNorm2d(49, eps=0.001, momentum=0.01, affine=True, track_running_stats=True),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.Conv2d(256+1, 49, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(49, eps=0.001, momentum=0.01, affine=True, track_running_stats=True),
                nn.ReLU(),
            ),
        ])
        
        self.score_net = nn.Sequential(
            nn.Linear(50+16, 1),
            nn.ReLU()
        )
        self.get_fore = nn.Sequential(
                nn.Conv2d(8, 64, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(64, eps=0.001, momentum=0.01, affine=True, track_running_stats=True),
                # nn.ReLU(),
                nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(1, eps=0.001, momentum=0.01, affine=True, track_running_stats=True),
                # nn.ReLU(),
        )
        self.bev_fusion = SPPModule(in_channels=256+128)
        self.bev_fusion2 = SPPModule(in_channels=229+256)
        self.init_weights(pretrained=kwargs.get('pretrained', None))


    def init_weights(self, pretrained=None):
        """Initialize model weights."""
        super(CSDSFusionDetector, self).init_weights(pretrained)

        if self.freeze_img:
            if self.with_img_backbone:
                for param in self.img_backbone.parameters():
                    param.requires_grad = False
            if self.with_img_neck:
                for param in self.img_neck.parameters():
                    param.requires_grad = False

    def extract_img_feat(self, img, img_metas):
        """Extract features of images."""
        if self.with_img_backbone and img is not None:
            
            input_shape = img.shape[-2:]
            # update real input shape of each single img
            for img_meta in img_metas:
                img_meta.update(input_shape=input_shape)

            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_(0)
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.view(B * N, C, H, W)
            img_feats = self.img_backbone(img.float())
        else:
            return None
        
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)
        
        return img_feats

    def extract_pts_feat(self, pts, img_feats, img_metas, img):
        """Extract features of points."""
        if not self.with_pts_bbox:
            return None
        img_down = []
        if len(img.shape) == 4:
            img = img.unsqueeze(0)
        for i in range(img.shape[0]):
            img_down.append(F.interpolate(img[i, :, :, :, :], scale_factor=0.25))
        img = torch.cat(img_down, 0)

        voxels, num_points, coors = self.voxelize(pts)

        virtual_img_depth, virtual_img, img_fore_loss = self.lidar_img_encoder(pts, img_metas, img)

        virtual_points_feature, depth = self.virtual_pts_encoder(img_feats, img_metas, virtual_img_depth, virtual_img)
        # virtual_points_feature = self.virtual_pts_encoder(img_feats, img_metas)

        voxel_features = self.pts_voxel_encoder(voxels, num_points, coors)

        batch_size = coors[-1, 0] + 1
        
        x, encode_features = self.pts_middle_encoder(voxel_features, coors, batch_size)

        x = self.pts_backbone(x)

        if self.with_pts_neck:
            pts_feature = self.pts_neck(x)

        stage_outs, fused_img_feats = self.pre_fusion_encoder(pts_feature[0], virtual_points_feature, img_feats, virtual_img, virtual_img_depth)

        return pts_feature, fused_img_feats, img_fore_loss, depth



    def extract_feat(self, points, img, img_metas):
        """Extract features from images and points."""
        img_feats = self.extract_img_feat(img, img_metas)
        pts_feats, fused_img_feats, img_fore_loss, depth = self.extract_pts_feat(points, img_feats, img_metas, img)
        return (fused_img_feats, pts_feats, img_fore_loss, depth)
        # pts_feats = self.extract_pts_feat(points, img_feats, img_metas)
        # return (img_feats, pts_feats)

    @torch.no_grad()
    @force_fp32()
    def voxelize(self, points, downscale_factor=1.0):
        """Apply dynamic voxelization to points.

        Args:
            points (list[torch.Tensor]): Points of each sample.

        Returns:
            tuple[torch.Tensor]: Concatenated points, number of points
                per voxel, and coordinates.
        """
        # reset voxel size to [0.075, 0.075, 0.2]

        self.pts_voxel_layer.voxel_size = [0.075, 0.075, 0.2]

        voxels, coors, num_points = [], [], []
        self.pts_voxel_layer.voxel_size = list(map(lambda x: x * downscale_factor, self.pts_voxel_layer.voxel_size))
        for res in points:
            res_voxels, res_coors, res_num_points = self.pts_voxel_layer(res)
            voxels.append(res_voxels)
            coors.append(res_coors)
            num_points.append(res_num_points)
        voxels = torch.cat(voxels, dim=0)
        num_points = torch.cat(num_points, dim=0)
        coors_batch = []
        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)
        return voxels, num_points, coors_batch

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img=None,
                      proposals=None,
                      gt_bboxes_ignore=None):
        """Forward training function.

        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.

        Returns:
            dict: Losses of different branches.
        """
        
        img_feats, pts_feats, img_fore_loss, depth = self.extract_feat(
            points, img=img, img_metas=img_metas)
        losses = dict()
        
        if pts_feats:
            losses_pts = self.forward_pts_train(pts_feats, img_feats, gt_bboxes_3d,
                                                gt_labels_3d, img_metas,
                                                gt_bboxes_ignore)
            losses.update(losses_pts)
        # if img_depth is not None:
        #     loss_depth = self.depth_dist_loss(depth_dist, img_depth, loss_method=self.img_depth_loss_method, img=img) * self.img_depth_loss_weight
        #     losses.update(img_depth_loss=loss_depth)
        
        if img_feats:
            losses_img = self.forward_img_train(
                img_feats,
                img_metas=img_metas,
                gt_bboxes=gt_bboxes,
                gt_labels=gt_labels,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposals=proposals)
            losses.update(losses_img)
            losses['loss_heatmap'] = img_fore_loss + losses['loss_heatmap']
        return losses

    def forward_pts_train(self,
                          pts_feats,
                          img_feats,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          img_metas,
                          gt_bboxes_ignore=None):
        """Forward function for point cloud branch.

        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.

        Returns:
            dict: Losses of each branch.
        """
        outs = self.pts_bbox_head(pts_feats, img_feats, img_metas)
        loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
        losses = self.pts_bbox_head.loss(*loss_inputs)
        return losses

    def simple_test_pts(self, x, x_img, img_metas, rescale=False):
        """Test function of point cloud branch."""
        outs = self.pts_bbox_head(x, x_img, img_metas)
        bbox_list = self.pts_bbox_head.get_bboxes(
            outs, img_metas, rescale=rescale)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return bbox_results

    def simple_test(self, points, img_metas, img=None, rescale=False):
        """Test function without augmentaiton."""
        # import time
        # torch.cuda.synchronize()
        # ckpt0 = time.time()
        
        img_feats, pts_feats, img_fore_loss, depth = self.extract_feat(
            points, img=img, img_metas=img_metas)


        bbox_list = [dict() for i in range(len(img_metas))]
        
        if pts_feats and self.with_pts_bbox:
            bbox_pts = self.simple_test_pts(
                pts_feats, img_feats, img_metas, rescale=rescale)
            for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
                result_dict['pts_bbox'] = pts_bbox
        if img_feats and self.with_img_bbox:
            bbox_img = self.simple_test_img(
                img_feats, img_metas, rescale=rescale)
            for result_dict, img_bbox in zip(bbox_list, bbox_img):
                result_dict['img_bbox'] = img_bbox
        return bbox_list
