import mmcv
import torch
import time
from torch import nn as nn
from torch.nn import functional as F
from mmcv.cnn import build_conv_layer
from torch.cuda.amp import autocast
import numpy as np
from ..registry import MIDDLE_ENCODERS
from nuscenes.utils.data_classes import Box, LidarPointCloud
from nuscenes.utils.geometry_utils import view_points
from pyquaternion import Quaternion
from torch.utils.data import Dataset
from math import sqrt
import cv2
from torch.autograd import Variable

def save_to_file(file_name, contents):
    fh = open(file_name, 'w')
    fh.write(contents)
    fh.close()

class FocalLoss(nn.Module):

    def __init__(self, gamma=2.0, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps

    def one_hot(self, index, classes):
        size = index.size() + (classes,)
        view = index.size() + (1,)

        mask = torch.Tensor(*size).fill_(0).to(index.device)

        index = index.view(*view)
        ones = 1.

        if isinstance(index, Variable):
            a = torch.Tensor(index.size()).fill_(1)
            ones = Variable(a.to(index.device))
            mask = Variable(mask, volatile=index.volatile)
            index = index.long()
            ones = ones.long()
            mask = mask.long()
        return mask.scatter_(1, index, ones)

    def forward(self, input, target):
        y = self.one_hot(target, input.size(-1))
        logit = F.softmax(input, dim=-1)
        logit = logit.clamp(self.eps, 1. - self.eps)
        loss = -1 * y * torch.log(logit) # cross entropy
        loss = loss * (1 - logit) ** self.gamma # focal loss
        return loss.mean()

@MIDDLE_ENCODERS.register_module()
class lidar_img_encoder(nn.Module):
    def __init__(self,
                 cam_feature_shape=[12, 112, 112, 200],
                 get_loss=False
                 ):
        super().__init__()
        self.get_loss = get_loss
        self.cam_feature_shape = cam_feature_shape
        self.get_fore = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32, eps=0.001, momentum=0.01, affine=True, track_running_stats=True),
            # nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(1, eps=0.001, momentum=0.01, affine=True, track_running_stats=True),
            # nn.ReLU(),
        )
        self.get_loss = FocalLoss()

    def lidar_trans(self, cam, pcd_rotation, pcd_trans):
        cam[:, 0:3] = cam[:, 0:3] - pcd_trans.unsqueeze(0).repeat(cam.shape[0], 1)
        cam[:, 0:3] = torch.linalg.inv(pcd_rotation).unsqueeze(0).repeat(cam.shape[0], 1, 1).matmul(cam[:, 0:3].unsqueeze(-1)).squeeze(-1)
        return cam

    def get_img_fore(self, img, depth, img_metas):
        depth = F.interpolate(depth.unsqueeze(0).unsqueeze(0), scale_factor=0.125)
        img_with_depth = torch.cat([img, depth.squeeze(0)], dim=0).unsqueeze(0)
        img_fore_back = self.get_fore(img_with_depth)
        img_fore_back = F.sigmoid(img_fore_back)
        img_fore_mask = img_fore_back > 0.5
        img_fore = img_fore_mask.int()
        return img_fore_back, img_fore, img_fore_mask

    def forward(self, get_voxels, img_metas, img):
        device = torch.device('cuda:0')
        b_cam_around_depth = []
        b_cam_around_discretize = []

        if 'pcd_trans' in img_metas[0] and 'pcd_rotation' in img_metas[0]:
            for i in range(len(get_voxels)):
                pcd_trans = (torch.tensor(img_metas[i]['pcd_trans'])).to(device)
                pcd_rotation = (torch.tensor(img_metas[i]['pcd_rotation'])).to(device)
                get_voxels[i] = self.lidar_trans(get_voxels[i], pcd_rotation, pcd_trans)
        img_fore_per_batch = []
        b_img_fore_base = []
        for i in range(len(get_voxels)):
            lidar2img_percam = img_metas[i]['lidar2img']

            pad = torch.ones([get_voxels[i].shape[0], 1]).to(device)
            get_voxels[i] = torch.cat([get_voxels[i][:, 0:3], pad, get_voxels[i][:, 3:5]], -1)

            B_N, D, H, W = self.cam_feature_shape[0], self.cam_feature_shape[1], self.cam_feature_shape[2], self.cam_feature_shape[3]
            B = int(B_N / 6)
            canvas = torch.zeros(6, 896, 1600).to(device)
            img_fore_base = torch.zeros(6, 896, 1600).to(device)
            canvas_depth1 = torch.zeros(6, D, H, W).to(device)
            canvas_depth2 = torch.zeros(6, D, H, W).to(device)
            # canvas = torch.zeros(6, 800, 1600).to(device)
            # filename = img_metas[i]['filename']
            # pts_filename = img_metas[i]['filename'][1]
            # cam_orders = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK',
            #               'CAM_BACK_LEFT']
            img_fore_percam = []
            for z in range(6):
                get_voxel = torch.clone(get_voxels[i])
                lidar2img = torch.tensor(lidar2img_percam[z]).unsqueeze(0).to(device)
                get_voxel[:, 0:4] = lidar2img.repeat(get_voxel.shape[0], 1, 1).to(torch.float32).matmul(get_voxel[:, 0:4].to(torch.float32).unsqueeze(-1)).squeeze(-1)
                get_voxel[:, 2] = torch.clamp(get_voxel[:, 2], 1e-4, 1e4)
                get_voxel[:, 0:2] = get_voxel[:, 0:2]/get_voxel[:, 2].unsqueeze(-1)
                get_voxel2 = torch.clone(get_voxel)
                get_voxel3 = torch.clone(get_voxels[i][:, 0:3])
                get_voxel3[:, 0:2] = get_voxel3[:, 0:2] / get_voxel3[:, 2].unsqueeze(-1)

                get_voxels_mask = get_voxel2[:, 0] > torch.tensor(0, dtype=float).to(device)
                get_voxel2 = get_voxel2[get_voxels_mask]
                get_voxel3 = get_voxel3[get_voxels_mask]
                get_voxels_mask = get_voxel2[:, 0] < torch.tensor(1600, dtype=float).to(device)
                get_voxel2 = get_voxel2[get_voxels_mask]
                get_voxel3 = get_voxel3[get_voxels_mask]
                get_voxels_mask = get_voxel2[:, 1] > torch.tensor(0, dtype=float).to(device)
                get_voxel2 = get_voxel2[get_voxels_mask]
                get_voxel3 = get_voxel3[get_voxels_mask]
                get_voxels_mask = get_voxel2[:, 1] < torch.tensor(896, dtype=float).to(device)
                get_voxel2 = get_voxel2[get_voxels_mask]
                get_voxel3 = get_voxel3[get_voxels_mask]
                get_voxels_mask = get_voxel2[:, 2] > torch.tensor(0, dtype=float).to(device)
                get_voxel2 = get_voxel2[get_voxels_mask]
                get_voxel3 = get_voxel3[get_voxels_mask]

                # if pts_filename == './data/nuscenes/samples/CAM_FRONT/n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151607012404.jpg':
                # # if True:
                # # if pts_filename == './data/nuscenes/samples/CAM_FRONT/n015-2018-10-08-15-36-50+0800__CAM_FRONT__1538984236912460.jpg':
                #     img = torch.tensor(cv2.imread(filename[z]), dtype=int).to(device)
                #
                #     depth_color3 = []
                #     depth_color = torch.clone(get_voxel2[:, 2])
                #
                #     for j in range(2):
                #         depth_color3.append((torch.zeros(depth_color.shape).to(device)).unsqueeze(-1).long())
                #     depth_color3.append(depth_color.unsqueeze(-1).long())
                #     depth_color = torch.cat(depth_color3, 1).to(device)
                #     # depth_try = torch.tensor([128, 0, 128]).to(device)
                #     index_try = (get_voxel2[:, 1].floor().long(), get_voxel2[:, 0].floor().long())
                #     img.index_put_(index_try, depth_color)
                #     # pth = '/home/zcj/MSMDFusion/version/' + cam_orders[z] + '.jpg'
                #     pth = '/home/zcj/MSMDFusion/version/' + cam_orders[z] + '.jpg'
                #     device2 = torch.device('cpu')
                #     image = torch.clone(img).to(device2)
                #     image = np.array(image)
                #     cv2.imwrite(pth, image)


                depth = get_voxel2[:, 2]
                index = (get_voxel2[:, 1].long(), get_voxel2[:, 0].long())
                canvas[z].index_put_(index, depth)

                get_voxel_mul = torch.tensor([1600/W, 896/H]).to(device)
                get_voxel2[:, :2] = (get_voxel2[:, :2] / get_voxel_mul).floor()
                get_voxel2[:, 2] = (get_voxel2[:, 2]*1.6).floor()
                get_voxel2[:, 2] = torch.where(get_voxel2[:, 2].long() > 111, 111, get_voxel2[:, 2].long())
                feature = get_voxel2[:, 4]
                index = (get_voxel2[:, 2].long(), get_voxel2[:, 1].long(), get_voxel2[:, 0].long())
                canvas_depth1[z].index_put_(index, feature)
                feature = get_voxel2[:, 5]
                canvas_depth2[z].index_put_(index, feature)

                img_fore_feature, img_fore, img_fore_mask = self.get_img_fore(img[z], canvas[z], img_metas)
                img_fore_percam.append(img_fore_feature)
                if 'pcd_trans' in img_metas[0] and 'pcd_rotation' in img_metas[0]:
                    if self.get_loss:
                        gt_bboxes_3d = img_metas[i]['ann_info']['gt_bboxes_3d']
                        gt_bboxes_3d_corners = gt_bboxes_3d.corners
                        gt_bboxes_3d_2 = torch.cat([torch.max(gt_bboxes_3d_corners[:, :, 0], dim=1)[0].unsqueeze(-1), torch.max(gt_bboxes_3d_corners[:, :, 1], dim=1)[0].unsqueeze(-1), torch.max(gt_bboxes_3d_corners[:, :, 2], dim=1)[0].unsqueeze(-1)], 1)
                        gt_bboxes_3d_1 = torch.cat([torch.min(gt_bboxes_3d_corners[:, :, 0], dim=1)[0].unsqueeze(-1), torch.min(gt_bboxes_3d_corners[:, :, 1], dim=1)[0].unsqueeze(-1), torch.min(gt_bboxes_3d_corners[:, :, 2], dim=1)[0].unsqueeze(-1)], 1)
                        # gt_bboxes_3d = torch.cat([gt_bboxes_3d_1.unsqueeze(1), gt_bboxes_3d_2.unsqueeze(1)], dim=1)
                        voxel = get_voxel3.unsqueeze(1).repeat(1, gt_bboxes_3d_1.shape[0], 1)
                        gt_bboxes_3d_1 = gt_bboxes_3d_1.unsqueeze(0).repeat(get_voxel3.shape[0], 1, 1)
                        gt_bboxes_3d_2 = gt_bboxes_3d_2.unsqueeze(0).repeat(get_voxel3.shape[0], 1, 1)
                        point_mask1 = voxel > gt_bboxes_3d_1.to(device)
                        point_mask2 = voxel < gt_bboxes_3d_2.to(device)
                        point_mask = torch.cat([point_mask1, point_mask2], dim=2).int()
                        point_mask = torch.sum(point_mask, dim=2)
                        point_mask = torch.sum((point_mask == 6).int(), dim=1)
                        point_mask = point_mask > 0
                        point_in_box = get_voxel3[point_mask]
                        point_mask = point_mask.int()
                        index = (get_voxel2[:, 1].long(), get_voxel2[:, 0].long())
                        img_fore_base[z].index_put_(index, point_mask.float())
            canvas_depth = torch.cat([canvas_depth1.unsqueeze(1), canvas_depth2.unsqueeze(1)], 1)
            canvas = F.interpolate(canvas.unsqueeze(0), scale_factor=0.125)
            if self.get_loss:
                b_img_fore_base.append(F.interpolate(img_fore_base.unsqueeze(0), scale_factor=0.125))
            b_cam_around_depth.append(canvas.squeeze(0))
            img_fore_per_batch.append(torch.cat(img_fore_percam, dim=1))

            # pts_filename = img_metas[i]['filename']
            # path = '/home/mxd/MSMDFusion/vision/1'
            # path2 = '/home/mxd/MSMDFusion/vision/2'
            # torch.save(canvas, path)
            # save_to_file(path2, pts_filename)

            b_cam_around_discretize.append(canvas_depth)
        if self.get_loss:
            b_img_fore = torch.cat(img_fore_per_batch, dim=0).reshape(-1, 1)
            b_img_fore_base = torch.cat(b_img_fore_base, dim=0)
            mask_two_classes = torch.cat([1 - b_img_fore, b_img_fore], dim=1)
            loss = self.get_loss(mask_two_classes, b_img_fore_base.reshape(-1))
        else:
            loss = torch.tensor(0, device='cuda:0')
        B = len(b_cam_around_depth)
        b_cam_around_depth = torch.cat(b_cam_around_depth, 0)
        b_cam_around_discretize = torch.cat(b_cam_around_discretize, 0)

        step = 5
        N = 6
        C = 1
        BN, H, W = b_cam_around_depth.size()
        depth_tmp = b_cam_around_depth.reshape(BN, C, H, W)
        pad = int((step - 1) // 2)
        depth_tmp = F.pad(depth_tmp, [pad, pad, pad, pad], mode='constant', value=0)
        patches = depth_tmp.unfold(dimension=2, size=step, step=1)
        patches = patches.unfold(dimension=3, size=step, step=1)
        max_depth, _ = patches.reshape(B, N, C, H, W, -1).max(dim=-1)  # [2, 6, 1, 256, 704]
        # img_metas[0].update({'max_depth': max_depth})

        step = float(step)
        shift_list = [[step / H, 0.0 / W], [-step / H, 0.0 / W], [0.0 / H, step / W], [0.0 / H, -step / W]]
        max_depth_tmp = max_depth.reshape(B * N, C, H, W)
        output_list = []
        for shift in shift_list:
            transform_matrix = torch.tensor([[1, 0, shift[0]], [0, 1, shift[1]]]).unsqueeze(0).repeat(B * N, 1, 1).cuda()
            grid = F.affine_grid(transform_matrix, max_depth_tmp.shape).float()
            output = F.grid_sample(max_depth_tmp, grid, mode='nearest').reshape(B, N, C, H, W)
            output = max_depth - output
            output_mask = ((output == max_depth) == False)
            output = output * output_mask
            output_list.append(output)
        grad = torch.cat(output_list, dim=2)
        b_cam_around_depth = torch.cat([b_cam_around_depth.view(B, N, C, H, W), grad], dim=2).reshape(BN, -1, H, W)
        img_fore_per_batch = torch.cat(img_fore_per_batch, dim=0).reshape(BN, 1, H, W)
        b_cam_around_depth = torch.cat([b_cam_around_depth, img_fore_per_batch], dim=1)
        return b_cam_around_depth, b_cam_around_discretize, loss
