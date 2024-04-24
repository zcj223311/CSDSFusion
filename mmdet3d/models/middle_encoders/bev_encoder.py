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
# from src.train import train

def save_to_file(file_name, contents):
    fh = open(file_name, 'w')
    fh.write(contents)
    fh.close()

class _ASPPModule(nn.Module):

    def __init__(self, inplanes, planes, kernel_size, padding, dilation,
                 BatchNorm):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes,
                                     planes,
                                     kernel_size=kernel_size,
                                     stride=1,
                                     padding=padding,
                                     dilation=dilation,
                                     bias=False)
        self.bn = BatchNorm(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)
class ASPP(nn.Module):

    def __init__(self, inplanes, mid_channels=256, BatchNorm=nn.BatchNorm2d):
        super(ASPP, self).__init__()

        dilations = [1, 6, 12, 18]

        self.aspp1 = _ASPPModule(inplanes,
                                 mid_channels,
                                 1,
                                 padding=0,
                                 dilation=dilations[0],
                                 BatchNorm=BatchNorm)
        self.aspp2 = _ASPPModule(inplanes,
                                 mid_channels,
                                 3,
                                 padding=dilations[1],
                                 dilation=dilations[1],
                                 BatchNorm=BatchNorm)
        self.aspp3 = _ASPPModule(inplanes,
                                 mid_channels,
                                 3,
                                 padding=dilations[2],
                                 dilation=dilations[2],
                                 BatchNorm=BatchNorm)
        self.aspp4 = _ASPPModule(inplanes,
                                 mid_channels,
                                 3,
                                 padding=dilations[3],
                                 dilation=dilations[3],
                                 BatchNorm=BatchNorm)

        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(inplanes, mid_channels, 1, stride=1, bias=False),
            BatchNorm(mid_channels),
            nn.ReLU(),
        )
        self.conv1 = nn.Conv2d(int(mid_channels * 5),
                               mid_channels,
                               1,
                               bias=False)
        self.bn1 = BatchNorm(mid_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5,
                           size=x4.size()[2:],
                           mode='bilinear',
                           align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        return self.dropout(x)

class Depth_net(nn.Module):
    def __init__(self,in_channels=256 ,mid_channels=256 ,context_channels=80 ,depth_channels=112):
        super(Depth_net, self).__init__()
        self.reduce_conv = nn.Sequential(
            nn.Conv2d(in_channels,
                      mid_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.reduce_conv1 = nn.Sequential(
            nn.Conv2d(6, context_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(context_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(context_channels, context_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(context_channels),
            nn.ReLU(inplace=True),

        )
        self.reduce_conv2 = nn.Sequential(
            nn.Conv2d(6+2, depth_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(depth_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(depth_channels, depth_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(depth_channels),
            nn.ReLU(inplace=True),

        )
        self.sigmoid = nn.Sigmoid()
        self.context_conv = nn.Conv2d(mid_channels,
                                      context_channels,
                                      kernel_size=1,
                                      stride=1,
                                      padding=0)
        self.Aspp = nn.Sequential(
        ASPP(mid_channels,mid_channels),
        build_conv_layer(cfg=dict(
            type='DCN',
            in_channels=mid_channels,
            out_channels=mid_channels,
            kernel_size=3,
            padding=1,
            groups=4,
        )),
        nn.Conv2d(mid_channels,
                  depth_channels,
                  kernel_size=1,
                  stride=1,
                  padding=0),
        )
        self.attention = nn.Conv2d(depth_channels*2,
                                      1,
                                      kernel_size=1,
                                      stride=1,
                                      padding=0)
        self.reduce_conv3 = nn.Conv2d(depth_channels*2,
                                      depth_channels,
                                      kernel_size=1,
                                      stride=1,
                                      padding=0)
    def forward(self,img_feature, virtual_depth, virtual_img):
        # img_feature = self.reduce_conv(img_feature)
        # virtual_feature = virtual_feature.unsqueeze(1)
        # virtual_context = self.reduce_conv1(virtual_feature.squeeze(1))
        # virtual_depth = self.reduce_conv2(virtual_feature.squeeze(1))
        # virtual_context = self.sigmoid(virtual_context)
        # virtual_depth = self.sigmoid(virtual_depth)
        # context_feature = self.context_conv(img_feature)
        # context_feature = context_feature * virtual_context

        # depth = self.Aspp(img_feature)
        # depth = depth * virtual_depth
        # img_feature = torch.cat([depth, context_feature], dim=1)
        # return img_feature
        img_feature = self.reduce_conv(img_feature)
        context_feature = self.context_conv(img_feature)
        depth = self.Aspp(img_feature)
        virtual_feature = torch.cat([virtual_depth, virtual_img.max(2)[0]], dim=1)
        virtual_feature = self.reduce_conv2(virtual_feature)
        depth = torch.cat([depth, virtual_feature], dim=1)
        attention = self.attention(depth)
        depth = self.reduce_conv3(depth)*attention

        img_feature = torch.cat([depth, context_feature], dim=1)
        return img_feature

class HoriConv(nn.Module):

    def __init__(self, in_channels, mid_channels, out_channels, cat_dim=0):
        """HoriConv that reduce the image feature
            in height dimension and refine it.

        Args:
            in_channels (int): in_channels
            mid_channels (int): mid_channels
            out_channels (int): output channels
            cat_dim (int, optional): channels of position
                embedding. Defaults to 0.
        """
        super().__init__()

        self.merger = nn.Sequential(
            nn.Conv2d(in_channels + cat_dim,
                      in_channels,
                      kernel_size=1,
                      bias=True),
            nn.Sigmoid(),
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=True),
        )

        self.reduce_conv = nn.Sequential(
            nn.Conv1d(
                in_channels,
                mid_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(inplace=True),
        )

        self.conv1 = nn.Sequential(
            nn.Conv1d(
                mid_channels,
                mid_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(
                mid_channels,
                mid_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(inplace=True),
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(
                mid_channels,
                mid_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(
                mid_channels,
                mid_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(inplace=True),
        )

        self.out_conv = nn.Sequential(
            nn.Conv1d(
                mid_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

    @autocast(False)
    def forward(self, x, pe=None):
        # [N,C,H,W]
        if pe is not None:
            x = self.merger(torch.cat([x, pe], 1))
        else:
            x = self.merger(x)
        x = x.max(2)[0]
        x = self.reduce_conv(x)
        x = self.conv1(x) + x
        x = self.conv2(x) + x
        x = self.out_conv(x)
        return x
class DepthReducer(nn.Module):

    def __init__(self, img_channels=80, mid_channels=112):
        """Module that compresses the predicted
            categorical depth in height dimension

        Args:
            img_channels (int): in_channels
            mid_channels (int): mid_channels
        """
        super().__init__()
        self.vertical_weighter = nn.Sequential(
            nn.Conv2d(img_channels,
                      mid_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, 1, kernel_size=3, stride=1, padding=1),
        )

    @autocast(False)
    def forward(self, feat, depth, img_metas):
        # depth2 = depth[:, 0, :, :]
        vert_weight = self.vertical_weighter(feat).softmax(2)  # [N,1,H,W]
        depth = (depth * vert_weight).sum(2)
        ###### ######

        # filename = img_metas[0]['filename']
        # pts_filename = img_metas[0]['filename'][1]
        # cam_orders = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK',
        #               'CAM_BACK_LEFT']
        # if pts_filename == './data/nuscenes/samples/CAM_FRONT/n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151607012404.jpg':
        #     path = '/home/zcj/MSMDFusion/version/1'
        #     path2 = '/home/zcj/MSMDFusion/version/2'
        #     torch.save(depth2, path)
        #     save_to_file(path2, pts_filename)
        return depth

@MIDDLE_ENCODERS.register_module()
class BEV_Encoder(nn.Module):
    def __init__(self,in_channels=256,
                 mid_channels=256,
                 context_channels=80,
                 depth_channels=112,
                 x_bound=[-54.0, 54.0, 0.6],
                 y_bound=[-54.0, 54.0, 0.6],
                 z_bound=[-5, 3, 8],
                 d_bound=[2.0, 58.0, 0.5],
                 voxel_num=[180, 180, 1],
                 voxel=[-54.0, -54.0, -5],
                 voxel_size=[0.6, 0.6, 8],
                 img_scale=(800, 448),
                 final_scale=(28 ,50)
                 ):
        super().__init__()
        self.grid_conf = {
            'xbound': x_bound,
            'ybound': y_bound,
            'zbound': z_bound,
            'dbound': d_bound,
        }
        self.depthnet = Depth_net(in_channels ,mid_channels ,context_channels ,depth_channels)
        self.depth_channels = depth_channels
        self.img_scale = img_scale
        self.final_scale = final_scale
        self.depth_reducer = DepthReducer(context_channels ,depth_channels)
        self.register_buffer('frustum', self.create_frustum())
        self.voxel_num = voxel_num
        self.voxel = voxel
        self.voxel_size = voxel_size
        self.context_channels = context_channels
        self.horiconv = HoriConv(self.context_channels, 512,
                                 self.context_channels)
        self.register_buffer('bev_anchors', self.create_bev_anchors(self.grid_conf['xbound'],self.grid_conf['ybound']))
    def create_bev_anchors(self, x_bound, y_bound, ds_rate=1):
        """Create anchors in BEV space

        Args:
            x_bound (list): xbound in meters [start, end, step]
            y_bound (list): ybound in meters [start, end, step]
            ds_rate (iint, optional): downsample rate. Defaults to 1.

        Returns:
            anchors: anchors in [W, H, 2]
        """
        x_coords = ((torch.linspace(
            x_bound[0],
            x_bound[1] - x_bound[2] * ds_rate,
            self.voxel_num[0] // ds_rate,
            dtype=torch.float,
        ) + x_bound[2] * ds_rate / 2).view(self.voxel_num[0] // ds_rate,
                                           1).expand(
                                               self.voxel_num[0] // ds_rate,
                                               self.voxel_num[1] // ds_rate))
        y_coords = ((torch.linspace(
            y_bound[0],
            y_bound[1] - y_bound[2] * ds_rate,
            self.voxel_num[1] // ds_rate,
            dtype=torch.float,
        ) + y_bound[2] * ds_rate / 2).view(
            1,
            self.voxel_num[1] // ds_rate).expand(self.voxel_num[0] // ds_rate,
                                                 self.voxel_num[1] // ds_rate))

        anchors = torch.stack([x_coords, y_coords]).permute(1, 2, 0)
        return anchors
    def create_frustum(self):
        # make grid in image plane
        ogfH, ogfW = self.img_scale[0], self.img_scale[1]
        fH, fW = self.final_scale[0], self.final_scale[1]
        d_coords = torch.arange(*self.grid_conf['dbound'], dtype=torch.float).view(-1, 1, 1).expand(-1, fH, fW)
        D, _, _ = d_coords.shape
        x_coords = torch.linspace(0, ogfW - 1, fW, dtype=torch.float).view(1, 1, fW).expand(D, fH, fW)
        y_coords = torch.linspace(0, ogfH - 1, fH, dtype=torch.float).view(1, fH, 1).expand(D, fH, fW)
        paddings = torch.ones_like(d_coords)

        # D x H x W x 3
        frustum = torch.stack((x_coords, y_coords, d_coords, paddings), -1)
        # return nn.Parameter(frustum, requires_grad=False)
        return frustum

    def get_geometry(self, img_metas, B):
        """Transfer points from camera coord to ego coord.

        Args:
            rots(Tensor): Rotation matrix from camera to ego.
            trans(Tensor): Translation matrix from camera to ego.
            intrins(Tensor): Intrinsic matrix.
            post_rots_ida(Tensor): Rotation matrix for ida.
            post_trans_ida(Tensor): Translation matrix for ida
            post_rot_bda(Tensor): Rotation matrix for bda.
            sensor2ego_mats:相机坐标系->车辆坐标系
            intrin_mats:相机内参
            ida_mats:图像数据增强矩阵
            sensor2sensor_mats：关键帧到过渡帧的变化矩阵
            bda_mat:bev特征增强矩阵
        Returns:
            Tensors: points ego coord.
        """
        # undo post-transformation
        # B x N x D x H x W x 3
        device = torch.device('cuda:0')
        points = self.frustum.unsqueeze(0)
        points = points.repeat(B * 6, 1, 1, 1, 1 )
        b_points = []
        for i in range(B):

            img2lidar = torch.inverse(torch.tensor(img_metas[i]['lidar2img'], dtype=torch.float32)).view(1, 6, 1, 1, 1, 4, 4).to(device)
            point = points[i*6:(i+1)*6].unsqueeze(0)
            point = torch.cat(
                (point[:, :, :, :, :, :2] * point[:, :, :, :, :, 2:3],
                 point[:, :, :, :, :, 2:]), 5)
            point = img2lidar.matmul(point.unsqueeze(-1))
            if 'pcd_trans' in img_metas[i] and 'pcd_rotation' in img_metas[i]:
                pcd_trans = torch.tensor(img_metas[i]['pcd_trans']).unsqueeze(0).repeat(6, 1).view(1, 6, 1, 1, 1, 3).to(device)
                pcd_rotation = torch.tensor(img_metas[i]['pcd_rotation']).unsqueeze(0).repeat(6, 1, 1).view(1, 6, 1, 1, 1, 3, 3).to(device)
                point[:, :, :, :, :, 0:3, :] = pcd_rotation.matmul(point[:, :, :, :, :, 0:3, :])+pcd_trans.unsqueeze(-1)
            point = point.squeeze(-1)
            b_points.append(point.squeeze(0))
        points = torch.cat(b_points)
        return points[..., :3]

    def reduce_and_project(self, feature, depth, B, img_metas, virtual_img_depth):
        """reduce the feature and depth in height
            dimension and make BEV feature

        Args:
            feature (Tensor): image feature in [B, C, H, W]
            depth (Tensor): Depth Prediction in [B, D, H, W]
            mats_dict (dict): dictionary that contains intrin-
                and extrin- parameters

        Returns:
            Tensor: BEV feature in B, C, L, L
                 sensor2ego_mats:相机坐标系->车辆坐标系
                 intrin_mats:相机内参
                 ida_mats:图像数据增强矩阵
                 sensor2sensor_mats：关键帧到过渡帧的变化矩阵
                 bda_mat:bev特征增强矩阵
        """
        # [N,112,H,W], [N,256,H,W]


        depth = self.depth_reducer(feature, depth, img_metas)

        feature = self.horiconv(feature)

        depth = depth.permute(0, 2, 1).reshape(B, -1, self.depth_channels)
        feature = feature.permute(0, 2, 1).reshape(B, -1, self.context_channels)
        circle_map, ray_map = self.get_proj_mat(img_metas ,B)
        proj_mat = depth.matmul(circle_map)
        proj_mat = (proj_mat * ray_map).permute(0, 2, 1)
        img_feat_with_depth = proj_mat.matmul(feature)
        img_feat_with_depth = img_feat_with_depth.permute(0, 2, 1).reshape(
            B, -1, *self.voxel_num[:2])

        return img_feat_with_depth

    def get_proj_mat(self, img_metas, B):
        """Create the Ring Matrix and Ray Matrix

        Args:
            mats_dict (dict, optional): dictionary that
                contains intrin- and extrin- parameters.
            Defaults to None.

        Returns:
            tuple: Ring Matrix in [B, D, L, L] and Ray Matrix in [B, W, L, L]
        """

        b_geom_sep = []
        bev_size = int(self.voxel_num[0])  # only consider square BEV
        geom_sep = self.get_geometry(img_metas, B)
        device = torch.device("cuda:0")
        # geom_sep = self.frustum[..., :3].unsqueeze(0)
        geom_sep = (geom_sep - torch.tensor(self.voxel).to(device)) / torch.tensor(self.voxel_size).to(device)
        geom_sep = geom_sep.mean(2).permute(0, 2, 1, 3).contiguous()  # B*Ncam,W,D,2
        for i in range(B):
            b_geom_sep.append(geom_sep[6*i:(i+1)*6].unsqueeze(0))
        geom_sep = torch.cat(b_geom_sep)
        B, Nc, W, D, _ = geom_sep.shape
        geom_sep = geom_sep.long().view(B, Nc * W, D, -1)[..., :2]

        invalid1 = torch.logical_or((geom_sep < 0)[..., 0], (geom_sep < 0)[...,
                                                                           1])
        invalid2 = torch.logical_or((geom_sep > (bev_size - 1))[..., 0],
                                    (geom_sep > (bev_size - 1))[..., 1])
        geom_sep[(invalid1 | invalid2)] = int(bev_size / 2)
        geom_idx = geom_sep[..., 1] * bev_size + geom_sep[..., 0]

        geom_uni = self.bev_anchors[None].repeat([B, 1, 1, 1])
        B, L, L, _ = geom_uni.shape

        circle_map = geom_uni.new_zeros((B, D, L * L))

        ray_map = geom_uni.new_zeros((B, Nc * W, L * L))
        for b in range(B):
            for dir in range(Nc * W):
                ray_map[b, dir, geom_idx[b, dir]] += 1
            for d in range(D):
                circle_map[b, d, geom_idx[b, :, d]] += 1
        null_point = int((bev_size / 2) * (bev_size + 1))
        circle_map[..., null_point] = 0
        ray_map[..., null_point] = 0
        circle_map = circle_map.view(B, D, L * L)
        ray_map = ray_map.view(B, -1, L * L)
        circle_map /= circle_map.max(1)[0].clip(min=1)[:, None]
        ray_map /= ray_map.max(1)[0].clip(min=1)[:, None]

        return circle_map, ray_map

    # def forward(self, img_features, img_metas, virtual_img_depth):
    #     img_feature = img_features[-3]
    #     B_N = img_feature.shape[0]
    #     B = int(B_N / int(6))
    #     mixed_feature = self.depthnet(img_feature, virtual_img_depth)
    #
    #     with autocast(enabled=False):
    #         feature = mixed_feature[:, self.depth_channels:(
    #             self.depth_channels + self.context_channels)].float()
    #         depth = mixed_feature[:, :self.depth_channels].float().softmax(1)
    #         img_feat_with_depth = self.reduce_and_project(
    #             feature, depth, B, img_metas, virtual_img_depth)
    #     return img_feat_with_depth.contiguous()

    def forward(self, img_features, img_metas, virtual_img_depth, virtual_img):
        img_feature = img_features[0]
        B_N = img_feature.shape[0]
        B = int(B_N / int(6))
        mixed_feature = self.depthnet(img_feature, virtual_img_depth, virtual_img)

        with autocast(enabled=False):
            feature = mixed_feature[:, self.depth_channels:(
                self.depth_channels + self.context_channels)].float()
            depth = mixed_feature[:, :self.depth_channels].float().softmax(1)
            img_feat_with_depth = self.reduce_and_project(
                feature, depth, B, img_metas, virtual_img_depth)
        return img_feat_with_depth.contiguous(), depth
