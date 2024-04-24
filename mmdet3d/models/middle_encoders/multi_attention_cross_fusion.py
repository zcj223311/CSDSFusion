from mmcv.runner import auto_fp16
from mmdet3d.core import voxel
from torch import nn as nn
from torch.nn import functional as F
import torch
import ipdb
from spconv.pytorch import functional as Fsp
from mmcv.cnn import ConvModule, build_conv_layer, kaiming_init ,build_norm_layer

from mmdet3d.ops import SparseBasicBlock, make_sparse_convmodule
from mmdet3d.ops import furthest_point_sample, gather_points, ball_query
# from mmdet3d.ops import spconv as spconv
import spconv.pytorch as spconv
from ..registry import MIDDLE_ENCODERS


class SIM(nn.Module):

    def __init__(self, channel_up, channel_down):
        super().__init__()

        self.param_free_norm = nn.InstanceNorm2d(channel_up, affine=False, track_running_stats=False)
        # self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)
        # The dimension of the intermediate embedding space. Yes, hardcoded.
        # nhidden = 128
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(channel_down, channel_down, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Sequential(
            nn.Conv2d(channel_down, channel_down, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        self.mlp_beta = nn.Conv2d(channel_down, channel_down, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(num_features=channel_down)
        self.down_channel = nn.Sequential(
            nn.Conv2d(channel_down, channel_down, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

    def forward(self, segmap, x):

        normalized = self.param_free_norm(x)
        segmap = self.down_channel(segmap)
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)
        out = self.bn(normalized * (1 + gamma)) + beta
        return out


class Fusion_module(nn.Module):

    def __init__(self, middle_channel=128, out_channel=128):
        super(Fusion_module, self).__init__()

        self.Recalibrate = nn.Sequential(
            nn.AvgPool2d(kernel_size=1 ,stride=1 ,padding=0),
            nn.Conv2d(2 * middle_channel, middle_channel//2, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(middle_channel//2, eps=0.001, momentum=0.01, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channel//2, 2 * middle_channel, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(2 * middle_channel),
            nn.Sigmoid(),
        ).cuda()

        self.channel_agg = nn.Sequential(
            nn.Conv2d(2 * middle_channel, middle_channel, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(middle_channel, eps=0.001, momentum=0.01, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            ).cuda()

        self.local_att = nn.Sequential(
            nn.Conv2d(middle_channel, middle_channel//2, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(middle_channel//2, eps=0.001, momentum=0.01, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channel//2, middle_channel, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(middle_channel, eps=0.001, momentum=0.01, affine=True, track_running_stats=True),
        ).cuda()

        self.global_att = nn.Sequential(
            nn.AvgPool2d(kernel_size=1, stride=1, padding=0),
            nn.Conv2d(middle_channel, middle_channel//2, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(middle_channel//2, eps=0.001, momentum=0.01, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channel//2, middle_channel, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(middle_channel, eps=0.001, momentum=0.01, affine=True, track_running_stats=True),
        ).cuda()
        self.sigmoid = nn.Sigmoid()

    def forward(self, v_f, r_f):
        device = v_f.device
        _, c, _, _ = v_f.shape
        input = torch.cat([v_f, r_f], dim=1).to(device)
        recal_w = self.Recalibrate(input)
        recal_input = recal_w * input
        recal_input = recal_input + input
        x1, x2 = torch.split(recal_input, c, dim=1)
        agg_input = self.channel_agg(recal_input)
        local_w = self.local_att(agg_input)
        global_w = self.global_att(agg_input)
        w = self.sigmoid(local_w * global_w)
        xo = torch.cat([w * x1, (1 - w) * x2], dim=1)
        return xo
class virtual_2_real(nn.Module):
    def __init__(self,
                 in_channels_virtual=256,
                 in_channels_real=256,
                 middle_channels=128,
                 out_channels=128,
                 bias='auto',
                 ):
        super().__init__()
        block1 = []
        block2 = []
        block1.append(build_conv_layer(
            dict(type='Conv2d'),
            in_channels_virtual,
            middle_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=bias, ))
        block2.append(build_conv_layer(
            dict(type='Conv2d'),
            in_channels_real,
            middle_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=bias, ))
        self.fused_features = Fusion_module(middle_channels, out_channels)
        self.block1 = nn.Sequential(*block1)
        self.block2 = nn.Sequential(*block2)
        # a = 128
        # self.sim1 = SIM(a, a)
        # self.sim2 = SIM(a, a)


    def forward(self, x1, x2):
        device = x1.device
        v_f = x1.to(device)
        r_f = x2.to(device)
        v_f = self.block1[0](v_f)
        r_f = self.block2[0](r_f)
        fused_feature = self.fused_features(v_f, r_f).to(device)
        return fused_feature

@MIDDLE_ENCODERS.register_module()
class MultiAttentionCrossFusion(nn.Module):
    def __init__(self,
                 in_channels_virtual_3D=80,
                 in_channels_real_3D=512,
                 in_channels_virtual_img=128,
                 in_channels_real_img=256,
                 use_img=True,
                 order=('conv', 'norm', 'act'),
                 block_type='conv_module'):
        super().__init__()
        assert block_type in ['conv_module', 'basicblock']
        self.order = order
        self.fp16_enabled = False
        self.virtual_2_real = virtual_2_real(in_channels_virtual_3D, in_channels_real_3D, 128, 512)
        self.virtual_2_real2 = virtual_2_real(in_channels_virtual_img, in_channels_real_img, 128, 128)
        self.use_img = use_img
        block_sum = []

        self.make_num_features_same = nn.ModuleList(block_sum)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels_virtual_img + in_channels_real_img, 128, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128, eps=0.001, momentum=0.01, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128, eps=0.001, momentum=0.01, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Conv2d(128, 128, 3, stride=1, padding=1, bias=False)
        self.attention = nn.Conv2d(128, 1, 1, stride=1, bias=False)

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels_real_3D + 80, 256, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256, eps=0.001, momentum=0.01, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128, eps=0.001, momentum=0.01, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
        )
        self.conv4 = nn.Conv2d(128, 128, 3, stride=1, padding=1, bias=False)
        self.attention1 = nn.Conv2d(128, 1, 1, stride=1, bias=False)
        self.upchannel = nn.Sequential(nn.Conv2d(6+2, 128, 3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(128, eps=0.001, momentum=0.01, affine=True, track_running_stats=True),
                                       nn.ReLU(inplace=True),)

    # @auto_fp16(apply_to=('voxel_features', ))
    def forward(self, voxel_3D_feature, voxel_2D_feature, img_feats, virtual_img, depth):
        fused_lidar_features = self.virtual_2_real(voxel_2D_feature, voxel_3D_feature)
        fused_img_feats = []
        if self.use_img:
            virtural_img_feats, _ = virtual_img.max(dim=2)
            fused_img_features = self.upchannel(torch.cat([virtural_img_feats, depth], dim=1))
            fused_img_feats.append(self.virtual_2_real2(fused_img_features, img_feats[0]))
        else:
            fused_img_feats.append(img_feats)
        ###### cat #####
        # feature = torch.cat([voxel_3D_feature, voxel_2D_feature], dim=1)
        # feature = self.conv3(feature)
        # attention = self.attention1(feature)
        # feature = self.conv4(feature)
        # fused_lidar_features = feature*attention
        #
        # img_feat = torch.tensor(img_feats[0])
        # feature1 = torch.cat([virtual_img.view(B * 6, 224, 112, 200), img_feat], dim=1)
        # feature1 = self.conv1(feature1)
        # attention1 = self.attention(feature1)
        # feature1 = self.conv2(feature1)
        # fused_img_feats.append(feature1*attention1)

        # for i in range(len(img_feats)-1):
        #     fused_img_feats.append(img_feats[i+1])
        # fused_img_feats = tuple(fused_img_feats)
        return fused_lidar_features, fused_img_feats
        # return fused_lidar_features
    