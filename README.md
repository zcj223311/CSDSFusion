# Cross-supervised LiDAR-camera Fusion for 3D Object Detection  
![image](https://github.com/user-attachments/assets/9a6103a8-f615-4f04-838b-c5134368749e)  
Fusing LiDAR and camera information is essential for achieving accurate and reliable 3D object
detection in autonomous driving systems. Due to inherent differences between different modalities, seeking
an efficient and accurate fusion method is of great importance. Recently, significant progress has been
made in 3D object detection methods based on lift-splat-shot (LSS-based) approaches. However, inaccurate
depth estimation and substantial semantic information loss remain significant factors limiting the accuracy
of 3D detection. In this paper, we propose a cross-fusion framework under a dual spatial representation,
by integrating information in different spatial representations, namely bird’s-eye view (BEV) and camera
view, and establishing soft links to fully utilize the information carried by different modalities. It consists
of two important components, gated lidar supervised BEV (GLS-BEV) and multi-attention cross fusion
(MACF) modules. The former achieves accurate depth estimation by supervising the transformation of
LiDAR data with clear depth into the image space, constructing point cloud features in vehicle’s perspective.
The latter utilizes three sub-attention modules with different roles to achieve cross-modal interaction within
the same space, effectively reducing semantic loss. On the nuScenes benchmark, our proposed method
achieves outstanding 3D object detection results with 71.8 mAP and 74.2 NDS.   
#Installation  
```
git clone https://github.com/zcj223311/CSDSFusion.git  
cd CSDSFusion
```
## Detection  
For basic installation, please refer to [getting_started.md](docs/getting_started.md) for installation.  
You can find requirements in [requirements.txt](requirements.txt)  
**Notice:** 
- [spconv-2.x](https://github.com/traveller59/spconv) is required for its ```sparse_add``` op.
- You should manually add mmcv register to [spconv library file](https://github.com/traveller59/spconv/blob/v2.1.21/spconv/pytorch/conv.py) following [this example](https://github.com/SxJyJay/MSMDFusion/blob/main/bug_fix/conv.py)

## Data Preparation

**Step 1**: Please refer to the [official site](https://github.com/ADLab-AutoDrive/BEVFusion/blob/main/docs/getting_started.md) for prepare nuscenes data. After data preparation, you will be able to see the following directory structure:
```
mmdetection3d
├── mmdet3d
├── tools
├── configs
├── data
│   ├── nuscenes
│   │   ├── maps
│   │   ├── samples
│   │   ├── sweeps
│   │   ├── v1.0-test
|   |   ├── v1.0-trainval
│   │   ├── nuscenes_database
│   │   ├── nuscenes_infos_train.pkl
│   │   ├── nuscenes_infos_val.pkl
│   │   ├── nuscenes_infos_test.pkl
│   │   ├── nuscenes_dbinfos_train.pkl
```
## train
For training, you need to first train a pure LiDAR backbone, such as TransFusion-L. Then, you can merge the checkpoints from pretrained TransFusion-L and ResNet-50 as suggested [here](https://github.com/XuyangBai/TransFusion/issues/7#issuecomment-1115499891). We also provide a merged 1-st stage checkpoint [here](https://pan.baidu.com/s/1Lj35HXc2Ajv0yWEH6H8g_A?pwd=69i7)(extraction code: 69i7)
```
# 1-st stage training
sh ./tools/dist_train.sh ./configs/transfusion_nusc_voxel_L.py 8
# 2-nd stage training
sh ./tools/dist_train.sh ./configs/CSDSFusion_nusc_voxel_LC.py 8
```
**Notice**: When training the 1-st stage of TransFusion-L, please follow the copy-and-paste fade strategy as suggested [here](https://github.com/XuyangBai/TransFusion/issues/7#issuecomment-1114113329).

## Acknowlegement

We sincerely thank the authors of [mmdetection3d](https://github.com/open-mmlab/mmdetection3d), [CenterPoint](https://github.com/tianweiy/CenterPoint), [TransFusion](https://github.com/XuyangBai/TransFusion), [MVP](https://github.com/tianweiy/MVP), [BEVFusion](https://github.com/ADLab-AutoDrive/BEVFusion) and [BEVFusion](https://github.com/mit-han-lab/bevfusion) for open sourcing their methods.


