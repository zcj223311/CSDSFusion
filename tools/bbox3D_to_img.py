
import argparse
import copy
import mmcv
import torch
from mmcv import Config, DictAction
import numpy as np
from mmdet3d.datasets import build_dataset
import cv2
import pickle
import re


def plot_rect3d_on_img(img,
                       num_rects,
                       rect_corners,
                       color=(0, 255, 0),
                       thickness=1):

    line_indices = ((0, 1), (0, 3), (0, 4), (1, 2), (1, 5), (3, 2), (3, 7),
                    (4, 5), (4, 7), (2, 6), (5, 6), (6, 7))
    # filter boxes out of range
    h,w,c = img.shape
    for i in range(num_rects):
        corners = rect_corners[i].astype(np.int)
        for idx, corner in enumerate(corners):
            corners[idx][0] = w if corner[0] > w else corner[0]
            corners[idx][0] = 0 if corner[0] < 0 else corner[0]
            corners[idx][1] = w if corner[1] > h else corner[1]
            corners[idx][1] = 0 if corner[1] < 0 else corner[1]
        # draw
        for start, end in line_indices:
            cv2.line(img, (corners[start, 0], corners[start, 1]),
                     (corners[end, 0], corners[end, 1]), color, thickness,
                     cv2.LINE_AA)

    return img.astype(np.uint8)



def draw_lidar_bbox3d_on_img(bboxes3d,
                             raw_img,
                             lidar2img_rt,
                             out_path,
                             color=(0, 255, 0),
                             thickness=1):
    """Project the 3D bbox on 2D plane and draw on input image.

    Args:
        bboxes3d (:obj:`LiDARInstance3DBoxes`):
            3d bbox in lidar coordinate system to visualize.
        raw_img (numpy.array): The numpy array of image.
        lidar2img_rt (numpy.array, shape=[4, 4]): The projection matrix
            according to the camera intrinsic parameters.
        img_metas (dict): Useless here.
        color (tuple[int], optional): The color to draw bboxes.
            Default: (0, 255, 0).
        thickness (int, optional): The thickness of bboxes. Default: 1.
    """
    img = raw_img.copy()
    corners_3d = bboxes3d.corners.cpu().numpy()
    num_bbox = corners_3d.shape[0]
    pts_4d = np.concatenate(
        [corners_3d.reshape(-1, 3),
         np.ones((num_bbox * 8, 1))], axis=-1)
    lidar2img_rt = copy.deepcopy(lidar2img_rt).reshape(4, 4)
    if isinstance(lidar2img_rt, torch.Tensor):
        lidar2img_rt = lidar2img_rt.cpu().numpy()
    # pts_2d = pts_4d @ lidar2img_rt.T
    lidar2img_rt = lidar2img_rt[np.newaxis, :, :].repeat(pts_4d.shape[0], axis=0)
    pts_4d = pts_4d[:, :, np.newaxis]
    pts_2d = lidar2img_rt@pts_4d
    pts_2d = pts_2d.squeeze(-1)
    # pts_2d = lidar2img_rt.matmul(pts_4d)
    pts_2d[:, 2] = np.clip(pts_2d[:, 2], a_min=1e-5, a_max=1e5)
    pts_2d[:, 0] /= pts_2d[:, 2]
    pts_2d[:, 1] /= pts_2d[:, 2]
    imgfov_pts_2d = pts_2d[..., :2].reshape(num_bbox, 8, 2)
    img = plot_rect3d_on_img(img, num_bbox, imgfov_pts_2d, color, thickness)
    cv2.imwrite(out_path, img)

def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('--config', default="/home/zcj/BEVFusion/configs/bevfusion/bevf_tf_4x8_6e_nusc.py",help='train config file path')
    parser.add_argument('--pkl_path', default="/home/zcj/BEVFusion/results/1.pkl")
    parser.add_argument('--out_path', default="/home/zcj/BEVFusion/vision2img/")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    datasets = [build_dataset(cfg.data.test)]
    datas = datasets[0].data_infos

    pkl_path = args.pkl_path
    pkl = open(pkl_path, 'rb')
    box_datas = pickle.load(pkl)
    for i in range(len(datas)):
        box_data = box_datas[i]
        pts_bbox = box_data['pts_bbox']
        score_box = pts_bbox['scores_3d']
        boxes = pts_bbox['boxes_3d']
        inds = score_box > 0.1
        boxes_in = boxes[inds]

        data = datas[i]
        lidar_path = data['lidar_path']
        out_lidar_path = re.split('/', lidar_path)
        cams = data['cams']
        # lidar2img_rts = []
        cam_orders = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
        for z in range(len(cams)):
            cam_path = cams[cam_orders[z]]['data_path']
            cam2lidar_r = cams[cam_orders[z]]['sensor2lidar_rotation']
            cam2lidar_t = cams[cam_orders[z]]['sensor2lidar_translation']
            intrinsic = cams[cam_orders[z]]['cam_intrinsic']
            lidar2cam_r = np.linalg.inv(cam2lidar_r)
            lidar2cam_t = cam2lidar_t @ lidar2cam_r.T
            lidar2cam_rt = np.eye(4)
            lidar2cam_rt[:3, :3] = lidar2cam_r.T
            lidar2cam_rt[3, :3] = -lidar2cam_t
            # intrinsic = cam_info['cam_intrinsic']
            viewpad = np.eye(4)
            viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
            lidar2img_rt = (viewpad @ lidar2cam_rt.T)
            img = cv2.imread(cam_path)
            out_path = args.out_path + str(i) + '/' + cam_orders[z]
            mmcv.mkdir_or_exist(out_path)

            out_path = out_path + '/' + out_lidar_path[-1] + '.jpg'
            draw_lidar_bbox3d_on_img(boxes_in, img, lidar2img_rt, out_path)

if __name__ == '__main__':
    main()