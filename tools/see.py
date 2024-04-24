import open3d as o3d
import os
from tqdm import tqdm
import numpy as np
import pickle
from open3d import geometry

def _draw_bboxes(bbox3d,
                 vis,
                 bbox_color=(0, 1, 0),
                 points_in_box_color=(1, 0, 0),
                 rot_axis=2,
                 center_mode='lidar_bottom',):
    # in_box_color = np.array(points_in_box_color)
    for i in range(bbox3d.bev.shape[0]):
        center = bbox3d.center
        dim = bbox3d.dim
        yaw = np.zeros(3)
        yaw[rot_axis] = -bbox3d[i, 6]
        # 在 Open3D 中需要将 yaw 朝向角转为旋转矩阵
        rot_mat = geometry.get_rotation_matrix_from_xyz(yaw)
        # 底部中心点转为几何中心店
        if center_mode == 'lidar_bottom':
            center[rot_axis] += dim[
                                    rot_axis] / 2
        elif center_mode == 'camera_bottom':
            center[rot_axis] -= dim[
                                    rot_axis] / 2
        box3d = geometry.OrientedBoundingBox(center, rot_mat, dim)

        line_set = geometry.LineSet.create_from_oriented_bounding_box(box3d)
        line_set.paint_uniform_color(bbox_color)
        # 在 Visualizer 中绘制 Box
        vis.add_geometry(line_set)

        # 更改 3D Box 中的点云颜色
        # if pcd is not None and mode == 'xyz':
        #     indices = box3d.get_point_indices_within_bounding_box(pcd.points)
        #     points_colors[indices] = in_box_color

            # 更新点云颜色
    # if pcd is not None:
    #     pcd.colors = o3d.utility.Vector3dVector(points_colors)
    #     vis.update_geometry(pcd)
def read_display_bin_pc(path, pred_points, box_data):
    points = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
    points = points[:, :3]  # open3d 只需xyz 与pcl不同

    # 将array格式的点云转换为open3d的点云格式,若直接用open3d打开的点云数据，则不用转换
    pcd = o3d.geometry.PointCloud()  # 传入3d点云格式
    pcd.points = o3d.utility.Vector3dVector(points)  # 转换格式
    # 设置颜色 只能是0 1 如[1,0,0]代表红色为既r
    pcd.paint_uniform_color([1, 0, 0])
    # 创建窗口对象
    vis = o3d.visualization.Visualizer()
    # 创建窗口,设置窗口名称
    vis.create_window(window_name="point_cloud")
    # 设置点云渲染参数
    opt = vis.get_render_option()
    # 设置背景色（这里为白色）
    opt.background_color = np.array([255, 255, 255])
    # 设置渲染点的大小
    opt.point_size = 1.0
    # 添加点云
    vis.add_geometry(pcd)
    _draw_bboxes(box_data['pts_bbox']['boxes_3d'], vis)
    vis.run()

pkl_path = "/home/zcj/MSMDFusion/vision/1.pkl"
pkl = open(pkl_path, 'rb')
box_datas = pickle.load(pkl)
filenames = os.listdir('/home/zcj/MSMDFusion/vision')
filenames = tqdm(filenames)
for i, filename in enumerate(filenames):
    # pred_path = "/home/zcj/MSMDFusion/vision/" + filename + '/' + filename + '_pred.ply'
    pred_path = '/home/zcj/MSMDFusion/vision2/n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151603547590/n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151603547590_gt.ply'
    pred_pcd = o3d.io.read_point_cloud(pred_path)
    o3d.visualization.draw_geometries([pred_pcd])
    pred_out = np.asarray(pred_pcd.points)
    point_path = '/home/zcj/MSMDFusion/data/nuscenes/samples/LIDAR_TOP/' + filename + '.pcd.bin'
    box_data = box_datas[i]
    read_display_bin_pc(point_path, pred_out, box_data)
    # print(pcd)
    # o3d.visualization.draw_geometries([pcd])

# textured_mesh = o3d.io.read_triangle_mesh('/home/zcj/MSMDFusion/vision/n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151604048025/n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151604048025_points.obj')
#
# textured_mesh.compute_vertex_normals()
# o3d.visualization.draw_geometries([textured_mesh], window_name="Open3D1")

