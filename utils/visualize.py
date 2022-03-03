
import os
from re import I
import tensorflow.compat.v1 as tf
# import tensorflow as tf
import math
import numpy as np
import itertools
import cv2

tf.enable_eager_execution()

import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_point_cloud(img, pointcloud):
    y = 1500 - pointcloud[:, 0] * 10
    x = 750 - pointcloud[:, 1] * 10
    x = x.astype(np.int)
    y = y.astype(np.int)
    for x_, y_ in zip(x, y):
        img = cv2.circle(img, (x_, y_), 1, (0, 0, 0))
    return img

def draw_bbox(img, bboxes, labels, scores):
    i = 0
    for box, label, score in zip(bboxes, labels, scores):
        # print('i:', i)
        # print('heading angle:', box[-1].item())
        angle = box[-1]
        angle = float('%.1f' % angle)
        if score < 0.3:
            continue
        corners = center_box_to_corners(box)[:4, :2]
        x = 1500 - corners[:, 0] * 10
        y = 750 - corners[:, 1] * 10
        z = y
        y = x.astype(np.int)
        x = z.astype(np.int) - 750
        if label == 0:
            color = (0, 0, 255)
        elif label == 1:
            color = (0, 255, 0)
        elif label == 2:
            color = (255, 0, 0)
        else:
            color = (0, 255, 255)
        img = cv2.line(img, (x[0], y[0]), (x[1], y[1]), color=color, thickness=3)
        img = cv2.line(img, (x[1], y[1]), (x[2], y[2]), color=color, thickness=3)
        img = cv2.line(img, (x[2], y[2]), (x[3], y[3]), color=color, thickness=3)
        img = cv2.line(img, (x[3], y[3]), (x[0], y[0]), color=color, thickness=3)
        # cv2.putText(img, '{}'.format(angle), (x[1], y[1]+20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        i += 1
    return img

def center_box_to_corners(box):
    pos_x, pos_y, pos_z, dim_x, dim_y, dim_z = box[:6]
    yaw = box[-1]
    # yaw = np.pi-yaw
    half_dim_x, half_dim_y, half_dim_z = dim_x / 2.0, dim_y / 2.0, dim_z / 2.0
    corners = np.array([[half_dim_x, half_dim_y, -half_dim_z],
                        [half_dim_x, -half_dim_y, -half_dim_z],
                        [-half_dim_x, -half_dim_y, -half_dim_z],
                        [-half_dim_x, half_dim_y, -half_dim_z],
                        [half_dim_x, half_dim_y, half_dim_z],
                        [half_dim_x, -half_dim_y, half_dim_z],
                        [-half_dim_x, -half_dim_y, half_dim_z],
                        [-half_dim_x, half_dim_y, half_dim_z]])
    transform_matrix = np.array([
        [np.cos(yaw), -np.sin(yaw), 0, pos_x],
        [np.sin(yaw), np.cos(yaw), 0, pos_y],
        [0, 0, 1.0, pos_z],
        [0, 0, 0, 1.0],
    ])
    corners = (transform_matrix[:3, :3] @ corners.T + transform_matrix[:3, [3]]).T
    return corners

if __name__ == '__main__':
    import pickle
    # detection_pkl = pickle.load(open('work_dirs/robosense_bs2_no_dpsampler/prediction.pkl', 'rb'))
    # detection_pkl = pickle.load(open('work_dirs/small/robosense_2_cls/prediction.pkl', 'rb'))
    # detection_pkl = pickle.load(open('work_dirs/big/robosense_2_cls/prediction.pkl', 'rb'))
    # detection_pkl = pickle.load(open('work_dirs/big/robosense_3_cls_no_dpsampler/prediction.pkl', 'rb'))
    detection_pkl = pickle.load(open('./data/WAYMO/prediction.pkl', 'rb'))

    lidar_path = 'data/Robosense/test/lidar'
    # lidar_path = 'data/Robosense_2/val/lidar'
    pcd_dir_list = os.listdir(lidar_path)
    # pcd_dir_list = os.listdir(pcd_dir)
    pcd_num_list = [pcd_name.split('_')[-1].split('.')[0] for pcd_name in pcd_dir_list if
                    pcd_name.split('.')[-1] == 'pkl']
    pcd_num_array = np.array(pcd_num_list).astype(np.int16)
    sort_idx = np.argsort(pcd_num_array)
    pcd_dir_list = np.array(pcd_dir_list)[sort_idx]
    from tqdm import tqdm
    for i in tqdm(range(len(pcd_dir_list))):
        # if i<200:
        #     continue
        pkl = pcd_dir_list[i]
        pointcloud = pickle.load(open(os.path.join(lidar_path, pkl), 'rb'))['lidars']['points_xyz']
        detection_results = detection_pkl[pkl]
        bboxes = detection_results['box3d_lidar']
        scores = detection_results['scores']
        labels = detection_results['label_preds']
        img = np.ones((3000, 1500, 3)) * 255
        img = draw_point_cloud(img, pointcloud)
        img = draw_bbox(img, bboxes, labels, scores)
        cv2.imwrite('output_test/{}.jpg'.format(pkl), img)
        i += 1
        # if i>=300:
        #     break