###############
import sys
print(sys.path)
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
print(sys.path)
import cv2
##############

import os
import argparse
import numpy as np
from utils.read_dir import ReadDir
from data_processing.KITTI_dataloader import KITTILoader
from utils.correspondece_constraint import *

from post_processing import gen_3D_box,draw_3D_box,draw_2D_box

import time
import math

from config import config as cfg

if cfg().network == 'vgg16':
    from model import vgg16 as nn
if cfg().network == 'mobilenet_v2':
    from model import mobilenet_v2_early_element as nn
if cfg().network == 'vgg16v2':
    from model import vgg16v2 as nn
if cfg().network == 'vgg16_one':
    from model import vgg16_minggu_pre as nn

# TAMBAHAN AING

def get_cam_data(calib_file):
    for line in open(calib_file):
        if 'P2:' in line:
            cam_to_img = line.strip().split(' ')
            cam_to_img = np.asarray([float(number) for number in cam_to_img[1:]])
            cam_to_img = np.reshape(cam_to_img, (3, 4))
            return cam_to_img

calib_file = '/media/ferdyan/NewDisk/3d_bounding_box/train_dataset/2011_09_26/own/calib_02/0000000001.txt'
cam_to_img = get_cam_data(calib_file)
fx = cam_to_img[0][0]
u0 = cam_to_img[0][2]
v0 = cam_to_img[1][2]

dims = np.loadtxt(r'/media/ferdyan/NewDisk/3d_bbox_depth/voc_dims.txt',delimiter=',')


def predict(args):
    # complie models
    model = nn.network()
    # model.load_weights('3dbox_weights_1st.hdf5')
    model.load_weights(args.w)

    # KITTI_train_gen = KITTILoader(subset='training')
    dims_avg, _ =KITTILoader(subset='training').get_average_dimension()

    print("dims_avg = ", dims_avg)
    # dims_avg =  {'Car': array([1.52608343, 1.62858987, 3.88395449])}

    # list all the validation images
    if args.a == 'training':
        all_imgs = sorted(os.listdir(test_image_dir))
        val_index = int(len(all_imgs)* cfg().split)
        val_imgs = all_imgs[val_index:]

    else:
        val_imgs = sorted(os.listdir(test_image_dir))

    start_time = time.time()

    for i in val_imgs:
        image_file = test_image_dir + i
        depth_file = test_depth_dir + i
        label_file = test_label_dir + i.replace('png', 'txt')
        prediction_file = prediction_path + i.replace('png', 'txt')
        calibration_file = test_calib_path + i.replace('png', 'txt')
        #calibration_file = os.path.join('/media/ferdyan/NewDisk/Trajectory_Final/bbox_3d/0000.txt')

        # write the prediction file
        with open(prediction_file, 'w') as predict:
            img = cv2.imread(image_file)
            img = np.array(img, dtype='float32')

            dpth = cv2.imread(depth_file)
            dpth = np.array(dpth, dtype='float32')

            P2 = np.array([])
            for line in open(calibration_file):
                if 'P2' in line:
                    P2 = line.split(' ')
                    P2 = np.asarray([float(i) for i in P2[1:]])
                    P2 = np.reshape(P2, (3,4))

            for line in open(label_file):
                line = line.strip().split(' ')
                #print("line = ", line)
                obj = detectionInfo(line)
                xmin = int(obj.xmin)
                xmax = int(obj.xmax)
                ymin = int(obj.ymin)
                ymax = int(obj.ymax)

                box2d = [xmin, ymin, xmax, ymax]
                box_2D = np.asarray(box2d,dtype=np.float)

                if obj.name in cfg().KITTI_cat:
                    # cropped 2d bounding box
                    if xmin == xmax or ymin == ymax:
                        continue
                    # 2D detection area RGB image
                    patch = img[ymin : ymax, xmin : xmax]
                    patch = cv2.resize(patch, (cfg().norm_h, cfg().norm_w))
                    patch -= np.array([[[103.939, 116.779, 123.68]]])
                    # extend it to match the training dimension
                    patch = np.expand_dims(patch, 0)

                    # 2D detection area depth map
                    #patch_d = dpth[ymin : ymax, xmin : xmax]
                    #patch_d = cv2.resize(patch_d, (cfg().norm_h, cfg().norm_w))
                    #patch_d -= np.array([[[103.939, 116.779, 123.68]]])
                    # extend it to match the training dimension
                    #patch_d = np.expand_dims(patch_d, 0)

                    # one
                    prediction = model.predict([patch])

                    # two
                    #prediction = model.predict([patch, patch_d])


                    # TAMBAHAN AING
                    # Transform regressed angle
                    box2d_center_x = (xmin + xmax) / 2.0
                    theta_ray = np.arctan(fx /(box2d_center_x - u0))

                    if theta_ray<0:
                        theta_ray = theta_ray+np.pi


                    max_anc = np.argmax(prediction[2][0])
                    anchors = prediction[1][0][max_anc]

                    if anchors[1] > 0:
                        angle_offset = np.arccos(anchors[0])
                    else:
                        angle_offset = -np.arccos(anchors[0])

                    bin_num = prediction[2][0].shape[0]
                    wedge = 2. * np.pi / bin_num
                    theta_loc = angle_offset + max_anc * wedge

                    theta = theta_loc + theta_ray
                    # object's yaw angle
                    yaw = np.pi/2 - theta

                    points2D = gen_3D_box(yaw, dims, cam_to_img, box_2D)
                    draw_3D_box(img, points2D)

        cv2.imshow('f', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        #cv2.imwrite('output/'+ f.replace('png','jpg'), img)



    end_time = time.time()
    process_time = (end_time - start_time) / len(val_imgs)
    print(process_time)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for prediction',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parser.add_argument('-d', '-dir', type=str, default='/media/ferdyan/NewDisk/3d_bbox_depth/train_dataset/', help='File to predict')
    parser.add_argument('-d', '-dir', type=str, default='/media/ferdyan/NewDisk/3d_bounding_box/train_dataset/', help='File to predict')
    parser.add_argument('-a', '-dataset', type=str, default='tracklet', help='training dataset or tracklet')
    parser.add_argument('-w', '-weight', type=str, default='/media/ferdyan/NewDisk/3d_bbox_depth/Weights_train/weight_7Okt_Sore/model00000125.hdf5', help ='Load trained weights')
    args = parser.parse_args()

    # Todo: subset = 'training' or 'tracklet'
    dir = ReadDir(args.d, subset=args.a,
                  tracklet_date='2011_09_26', tracklet_file='own')
    test_label_dir = dir.label_dir
    test_image_dir = dir.image_dir
    test_depth_dir = dir.depth_dir
    test_calib_path = dir.calib_dir
    prediction_path = dir.prediction_dir

    if not os.path.exists(prediction_path):
        os.mkdir(prediction_path)

    predict(args)