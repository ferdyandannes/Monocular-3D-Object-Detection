

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

from utils.post_processing import gen_3D_box,draw_3D_box,draw_2D_box
from utils.process_data import get_cam_data, get_dect2D_data

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

# Construct the network
model = nn.network()

model.load_weights(r'/media/ferdyan/NewDisk/3d_bbox_depth/Weights_train/weight_7Okt_Sore/model00000125.hdf5')

image_dir = '/media/ferdyan/NewDisk/3d_bounding_box/train_dataset/2011_09_26/own/image_02/data/'
calib_dir = '/media/ferdyan/NewDisk/3d_bounding_box/train_dataset/2011_09_26/own/calib_02/'
box2d_dir = '/media/ferdyan/NewDisk/3d_bounding_box/train_dataset/2011_09_26/own/label_02/'

classes = ['Car']
cls_to_ind = {cls:i for i,cls in enumerate(classes)}

dims_avg = np.loadtxt(r'/media/ferdyan/NewDisk/3d_bbox_depth/voc_dims.txt',delimiter=',')


all_image = sorted(os.listdir(image_dir))
# np.random.shuffle(all_image)



for f in all_image:
    image_file = image_dir + f
    box2d_file = box2d_dir + f.replace('png', 'txt')
    calib_file = calib_dir + f.replace('png', 'txt')

    cam_to_img = get_cam_data(calib_file)
    fx = cam_to_img[0][0]
    u0 = cam_to_img[0][2]
    v0 = cam_to_img[1][2]

    img = cv2.imread(image_file)

    print("image_file = ", image_file)
    print("box2d_file = ", box2d_file)

    dect2D_data,box2d_reserved = get_dect2D_data(box2d_file,classes)

    for data in dect2D_data:
        cls = data[0]
        box_2D = np.asarray(data[1],dtype=np.float)
        xmin = box_2D[0]
        ymin = box_2D[1]
        xmax = box_2D[2]
        ymax = box_2D[3]

        patch = img[int(ymin):int(ymax), int(xmin):int(xmax)]
        patch = cv2.resize(patch, (224, 224))
        patch = patch - np.array([[[103.939, 116.779, 123.68]]])
        patch = np.expand_dims(patch, 0)

        prediction = model.predict(patch)

        # compute dims
        dims = dims_avg[cls_to_ind[cls]] + prediction[0][0]

        # Transform regressed angle
        box2d_center_x = (xmin + xmax) / 2.0
        # Transfer arctan() from (-pi/2,pi/2) to (0,pi)
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

    for cls,box in box2d_reserved:
        draw_2D_box(img,box)

    cv2.imshow(f, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    #cv2.imwrite('output/'+ f.replace('png','jpg'), img)
