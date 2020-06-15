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

import time
import math

from config import config as cfg

if cfg().network == 'vgg16':
    from model import vgg16 as nn
if cfg().network == 'mobilenet_v2':
    from model import mobilenet_v2_early_element as nn
if cfg().network == 'vgg16v2':
    from model import vgg16v2 as nn
# if cfg().network == 'vgg16_one':
#     from model import vgg16_one as nn
# if cfg().network == 'vgg16_one':
#     from model import vgg16_1010 as nn
# if cfg().network == 'vgg16_one':
#     from model import vgg19 as nn
if cfg().network == 'vgg16_one':
    from model import vgg16_minggu_pre as nn

def read_kitti_cal(calfile):
    """
    Reads the kitti calibration projection matrix (p2) file from disc.

    Args:
        calfile (str): path to single calibration file
    """

    text_file = open(calfile, 'r')

    p2pat = re.compile(('(P2:)\s+(fpat)\s+(fpat)\s+(fpat)\s+(fpat)\s+(fpat)\s+(fpat)\s+(fpat)\s+(fpat)\s+(fpat)' +
                        '\s+(fpat)\s+(fpat)\s+(fpat)\s*\n').replace('fpat', '[-+]?[\d]+\.?[\d]*[Ee](?:[-+]?[\d]+)?'))

    for line in text_file:

        parsed = p2pat.fullmatch(line)

        # bbGt annotation in text format of:
        # cls x y w h occ x y w h ign ang
        if parsed is not None:
            p2 = np.zeros([4, 4], dtype=float)
            p2[0, 0] = parsed.group(2)
            p2[0, 1] = parsed.group(3)
            p2[0, 2] = parsed.group(4)
            p2[0, 3] = parsed.group(5)
            p2[1, 0] = parsed.group(6)
            p2[1, 1] = parsed.group(7)
            p2[1, 2] = parsed.group(8)
            p2[1, 3] = parsed.group(9)
            p2[2, 0] = parsed.group(10)
            p2[2, 1] = parsed.group(11)
            p2[2, 2] = parsed.group(12)
            p2[2, 3] = parsed.group(13)

            p2[3, 3] = 1

    text_file.close()

    return p2

def post__processing(calibration_file, ):
    p2 = read_kitti_cal(calibration_file)

def predict(args):
    # complie models
    model = nn.network()
    # model.load_weights('3dbox_weights_1st.hdf5')
    model.load_weights(args.w)

    # KITTI_train_gen = KITTILoader(subset='training')
    dims_avg, _ =KITTILoader(subset='training').get_average_dimension()

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
        #calibration_file = test_calib_path + i.replace('png', 'txt')
        calibration_file = os.path.join('/media/ferdyan/NewDisk/Trajectory_Final/bbox_3d/0000.txt')

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
                    patch_d = dpth[ymin : ymax, xmin : xmax]
                    patch_d = cv2.resize(patch_d, (cfg().norm_h, cfg().norm_w))
                    patch_d -= np.array([[[103.939, 116.779, 123.68]]])
                    # extend it to match the training dimension
                    patch_d = np.expand_dims(patch_d, 0)

                    # one
                    #prediction = model.predict([patch])

                    # two
                    prediction = model.predict([patch, patch_d])

                    dim = prediction[0][0]
                    bin_anchor = prediction[1][0]
                    bin_confidence = prediction[2][0]

                    # update with predict dimension
                    dims = dims_avg[obj.name] + dim
                    obj.h, obj.w, obj.l = np.array([round(dim, 2) for dim in dims])

                    # update with predicted alpha, [-pi, pi]
                    obj.alpha = recover_angle(bin_anchor, bin_confidence, cfg().bin)
                    print("obj.alpha = ", obj.alpha)

                    if math.isnan(obj.alpha) :
                        continue

                    # compute global and local orientation
                    print("P2 = ", P2)
                    print("P2[0, 2] = ", P2[0, 2])
                    obj.rot_global, rot_local = compute_orientaion(P2, obj)

                    # compute and update translation, (x, y, z)
                    obj.tx, obj.ty, obj.tz = translation_constraints(P2, obj, rot_local)

                    # TAMBAHAN AING
                    # print("obj.name = ", obj.name)
                    # print("obj.truncation = ", obj.truncation)
                    # print("obj.occlusion = ", obj.occlusion)
                    # print("obj.alpha = ", obj.alpha)
                    # print("obj.xmin = ", obj.xmin)
                    # print("obj.ymin = ", obj.ymin)
                    # print("obj.xmax = ", obj.xmax)
                    # print("obj.ymax = ", obj.ymax)
                    # print("obj.h = ", obj.h)
                    # print("obj.w = ", obj.w)
                    # print("obj.l = ", obj.l)
                    # print("obj.tx = ", obj.tx)
                    # print("obj.ty = ", obj.ty)
                    # print("obj.tz = ", obj.tz)
                    # print("obj.rot_global = ", obj.rot_global)
                    
                    # output prediction label
                    output_line = obj.member_to_list()
                    output_line.append(1.0)
                    # Write regressed 3D dim and orientation to file
                    output_line = ' '.join([str(item) for item in output_line]) + '\n'

                    # output_organized = format(obj.name) + ' ' + format(obj.truncation) + ' ' + format(obj.occlusion) + ' ' + format(obj.alpha) 
                    # + ' ' + format(obj.xmin) + ' ' + format(obj.ymin) + ' ' + format(obj.xmax) + ' ' + format(obj.ymax) + ' ' + format(obj.h) 
                    # + ' ' + format(obj.w) + ' ' + format(obj.l) + ' ' + format(obj.tx) + ' ' + format(obj.ty) + ' ' + format(obj.tz) + ' ' + format(obj.rot_global)
                    
                    output_organized = str(obj.name) + ' ' + str(obj.truncation) + ' ' + str(obj.occlusion) + ' ' + str(obj.alpha) + ' ' + str(obj.xmin) + ' ' + str(obj.ymin) + ' ' + str(obj.xmax) + ' ' + str(obj.ymax) + ' ' + str(obj.h) + ' ' + str(obj.w) + ' ' + str(obj.l) + ' ' + str(obj.tx) + ' ' + str(obj.ty) + ' ' + str(obj.tz) + ' ' + str(obj.rot_global) + ' 0.7' + '\n'

                    predict.write(output_organized)
                    print('Write predicted labels for: ' + str(i))
                    print("")
                    print("")

    end_time = time.time()
    process_time = (end_time - start_time) / len(val_imgs)
    print(process_time)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for prediction',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '-dir', type=str, default='/media/ferdyan/NewDisk/3d_bbox_depth/train_dataset/', help='File to predict')
    #parser.add_argument('-d', '-dir', type=str, default='/media/ferdyan/NewDisk/3d_bounding_box/train_dataset/', help='File to predict')
    parser.add_argument('-a', '-dataset', type=str, default='tracklet', help='training dataset or tracklet')
    parser.add_argument('-w', '-weight', type=str, default='/media/ferdyan/NewDisk/3d_bbox_depth/model00000312.hdf5', help ='Load trained weights')
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