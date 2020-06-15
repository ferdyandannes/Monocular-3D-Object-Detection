###############
import sys
print(sys.path)
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
print(sys.path)
import cv2
import copy
import numpy as np

# shape =  (21, 0, 3)

ymin = 21
ymax = 34
xmin = 20
xmax = 50


img = cv2.imread('/media/ferdyan/NewDisk/3d_bounding_box/tes/training/0000003840.png')
img = copy.deepcopy(img[ymin:ymax + 1, xmin:xmax + 1]).astype(np.float32)
img_temp = copy.deepcopy(img)

print(img.shape)

x = img.shape[0]
y = img.shape[1]

if x == 0 :
	img[0] = copy.deepcopy(img_temp[20:20+1,:])
	print("1")

if y == 0 :
    img[1] = copy.deepcopy(img_temp[:,21:21+1])
    print("2")

print(img.shape)

