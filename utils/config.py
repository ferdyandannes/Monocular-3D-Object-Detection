import os
class config():
    def __init__(self):
        # Todo: set base_dir to kitti/image_2
        self.base_dir = '/media/ferdyan/NewDisk/3d_bbox_depth/train_dataset/'

        # Todo: set the base network: vgg16, vgg16_conv or mobilenet_v2
        self.network = 'vgg16'

        # set the bin size
        self.bin = 2

        # set the train/val split
        self.split = 0.8

        # set overlapping
        self.overlap = 0.1

        # set jittered
        self.jit = 3

        # set the normalized image size
        self.norm_w = 224
        self.norm_h = 224

        # set the batch size
        self.batch_size = 8

        # set the categories
        #self.KITTI_cat = ['Car', 'Cyclist', 'Pedestrian']
        self.KITTI_cat = ['Car']
