import sys
ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if ros_path in sys.path:
    sys.path.remove(ros_path)
import cv2

import numpy as np
import tensorflow as tf
optimizer = tf.keras.optimizers
callbacks = tf.keras.callbacks

from data_processing.KITTI_dataloader import KITTILoader
from data_processing.preprocessing import orientation_confidence_flip

from utils.data_generation import data_gen
from utils.loss import orientation_loss

from config import config as cfg

if cfg().network == 'vgg16':
    from model import vgg16 as nn
if cfg().network == 'mobilenet_v2':
    from model import mobilenet_v2_early_element as nn
if cfg().network == 'vgg16v2':
    from model import vgg16v2 as nn
if cfg().network == 'vgg16_one':
    from model import vgg16_one as nn

from tensorflow.python.keras import backend as K


# def generator_two_img(X1, X2, y, batch_size):
#     genX1 = gen.flow(X1, y,  batch_size=cfg().batch_size, seed=1)
#     genX2 = gen.flow(X2, y, batch_size=cfg().batch_size, seed=1)
#     while True:
#         X1i = genX1.next()
#         X2i = genX2.next()
#         yield [X1i[0], X2i[0]], X1i[1]

# def scheduler(epoch):
#     if epoch%10==0 and epoch!=0:
#         lr = K.get_value(K.eval(model.optimizer.lr))
#         K.set_value(K.eval(model.optimizer.lr, lr*.8))
#         print("lr changed to {}".format(lr*.8))
#     return K.get_value(K.eval(model.optimizer.lr))

def train():
    KITTI_train_gen = KITTILoader(subset='training')
    dim_avg, dim_cnt = KITTI_train_gen.get_average_dimension()

    new_data = orientation_confidence_flip(KITTI_train_gen.image_data, dim_avg)

    model = nn.network()
    #model.load_weights('model00000296.hdf5')

    early_stop = callbacks.EarlyStopping(monitor='val_loss', min_delta=0.001, patience=10, mode='min', verbose=1)
    checkpoint = callbacks.ModelCheckpoint('model{epoch:08d}.hdf5', monitor='val_loss', verbose=1, save_best_only=False, mode='min', period=1)
    tensorboard = callbacks.TensorBoard(log_dir='logs/', histogram_freq=0, write_graph=True, write_images=False)

    

    all_examples = len(new_data)
    trv_split = int(cfg().split * all_examples) # train val split

    train_gen = data_gen(new_data[: trv_split])
    valid_gen = data_gen(new_data[trv_split : all_examples])

    print("READY FOR TRAINING")

    train_num = int(np.ceil(trv_split / cfg().batch_size))
    valid_num = int(np.ceil((all_examples - trv_split) / cfg().batch_size))

    #gen_flow = gen_flow_for_two_inputs(X_train, X_angle_train, y_train)

    # choose the minimizer to be sgd
    # minimizer = optimizer.SGD(lr=0.0001, momentum = 0.9)
    minimizer = optimizer.Adam(lr=0.0001)

    # multi task learning
    model.compile(optimizer=minimizer,  #minimizer,
                  loss={'dimensions': 'mean_squared_error', 'orientation': orientation_loss, 'confidence': 'categorical_crossentropy'},
                  loss_weights={'dimensions': 1., 'orientation': 10., 'confidence': 5.})

    print("####################################################")
    print(K.get_value(model.optimizer.lr))

    # Tambahan aing
    def scheduler(epoch):
        if epoch%10==0 and epoch!=0:
            lr = K.get_value(model.optimizer.lr)
            K.set_value(model.optimizer.lr, lr*.8)
            print("lr changed to {}".format(lr*.8))
            print("lr = ", K.get_value(model.optimizer.lr))
        return K.get_value(model.optimizer.lr)

    lr_sched = callbacks.LearningRateScheduler(scheduler)


    # d:0.0088 o:0.0042, c:0.0098
    # steps_per_epoch=train_num,
    # validation_steps=valid_num,
    # callbacks=[early_stop, checkpoint, tensorboard],
    model.fit_generator(generator=train_gen,
                        steps_per_epoch=train_num,
                        epochs=500,
                        verbose=1,
                        validation_data=valid_gen,
                        validation_steps=valid_num,
                        shuffle=True,
                        callbacks=[checkpoint, tensorboard, lr_sched],
                        max_queue_size=3)

if __name__ == '__main__':
    train()