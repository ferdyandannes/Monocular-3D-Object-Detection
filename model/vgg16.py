'''
Refs:
    Very Deep Convolutional Networks for Large-Scale Image Recognition -- https://arxiv.org/abs/1409.1556
'''

import tensorflow as tf
layers = tf.keras.layers
reg = tf.keras.regularizers

from config import config as cfg
from tensorflow.python.keras.layers import Lambda;
from tensorflow.python.keras.layers import Multiply

def l2_normalize(x):
    return tf.nn.l2_normalize(x, axis=2)

def network():

    inputs = layers.Input(shape=(cfg().norm_h, cfg().norm_w, 3))
    inputs_depth = layers.Input(shape=(cfg().norm_h, cfg().norm_w, 3))

    ##################################################### 1 ############################################################
    # Block 1__
    x = layers.Conv2D(64, (3,3), padding='same', kernel_initializer='he_normal', kernel_regularizer=reg.l2(1e-4), name='block1_conv1')(inputs)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(64, (3,3), padding='same', kernel_initializer='he_normal', kernel_regularizer=reg.l2(1e-4), name='block1_conv2')(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(strides=(2,2), name='block1_pool')(x)

    # Block 2
    x = layers.Conv2D(128, (3,3), padding='same', kernel_initializer='he_normal', kernel_regularizer=reg.l2(1e-4), name='block2_conv1')(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(128, (3,3), padding='same', kernel_initializer='he_normal', kernel_regularizer=reg.l2(1e-4), name='block2_conv2')(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(strides=(2,2), name='block2_pool')(x)

    # Block 3
    x = layers.Conv2D(256, (3,3), padding='same', kernel_initializer='he_normal', kernel_regularizer=reg.l2(1e-4), name='block3_conv1')(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(256, (3,3), padding='same', kernel_initializer='he_normal', kernel_regularizer=reg.l2(1e-4), name='block3_conv2')(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(256, (3,3), padding='same', kernel_initializer='he_normal', kernel_regularizer=reg.l2(1e-4), name='block3_conv3')(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(strides=(2,2), name='block3_pool')(x)

    # Block 4
    x = layers.Conv2D(512, (3,3), padding='same', kernel_initializer='he_normal', kernel_regularizer=reg.l2(1e-4), name='block4_conv1')(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(512, (3,3), padding='same', kernel_initializer='he_normal', kernel_regularizer=reg.l2(1e-4), name='block4_conv2')(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(512, (3,3), padding='same', kernel_initializer='he_normal', kernel_regularizer=reg.l2(1e-4), name='block4_conv3')(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(strides=(2,2), name='block4_pool')(x)

    # Block 5
    x = layers.Conv2D(512, (3,3), padding='same', kernel_initializer='he_normal', kernel_regularizer=reg.l2(1e-4), name='block5_conv1')(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(512, (3,3), padding='same', kernel_initializer='he_normal', kernel_regularizer=reg.l2(1e-4), name='block5_conv2')(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(512, (3,3), padding='same', kernel_initializer='he_normal', kernel_regularizer=reg.l2(1e-4), name='block5_conv3')(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(strides=(2,2), name='block5_pool')(x)

    # layers.Flatten
    x = layers.Flatten(name='Flatten')(x)


    ##################################################### 2 ############################################################
    # Block 1__
    y = layers.Conv2D(64, (3,3), padding='same', kernel_initializer='he_normal', kernel_regularizer=reg.l2(1e-4), name='block1_conv1_d')(inputs_depth)
    y = layers.Activation('relu')(y)
    y = layers.Conv2D(64, (3,3), padding='same', kernel_initializer='he_normal', kernel_regularizer=reg.l2(1e-4), name='block1_conv2_d')(y)
    y = layers.Activation('relu')(y)
    y = layers.MaxPooling2D(strides=(2,2), name='block1_pool_d')(y)

    # Block 2
    y = layers.Conv2D(128, (3,3), padding='same', kernel_initializer='he_normal', kernel_regularizer=reg.l2(1e-4), name='block2_conv1_d')(y)
    y = layers.Activation('relu')(y)
    y = layers.Conv2D(128, (3,3), padding='same', kernel_initializer='he_normal', kernel_regularizer=reg.l2(1e-4), name='block2_conv2_d')(y)
    y = layers.Activation('relu')(y)
    y = layers.MaxPooling2D(strides=(2,2), name='block2_pool_d')(y)

    # Block 3
    y = layers.Conv2D(256, (3,3), padding='same', kernel_initializer='he_normal', kernel_regularizer=reg.l2(1e-4), name='block3_conv1_d')(y)
    y = layers.Activation('relu')(y)
    y = layers.Conv2D(256, (3,3), padding='same', kernel_initializer='he_normal', kernel_regularizer=reg.l2(1e-4), name='block3_conv2_d')(y)
    y = layers.Activation('relu')(y)
    y = layers.Conv2D(256, (3,3), padding='same', kernel_initializer='he_normal', kernel_regularizer=reg.l2(1e-4), name='block3_conv3_d')(y)
    y = layers.Activation('relu')(y)
    y = layers.MaxPooling2D(strides=(2,2), name='block3_pool_d')(y)

    # Block 4
    y = layers.Conv2D(512, (3,3), padding='same', kernel_initializer='he_normal', kernel_regularizer=reg.l2(1e-4), name='block4_conv1_d')(y)
    y = layers.Activation('relu')(y)
    y = layers.Conv2D(512, (3,3), padding='same', kernel_initializer='he_normal', kernel_regularizer=reg.l2(1e-4), name='block4_conv2_d')(y)
    y = layers.Activation('relu')(y)
    y = layers.Conv2D(512, (3,3), padding='same', kernel_initializer='he_normal', kernel_regularizer=reg.l2(1e-4), name='block4_conv3_d')(y)
    y = layers.Activation('relu')(y)
    y = layers.MaxPooling2D(strides=(2,2), name='block4_pool_d')(y)

    # Block 5
    y = layers.Conv2D(512, (3,3), padding='same', kernel_initializer='he_normal', kernel_regularizer=reg.l2(1e-4), name='block5_conv1_d')(y)
    y = layers.Activation('relu')(y)
    y = layers.Conv2D(512, (3,3), padding='same', kernel_initializer='he_normal', kernel_regularizer=reg.l2(1e-4), name='block5_conv2_d')(y)
    y = layers.Activation('relu')(y)
    y = layers.Conv2D(512, (3,3), padding='same', kernel_initializer='he_normal', kernel_regularizer=reg.l2(1e-4), name='block5_conv3_d')(y)
    y = layers.Activation('relu')(y)
    y = layers.MaxPooling2D(strides=(2,2), name='block5_pool_d')(y)

    # layers.Flatten
    y = layers.Flatten(name='Flatten_d')(y)

    ############################################### COMBINE ##################################################
    # xy = layers.Concatenate()([x, y])
    xy = Multiply()([x, y])

    # Dimensions branch
    dimensions = layers.Dense(512, name='d_fc_1')(xy)
    dimensions = layers.LeakyReLU(alpha=0.1)(dimensions)
    dimensions = layers.Dropout(0.5)(dimensions)
    dimensions = layers.Dense(3, name='d_fc_2')(dimensions)
    dimensions = layers.LeakyReLU(alpha=0.1, name='dimensions')(dimensions)

    # Orientation branch
    orientation = layers.Dense(256, name='o_fc_1')(xy)
    orientation = layers.LeakyReLU(alpha=0.1)(orientation)
    orientation = layers.Dropout(0.5)(orientation)
    orientation = layers.Dense(cfg().bin * 2, name='o_fc_2')(orientation)
    orientation = layers.LeakyReLU(alpha=0.1)(orientation)
    orientation = layers.Reshape((cfg().bin, -1))(orientation)
    orientation = layers.Lambda(l2_normalize, name='orientation')(orientation)

    # Confidence branch
    confidence = layers.Dense(256, name='c_fc_1')(xy)
    confidence = layers.LeakyReLU(alpha=0.1)(confidence)
    confidence = layers.Dropout(0.5)(confidence)
    confidence = layers.Dense(cfg().bin, activation='softmax', name='confidence')(confidence)

    # Build model
    model = tf.keras.Model([inputs, inputs_depth], [dimensions, orientation, confidence])
    model.summary()

    return model
