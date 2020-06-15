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

from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Flatten, Dense, Dropout
from keras.layers.normalization import BatchNormalization

def l2_normalize(x):
    return tf.nn.l2_normalize(x, axis=2)

def network():
    inputs = layers.Input(shape=(cfg().norm_h, cfg().norm_w, 3))
    inputs_depth = layers.Input(shape=(cfg().norm_h, cfg().norm_w, 3))


    x = layers.Conv2D(64, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block1_conv1')(inputs)
    x = layers.Conv2D(64, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block1_conv2')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = layers.Conv2D(128, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block2_conv1')(x)
    x = layers.Conv2D(128, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block2_conv2')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv1')(x)
    x = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv2')(x)
    x = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv3')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv1')(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv2')(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv3')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv1')(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv2')(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv3')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    x = layers.GlobalAveragePooling2D()(x)

    model1 = tf.keras.Model([inputs], [x], name='vgg16_1')
    model1.load_weights('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')
    model1.summary()

    for layer in model1.layers[:]:
      layer.trainable = False

    for layer in model1.layers:
      print(layer, layer.trainable)

    y = layers.Conv2D(64, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block1_conv1_d')(inputs_depth)
    y = layers.Conv2D(64, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block1_conv2_d')(y)
    y = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool_d')(y)

    # Block 2
    y = layers.Conv2D(128, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block2_conv1_d')(y)
    y = layers.Conv2D(128, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block2_conv2_d')(y)
    y = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool_d')(y)

    # Block 3
    y = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv1_d')(y)
    y = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv2_d')(y)
    y = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv3_d')(y)
    y = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool_d')(y)

    # Block 4
    y = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv1_d')(y)
    y = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv2_d')(y)
    y = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv3_d')(y)
    y = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool_d')(y)

    # Block 5
    y = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv1_d')(y)
    y = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv2_d')(y)
    y = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv3_d')(y)
    y = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool_d')(y)

    y = layers.GlobalAveragePooling2D()(y)

    model2 = tf.keras.Model([inputs_depth], [y], name='vgg16_1')
    model2.load_weights('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')
    model2.summary()

    for layer in model2.layers[:]:
      layer.trainable = False

    for layer in model2.layers:
      print(layer, layer.trainable)

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

    model = tf.keras.Model([inputs, inputs_depth], [dimensions, orientation, confidence])
    model.summary()

    return model
