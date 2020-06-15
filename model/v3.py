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

from tf.keras.applications.vgg16 import VGG16

def l2_normalize(x):
    return tf.nn.l2_normalize(x, axis=2)

def network():
    vgg16_model = VGG16(include_top=False, weights='imagenet', input_shape=(224,224,3))

    for layer in vgg16_model.layers:
      layer.trainable = False

    for layer in vgg16_model.layers:
      print(layer, layer.trainable)

    x = layers.Flatten()(vgg16_model.output)

    # Dimensions branch
    dimensions = layers.Dense(512, name='d_fc_1')(x)
    dimensions = layers.LeakyReLU(alpha=0.1)(dimensions)
    dimensions = layers.Dropout(0.5)(dimensions)
    dimensions = layers.Dense(3, name='d_fc_2')(dimensions)
    dimensions = layers.LeakyReLU(alpha=0.1, name='dimensions')(dimensions)

    # Orientation branch
    orientation = layers.Dense(256, name='o_fc_1')(x)
    orientation = layers.LeakyReLU(alpha=0.1)(orientation)
    orientation = layers.Dropout(0.5)(orientation)
    orientation = layers.Dense(cfg().bin * 2, name='o_fc_2')(orientation)
    orientation = layers.LeakyReLU(alpha=0.1)(orientation)
    orientation = layers.Reshape((cfg().bin, -1))(orientation)
    orientation = layers.Lambda(l2_normalize, name='orientation')(orientation)

    # Confidence branch
    confidence = layers.Dense(256, name='c_fc_1')(x)
    confidence = layers.LeakyReLU(alpha=0.1)(confidence)
    confidence = layers.Dropout(0.5)(confidence)
    confidence = layers.Dense(cfg().bin, activation='softmax', name='confidence')(confidence)


    model = tf.keras.Model([inputs], [dimensions, orientation, confidence])
    model.summary()

    return model
