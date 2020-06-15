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
from keras.layers import Flatten, Dense, Dropout, Reshape
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU

from keras.layers import Input, Dense
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Flatten, Dropout, Reshape, Lambda
from keras.models import Model
from keras.applications.vgg16 import VGG16
import tensorflow as tf

def l2_normalize(x):
    return tf.nn.l2_normalize(x, axis=2)

def network():
    vgg16_model = VGG16(include_top=False, weights='imagenet', input_shape=(224,224,3))

    for layer in vgg16_model.layers:
      layer.trainable = False

    for layer in vgg16_model.layers:
      print(layer, layer.trainable)

    x = Flatten()(vgg16_model.output)

    # Dimensions branch
    dimensions = Dense(512)(x)
    dimensions = LeakyReLU(alpha=0.1)(dimensions)
    dimensions = Dropout(0.5)(dimensions)
    dimensions = Dense(3)(dimensions)
    dimensions = LeakyReLU(alpha=0.1, name='dimensions')(dimensions)

    orientation = Dense(256)(x)
    orientation = LeakyReLU(alpha=0.1)(orientation)
    orientation = Dropout(0.5)(orientation)
    orientation = Dense(cfg().bin * 2)(orientation)
    orientation = LeakyReLU(alpha=0.1)(orientation)
    orientation = Reshape((cfg().bin, -1))(orientation)
    orientation = Lambda(l2_normalize, name='orientation')(orientation)

    confidence = Dense(256)(x)
    confidence = LeakyReLU(alpha=0.1)(confidence)
    confidence = Dropout(0.5)(confidence)
    confidence = Dense(cfg().bin, activation='softmax', name='confidence')(confidence)


    model = Model(vgg16_model.input, [dimensions, orientation, confidence])
    model.summary()

    return model
