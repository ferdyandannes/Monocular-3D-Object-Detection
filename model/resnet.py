'''
Refs:
    Very Deep Convolutional Networks for Large-Scale Image Recognition -- https://arxiv.org/abs/1409.1556
'''

import tensorflow as tf
layers = tf.keras.layers
reg = tf.keras.regularizers

from config import config as cfg
#from tensorflow.python.keras.layers import Lambda;
#from tensorflow.python.keras.layers import Multiply

from keras.layers import Multiply

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
#from keras.applications.vgg19 import VGG19
from keras.applications.resnet50 import ResNet50
import tensorflow as tf

def l2_normalize(x):
    return tf.nn.l2_normalize(x, axis=2)

def network():

    #vgg16_model_rgb = VGG19(include_top=False, weights='imagenet', input_shape=(224,224,3))
    vgg16_model_rgb = ResNet50(include_top=False, weights='imagenet', input_shape=(224,224,3))

    for layer_rgb in vgg16_model_rgb.layers:
      layer_rgb.trainable = False

    for layer_rgb in vgg16_model_rgb.layers:
      print(layer_rgb, layer_rgb.trainable)

    x = Flatten(name='Flatten_rgb')(vgg16_model_rgb.output)

    #vgg16_model_depth = VGG19(include_top=False, weights='imagenet', input_shape=(224,224,3))
    vgg16_model_depth = ResNet50(include_top=False, weights='imagenet', input_shape=(224,224,3))

    for layer_depth in vgg16_model_depth.layers:
        layer_depth.name = layer_depth.name + str("_2")

    for layer_depth in vgg16_model_depth.layers:
      layer_depth.trainable = False

    for layer_depth in vgg16_model_depth.layers:
      print(layer_depth, layer_depth.trainable)

    y = Flatten(name='Flatten_depth')(vgg16_model_depth.output)

    ############################################### COMBINE ##################################################
    # xy = layers.Concatenate()([x, y])
    xy = Multiply()([x, y])
    #xy = Flatten(name='Flatten_d')(xy)

    # Dimensions branch
    dimensions = Dense(512)(xy)
    dimensions = LeakyReLU(alpha=0.1)(dimensions)
    dimensions = Dropout(0.5)(dimensions)
    dimensions = Dense(3, name='d_fc_2')(dimensions)
    dimensions = LeakyReLU(alpha=0.1, name='dimensions')(dimensions)

    # Orientation branch
    orientation = Dense(256)(xy)
    orientation = LeakyReLU(alpha=0.1)(orientation)
    orientation = Dropout(0.5)(orientation)
    orientation = Dense(cfg().bin * 2)(orientation)
    orientation = LeakyReLU(alpha=0.1)(orientation)
    orientation = Reshape((cfg().bin, -1))(orientation)
    orientation = Lambda(l2_normalize, name='orientation')(orientation)

    # Confidence branch
    confidence = Dense(256)(xy)
    confidence = LeakyReLU(alpha=0.1)(confidence)
    confidence = Dropout(0.5)(confidence)
    confidence = Dense(cfg().bin, activation='softmax', name='confidence')(confidence)

    print("a")

    # Build model
    model = Model([vgg16_model_rgb.input, vgg16_model_depth.input], [dimensions, orientation, confidence])
    model.summary()

    return model
