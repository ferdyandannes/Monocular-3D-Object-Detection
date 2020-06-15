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
from keras.applications.vgg16 import VGG16
import tensorflow as tf

def l2_normalize(x):
    return tf.nn.l2_normalize(x, axis=2)

def network():

    # inputs = layers.Input(shape=(cfg().norm_h, cfg().norm_w, 3))
    # inputs_depth = layers.Input(shape=(cfg().norm_h, cfg().norm_w, 3))

    # ##################################################### 1 ############################################################
    # # Block 1__
    # x = layers.Conv2D(64, (3,3), padding='same', kernel_initializer='he_normal', kernel_regularizer=reg.l2(1e-4), name='block1_conv1')(inputs)
    # x = layers.Activation('relu')(x)
    # x = layers.Conv2D(64, (3,3), padding='same', kernel_initializer='he_normal', kernel_regularizer=reg.l2(1e-4), name='block1_conv2')(x)
    # x = layers.Activation('relu')(x)
    # x = layers.MaxPooling2D(strides=(2,2), name='block1_pool')(x)

    # # Block 2
    # x = layers.Conv2D(128, (3,3), padding='same', kernel_initializer='he_normal', kernel_regularizer=reg.l2(1e-4), name='block2_conv1')(x)
    # x = layers.Activation('relu')(x)
    # x = layers.Conv2D(128, (3,3), padding='same', kernel_initializer='he_normal', kernel_regularizer=reg.l2(1e-4), name='block2_conv2')(x)
    # x = layers.Activation('relu')(x)
    # x = layers.MaxPooling2D(strides=(2,2), name='block2_pool')(x)

    # # Block 3
    # x = layers.Conv2D(256, (3,3), padding='same', kernel_initializer='he_normal', kernel_regularizer=reg.l2(1e-4), name='block3_conv1')(x)
    # x = layers.Activation('relu')(x)
    # x = layers.Conv2D(256, (3,3), padding='same', kernel_initializer='he_normal', kernel_regularizer=reg.l2(1e-4), name='block3_conv2')(x)
    # x = layers.Activation('relu')(x)
    # x = layers.Conv2D(256, (3,3), padding='same', kernel_initializer='he_normal', kernel_regularizer=reg.l2(1e-4), name='block3_conv3')(x)
    # x = layers.Activation('relu')(x)
    # x = layers.MaxPooling2D(strides=(2,2), name='block3_pool')(x)

    # # Block 4
    # x = layers.Conv2D(512, (3,3), padding='same', kernel_initializer='he_normal', kernel_regularizer=reg.l2(1e-4), name='block4_conv1')(x)
    # x = layers.Activation('relu')(x)
    # x = layers.Conv2D(512, (3,3), padding='same', kernel_initializer='he_normal', kernel_regularizer=reg.l2(1e-4), name='block4_conv2')(x)
    # x = layers.Activation('relu')(x)
    # x = layers.Conv2D(512, (3,3), padding='same', kernel_initializer='he_normal', kernel_regularizer=reg.l2(1e-4), name='block4_conv3')(x)
    # x = layers.Activation('relu')(x)
    # x = layers.MaxPooling2D(strides=(2,2), name='block4_pool')(x)

    # # Block 5
    # x = layers.Conv2D(512, (3,3), padding='same', kernel_initializer='he_normal', kernel_regularizer=reg.l2(1e-4), name='block5_conv1')(x)
    # x = layers.Activation('relu')(x)
    # x = layers.Conv2D(512, (3,3), padding='same', kernel_initializer='he_normal', kernel_regularizer=reg.l2(1e-4), name='block5_conv2')(x)
    # x = layers.Activation('relu')(x)
    # x = layers.Conv2D(512, (3,3), padding='same', kernel_initializer='he_normal', kernel_regularizer=reg.l2(1e-4), name='block5_conv3')(x)
    # x = layers.Activation('relu')(x)
    # x = layers.MaxPooling2D(strides=(2,2), name='block5_pool')(x)

    # # layers.Flatten
    # #x = layers.Flatten(name='Flatten')(x)


    # ##################################################### 2 ############################################################
    # # Block 1__
    # y = layers.Conv2D(64, (3,3), padding='same', kernel_initializer='he_normal', kernel_regularizer=reg.l2(1e-4), name='block1_conv1_d')(inputs_depth)
    # y = layers.Activation('relu')(y)
    # y = layers.Conv2D(64, (3,3), padding='same', kernel_initializer='he_normal', kernel_regularizer=reg.l2(1e-4), name='block1_conv2_d')(y)
    # y = layers.Activation('relu')(y)
    # y = layers.MaxPooling2D(strides=(2,2), name='block1_pool_d')(y)

    # # Block 2
    # y = layers.Conv2D(128, (3,3), padding='same', kernel_initializer='he_normal', kernel_regularizer=reg.l2(1e-4), name='block2_conv1_d')(y)
    # y = layers.Activation('relu')(y)
    # y = layers.Conv2D(128, (3,3), padding='same', kernel_initializer='he_normal', kernel_regularizer=reg.l2(1e-4), name='block2_conv2_d')(y)
    # y = layers.Activation('relu')(y)
    # y = layers.MaxPooling2D(strides=(2,2), name='block2_pool_d')(y)

    # # Block 3
    # y = layers.Conv2D(256, (3,3), padding='same', kernel_initializer='he_normal', kernel_regularizer=reg.l2(1e-4), name='block3_conv1_d')(y)
    # y = layers.Activation('relu')(y)
    # y = layers.Conv2D(256, (3,3), padding='same', kernel_initializer='he_normal', kernel_regularizer=reg.l2(1e-4), name='block3_conv2_d')(y)
    # y = layers.Activation('relu')(y)
    # y = layers.Conv2D(256, (3,3), padding='same', kernel_initializer='he_normal', kernel_regularizer=reg.l2(1e-4), name='block3_conv3_d')(y)
    # y = layers.Activation('relu')(y)
    # y = layers.MaxPooling2D(strides=(2,2), name='block3_pool_d')(y)

    # # Block 4
    # y = layers.Conv2D(512, (3,3), padding='same', kernel_initializer='he_normal', kernel_regularizer=reg.l2(1e-4), name='block4_conv1_d')(y)
    # y = layers.Activation('relu')(y)
    # y = layers.Conv2D(512, (3,3), padding='same', kernel_initializer='he_normal', kernel_regularizer=reg.l2(1e-4), name='block4_conv2_d')(y)
    # y = layers.Activation('relu')(y)
    # y = layers.Conv2D(512, (3,3), padding='same', kernel_initializer='he_normal', kernel_regularizer=reg.l2(1e-4), name='block4_conv3_d')(y)
    # y = layers.Activation('relu')(y)
    # y = layers.MaxPooling2D(strides=(2,2), name='block4_pool_d')(y)

    # # Block 5
    # y = layers.Conv2D(512, (3,3), padding='same', kernel_initializer='he_normal', kernel_regularizer=reg.l2(1e-4), name='block5_conv1_d')(y)
    # y = layers.Activation('relu')(y)
    # y = layers.Conv2D(512, (3,3), padding='same', kernel_initializer='he_normal', kernel_regularizer=reg.l2(1e-4), name='block5_conv2_d')(y)
    # y = layers.Activation('relu')(y)
    # y = layers.Conv2D(512, (3,3), padding='same', kernel_initializer='he_normal', kernel_regularizer=reg.l2(1e-4), name='block5_conv3_d')(y)
    # y = layers.Activation('relu')(y)
    # y = layers.MaxPooling2D(strides=(2,2), name='block5_pool_d')(y)

    # # layers.Flatten
    # #y = layers.Flatten(name='Flatten_d')(y)


    vgg16_model_rgb = VGG16(include_top=False, weights='imagenet', input_shape=(224,224,3))

    for layer_rgb in vgg16_model_rgb.layers:
      layer_rgb.trainable = False

    for layer_rgb in vgg16_model_rgb.layers:
      print(layer_rgb, layer_rgb.trainable)


    vgg16_model_depth = VGG16(include_top=False, weights='imagenet', input_shape=(224,224,3))

    for layer_depth in vgg16_model_depth.layers:
        layer_depth.name = layer_depth.name + str("_2")

    for layer_depth in vgg16_model_depth.layers:
      layer_depth.trainable = False

    for layer_depth in vgg16_model_depth.layers:
      print(layer_depth, layer_depth.trainable)

    ############################################### COMBINE ##################################################
    # xy = layers.Concatenate()([x, y])
    xy = Multiply()([vgg16_model_rgb.output, vgg16_model_depth.output])
    xy = Flatten(name='Flatten_d')(xy)

    # Dimensions branch
    dimensions = Dense(512)(xy)
    dimensions = LeakyReLU(alpha=0.1)(dimensions)
    dimensions = Dropout(0.5)(dimensions)
    dimensions = Dense(3, name='dimensions')(dimensions)

    # Orientation branch
    orientation = Dense(256)(xy)
    orientation = LeakyReLU(alpha=0.1)(orientation)
    orientation = Dropout(0.5)(orientation)
    orientation = Dense(cfg().bin * 2)(orientation)
    #orientation = LeakyReLU(alpha=0.1)(orientation)
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
