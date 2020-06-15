'''
Refs:
    MobileNetV2: Inverted Residuals and Linear Bottlenecks -- https://arxiv.org/abs/1801.04381
'''

import tensorflow as tf
layers = tf.keras.layers
K = tf.keras.backend
import keras

from tensorflow.python.keras.layers import Lambda;
from tensorflow.python.keras.layers import Multiply

from config import config as cfg

def l2_normalize(x):
    return tf.nn.l2_normalize(x, axis=2)


def _conv_block(inputs, filters, kernel, strides):
    """Convolution Block
    This function defines a 2D convolution operation with BN and relu6.

    # Arguments
        inputs: Tensor, input tensor of conv layer.
        filters: Integer, the dimensionality of the output space.
        kernel: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
        strides: An integer or tuple/list of 2 integers,
            specifying the strides of the convolution along the width and height.
            Can be a single integer to specify the same value for
            all spatial dimensions.

    # Returns
        Output tensor.
    """

    x = layers.Conv2D(filters, kernel, padding='same', strides=strides)(inputs)
    x = layers.BatchNormalization(axis= -1)(x)
    x = layers.ReLU(6., )(x)
    #x = tf.keras.layers.Dense(6., activation=tf.nn.relu)(x)
    #x.add(Activation('tanh'))
    #keras.activations.elu(x, alpha=1.0)
    return x


def _bottleneck(inputs, filters, kernel, t, s, r=False):
    """Bottleneck
    This function defines a basic bottleneck structure.

    # Arguments
        inputs: Tensor, input tensor of conv layer.
        filters: Integer, the dimensionality of the output space.
        kernel: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
        t: Integer, expansion factor.
            t is always applied to the input size.
        s: An integer or tuple/list of 2 integers,specifying the strides
            of the convolution along the width and height.Can be a single
            integer to specify the same value for all spatial dimensions.
        r: Boolean, Whether to use the residuallayerss.

    # Returns
        Output tensor.
    """

    channel_axis = -1
    tchannel = K.int_shape(inputs)[channel_axis] * t

    x = _conv_block(inputs, tchannel, (1, 1), (1, 1))

    x = layers.DepthwiseConv2D(kernel, strides=(s, s), depth_multiplier=1, padding='same')(x)
    x = layers.BatchNormalization(axis=channel_axis)(x)
    # x = Activation(relu6)(x)
    x = layers.ReLU(6., )(x)

    x = layers.Conv2D(filters, (1, 1), strides=(1, 1), padding='same')(x)
    x = layers.BatchNormalization(axis=channel_axis)(x)

    if r:
        x = layers.add([x, inputs])
    return x


def _inverted_residual_block(inputs, filters, kernel, t, strides, n):
    """Inverted Residual Block
    This function defines a sequence of 1 or more identical layers.

    # Arguments
        inputs: Tensor, input tensor of conv layer.
        filters: Integer, the dimensionality of the output space.
        kernel: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
        t: Integer, expansion factor.
            t is always applied to the input size.
        s: An integer or tuple/list of 2 integers,specifying the strides
            of the convolution along the width and height.Can be a single
            integer to specify the same value for all spatial dimensions.
        n: Integer, layer repeat times.
    # Returns
        Output tensor.
    """

    x = _bottleneck(inputs, filters, kernel, t, strides)

    for i in range(1, n):
        x = _bottleneck(x, filters, kernel, t, 1, True)

    return x

# def multiply(x):
#     image,mask = x
#     mask = K.expand_dims(mask, axis=-1) #could be K.stack([mask]*3, axis=-1) too 
#     return mask*image

def network():
    inputs = layers.Input(shape=(cfg().norm_h, cfg().norm_w, 3))
    inputs_depth = layers.Input(shape=(cfg().norm_h, cfg().norm_w, 3))

    print("")
    print("")
    print("")
    print("")
    print("")
    print("")
    print("inputs = ", inputs.shape)
    print("inputs_depth = ", inputs_depth.shape)
    print("")
    print("")
    print("")
    print("")
    print("")

    ######################################### 1 ######################################
    x = _conv_block(inputs, 32, (3, 3), strides=(2, 2))

    x = _inverted_residual_block(x, 16, (3, 3), t=1, strides=1, n=1)
    x = _inverted_residual_block(x, 24, (3, 3), t=6, strides=2, n=2)
    x = _inverted_residual_block(x, 32, (3, 3), t=6, strides=2, n=3)
    x = _inverted_residual_block(x, 64, (3, 3), t=6, strides=2, n=4)
    x = _inverted_residual_block(x, 96, (3, 3), t=6, strides=1, n=3)
    x = _inverted_residual_block(x, 160, (3, 3), t=6, strides=2, n=3)
    x = _inverted_residual_block(x, 320, (3, 3), t=6, strides=1, n=1)

    x = _conv_block(x, 1280, (1, 1), strides=(1, 1))
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Reshape((1, 1, 1280))(x)
    x = layers.Dropout(0.3, name='Dropout_1')(x)
    print("x = ", x)

    ##################################################################################
    ######################################### 2 ######################################
    y = _conv_block(inputs_depth, 32, (3, 3), strides=(2, 2))

    y = _inverted_residual_block(y, 16, (3, 3), t=1, strides=1, n=1)
    y = _inverted_residual_block(y, 24, (3, 3), t=6, strides=2, n=2)
    y = _inverted_residual_block(y, 32, (3, 3), t=6, strides=2, n=3)
    y = _inverted_residual_block(y, 64, (3, 3), t=6, strides=2, n=4)
    y = _inverted_residual_block(y, 96, (3, 3), t=6, strides=1, n=3)
    y = _inverted_residual_block(y, 160, (3, 3), t=6, strides=2, n=3)
    y = _inverted_residual_block(y, 320, (3, 3), t=6, strides=1, n=1)

    y = _conv_block(y, 1280, (1, 1), strides=(1, 1))
    y = layers.GlobalAveragePooling2D()(y)
    y = layers.Reshape((1, 1, 1280))(y)
    y = layers.Dropout(0.3, name='Dropout_2')(y)
    print("y = ", y)

    ##################################################################################
    ################################### COMBINE ######################################
    # xy = x*y
    # xy = Lambda(multiply)([x,y])

    # Bisaaaaaaa
    xy = layers.Concatenate()([x, y])

    # Belom bisa
    #xy = Lambda(lambda x: x[0] * K.expand_dims(x[1], axis=-1))([x, y])

    # Terbaru
    #xy = Multiply()([x, y])

    print("xy = ", xy)
    #print(akmj)
    ##################################################################################

    # Dimensions branch
    dimensions = layers.Conv2D(3, (1,1), padding='same', name='d_conv')(xy)
    dimensions = layers.Reshape((3,), name='dimensions')(dimensions)

    # Orientation branch
    orientation = layers.Conv2D(4, (1,1), padding='same', name='o_conv')(xy)
    orientation = layers.Reshape((cfg().bin, -1))(orientation)
    orientation = layers.Lambda(l2_normalize, name='orientation')(orientation)

    # Confidence branch
    confidence = layers.Conv2D(cfg().bin, (1,1), padding='same', name='c_conv')(xy)
    confidence = layers.Activation('softmax', name='softmax')(confidence)
    confidence = layers.Reshape((2,), name='confidence')(confidence)

    # Build model
    model = tf.keras.Model([inputs, inputs_depth], [dimensions, orientation, confidence])
    model.summary()

    return model

if __name__ == '__main__':
    model = network()