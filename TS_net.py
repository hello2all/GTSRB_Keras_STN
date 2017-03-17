from keras.layers import Activation, Dense, Dropout, Flatten, Input, merge
from keras.layers.convolutional import (AveragePooling2D, Convolution2D,
                                        MaxPooling2D)
from keras.layers.normalization import BatchNormalization
from keras.models import Model, Sequential
from keras.utils.data_utils import get_file
from spatial_transformer import SpatialTransformer
import numpy as np


def locnet(type):
    b = np.zeros((2, 3), dtype='float32')
    b[0, 0] = 1
    b[1, 1] = 1
    W = np.zeros((192, 6), dtype='float32')
    weights = [W, b.flatten()]
    locnet = Sequential()
    if type == 1:
        locnet.add(Convolution2D(128, 5, 5, subsample=(2, 2) , input_shape=input_shape))
        locnet.add(MaxPooling2D(pool_size=(2, 2)))
        locnet.add(Convolution2D(128, 3, 3, subsample=(2, 2)))
        locnet.add(MaxPooling2D(pool_size=(2, 2)))
    elif type == 2:
        locnet.add(Convolution2D(128, 3, 3, subsample=(2, 2) , input_shape=(32, 32, 100)))
        locnet.add(MaxPooling2D(pool_size=(2, 2)))
        locnet.add(Convolution2D(128, 3, 3, subsample=(2, 2)))
    # elif type == 3:
    #     locnet.add(Convolution2D(128, 3, 3, subsample=(2, 2) , input_shape=input_shape))
    #     locnet.add(Convolution2D(128, 3, 3))
    #     locnet.add(MaxPooling2D(pool_size=(2, 2))) 
    else:
        print("Error: wrong localization network type")
        exit()

    locnet.add(Flatten())
    locnet.add(Dense(192))
    locnet.add(Activation('relu'))
    locnet.add(Dense(192))
    locnet.add(Activation('relu'))
    locnet.add(Dense(6, weights=weights))

    return locnet

def conv_block(x, nb_filter, nb_row, nb_col, border_mode='same', subsample=(1, 1), bias=False):

    channel_axis = -1

    x = Convolution2D(nb_filter, nb_row, nb_col, subsample=subsample, border_mode=border_mode, bias=bias)(x)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)
    return x


def inception_stem(input):

    channel_axis = -1

    # Input Shape is 299 x 299 x 3 (th) or 3 x 299 x 299 (th)
    x = conv_block(input, 32, 3, 3, subsample=(2, 2), border_mode='valid')
    x = conv_block(x, 32, 3, 3, border_mode='valid')
    x = conv_block(x, 64, 3, 3)

    x1 = MaxPooling2D((3, 3), strides=(2, 2), border_mode='valid')(x)
    x2 = conv_block(x, 96, 3, 3, subsample=(2, 2), border_mode='valid')

    x = merge([x1, x2], mode='concat', concat_axis=channel_axis)

    x1 = conv_block(x, 64, 1, 1)
    x1 = conv_block(x1, 96, 3, 3, border_mode='valid')

    x2 = conv_block(x, 64, 1, 1)
    x2 = conv_block(x2, 64, 1, 7)
    x2 = conv_block(x2, 64, 7, 1)
    x2 = conv_block(x2, 96, 3, 3, border_mode='valid')

    x = merge([x1, x2], mode='concat', concat_axis=channel_axis)

    x1 = conv_block(x, 192, 3, 3, subsample=(2, 2), border_mode='valid')
    x2 = MaxPooling2D((3, 3), strides=(2, 2), border_mode='valid')(x)

    x = merge([x1, x2], mode='concat', concat_axis=channel_axis)
    return x


def inception_A(input):

    channel_axis = -1

    a1 = conv_block(input, 96, 1, 1)

    a2 = conv_block(input, 64, 1, 1)
    a2 = conv_block(a2, 96, 3, 3)

    a3 = conv_block(input, 64, 1, 1)
    a3 = conv_block(a3, 96, 3, 3)
    a3 = conv_block(a3, 96, 3, 3)

    a4 = AveragePooling2D((3, 3), strides=(1, 1), border_mode='same')(input)
    a4 = conv_block(a4, 96, 1, 1)

    m = merge([a1, a2, a3, a4], mode='concat', concat_axis=channel_axis)
    return m


def inception_B(input):

    channel_axis = -1

    b1 = conv_block(input, 384, 1, 1)

    b2 = conv_block(input, 192, 1, 1)
    b2 = conv_block(b2, 224, 1, 7)
    b2 = conv_block(b2, 256, 7, 1)

    b3 = conv_block(input, 192, 1, 1)
    b3 = conv_block(b3, 192, 7, 1)
    b3 = conv_block(b3, 224, 1, 7)
    b3 = conv_block(b3, 224, 7, 1)
    b3 = conv_block(b3, 256, 1, 7)

    b4 = AveragePooling2D((3, 3), strides=(1, 1), border_mode='same')(input)
    b4 = conv_block(b4, 128, 1, 1)

    m = merge([b1, b2, b3, b4], mode='concat', concat_axis=channel_axis)
    return m


def inception_C(input):

    channel_axis = -1

    c1 = conv_block(input, 256, 1, 1)

    c2 = conv_block(input, 384, 1, 1)
    c2_1 = conv_block(c2, 256, 1, 3)
    c2_2 = conv_block(c2, 256, 3, 1)
    c2 = merge([c2_1, c2_2], mode='concat', concat_axis=channel_axis)

    c3 = conv_block(input, 384, 1, 1)
    c3 = conv_block(c3, 448, 3, 1)
    c3 = conv_block(c3, 512, 1, 3)
    c3_1 = conv_block(c3, 256, 1, 3)
    c3_2 = conv_block(c3, 256, 3, 1)
    c3 = merge([c3_1, c3_2], mode='concat', concat_axis=channel_axis)

    c4 = AveragePooling2D((3, 3), strides=(1, 1), border_mode='same')(input)
    c4 = conv_block(c4, 256, 1, 1)

    m = merge([c1, c2, c3, c4], mode='concat', concat_axis=channel_axis)
    return m


def reduction_A(input):

    channel_axis = -1

    r1 = conv_block(input, 384, 3, 3, subsample=(2, 2), border_mode='valid')

    r2 = conv_block(input, 192, 1, 1)
    r2 = conv_block(r2, 224, 3, 3)
    r2 = conv_block(r2, 256, 3, 3, subsample=(2, 2), border_mode='valid')

    r3 = MaxPooling2D((3, 3), strides=(2, 2), border_mode='valid')(input)

    m = merge([r1, r2, r3], mode='concat', concat_axis=channel_axis)
    return m


def reduction_B(input):

    channel_axis = -1

    r1 = conv_block(input, 192, 1, 1)
    r1 = conv_block(r1, 192, 3, 3, subsample=(2, 2), border_mode='valid')

    r2 = conv_block(input, 256, 1, 1)
    r2 = conv_block(r2, 256, 1, 7)
    r2 = conv_block(r2, 320, 7, 1)
    r2 = conv_block(r2, 320, 3, 3, subsample=(2, 2), border_mode='valid')

    r3 = MaxPooling2D((3, 3), strides=(2, 2), border_mode='valid')(input)

    m = merge([r1, r2, r3], mode='concat', concat_axis=channel_axis)
    return m


def create_inception_v4(nb_classes=1001, load_weights=True):
    '''
    Creates a inception v4 network

    :param nb_classes: number of classes.txt
    :return: Keras Model with 1 input and 1 output
    '''
    init = Input((299, 299, 3))

    # Input Shape is 299 x 299 x 3 (tf) or 3 x 299 x 299 (th)
    x = inception_stem(init)

    # 4 x Inception A
    for i in range(4):
        x = inception_A(x)

    # Reduction A
    x = reduction_A(x)

    # 7 x Inception B
    for i in range(7):
        x = inception_B(x)

    # Reduction B
    x = reduction_B(x)

    # 3 x Inception C
    for i in range(3):
        x = inception_C(x)

    # Average Pooling
    x = AveragePooling2D((8, 8))(x)

    # Dropout
    x = Dropout(0.8)(x)
    x = Flatten(name='flatten')(x)

    # Output
    out = Dense(output_dim=nb_classes, activation='softmax')(x)

    model = Model(init, out, name='Inception-v4')

    return model


if __name__ == "__main__":
    # from keras.utils.visualize_util import plot

    inception_v4 = create_inception_v4(load_weights=True)
    # inception_v4.summary()

    # plot(inception_v4, to_file="Inception-v4.png", show_shapes=True)
