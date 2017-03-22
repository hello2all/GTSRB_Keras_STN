import numpy as np

from keras.layers import (Activation, Dense, Dropout, Flatten,
                          Lambda, MaxPooling2D)
from keras.layers.convolutional import Conv2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential, model_from_json
from keras.regularizers import l2
from keras.utils import np_utils
from spatial_transformer import SpatialTransformer

def locnet():
    b = np.zeros((2, 3), dtype='float32')
    b[0, 0] = 1
    b[1, 1] = 1
    W = np.zeros((64, 6), dtype='float32')
    weights = [W, b.flatten()]
    locnet = Sequential()

    locnet.add(Conv2D(16, (7, 7), padding='valid', input_shape=(32, 32, 3)))
    locnet.add(MaxPooling2D(pool_size=(2, 2)))
    locnet.add(Conv2D(32, (5, 5), padding='valid'))
    locnet.add(MaxPooling2D(pool_size=(2, 2)))
    locnet.add(Conv2D(64, (3, 3), padding='valid'))
    locnet.add(MaxPooling2D(pool_size=(2, 2)))

    locnet.add(Flatten())
    locnet.add(Dense(128))
    locnet.add(Activation('elu'))
    locnet.add(Dense(64))
    locnet.add(Activation('elu'))
    locnet.add(Dense(6, weights=weights))

    return locnet

def conv_model(input_shape=(32, 32, 3)):
    model = Sequential()
    model.add(Lambda(
        lambda x: x/127.5 - 1.,
        input_shape=(32, 32, 3),
        output_shape=(32, 32, 3)))
    model.add(BatchNormalization())
    model.add(Conv2D(10, (1, 1), padding='same', kernel_regularizer=l2(0.04)))
    model.add(LeakyReLU(alpha=0.5))
    model.add(BatchNormalization())
    model.add(Conv2D(3, (1, 1), padding='same', kernel_regularizer=l2(0.04)))
    model.add(LeakyReLU(alpha=0.5))
    model.add(BatchNormalization())
    model.add(SpatialTransformer(localization_net=locnet(),
                                 output_size=(32, 32)))
    model.add(Conv2D(16, (5, 5), padding='same', activation='relu', kernel_regularizer=l2(0.04)))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (5, 5), padding='same', activation='relu', kernel_regularizer=l2(0.04)))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (5, 5), padding='same', activation='relu', kernel_regularizer=l2(0.04)))
    model.add(BatchNormalization())
    model.add(Conv2D(96, (5, 5), padding='same', activation='relu', kernel_regularizer=l2(0.04)))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (5, 5), padding='same', activation='relu', kernel_regularizer=l2(0.04)))
    model.add(BatchNormalization())
    model.add(Conv2D(192, (5, 5), padding='same', activation='relu', kernel_regularizer=l2(0.04)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(256, (5, 5), padding='same', activation='relu', kernel_regularizer=l2(0.04)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (5, 5), padding='same', activation='relu', kernel_regularizer=l2(0.04)))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (5, 5), padding='same', activation='relu', kernel_regularizer=l2(0.04)))
    model.add(MaxPooling2D(pool_size=(8, 8)))
    model.add(Flatten())
    model.add(Dropout(0.6))
    model.add(Dense(43, activation="softmax"))
    return model
