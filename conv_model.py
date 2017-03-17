import numpy as np

from keras.layers import (Activation, Convolution2D, Dense, Dropout, Flatten,
                          Lambda, MaxPooling2D)
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential, model_from_json
from keras.preprocessing.image import ImageDataGenerator
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

    locnet.add(Convolution2D(16, 7, 7 , border_mode='valid', input_shape=(32, 32, 3)))
    locnet.add(MaxPooling2D(pool_size=(2, 2)))
    locnet.add(Convolution2D(32, 5, 5, border_mode='valid'))
    locnet.add(MaxPooling2D(pool_size=(2, 2)))
    locnet.add(Convolution2D(64, 3, 3, border_mode='valid'))
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
    model.add(Convolution2D(10, 1, 1, border_mode='same', W_regularizer=l2(0.012)))
    model.add(LeakyReLU(alpha=0.5))
    model.add(BatchNormalization())
    model.add(Convolution2D(3, 1, 1, border_mode='same', W_regularizer=l2(0.012)))
    model.add(LeakyReLU(alpha=0.5))
    model.add(BatchNormalization())
    model.add(SpatialTransformer(localization_net=locnet(),
                                 output_size=(32, 32)))
    model.add(Convolution2D(16, 5, 5, border_mode='same', activation='relu', W_regularizer=l2(0.012)))
    model.add(BatchNormalization())
    model.add(Convolution2D(32, 5, 5, border_mode='same', activation='relu', W_regularizer=l2(0.012)))
    model.add(BatchNormalization())
    model.add(Convolution2D(64, 5, 5, border_mode='same', activation='relu', W_regularizer=l2(0.012)))
    model.add(BatchNormalization())
    model.add(Convolution2D(96, 5, 5, border_mode='same', activation='relu', W_regularizer=l2(0.012)))
    model.add(BatchNormalization())
    model.add(Convolution2D(128, 5, 5, border_mode='same', activation='relu', W_regularizer=l2(0.012)))
    model.add(BatchNormalization())
    model.add(Convolution2D(192, 5, 5, border_mode='same', activation='relu', W_regularizer=l2(0.012)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(256, 5, 5, border_mode='same', activation='relu', W_regularizer=l2(0.012)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(128, 5, 5, border_mode='same', activation='relu', W_regularizer=l2(0.012)))
    model.add(BatchNormalization())
    model.add(Convolution2D(64, 5, 5, border_mode='same', activation='relu', W_regularizer=l2(0.012)))
    model.add(MaxPooling2D(pool_size=(8, 8)))
    model.add(Flatten())
    model.add(Dropout(0.6))
    model.add(Dense(43, activation="softmax"))
    return model
