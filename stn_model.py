import numpy as np

from keras.layers import (Activation, AveragePooling2D, Convolution2D, Dense,
                          Dropout, Flatten, Lambda, MaxPooling2D)
from keras.models import Sequential, model_from_json
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras.utils import np_utils
from spatial_transformer import SpatialTransformer

IMG_SIZE = 32
NUM_CLASSES = 43
input_shape = (IMG_SIZE, IMG_SIZE, 3)

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

def stn_model():
    model = Sequential()
    lamda = 0.0001
    # Use a lambda layer to normalize the input data
    model.add(Lambda(
        lambda x: x/127.5 - 1.,
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        output_shape=(IMG_SIZE, IMG_SIZE, 3)))
    model.add(SpatialTransformer(localization_net=locnet(1),
                                 output_size=(32, 32)))

    model.add(Convolution2D(3, 1, 1, border_mode='valid',
                            activation='relu'))
 
    model.add(Convolution2D(128, 5, 5, border_mode='valid', 
                            activation='relu', W_regularizer=l2(lamda)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.2))
    model.add(Convolution2D(256, 5, 5, border_mode='valid',
                            activation='relu', W_regularizer=l2(lamda)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.2))
    model.add(Convolution2D(512, 3, 3, border_mode='valid',
                            activation='relu', W_regularizer=l2(lamda)))
    model.add(AveragePooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(1024, activation='relu', W_regularizer=l2(lamda)))
    #model.add(Dropout(0.4))
    model.add(Dense(NUM_CLASSES, activation='softmax'))
    return model
