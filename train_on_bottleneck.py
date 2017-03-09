import numpy as np
from floyd import dot

from inception_v4 import create_inception_v4
from keras.callbacks import ModelCheckpoint
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.models import Model
from keras.optimizers import RMSprop
from keras.models import Sequential
from keras.regularizers import l2

with open(dot + '/output/bottleneck_features_train.npy', 'rb') as f_in:
    bottleneck_features = np.load(f_in)
with open(dot + '/output/labels.npy', 'rb') as f_in:
    labels = np.load(f_in)


model = Sequential()
# model.add(Flatten(input_shape=bottleneck_features.shape))
model.add(Dense(1024, W_regularizer=l2(0.001), input_shape=bottleneck_features.shape[1:], activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(43, activation='softmax'))
rmsprop = RMSprop(lr=0.0001)
model.compile(optimizer=rmsprop,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

checkpointer = ModelCheckpoint(filepath=dot+"/output/weights.hdf5", verbose=1, save_best_only=True)
model.fit(bottleneck_features, labels,
          nb_epoch=1000, batch_size=32,
          validation_split=0.2,
          callbacks=[checkpointer])