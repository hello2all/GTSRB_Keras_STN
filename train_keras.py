import pickle

import numpy as np
import tensorflow as tf
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.optimizers import Adagrad
from keras.callbacks import ModelCheckpoint

from floyd import dot
from stn_model import stn_model

training_file = dot + '/input/train.p'
testing_file = dot + '/input/test.p'
validating_file = dot + '/input/valid.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validating_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']

print("Number of training examples =", X_train.shape[0])
print("Number of validating examples =", X_valid.shape[0])
print("Number of testing examples =", X_test.shape[0])
print("Image data shape =", X_train[0].shape)
print("Number of classes =", len(np.unique(y_train)))

batch_size = 20
nb_epoch = 10000
model = stn_model()
adagrad = Adagrad(lr=0.00032, epsilon=1e-08, decay=0.0)

model.compile(loss='sparse_categorical_crossentropy',
              optimizer=adagrad,
              metrics=['accuracy'])
checkpointer = ModelCheckpoint(filepath=dot + "/output/weights.hdf5", verbose=1, save_best_only=True)
model.fit(X_train, y_train,
          batch_size=batch_size,
          nb_epoch=nb_epoch,
          validation_data=(X_valid, y_valid),
          shuffle=True,
          callbacks=[checkpointer])

y_pred = model.predict_classes(X_test)
acc = np.sum(y_pred==y_test)/np.size(y_pred)
print("Test accuracy = {}".format(acc))
