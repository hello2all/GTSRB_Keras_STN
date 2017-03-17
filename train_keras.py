import pickle
from conv_model import conv_model
import numpy as np
from floyd import dot
import sys

from keras.callbacks import LearningRateScheduler, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adagrad, SGD, Adam
from keras.utils.np_utils import to_categorical
from stn_model import stn_model

training_file = dot + '/input/train.p'
testing_file = dot + '/input/test.p'
validating_file = dot + '/input/valid.p'

# training_file = dot + '/input/train_norm.p'
# testing_file = dot + '/input/test_norm.p'
# validating_file = dot + '/input/valid_norm.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validating_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']


# y_train = to_categorical(y_train, nb_classes=43)
# y_valid = to_categorical(y_valid, nb_classes=43)
# y_test = to_categorical(y_test, nb_classes=43)

print("Number of training examples =", X_train.shape[0])
print("Number of validating examples =", X_valid.shape[0])
print("Number of testing examples =", X_test.shape[0])
print("Image data shape =", X_train[0].shape)
print("Number of classes =", len(np.unique(y_train)))

batch_size = 128
nb_epoch = 100
model = conv_model()
adagrad = Adagrad(lr=0.0002, epsilon=1e-08, decay=0.01)
sgd = SGD(lr=0.001, momentum=0.9, decay=0.0, nesterov=True)
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.02)

model.compile(loss='sparse_categorical_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])

checkpointer = ModelCheckpoint(filepath=dot + "/output/weights.hdf5", verbose=1, save_best_only=True)
# model.load_weights(dot + "/output/weights.hdf5")
try:
    model.fit(X_train, y_train,
                batch_size=batch_size,
                nb_epoch=nb_epoch,
                validation_data=(X_valid, y_valid),
                shuffle=True,
                callbacks=[checkpointer])
except:
    print("training interrupted")
    print(sys.exc_info()[0])

y_pred = model.predict_classes(X_test)
acc = np.sum(y_pred==y_test)/np.size(y_pred)
print("Test accuracy = {}".format(acc))
