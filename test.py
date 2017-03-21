import pickle

import numpy as np

from conv_model import conv_model

testing_file = './input/test.p'
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
X_test, y_test = test['features'], test['labels']
print("Number of testing examples =", X_test.shape[0])

model = conv_model()
model.load_weights("./output/weights.hdf5")
y_pred = model.predict_classes(X_test)
acc = np.sum(y_pred==y_test)/np.size(y_pred)
print("Test accuracy = {}".format(acc))
