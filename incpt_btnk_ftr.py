import numpy as np
from floyd import dot

from generator import generator_train
from inception_v4 import create_inception_v4
from keras.models import Model
from preprocessing import test_data, train_data

train_samples, labels = train_data()
labels = np.array(labels)
train_generator = generator_train(train_samples, labels, batch_size=32, shuffle=False)
inception = create_inception_v4()
base_model = Model(input=inception.input, output=inception.get_layer('flatten').output)
bottleneck_features = base_model.predict_generator(train_generator, val_samples=len(train_samples))

with open(dot + '/output/bottleneck_features_train.npy', 'wb') as f_out:
    np.save(f_out, bottleneck_features)
with open(dot + '/output/labels.npy', 'wb') as f_out:
    np.save(f_out, labels)
