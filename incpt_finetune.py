from inception_v4 import create_inception_v4
from generator import generator_train
from keras.callbacks import ModelCheckpoint
from floyd import dot
from keras.layers import Dense
from keras.models import Model
from keras.optimizers import SGD

from preprocessing import train_data, test_data

train_samples, labels = train_data()
train_generator = generator_train(train_samples, labels, batch_size=32, shuffle=True)
inception = create_inception_v4()
base_model = Model(input=inception.input, output=inception.get_layer('flatten').output)
x = base_model.output
predictions = Dense(43, activation='softmax')(x)
model = Model(input=base_model.input, output=predictions)
for layer in model.layers[:-1]:
    layer.trainable = False
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='sparse_categorical_crossentropy')
# checkpointer = ModelCheckpoint(filepath=dot+"/output/weights.hdf5", verbose=1, save_best_only=True)
history_object = model.fit_generator(train_generator,
                                     samples_per_epoch=len(train_samples),
                                     nb_epoch=10,
                                     callbacks=[])

