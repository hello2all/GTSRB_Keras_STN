import numpy as np
import cv2
import sklearn
from preprocessing import preprocess_img
# import matplotlib.pyplot as plt

def generator_train(samples, labels, batch_size=32, shuffle=False):
    num_samples = len(samples)

    while 1: # Loop forever so the generator never terminates
        if shuffle:
            samples, labels = sklearn.utils.shuffle(samples, labels)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            batch_labels = labels[offset:offset+batch_size]
            images = []
            for batch_sample in batch_samples:
                image = cv2.imread(batch_sample)
                image = preprocess_img(image)
                images.append(image)
            if shuffle:
                images, batch_labels = sklearn.utils.shuffle(images, batch_labels)
            X_train = np.array(images)
            y_train = np.array(batch_labels)
            # return sklearn.utils.shuffle(X_train, y_train)
            yield X_train, y_train

def main():
    # from preprocessing import train_data, test_data
    # train_samples, labels = train_data()
    # X_train, y_train = generator_train(train_samples, labels)
    # print(X_train.shape, y_train.shape)

    # plt.imshow(X_train[0])
    # plt.show()
    pass


if __name__ == "__main__":
    main()