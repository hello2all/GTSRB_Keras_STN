import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage import color, exposure, io, transform


def preprocess_img(img):
    IMG_SIZE = 32
    # Histogram normalization in v channel
    hsv = color.rgb2hsv(img)
    hsv[:, :, 2] = exposure.equalize_hist(hsv[:, :, 2])
    img = color.hsv2rgb(hsv)

    # central square crop
    min_side = min(img.shape[:-1])
    centre = img.shape[0]//2, img.shape[1]//2
    img = img[centre[0]-min_side//2:centre[0]+min_side//2,
              centre[1]-min_side//2:centre[1]+min_side//2,
              :]

    # rescale to standard size
    img = transform.resize(img, (IMG_SIZE, IMG_SIZE))
    return img

def get_path_and_label(IMG_DIR):
    image_paths = []
    labels = []
    for root, dirs, files in os.walk("./GTSRB/"):
        for f in files:
            if f.endswith(".ppm"):
                labels.append(int(root.split("/")[4]))
                image_paths.append(os.path.join(root, f))

    print "{} image names.".format(len(image_paths))
    print "{0} images labels with {1} unique labels".format(len(labels), len(set(labels)))
    return image_paths, labels

def get_arrays(image_paths):
    imgs = []
    NUM_CLASSES = 43
    for img_path in image_paths:
        img = preprocess_img(io.imread(img_path))
        imgs.append(img)
    X = np.array(imgs, dtype='float32')
    return X

def main():

    TRAIN_DIR = "./GTSRB/"
    TEST_DIR = "./GTSRB_TEST/"

    # image_paths, labels = get_path_and_label(TRAIN_DIR)
    # labels = np.array(labels)
    # images = get_arrays(image_paths)

    # test_data = {}
    # test_data["features"] = images
    # test_data["labels"] = labels

    # with open("test.pkl", "wb") as f_output:
    #     pickle.dump(test_data, f_output)
    #     print "Output complete"

    image_paths, labels = get_path_and_label(TEST_DIR)
    df = pd.read_csv(TEST_DIR + "GT-final_test.csv")
    print df.head()

if __name__ == "__main__":
    main()
