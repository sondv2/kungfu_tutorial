import os

import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

img_width = 64
img_height = 64

# Loading dataset
def load_datasets_rgb(dataset='dataset_image/'):
    X = []
    y = []
    label = os.listdir(dataset)
    for image_label in label:
        images = os.listdir(dataset + image_label)
        for image in tqdm(images):
            path = os.path.join(dataset + image_label + '/', image)
            img = cv2.imread(path)
            if img is not None:
                img = cv2.resize(img, (img_width, img_height))
                X.append(img)
                y.append(label.index(image_label))

    X = np.asarray(X)
    y = np.asarray(y)
    # return X, y
    # split dataset
    X_train, x_val, Y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=7)

    return  X_train, Y_train, x_val, y_val
