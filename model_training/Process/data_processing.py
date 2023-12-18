import numpy as np
from keras.utils import to_categorical  # one-hot
import cv2
import os
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split


def load_data(directory, norm_size):
    IMG = []
    read = lambda imagename: np.asarray(Image.open(imagename).convert("RGB"))
    for IMAGE_NAME in tqdm(os.listdir(directory)):
        PATH = os.path.join(directory, IMAGE_NAME)
        _, figuretype = os.path.splitext(PATH)
        if figuretype == ".png":
            img = read(PATH)
            img = cv2.resize(img, (norm_size, norm_size))
            IMG.append(np.array(img))
    return IMG


# Read raw data
benign_train = np.array(load_data('../Data/BreaKHis 400X/train/benign', 224))
malign_train = np.array(load_data('../Data/BreaKHis 400X/train/malignant', 224))

# Label
benign_train_label = np.zeros(len(benign_train))
malign_train_label = np.ones(len(malign_train))
label = {0: "benign", 1: "malignant"}

# Merge data for model training. X is the data while Y is the label.
X_train = np.concatenate((benign_train, malign_train), axis=0)
Y_train = np.concatenate((benign_train_label, malign_train_label), axis=0)

# Shuffle train data
s = np.arange(X_train.shape[0])
np.random.shuffle(s)
X_train = X_train[s]
Y_train = Y_train[s]

# One-hot
Y_train = to_categorical(Y_train, num_classes=2)

# Group train data into training and validation data
x_train, x_validation, y_train, y_validation = train_test_split(X_train, Y_train, random_state=10)
