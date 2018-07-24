from __future__ import division
import cv2
import os
import csv
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from itertools import islice
from tqdm import tqdm
from pkl import pickle_dump

MAX = None

DATA_PATH = './dataset/IMG'
TRAINING_FILE_PATH = './dataset'
TRAINING_FILE = os.path.join(TRAINING_FILE_PATH, 'driving_log.csv')

def preprocess(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    return cv2.resize((img)[:, :, 1], (100, 100))

def return_data(delta):
    X, y, preprocessed = [], [], []

    with open(TRAINING_FILE, 'rt') as f:
        reader = csv.reader(f)
        for line in reader:
            X.append(line)
        log_y = X.pop(0)

    for i in tqdm(range(len(X))):
        for j in range(3):
            img_path = X[i][j]
            img_path = DATA_PATH + (img_path.split('IMG')[1]).strip()
            img = plt.imread(img_path)
            preprocessed.append(preprocess(img))
            if j == 0:
                y.append(float(X[i][3]))
            elif j == 1:
                y.append(float(X[i][3]) + delta)
            else:
                y.append(float(X[i][3]) - delta)
    return preprocessed, y    

if __name__ == "__main__":
    print("preprocessing...")
    
    delta = 0.2
    preprocessed, labels = return_data(delta)

    preprocessed = np.array(preprocessed).astype('float32')
    labels = np.array(labels).astype('float32')

    pickle_dump(preprocessed, "preprocessed")
    pickle_dump(labels, "labels")

    print("done")