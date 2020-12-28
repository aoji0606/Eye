import os
import numpy as np
import cv2 as cv
from pprint import pprint


def file_name(file_dir):
    for files in os.walk(file_dir):
        pass
    return files


if __name__ == '__main__':
    X = np.array([])
    files = file_name("./data/temp/")

    num = 0
    for file in files[2]:
        img = cv.imread("./data/temp/" + file)
        img = cv.resize(img, (256, 256))
        arr = np.array(img).reshape(1, -1)
        X = np.append(X, arr)
        num += 1
        print(num)
    X = X.reshape(len(files[2]), -1)
    print(X.shape)
    np.save("./X.npy",X)

    num = 0
    Y = np.array([])
    files = file_name("./data/temp/")
    for file in files[2]:
        img = cv.imread("./data/temp/" + file, 0)
        img = cv.resize(img, (256, 256))
        arr = np.array(img).reshape(1, -1)
        Y = np.append(Y, arr)
        num += 1
        print(num)
    Y = Y.reshape(len(files[2]), -1)
    print(Y.shape)
    np.save("./Y.npy",Y)
