import os
import sys
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, precision_score, f1_score

num = int(sys.argv[1])


def LoadData():
    X = np.load("./data/test/%d/X.npy" % num)
    y = np.load("./data/test/%d/y.npy" % num)

    return X, y


def GetModel():
    shared_model = tf.keras.models.load_model("./model/shared_model.h5")
    model = tf.keras.models.load_model("./model/model_%d.h5" % num)
    for i in range(8, len(model.layers), 1):
        model.layers[i] = shared_model.layers[i]

    return model


def Model(X_test, y_test):
    model = GetModel()
    res = model.evaluate(X_test, y_test, verbose=0)
    
    print(num)
    print(res)
    print()


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    X_test, y_test = LoadData()
    Model(X_test, y_test)
