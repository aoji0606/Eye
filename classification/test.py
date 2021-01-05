import os
import sys
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, precision_score, f1_score

num = int(sys.argv[1])


def LoadData():
    X = np.load("./data/%d/X.npy" % num)
    y = np.load("./data/%d/y.npy" % num)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=666)

    return X_test, y_test


def GetModel():
    shared_model = tf.keras.models.load_model("model/shared_model.h5")
    model = tf.keras.models.load_model("./model/model_%d.h5" % num)
    for i in range(8, len(model.layers), 1):
        model.layers[i] = shared_model.layers[i]

    return model


def Model(X_test, y_test):
    model = GetModel()
    model.evaluate(X_test, y_test)

    # y_pre = model.predict(X_test)
    # y_pre = y_pre.reshape(-1)
    # y_pre[y_pre > 0.5] = 1
    # y_pre[y_pre != 1] = 0
    # y_test = y_test.reshape(-1)
    #
    # print(y_pre)
    # print(y_test)
    #
    # print("precision_score: ", precision_score(y_test, y_pre))
    # print("recall_score: ", recall_score(y_test, y_pre))
    # print("f1_score: ", f1_score(y_test, y_pre))


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    X_test, y_test = LoadData()
    Model(X_test, y_test)
