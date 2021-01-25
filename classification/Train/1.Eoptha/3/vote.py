import os
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


def LoadData(num):
    X = np.load("./data/%d/X.npy" % num)
    y = np.load("./data/%d/y.npy" % num)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=666)

    return X_test, y_test


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    X, y = LoadData(19)
    pred = np.zeros(y.shape, np.int)

    for num in range(19, 35, 2):
        X_test, y_test = LoadData(num)

        shared_model = tf.keras.models.load_model("./model/shared_model.h5")
        model = tf.keras.models.load_model("./model/model_%d.h5" % num)
        for i in range(8, len(model.layers), 1):
            model.layers[i] = shared_model.layers[i]

        y_pred = model.predict(X_test)
        y_pred[y_pred >= 0.5] = 1
        y_pred[y_pred != 1] = 0
        y_pred = np.array(y_pred, np.int)

        for i in range(y_test.shape[0]):
            if y_pred[i] == 1:
                pred[i] += 1

    pred[pred >= 4] = 1
    pred[pred != 1] = 0

    print("vote:")
    print("accuracy=", accuracy_score(y, pred))
    print("precision=", precision_score(y, pred))
    print("recall=", recall_score(y, pred))
    print("f1=", f1_score(y, pred))
    print()
