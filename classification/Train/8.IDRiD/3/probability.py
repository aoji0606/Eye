import os
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


def LoadData(num):
    X = np.load("./data/test/%d/X.npy" % num)
    y = np.load("./data/test/%d/y.npy" % num)

    return X, y


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    X, y = LoadData(19)
    a = np.zeros(y.shape, np.float32)
    b = np.zeros(y.shape, np.float32)

    for num in range(19, 35, 2):
        X_test, y_test = LoadData(num)

        shared_model = tf.keras.models.load_model("./model/shared_model.h5")
        model = tf.keras.models.load_model("./model/model_%d.h5" % num)
        for i in range(8, len(model.layers), 1):
            model.layers[i] = shared_model.layers[i]

        y_pred = model.predict(X_test)

        for i in range(y_test.shape[0]):
            temp = np.abs(y_pred[i][0] - 0.5)
            if temp >= a[i][0]:
                a[i][0] = temp
                b[i][0] = y_pred[i]

    b[b >= 0.5] = 1
    b[b != 1] = 0

    print("probability:")
    print("accuracy=", accuracy_score(y, b))
    print("precision=", precision_score(y, b))
    print("recall=", recall_score(y, b))
    print("f1=", f1_score(y, b))
    print()
