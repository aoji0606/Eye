import os
import sys
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

num = int(sys.argv[1])
shape = 0
ACTIVATION = "selu"
shape_list = []
k_s_list = []


def LoadData():
    X = np.load("./data/%d/X.npy" % num)
    y = np.load("./data/%d/y.npy" % num)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=666)

    return X_train, X_test, y_train, y_test


def Model(X_train, X_test, y_train, y_test):
    shared_model = tf.keras.models.load_model("model/shared_model.h5")
    model = tf.keras.models.load_model("./model/model_%d.h5" % num)
    for i in range(8, len(model.layers), 1):
        model.layers[i] = shared_model.layers[i]
    model.compile(optimizer=tf.keras.optimizers.Adam(0.0001), loss="binary_crossentropy",
                  metrics=[tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
    history = model.fit(X_train, y_train, batch_size=64, epochs=50, validation_split=0.2)

    loss = np.array(history.history["loss"])
    precision = np.array(history.history["precision"])
    recall = np.array(history.history["recall"])
    val_loss = np.array(history.history["val_loss"])
    val_precision = np.array(history.history["val_precision"])
    val_recall = np.array(history.history["val_recall"])
    np.savetxt("./shared/%d/loss.txt" % num, loss)
    np.savetxt("./shared/%d/precision.txt" % num, precision)
    np.savetxt("./shared/%d/recall.txt" % num, recall)
    np.savetxt("./shared/%d/val_loss.txt" % num, val_loss)
    np.savetxt("./shared/%d/val_precision.txt" % num, val_precision)
    np.savetxt("./shared/%d/val_recall.txt" % num, val_recall)

    model.save("./model/model_%d.h5" % num)

    # model.evaluate(X_test, y_test)

    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)
    plt.savefig("./shared/%d/result_%d.png" % (num, num))


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    X_train, X_test, y_train, y_test = LoadData()
    Model(X_train, X_test, y_train, y_test)
