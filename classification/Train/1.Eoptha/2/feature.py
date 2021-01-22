import os
import sys
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

num = int(sys.argv[1])


def LoadData():
    X = np.load("./data/%d/X.npy" % num)
    y = np.load("./data/%d/y.npy" % num)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=666)

    X_test = X_test[0]
    X_test = X_test.reshape([1, X.shape[1], X.shape[2], X.shape[3]])

    return X_test


def GetModel():
    shared_model = tf.keras.models.load_model("model/shared_model.h5")
    model = tf.keras.models.load_model("./model/model_%d.h5" % num)
    for i in range(8, len(model.layers), 1):
        model.layers[i] = shared_model.layers[i]

    return model


def GetFeature(X):
    model = GetModel()
    names = ["img", "img_ch", "img_sp"]
    for name in names:
        output = model.get_layer(name).output
        feature_model = tf.keras.models.Model(inputs=model.input, outputs=output)
        img = feature_model.predict(X)
        img = img[0]

        for channle in range(img.shape[2]):
        #for channle in range(10):
            # plt.imshow(img[:, :, channle])
            # dir = "./feature/%d/%s/%d.png" % (num, name, channle + 1)
            # plt.savefig(dir)
            # print("save %s successfully" % dir)
            fig, ax = plt.subplots()
            ax.imshow(img[:, :, channle], aspect='equal')
            plt.axis('off')
            fig.set_size_inches(5 / 300.0, 5 / 300.0)  # 输出width*height像素
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
            plt.margins(0, 0)
            plt.imshow(img[:, :, channle])
            dir = "./feature/%d/%s/%d.png" % (num, name, channle + 1)
            fig.savefig(dir, dpi=300)
            print("save %s successfully" % dir)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    X = LoadData()
    GetFeature(X)
