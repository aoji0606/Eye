import os
import sys
import tensorflow as tf
import tensorflow.keras.layers as layer
from tensorflow.keras import layers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

num = int(sys.argv[1])
shape = 0
ACTIVATION = "selu"
shape_list = []
k_s_list = []


def ChannelAttention(conv):
    avg = layer.GlobalAveragePooling2D()(conv)
    max = layer.GlobalMaxPool2D()(conv)
    avg = layer.Reshape((1, 1, avg.shape[1]))(avg)
    max = layer.Reshape((1, 1, max.shape[1]))(max)
    avg_out = layer.Conv2D(int(conv.shape[-1] / 16), kernel_size=1, strides=1, padding="same",
                           kernel_regularizer=tf.keras.regularizers.l2(1e-4), use_bias=True, activation="relu")(avg)
    avg_out = layer.Conv2D(int(conv.shape[-1]), kernel_size=1, strides=1, padding="same",
                           kernel_regularizer=tf.keras.regularizers.l2(1e-4), use_bias=True)(avg_out)
    max_out = layer.Conv2D(int(conv.shape[-1] / 16), kernel_size=1, strides=1, padding="same",
                           kernel_regularizer=tf.keras.regularizers.l2(1e-4), use_bias=True, activation="relu")(max)
    max_out = layer.Conv2D(conv.shape[-1], kernel_size=1, strides=1, padding="same",
                           kernel_regularizer=tf.keras.regularizers.l2(1e-4), use_bias=True)(max_out)
    out = avg_out + max_out
    out = layer.Activation("sigmoid")(out)

    return out


def SpatialAttention(inputs):
    avg_out = tf.reduce_mean(inputs, axis=3)
    max_out = tf.reduce_max(inputs, axis=3)
    out = tf.stack([avg_out, max_out], axis=3)
    out = layers.Conv2D(filters=1, kernel_size=7, strides=1, activation="sigmoid", padding="same", use_bias=False,
                        kernel_initializer="he_normal", kernel_regularizer=tf.keras.regularizers.l2(5e-4))(out)

    return out


def LoadData():
    if not os.path.exists("model/shared_model.h5"):
        global num
        num = 19

    X = np.load("./data/%d/X.npy" % num)
    y = np.load("./data/%d/y.npy" % num)

    global shape
    shape = X.shape[1]
    input = layer.Input([X.shape[1], X.shape[2], X.shape[3]])

    while (len(shape_list) < 3):
        for k in range(2, int(shape / 2) + 1, 1):
            for s in range(2, k + 1, 1):
                pool = layer.MaxPool2D(pool_size=(k, k), strides=(s, s))(input)
                if pool.shape[1] % 2 and pool.shape[1] >= 5 and pool.shape[1] <= 9 and pool.shape[
                    1] not in shape_list and len(shape_list) < 3:
                    shape_list.append(pool.shape[1])
                    k_s_list.append([k, s])
        break

    print(shape_list)
    print(k_s_list)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=666)

    return X_train, X_test, y_train, y_test


def GetModel():
    if os.path.exists("model/shared_model.h5"):
        shared_model = tf.keras.models.load_model("model/shared_model.h5")
        model = tf.keras.models.load_model("./model/model_%d.h5" % num)
        for i in range(8, len(model.layers), 1):
            model.layers[i] = shared_model.layers[i]
    else:
        arg = np.argsort(shape_list)
        k_min = k_s_list[arg[0]][0]
        s_min = k_s_list[arg[0]][1]
        k_mid = k_s_list[arg[1]][0]
        s_mid = k_s_list[arg[1]][1]
        k_max = k_s_list[arg[2]][0]
        s_max = k_s_list[arg[2]][1]

        input = layer.Input(shape=[shape, shape, 3])

        conv = layer.Conv2D(filters=32, kernel_size=3, strides=1, padding="same", activation=ACTIVATION,
                            kernel_initializer=tf.keras.initializers.he_uniform())(input)

        # SPP
        pool_min = layer.MaxPool2D(pool_size=(k_min, k_min), strides=(s_min, s_min))(conv)
        pool_mid = layer.MaxPool2D(pool_size=(k_mid, k_mid), strides=(s_mid, s_mid))(conv)
        pool_max = layer.MaxPool2D(pool_size=(k_max, k_max), strides=(s_max, s_max))(conv)

        # print(pool_max.shape)
        # print(pool_mid.shape)
        # print(pool_min.shape)

        padding_mid = layer.ZeroPadding2D(1)(pool_mid)
        padding_min = layer.ZeroPadding2D(2)(pool_min)

        # print(pool_max.shape)
        # print(padding_mid.shape)
        # print(padding_min.shape)

        add = layer.Add()([pool_max, padding_mid, padding_min])
        # print(add.shape)

        conv1 = layer.Conv2D(filters=64, kernel_size=3, strides=1, padding="valid", activation=ACTIVATION,
                             kernel_initializer=tf.keras.initializers.he_uniform())(add)
        conv2 = layer.Conv2D(filters=128, kernel_size=3, strides=1, padding="valid", activation=ACTIVATION,
                             kernel_initializer=tf.keras.initializers.he_uniform())(conv1)
        conv3 = layer.Conv2D(filters=256, kernel_size=1, strides=1, padding="valid", activation=ACTIVATION,
                             kernel_initializer=tf.keras.initializers.he_uniform())(conv2)
        conv1_c = layer.Conv2D(filters=256, kernel_size=1, strides=1, padding="valid", activation=ACTIVATION,
                               kernel_initializer=tf.keras.initializers.he_uniform())(conv1)
        conv2_c = layer.Conv2D(filters=256, kernel_size=1, strides=1, padding="valid", activation=ACTIVATION,
                               kernel_initializer=tf.keras.initializers.he_uniform())(conv2)
        conv3_c = layer.Conv2D(filters=256, kernel_size=1, strides=1, padding="valid", activation=ACTIVATION,
                               kernel_initializer=tf.keras.initializers.he_uniform(), name="img")(conv3)

        # conv3_c Attention
        out_channel = ChannelAttention(conv3_c)
        out_channel = layer.multiply([out_channel, conv3_c], name="img_ch")
        out_spatial = SpatialAttention(out_channel)
        out_cbam_conv3_c = layer.multiply([out_spatial, out_channel], name="img_sp")

        dconv1 = layer.Conv2DTranspose(filters=256, kernel_size=1, strides=1, padding="valid", activation=ACTIVATION)(
            out_cbam_conv3_c)
        add_dconv1_conv2_c = layer.Add()([dconv1, conv2_c])

        # add_dconv1_conv2_c Attention
        out_channel = ChannelAttention(add_dconv1_conv2_c)
        out_channel = out_channel * add_dconv1_conv2_c
        out_spatial = SpatialAttention(out_channel)
        out_cbam_add_dconv1_conv2_c = out_spatial * out_channel

        dconv2 = layer.Conv2DTranspose(filters=256, kernel_size=3, strides=1, padding="valid", activation=ACTIVATION)(
            out_cbam_add_dconv1_conv2_c)
        add_dconv2_conv1_c = layer.Add()([dconv2, conv1_c])
        conv_temp = layer.Conv2D(filters=256, kernel_size=3, strides=1, padding="valid", activation=ACTIVATION,
                                 kernel_initializer=tf.keras.initializers.he_uniform())(
            add_dconv2_conv1_c)
        add_final = layer.Add()([add_dconv1_conv2_c, conv_temp, conv3_c])
        flatten = layer.Flatten()(add_final)
        output = layer.Dense(1, activation="sigmoid", kernel_initializer=tf.keras.initializers.he_uniform())(flatten)

        # Model
        model = tf.keras.Model(inputs=input, outputs=output)
        model.summary()

    return model


def Model(X_train, X_test, y_train, y_test):
    model = GetModel()
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

    model.evaluate(X_test, y_test)

    model.save("./model/shared_model.h5")

    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)
    plt.savefig("./shared/%d/result_%d.png" % (num, num))


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    X_train, X_test, y_train, y_test = LoadData()
    Model(X_train, X_test, y_train, y_test)
