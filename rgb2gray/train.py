import os
import tensorflow as tf
import tensorflow.keras.layers as layer
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras import backend as K

BATCHSIZE = 32
DROPOUT = 0.2
ACTIVATION = "relu"


def LoadData():
    X = np.load("./X.npy") / 255.0
    Y = np.load("./Y.npy") / 255.0
    X = X.reshape((-1, 256, 256, 3))
    Y = Y.reshape((-1, 256, 256, 1))

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1)

    # return X_train, X_test, Y_train, Y_test
    return X, 0, Y, 0


def LossClass(y_true, y_pred):
    return -np.log(y_true * np.sum(np.exp(y_pred)))


def LossDecolor(y_true, y_pred):
    return K.mean(K.sum(K.square(y_pred - y_true)) / (256 * 256)) / 2


def Loss(y_true, y_pred):
    return LossDecolor(y_true, y_pred)


def Model(X_train, X_test, Y_train, Y_test):
    input = layer.Input(shape=[256, 256, 3])

    # Basic Feature Network
    conv1_1 = layer.Conv2D(filters=64, kernel_size=3, strides=1, padding="same", activation=ACTIVATION, name="conv1_1")(
        input)
    # conv1_1 = layer.Dropout(DROPOUT)(conv1_1)
    conv1_2 = layer.Conv2D(filters=64, kernel_size=3, strides=1, padding="same", activation=ACTIVATION, name="conv1_2")(
        conv1_1)
    # conv1_2 = layer.Dropout(DROPOUT)(conv1_2)
    pool1 = layer.MaxPool2D(name="pool1")(conv1_2)
    conv2_1 = layer.Conv2D(filters=128, kernel_size=3, strides=1, padding="same", activation=ACTIVATION,
                           name="conv2_1")(pool1)
    # conv2_1 = layer.Dropout(DROPOUT)(conv2_1)
    conv2_2 = layer.Conv2D(filters=128, kernel_size=3, strides=1, padding="same", activation=ACTIVATION,
                           name="conv2_2")(conv2_1)
    # conv2_2 = layer.Dropout(DROPOUT)(conv2_2)
    pool2 = layer.MaxPool2D(name="pool2")(conv2_2)
    conv3_1 = layer.Conv2D(filters=256, kernel_size=3, strides=1, padding="same", activation=ACTIVATION,
                           name="conv3_1")(pool2)
    # conv3_1 = layer.Dropout(DROPOUT)(conv3_1)
    conv3_2 = layer.Conv2D(filters=256, kernel_size=3, strides=1, padding="same", activation=ACTIVATION,
                           name="conv3_2")(conv3_1)
    # conv3_2 = layer.Dropout(DROPOUT)(conv3_2)
    conv3_3 = layer.Conv2D(filters=256, kernel_size=3, strides=1, padding="same", activation=ACTIVATION,
                           name="conv3_3")(conv3_2)
    # conv3_3 = layer.Dropout(DROPOUT)(conv3_3)
    pool3 = layer.MaxPool2D(name="pool3")(conv3_3)
    conv4_1 = layer.Conv2D(filters=512, kernel_size=3, strides=1, padding="same", activation=ACTIVATION,
                           name="conv4_1")(pool3)
    # conv4_1 = layer.Dropout(DROPOUT)(conv4_1)
    conv4_2 = layer.Conv2D(filters=512, kernel_size=3, strides=1, padding="same", activation=ACTIVATION,
                           name="conv4_2")(conv4_1)
    # conv4_2 = layer.Dropout(DROPOUT)(conv4_2)
    conv4_3 = layer.Conv2D(filters=512, kernel_size=3, strides=1, padding="same", activation=ACTIVATION,
                           name="conv4_3")(conv4_2)
    # conv4_3 = layer.Dropout(DROPOUT)(conv4_3)
    pool4 = layer.MaxPool2D(name="pool4")(conv4_3)
    basic_features_output = pool4

    # Local Features
    conv5_1 = layer.Conv2D(filters=512, kernel_size=3, strides=1, padding="same", activation=ACTIVATION,
                           name="conv5_1")(basic_features_output)
    # conv5_1 = layer.Dropout(DROPOUT)(conv5_1)
    pool5_1 = layer.MaxPool2D(name="pool5_1")(conv5_1)
    conv5_2 = layer.Conv2D(filters=256, kernel_size=3, strides=1, padding="same", activation=ACTIVATION,
                           name="conv5_2")(pool5_1)
    # conv5_2 = layer.Dropout(DROPOUT)(conv5_2)
    pool5_2 = layer.MaxPool2D(name="pool5_2")(conv5_2)
    conv6_1 = layer.Conv2D(filters=384, kernel_size=1, strides=1, padding="same", activation=ACTIVATION,
                           name="conv6_1")(pool5_2)
    # conv6_1 = layer.Dropout(DROPOUT)(conv6_1)
    conv6_2 = layer.Conv2D(filters=384, kernel_size=1, strides=1, padding="same", activation=ACTIVATION,
                           name="conv6_2")(conv6_1)
    # conv6_2 = layer.Dropout(DROPOUT)(conv6_2)
    conv6_3 = layer.Conv2D(filters=384, kernel_size=1, strides=1, padding="same", activation=ACTIVATION,
                           name="conv6_3")(conv6_2)
    # conv6_3 = layer.Dropout(DROPOUT)(conv6_3)
    pool6 = layer.MaxPool2D(name="pool6")(conv6_3)
    flatten1 = layer.Flatten(name="flatten1")(pool6)
    fcn1 = layer.Dense(units=4096, activation=ACTIVATION, name="fcn1")(flatten1)
    # fcn1 = layer.Dropout(DROPOUT)(fcn1)
    fcn2 = layer.Dense(units=65536, activation=ACTIVATION, name="fcn2")(fcn1)
    # fcn2 = layer.Dropout(DROPOUT)(fcn2)
    reshape1 = layer.Reshape([256, 256, 1], name="reshape1")(fcn2)
    conv6_4 = layer.Conv2D(filters=1, kernel_size=1, strides=1, padding="same", activation=ACTIVATION, name="conv6_4")(
        reshape1)
    # conv6_4 = layer.Dropout(DROPOUT)(conv6_4)
    local_features_output = conv6_4

    # Exposure Features
    conv7_1 = layer.Conv2D(filters=512, kernel_size=3, strides=1, padding="same", activation=ACTIVATION,
                           name="conv7_1")(
        basic_features_output)
    # conv7_1 = layer.Dropout(DROPOUT)(conv7_1)
    conv7_2 = layer.Conv2D(filters=512, kernel_size=3, strides=1, padding="same", activation=ACTIVATION,
                           name="conv7_2")(conv7_1)
    # conv7_2 = layer.Dropout(DROPOUT)(conv7_2)
    conv7_3 = layer.Conv2D(filters=512, kernel_size=3, strides=1, padding="same", activation=ACTIVATION,
                           name="conv7_3")(conv7_2)
    # conv7_3 = layer.Dropout(DROPOUT)(conv7_3)
    pool7 = layer.MaxPool2D(name="pool7")(conv7_3)
    conv8_1 = layer.Conv2D(filters=256, kernel_size=3, strides=1, padding="same", activation=ACTIVATION,
                           name="conv8_1")(pool7)
    # conv8_1 = layer.Dropout(DROPOUT)(conv8_1)
    pool8_1 = layer.MaxPool2D(name="pool8_1")(conv8_1)
    conv8_2 = layer.Conv2D(filters=384, kernel_size=1, strides=1, padding="same", activation=ACTIVATION,
                           name="conv8_2")(pool8_1)
    # conv8_2 = layer.Dropout(DROPOUT)(conv8_2)
    conv8_3 = layer.Conv2D(filters=384, kernel_size=1, strides=1, padding="same", activation=ACTIVATION,
                           name="conv8_3")(conv8_2)
    # conv8_3 = layer.Dropout(DROPOUT)(conv8_3)
    pool8_2 = layer.MaxPool2D(name="pool8_2")(conv8_3)
    flatten2 = layer.Flatten(name="flatten2")(pool8_2)
    fcn3 = layer.Dense(units=4096, activation=ACTIVATION, name="fcn3")(flatten2)
    # fcn3 = layer.Dropout(DROPOUT)(fcn3)
    fcn4 = layer.Dense(units=65536, activation=ACTIVATION, name="fcn4")(fcn3)
    # fcn4 = layer.Dropout(DROPOUT)(fcn4)
    # classes_output = layer.Dense(units=3, activation="softmax")(fcn4)
    reshape2 = layer.Reshape([256, 256, 1], name="reshape2")(fcn4)
    conv8_4 = layer.Conv2D(filters=1, kernel_size=1, strides=1, padding="same", activation=ACTIVATION, name="conv8_4")(
        reshape2)
    # conv8_4 = layer.Dropout(DROPOUT)(conv8_4)
    exposure_features_output = conv8_4

    # Decolorization
    upsample1_local = layer.UpSampling2D()(local_features_output)
    upsample1_exposure = layer.UpSampling2D()(exposure_features_output)
    upsample2_local = layer.UpSampling2D()(upsample1_local)
    upsample2_exposure = layer.UpSampling2D()(upsample1_exposure)
    upsample3_local = layer.UpSampling2D()(upsample2_local)
    upsample3_exposure = layer.UpSampling2D()(upsample2_exposure)
    conv9_local = layer.Conv2D(filters=1, kernel_size=1, strides=1, padding="same")(local_features_output)
    conv9_exposure = layer.Conv2D(filters=1, kernel_size=1, strides=1, padding="same")(exposure_features_output)
    crop1_local = layer.Cropping2D()(conv9_local)
    crop1_exposure = layer.Cropping2D()(conv9_exposure)
    add = layer.Add()([crop1_local, crop1_exposure])
    deconv = layer.Conv2DTranspose(filters=1, kernel_size=1, strides=1)(add)
    crop2 = layer.Cropping2D()(deconv)
    output = crop2

    # conv1 = layer.Conv2D(filters=256, kernel_size=1, strides=1, padding="same", activation=ACTIVATION)(
    #     local_features_output)
    # # conv1 = layer.Dropout(DROPOUT)(conv1)
    # conv2 = layer.Conv2D(filters=256, kernel_size=1, strides=1, padding="same", activation=ACTIVATION)(
    #     exposure_features_output)
    # # conv2 = layer.Dropout(DROPOUT)(conv2)
    # add = layer.Add()([conv1, conv2])
    # conv = layer.Conv2D(filters=128, kernel_size=1, strides=1, padding="same", activation=ACTIVATION)(add)
    # # conv = layer.Dropout(DROPOUT)(conv)
    # up = layer.UpSampling2D()(conv)
    # conv = layer.Conv2D(filters=64, kernel_size=1, strides=1, padding="same", activation=ACTIVATION)(up)
    # # conv = layer.Dropout(DROPOUT)(conv)
    # conv = layer.Conv2D(filters=64, kernel_size=1, strides=1, padding="same", activation=ACTIVATION)(conv)
    # # conv = layer.Dropout(DROPOUT)(conv)
    # up = layer.UpSampling2D()(conv)
    # conv = layer.Conv2D(filters=32, kernel_size=1, strides=1, padding="same", activation=ACTIVATION)(up)
    # # conv = layer.Dropout(DROPOUT)(conv)
    # conv = layer.Conv2D(filters=32, kernel_size=1, strides=1, padding="same", activation=ACTIVATION)(conv)
    # # conv = layer.Dropout(DROPOUT)(conv)
    # up = layer.UpSampling2D()(conv)
    # conv = layer.Conv2D(filters=16, kernel_size=1, strides=1, padding="same", activation=ACTIVATION)(up)
    # # conv = layer.Dropout(DROPOUT)(conv)
    # conv = layer.Conv2D(filters=16, kernel_size=1, strides=1, padding="same", activation=ACTIVATION)(conv)
    # # conv = layer.Dropout(DROPOUT)(conv)
    # conv = layer.Conv2D(filters=1, kernel_size=1, strides=1, padding="same", activation=ACTIVATION)(conv)
    # # conv = layer.Dropout(DROPOUT)(conv)
    # output = conv

    # Classes
    # classes = tf.keras.Model(inputs=input, outputs=classes_output)
    # classes.summary()

    # Model
    model = tf.keras.Model(inputs=input, outputs=output)
    model.summary()
    model.compile(optimizer=tf.keras.optimizers.Adam(0.0001), loss=Loss)
    # history = model.fit(X_train, Y_train, batch_size=BATCHSIZE, epochs=100, validation_split=0.2)
    history = model.fit(X_train, Y_train, batch_size=BATCHSIZE, epochs=1000)

    # model.evaluate(X_test, Y_test)

    plt.figure(1, (8, 5))
    plt.gca().set_ylim(0, 0.1)
    plt.grid(True)
    plt.plot(history.epoch, history.history.get("loss"), label="loss")
    # plt.plot(history.epoch, history.history.get("val_loss"), label="val_loss")
    plt.legend()
    plt.show()

    plt.savefig("./loss.png")
    model.save("./model.h5")


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    X_train, X_test, Y_train, Y_test = LoadData()
    Model(X_train, X_test, Y_train, Y_test)
