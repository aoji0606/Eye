import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio

arr = [19, 21, 23, 25, 27, 29, 31, 33]

for shape in arr:
    print(shape)
    R = scio.loadmat("./data/%d/Eoptha_R_%d.mat" % (shape, shape))
    R = np.asarray(R["Eoptha_R"])
    num = int(R.shape[0] / R.shape[1])
    row = 0
    X_R = []
    for i in range(num):
        patch = R[row:row + R.shape[1], :]
        patch = patch.reshape(1, -1)
        X_R.append(patch)
        row += R.shape[1]
    X_R = np.asarray(X_R)

    G = scio.loadmat("./data/%d/Eoptha_G_%d.mat" % (shape, shape))
    G = np.asarray(G["Eoptha_G"])
    num = int(G.shape[0] / G.shape[1])
    row = 0
    X_G = []
    for i in range(num):
        patch = G[row:row + G.shape[1], :]
        patch = patch.reshape(1, -1)
        X_G.append(patch)
        row += G.shape[1]
    X_G = np.asarray(X_G)

    B = scio.loadmat("./data/%d/Eoptha_B_%d.mat" % (shape, shape))
    B = np.asarray(B["Eoptha_B"])
    num = int(B.shape[0] / B.shape[1])
    row = 0
    X_B = []
    for i in range(num):
        patch = B[row:row + B.shape[1], :]
        patch = patch.reshape(1, -1)
        X_B.append(patch)
        row += B.shape[1]
    X_B = np.asarray(X_B)

    X_R = X_R.reshape((-1, shape, shape, 1)) / 255.0
    X_G = X_G.reshape((-1, shape, shape, 1)) / 255.0
    X_B = X_B.reshape((-1, shape, shape, 1)) / 255.0
    X = np.concatenate((X_R, X_G, X_B), axis=-1)
    print(X.shape)

    label = scio.loadmat("./data/%d/Eoptha_label_%d.mat" % (shape, shape))
    data = np.asarray(label["Label"])
    y = data.reshape(-1, 1)
    print(y.shape)

    np.save("./data/%d/X.npy" % shape, X)
    np.save("./data/%d/y.npy" % shape, y)
    print()
