import sys
import numpy as np
import scipy.io as scio

arr = [19, 21, 23, 25, 27, 29, 31, 33]

num1 = int(sys.argv[1])
num2 = int(sys.argv[2])


for shape in arr:
    print(shape)
    
    R = scio.loadmat("./data/%d/%d/test/%d/IDRiD_R_%d.mat" % (num1, num2, shape, shape))
    R = np.asarray(R["IDRiD_test_R"])
    num = int(R.shape[0] / R.shape[1])
    row = 0
    X_R = []
    for i in range(num):
        patch = R[row:row + R.shape[1], :]
        patch = patch.reshape(1, -1)
        X_R.append(patch)
        row += R.shape[1]
    X_R = np.asarray(X_R)

    G = scio.loadmat("./data/%d/%d/test/%d/IDRiD_G_%d.mat" % (num1, num2, shape, shape))
    G = np.asarray(G["IDRiD_test_G"])
    num = int(G.shape[0] / G.shape[1])
    row = 0
    X_G = []
    for i in range(num):
        patch = G[row:row + G.shape[1], :]
        patch = patch.reshape(1, -1)
        X_G.append(patch)
        row += G.shape[1]
    X_G = np.asarray(X_G)

    B = scio.loadmat("./data/%d/%d/test/%d/IDRiD_B_%d.mat" % (num1, num2, shape, shape))
    B = np.asarray(B["IDRiD_test_B"])
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

    label = scio.loadmat("./data/%d/%d/test/%d/IDRiD_label_%d.mat" % (num1, num2, shape, shape))
    data = np.asarray(label["IDRiD_test_Label"])
    y = data.reshape(-1, 1)
    print(y.shape)

    np.save("./data/%d/%d/test/%d/X.npy" % (num1, num2, shape), X)
    np.save("./data/%d/%d/test/%d/y.npy" % (num1, num2, shape), y)
    print()


'''
for shape in arr:
    print(shape)
    gray = scio.loadmat("./data/%d/%d/%d/SLOI_%d.mat" % (num1, num2, shape, shape))
    gray = np.asarray(gray["SLOI"])
    num = int(gray.shape[0] / gray.shape[1])
    row = 0
    X = []
    for i in range(num):
        patch = gray[row:row + gray.shape[1], :]
        patch = patch.reshape(1, -1)
        X.append(patch)
        row += gray.shape[1]
    X = np.asarray(X)

    X = X.reshape((-1, shape, shape, 1)) / 255.0
    print(X.shape)

    label = scio.loadmat("./data/%d/%d/%d/SLOI_label_%d.mat" % (num1, num2, shape, shape))
    data = np.asarray(label["Label"])
    y = data.reshape(-1, 1)
    print(y.shape)

    np.save("./data/%d/%d/%d/X.npy" % (num1, num2, shape), X)
    np.save("./data/%d/%d/%d/y.npy" % (num1, num2, shape), y)
    print()
'''
