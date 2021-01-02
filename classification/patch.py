import tensorflow.keras.layers as l

arr = [13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33]

for i in arr:
    print(i)
    shape = []
    k_s = []
    input = l.Input([i, i, 1])

    while (len(shape) < 3):
        for k in range(2, int(i / 2) + 1, 1):
            for s in range(2, k + 1, 1):
                pool = l.MaxPool2D(pool_size=(k, k), strides=(s, s))(input)
                if pool.shape[1] % 2 and pool.shape[1] >= 5 and pool.shape[1] <=9 and pool.shape[1] not in shape and len(shape) < 3:
                    shape.append(pool.shape[1])
                    k_s.append([k, s])
        break
    print(shape)
    print(k_s)
    print()
