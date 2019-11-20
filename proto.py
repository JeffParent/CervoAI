import numpy as np


def load_data():
    X = np.load("data/binary/binary_%0.npy")
    for i in range(8):
        X = np.concatenate(X, np.load("data/binary/binary_%s".format(i)))
    Y = np.load("data/binary/binary_y.npy")
    return X, Y


X, Y = load_data()
print(X.size)
print(Y.size)


