import numpy as np


def load_data():
    X = np.load("data/binary/binary_0.npy")
    for i in range(8):
        print()
        X = np.concatenate((X, np.load("data/binary/binary_{}.npy".format(i))))
    Y = np.load("data/binary/binary_y.npy")
    return X, Y


X, Y = load_data()
print(X.size)
print(Y.size)


