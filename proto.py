import numpy as np


def load_data():
    X = list()
    for i in range(8):
        X.append(np.load("data/binary/binary_%s.npy"%i))
    X = np.array(X)
    Y = np.load("data/binary/binary_y.npy")
    return X,Y


X, Y = load_data()
print(X.size)
print(Y.size)


