import numpy as np


def load_data():
    X = list()
    for i in range(8):
        X.append(np.load("data/binary/binary_%s"%i))
    X = np.array(X)
    Y = np.load("data/binary/binary_y")
    return X,Y


X, Y = load_data()
print(X.size)
print(Y.size)


