import numpy as np

X = np.load("zone_0_scores_X.npy")
y = np.load("zone_0_scores_y.npy")

print(np.max(X))
mean = np.sum(X, axis = 1)/len(X[0])

X_pass = mean[np.where(y == 0)]
X_fail = mean[np.where(y == 1)]

print(sum(X_pass)/len(X_pass), sum(X_fail)/len(X_fail))
