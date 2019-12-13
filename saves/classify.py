import numpy as np

X = np.load("zone_0_scores_X.npy")
y = np.load("zone_0_scores_y.npy")

mean = np.sum(X, axis = 0)/len(X)

X_pass = X[np.where(y == 0),:][0]
mean_pass = np.sum(X_pass, axis = 0)/len(X_pass)
X_fail = X[np.where(y == 1),:][0]
mean_fail = np.sum(X_fail, axis = 0)/len(X_fail)

#print(mean_fail, mean_pass)
for i in range(len(mean_fail)):
    print(mean_fail[i], mean_pass[i])
