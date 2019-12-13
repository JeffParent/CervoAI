import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

X0 = np.load("zone_1_scores_X.npy")
X1 = np.load("zone_0_scores_X.npy")
X = np.concatenate((X0,X1),axis = 1)
y = np.load("zone_0_scores_y.npy")


X_pass = X[np.where(y == 0),:][0]
X_fail = X[np.where(y == 1),:][0]

#X = np.concatenate((X_pass[:len(X_fail)], X_fail), axis = 0)
#y = y[len(X_pass)-len(X_fail):]
#print(X.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42, shuffle = True)


gamma_list = [0.0001,0.001, 0.005,0.01,0.1,1,10, 100]
C_list = [0.01,0.1,1,10,100,1000, 5000, 10000,20000, 100000]
maximum_score_SVM = [0,[0,0]]
for gamma in gamma_list:
    for C in C_list:
        SVM = SVC(gamma = gamma, C = C)
        SVM.fit(X_train[:500], y_train[:500])
        score = SVM.score(X_train[500:], y_train[500:])
        if score > maximum_score_SVM[0]:
            maximum_score_SVM[0] = score
            maximum_score_SVM[1] = [gamma, C]
print("SVM: Meilleurs paramètres: gamma: %s, C: %s" %(maximum_score_SVM[1][0], maximum_score_SVM[1][1]))


clf = SVC(gamma=0.001, C = 20000)
clf.fit(X_train, y_train)
print(clf.score(X_train,y_train))
print(clf.score(X_test,y_test))

