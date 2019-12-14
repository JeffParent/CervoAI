import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as pyplot

X0 = np.load("Axial_zone_1_scores_X.npy")
X1 = np.load("Axial_zone_0_scores_X.npy")
X2 = np.load("zone_1_scores_X.npy")
X3 = np.load("zone_0_scores_X.npy")
X6 = np.load("Axial_simple_scores_X.npy")
X7 = np.load("Coranal_simple_scores_X.npy")


X8 = np.load("One_for_all_X.npy")
X = np.concatenate((X0,X1,X2,X3),axis = 1)
#X = np.nan_to_num(np.load("One_for_all_X_2.npy"),copy=True)
y = np.load("zone_0_scores_y.npy")


X = np.load("Coronal_simple_scores_X_total.npy")
y = np.load("Coronal_simple_scores_y_total.npy")

num_bins = 20
fig, ax = pyplot.subplots()
# the histogram of the data
Pass = X[np.where(y == 0)]
Fail = X[np.where(y == 1)]

n, bins, patches = ax.hist(np.reshape(Pass, (len(Pass)*20,)), num_bins, density=1, label = "Pass distribution")
n, bins, patches = ax.hist(np.reshape(Fail, (len(Fail)*20,)), num_bins, density=1, label = "Fail distribution")
pyplot.legend()
#pyplot.ylabel("Proportion de l'aire de segmentation sur l'aire totale")
pyplot.xlabel("Proportion de l'aire de segmentation sur l'aire totale")
pyplot.title("Distribution des proportions")
pyplot.show()

print(X.shape)
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
clf = SVC(gamma=maximum_score_SVM[1][0], C = maximum_score_SVM[1][1], probability=True)
#clf = SVC(gamma=0.001, C = 20000)

clf.fit(X_train, y_train)

zeros_list = []
ones_list = []
for i in range(0,40):
    y_pred_proba = clf.predict_proba(X_test)
    def predict(y_pred_proba, seuil = 0.5):
        y_pred = np.zeros((len(y_pred_proba)))
        y_pred[np.where(y_pred_proba[:,1] > seuil)] = 1
        return y_pred
    y_pred = predict(y_pred_proba, i/40)
    conf = confusion_matrix(y_test, y_pred)
    zeros_list.append(conf[0,0]/(conf[0,0]+conf[0,1]))
    ones_list.append(conf[1,1]/(conf[1,1]+conf[1,0]))
zeros_list = np.array(zeros_list)
ones_list = np.array(ones_list)
a1 = np.reshape(zeros_list, (len(zeros_list),1))
a2 = np.reshape(ones_list, (len(ones_list),1))
a = np.concatenate((a1,a2),axis = 1)
pyplot.plot(np.linspace(0,1,40),a[:,0], label = "Rappel fail")
pyplot.plot(np.linspace(0,1,40),a[:,1], label = "Rappel pass")
pyplot.plot(np.linspace(0,1,40),np.sum(a,axis = 1)/2, label = "Rappel total")

pyplot.ylabel('Rappel')
pyplot.xlabel('Threshold de classification')
pyplot.title("Rappel en fonction du threshold")
pyplot.legend()
pyplot.show()

#print("X_test len: ",len(X_test))
y_pred = clf.predict(X_test)
conf = confusion_matrix(y_test, y_pred)
print("One model per label: Coronal + Axial")
print(conf)
#print(np.sum(conf,axis=1))
print("Train score: ", clf.score(X_train,y_train))
print("Test score: ", clf.score(X_test,y_test))

