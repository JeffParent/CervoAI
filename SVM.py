import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as pyplot

class SVM_classifier():
    def __init__(self):
        pass

    def optimize_hyperparameters(self, X_train, y_train):
        '''
        Optimise les hyperparamètres gamma et C du SVM sur un jeu de validation
        qui vient du train
        In
            X_train
            y_train
        Out
            SVM optimisé avec les bons hyperparamètres
        '''
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
        self.clf = clf
        return clf

    def variate_classification_threshold(self, X_test, y_test):
        '''
        Calcule le rapel de chaque classe en fonction du seuille de classification
        In
            X_test
            y_test
        '''
        zeros_list = []
        ones_list = []
        for i in range(0,40):
            y_pred_proba = self.clf.predict_proba(X_test)
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
        pyplot.plot(np.linspace(0,1,40),a[:,0], label = "Fail")
        pyplot.plot(np.linspace(0,1,40),a[:,1], label = "Pass")
        #pyplot.plot(np.linspace(0,1,40),np.sum(a,axis = 1)/2, label = "Rappel total")
        pyplot.ylabel('Rappel')
        pyplot.xlabel('Threshold de classification')
        pyplot.title("Rappel en fonction du threshold")
        pyplot.legend()
        pyplot.ylim(0)
        
        pyplot.show()

    def fit(self,X_train, y_train):
        self.clf.fit(X_train, y_train)

    def score(self,X_train,y_train,X_test,y_test):
        '''
        Calcule le score en train et test
        '''
        print("Train score: ", self.clf.score(X_train,y_train))
        print("Test score: ", self.clf.score(X_test,y_test))


def plot_proportion_distribution(X, y):
    '''
    Fait un histogramme de la distribution des proportions pour chaque classe
    In
        X et y
    '''
    num_bins = 20
    fig, ax = pyplot.subplots()
    Pass = X[np.where(y == 0)]
    Fail = X[np.where(y == 1)]

    n, bins, patches = ax.hist(np.reshape(Pass, (len(Pass)*len(Pass[0]),)), num_bins, density=1, label = "Pass distribution")
    n, bins, patches = ax.hist(np.reshape(Fail, (len(Fail)*len(Fail[0]),)), num_bins, density=1, label = "Fail distribution")
    pyplot.legend()
    pyplot.xlabel("Proportion de l'aire de segmentation sur l'aire totale")
    pyplot.title("Distribution des proportions")
    pyplot.show()

if __name__ == '__main__':
    '''
    Prends les scores calculés pour chaque méthode de segmentation 
    et les donne à un SVM pour prédire si un cerveau passe ou fail
    '''
    X0 = np.load("saves/Axial_zone_1_scores_X.npy")
    X1 = np.load("saves/Axial_zone_0_scores_X.npy")
    X2 = np.load("saves/zone_1_scores_X.npy")
    X3 = np.load("saves/zone_0_scores_X.npy")
    X4 = np.load("saves/Axial_simple_scores_X.npy")
    X5 = np.load("saves/Coranal_simple_scores_X.npy")
    X6 = np.load("saves/One_for_all_X.npy")
    #print(np.max(X), np.min(X))
    X6 -= np.min(X6)
    X = X6/(np.max(X6)-np.min(X6))


    #X = np.concatenate((X0,X1,X2,X3,X6),axis = 1)
    
    y = np.load("saves/zone_0_scores_y.npy")

    plot_proportion_distribution(X, y)

    print(X.shape)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42, shuffle = True)
    SVM = SVM_classifier()
    SVM.optimize_hyperparameters(X_train, y_train)
    SVM.fit(X_train, y_train)
    SVM.variate_classification_threshold(X_test, y_test)
    SVM.score(X_train,y_train,X_test,y_test)

