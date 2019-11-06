import cv2
import os
import pandas as pd
import numpy as np

def load_images():
    csv = pd.read_csv('data_AI_QC.csv')
    ID = csv["id"].tolist()
    y = csv["fail"].to_numpy()

    #crée une liste qui contient les cerveaux
    brain_list = []
    n_brains = len(ID)#à changer si vous voulez loader moins de cerveaux
    for i in range(n_brains):
        print("Progression: ", i*100/n_brains, " %")
        #pour chaque cerveau, crée une liste qui contient toutes les coupes supperposées
        observation_list = []
        path = "Data/AI_FS_QC_img/"
        path += str(ID[i])+"/Coronal/merged/"
        #cette partie extrait les images merged pour chaque cerveau
        image_list = []
        try:
            directory = os.fsencode(path)
            for file in os.listdir(directory):
                filename = os.fsdecode(file)
                if filename[-3:] == "png": 
                    img = cv2.imread(path+filename)
                    observation_list.append(img)
            brain_list.append(observation_list)
        except:
            print("directory %s not found"%(path))
    return np.array(brain_list),y

if __name__ == '__main__':
    X,y = load_images()
    print("X: ", X.shape)
    print("y: ", y.shape)
