import cv2
import os
import pandas as pd
import numpy as np



def load_images():
    csv = pd.read_csv("data/raw/AI_FS_QC_img/data_AI_QC.csv")
    ID = csv["id"].tolist()
    y = csv["fail"].to_numpy()

    #fonction qui supperpose l'image en couleur (label) et l'image grise IRM (t1)
    def merge_label_t1(label_img, t1_img):
        foreground = label_img
        background = t1_img
        background[np.where((foreground!=[0,0,0]).all(axis=2))] = np.array([0,0,0])
        return cv2.add(foreground, background)

    #crée une liste qui contient les cerveaux
    brain_list = []
    n_brains = len(ID)#à changer si vous voulez loader moins de cerveaux
    for i in range(n_brains):
        print("Progression: ", i*100/n_brains, " %")
        #pour chaque cerveau, crée une liste qui contient toutes les coupes supperposées
        observation_list = []
        path = "data/raw/AI_FS_QC_img/"
        path += str(ID[i])+"/Coronal/"
        #cette partie extrait les images labels et t1 pour chaque cerveau. Elle fait ensuite deux listes (une pour
        #les coupes label et une pour t1). Elle met ces deux listes dans le dictionnaire image_dict. 
        image_dict = {} 
        for j in ["labels/","t1/"]:
            image_list = []
            try:
                directory = os.fsencode(path+j)
                for file in os.listdir(directory):
                    filename = os.fsdecode(file)
                    if filename[-3:] == "png": 
                        img = cv2.imread(path+j+filename)
                        image_list.append(img)
            except:
                print("directory %s not found"%(path+j))
            image_dict[j] = image_list
        #cette partie appelle la fonction merge_label_t1(). Elle ajoute ensuite l'image supperposée dans observation_list.
        #lorsque toutes les coupes pour un cerveau ont été traitées, elle ajoute observation_list à brain_list.
        #Elle save aussi chaque coupe dans un nouveau dossier "merged"
        if len(image_dict["labels/"]) != 0 and len(image_dict["t1/"]) != 0:
            try:
                os.mkdir(path+"merged")
            except:
                pass
            try:
                for j in range(20):
                    merged_image = merge_label_t1(image_dict["labels/"][j], image_dict["t1/"][j])
                    cv2.imwrite(path+"merged/img_"+str(j)+".png", merged_image)
                    observation_list.append(merged_image)
                brain_list.append(observation_list)
            except:
                pass
    return brain_list,y

if __name__ == '__main__':
    X,y = load_images()
    n_brains_per_save = 1000 #save par batch pour ne pas exploser la mémoire Ram. 
    for i in range(int(len(X)/n_brains_per_save)):
        np.save("data/binary/binary_%s"%(i), np.array(X[i*n_brains_per_save:i*n_brains_per_save+n_brains_per_save]))
    np.save("data/binary/binary_%s"%(i+1), np.array(X[(i+1)*n_brains_per_save:]))
    np.save("data/binary/binary_y", y)
    print("X: ", len(X))
    print("y: ", len(y))
