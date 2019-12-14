import os
import time
import torch
from skimage import io
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
import cv2


class CervoDataset(Dataset):
    def __init__(self, root_dir, index, transform=None):
        """
        Dataset pour les données
        In
            root_dir: dossier qui contient les données
            index: Nom des cerveaux
        """
        self.index = index
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.index)*20


    def extract_image(self, img_folder_path, idx):
        '''
        Extrait l'image demandée
        In
            img_folder_path: dossier du cerveau
            idx: indexe de l'image à extraire du cerveau
        Out
            Image 256x256x3
        '''
        file = os.listdir(img_folder_path)[idx]
        filename = os.fsdecode(file)
        if filename[-3:] == "png":
            img_name = os.path.join(img_folder_path, filename)
            image = io.imread(img_name)
            return image
        
    def __getitem__(self, idx):
        '''
        Extrait l'image du cerveau demandé
        In
            Idx: indexe entre 0 et __len__
        Out
            X: Image grise 265x256x1
            y: Image du label 265x256x1
        '''
        rest = idx%20
        
        if torch.is_tensor(idx):
            idx = idx.tolist()

        X_folder_path = os.path.join(self.root_dir, self.index[int(idx/20), 0], "Coronal", "t1", "") 
        y_folder_path = os.path.join(self.root_dir, self.index[int(idx/20), 0], "Coronal", "labels", "")
        
        X = self.extract_image(X_folder_path, rest)
        X = X[:,:,:3]
        X = cv2.cvtColor(X, cv2.COLOR_BGR2GRAY)

        y = self.extract_image(y_folder_path, rest)
        y = y[:,:,:3]
        y = cv2.cvtColor(y, cv2.COLOR_BGR2GRAY)

        return X, y


def compute_Score(X,y):
    '''
    Calcule la propostion de l'aire labelée par le logiel et l'aire totale
    In
       X: Image grise
       y: Image labelée
    Out: Score
    '''
    X[np.where(X > 0)] = 1
    y[np.where(y > 0)] = 1
    good_pred = len(np.where((X + y) == 2)[0])
    total = good_pred + len(np.where((X + y) == 1)[0])
        
    if total == 0:
        return 0
    return good_pred/total

def trainTestSplit(dataLen = 7000, trainTestRatio = 0.8, csv_file = 'data/raw/AI_FS_QC_img/data_AI_QC.csv'):
    '''
    Sépare les données en train et test. 
    In:
        datalen: nombre de cerveaux à traîter
        traintestRatio: ration entre train et test
        csv_file: path vers le csv
    Out:
        Array qui contient les noms des cerveaux pour train et test des Pass ainsi que pour les Fails
    '''
    labels = pd.read_csv(csv_file).values
    Pass = labels[np.where(labels[:,1] == 0)]
    Fail = labels[np.where(labels[:,1] == 1)]
    dataLen -= len(Fail)
    linspace = np.arange(dataLen)
    np.random.seed(seed=42)
    np.random.shuffle(linspace)
    train_index = linspace[:int(dataLen*trainTestRatio)]
    test_index = linspace[int(dataLen*trainTestRatio):]
    train_index = Pass[train_index]
    test_index = Pass[test_index]
    print("nb. pass: ", len(Pass))
    print("nb. fail: ", len(Fail))
    return train_index, test_index, Fail  

if __name__ == '__main__':
    train_index, test_index, Fail_index = trainTestSplit(dataLen = 7100, trainTestRatio = 0.05)

    X = []
    y = []
    cervo_dataset_test = CervoDataset(root_dir='data/raw/AI_FS_QC_img/', index = test_index)
    cervo_dataset_Fail = CervoDataset(root_dir='data/raw/AI_FS_QC_img/', index = Fail_index)
    
    for brain_no in range(len(test_index)):
        score = []
        for img in range(20):
            image, label = cervo_dataset_test.__getitem__(brain_no*20 + img)

            score.append(compute_Score(image, label))
        X.append(score)
        y.append(0)
    
    for brain_no in range(len(Fail_index)):
        score = []
        for img in range(20):
            image, label = cervo_dataset_Fail.__getitem__(brain_no*20 + img)

            score.append(compute_Score(image, label))
        X.append(score)
        y.append(1)

    X = np.array(X)
    y = np.array(y)
    np.save("saves/Coronal_simple_scores_X", X)
    np.save("saves/Coronal_simple_scores_y", y)

