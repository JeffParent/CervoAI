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
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.index = index
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.index)*20


    def extract_image(self, img_folder_path, idx):
        file = os.listdir(img_folder_path)[idx]
        filename = os.fsdecode(file)
        if filename[-3:] == "png":
            img_name = os.path.join(img_folder_path, filename)
            image = io.imread(img_name)
            return image
        
    def __getitem__(self, idx):
        rest = idx%20
        
        if torch.is_tensor(idx):
            idx = idx.tolist()

        X_folder_path = os.path.join(self.root_dir, self.index[int(idx/20), 0], "Axial", "t1", "") #self.index[int(idx/20), 0] à la place de "12648-10464"
        y_folder_path = os.path.join(self.root_dir, self.index[int(idx/20), 0], "Axial", "labels", "")
        
        X = self.extract_image(X_folder_path, rest)
        X = X[:,:,:3]
        X = cv2.cvtColor(X, cv2.COLOR_BGR2GRAY)

        y = self.extract_image(y_folder_path, rest)
        y = y[:,:,:3]
        y = cv2.cvtColor(y, cv2.COLOR_BGR2GRAY)

        return X, y


def compute_Score(X,y):
    X[np.where(X > 0)] = 1
    y[np.where(y > 0)] = 1
    good_pred = len(np.where((X + y) == 2)[0])
    total = good_pred + len(np.where((X + y) == 1)[0])
        
    if total == 0:
        return 0
    return good_pred/total

def trainTestSplit(dataLen = 7000, trainTestRatio = 0.8, csv_file = 'data/raw/AI_FS_QC_img/data_AI_QC.csv'):
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
    print("Version 1.1")

    train_index, test_index, Fail_index = trainTestSplit(dataLen = 7100, trainTestRatio = 0.9)

    X = []
    y = []
    cervo_dataset_test = CervoDataset(root_dir='data/raw/AI_FS_QC_img/', index = test_index)
    cervo_dataset_Fail = CervoDataset(root_dir='data/raw/AI_FS_QC_img/', index = Fail_index)
    
    for brain_no in range(len(test_index)):
        score = []
        for img in range(20):
            image, label = cervo_dataset_test.__getitem__(brain_no*20 + img)

            score.append(compute_Score(image, Label))
        X.append(score)
        y.append(0)
    
    for brain_no in range(len(Fail_index)):
        score = []
        for img in range(20):
            image, label = cervo_dataset_Fail.__getitem__(brain_no*20 + img)

            score.append(compute_Score(image, Label))
        X.append(score)
        y.append(1)

    X = np.array(X)
    y = np.array(y)
    np.save("saves/Axial_simple_scores_X", X)
    np.save("saves/Axial_simple_scores_y", y)
    #trained = unet.train(nb_epoch = 3, learning_rate = 0.01, momentum = 0.99, batch_size = 32, train_index = train_index)
    #torch.save(trained.state_dict(), "models/model_zone_%s" %(label))

