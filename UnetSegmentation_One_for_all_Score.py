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
import imagehash
from PIL import Image



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
            X: Image en couleur 265x256x3
            y: Image du label 265x256x3
        '''
        rest = idx%20
        
        if torch.is_tensor(idx):
            idx = idx.tolist()

        X_folder_path = os.path.join(self.root_dir, self.index[int(idx/20), 0], "Coronal", "t1", "") #self.index[int(idx/20), 0] à la place de "12648-10464"
        y_folder_path = os.path.join(self.root_dir, self.index[int(idx/20), 0], "Coronal", "labels", "")
        
        X = self.extract_image(X_folder_path, rest)
        X = X[:,:,:3]

        y = self.extract_image(y_folder_path, rest)
        y = y[:,:,:3]

        if self.transform is not None:
            trans = transforms.Compose([transforms.ToTensor()]+self.transform)
            X = (trans(X))
            y = (trans(y))
        else:
            X = transforms.ToTensor()(X)
            y = transforms.ToTensor()(y)

        return X, y


class u_net():
    def __init__(self, data_path, trained_model = None, device = "cuda"):
        '''
        In
            data_path: path vers les images
            trained_model: modèle unet qui predit la segmetation
            device = "cpu" ou "cuda"

        '''
        self.data_path = data_path
        self.device = device
        if trained_model != None:
            self.model = trained_model



    def predict(self, image_index, test_index):
        '''
        Fait la segmentation en couleur de toutes les zones pour le cerveau 
        In 
            image_index: indexe du cerveau et de l'image à aller chercher pour segmenter
            test_index: noms des cerveaux à tester. 
        Out
            Image brute 256x256x3, image segmentée pour u-net 256x256x3, image labelée par le logiciel 256x256x3
        '''
        self.cervo_dataset = CervoDataset(root_dir=self.data_path, index = test_index)
        self.model.eval()
        image, label = self.cervo_dataset.__getitem__(image_index)
        image = (image.unsqueeze(0)).to(self.device)
        label = (label.unsqueeze(0)).to("cpu")
        with torch.no_grad():
            prediction = self.model(image)

        image = image.cpu()
        prediction = prediction.cpu()
        label = label.cpu()

        image = image[0].permute(1, 2, 0).numpy()
        prediction = prediction.detach()[0].permute(1, 2, 0).numpy()
        label = label[0].permute(1, 2, 0).numpy()

        return image, prediction, label 

    def score(self, prediction, label):
        '''
        Calcule la multiplication des normes entre les deux images
        In
           X: Image prédite en couleur par le u-net
           y: Image labelée
        Out: Score
        '''
        prediction = (prediction*255).astype(np.uint8)
        label = (label*255).astype(np.uint8)
        prediction= Image.fromarray(prediction)
        label= Image.fromarray(label)
        pred = imagehash.average_hash(prediction, hash_size=256)
        lab = imagehash.average_hash(label, hash_size=256)
        score = abs(lab-pred)
        if score > 33000:
            prediction.save("saves/pred", "JPEG")
            label.save("saves/label", "JPEG")
            print(1/0)

        '''
        score = 0
        for i in range(3):
            pred = prediction[:,:,i]
            lab = label[:,:,i]
            picture1_norm = pred/np.sqrt(np.sum(pred**2))
            picture2_norm = lab/np.sqrt(np.sum(lab**2))
            score += np.sum(picture2_norm*picture1_norm)/3
        if score > 0.1:
            print(np.max(label), np.max(prediction))
            print(1/0)
        '''
        return score

     
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
    '''
    Calcule le score pour la segmetation en couleur du u-net. 
    Enregistre les données dans saves pour qu'elles soient
    utilisées plus tard par le SVM
    '''
    trained = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet', in_channels=3, out_channels=3, init_features=32, pretrained=False)
    trained.load_state_dict(torch.load("models/model0"))
    trained.cuda()
    unet = u_net(data_path = 'data/raw/AI_FS_QC_img/', device = "cuda", trained_model = trained)

    train_index, test_index, Fail_index = trainTestSplit(dataLen = 7100, trainTestRatio = 0.9)

    X = []
    y = []
    for brain_no in range(len(test_index)):
        score = []
        for img in range(20):
            gray, prediction, Label = unet.predict(brain_no*20 + img, test_index)
            score.append(unet.score(prediction, Label))
        X.append(score)
        y.append(0)
    
    for brain_no in range(len(Fail_index)):
        score = []
        for img in range(20):
            gray, prediction, Label = unet.predict(brain_no*20 + img, Fail_index)
            score.append(unet.score(prediction, Label))
        X.append(score)
        y.append(1)
    X = np.array(X)
    y = np.array(y)
    np.save("saves/One_for_all_X", X)
    np.save("saves/One_for_all_y", y)


