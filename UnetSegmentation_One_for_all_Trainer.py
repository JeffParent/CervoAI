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

    def __init__(self, csv_file, root_dir, index, transform=None):
        """
        Dataset pour les données
        In
            root_dir: dossier qui contient les données
            index: Nom des cerveaux
        """
        self.index = index
        self.labels = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.index)*20 #len(self.labels)*20

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
            X: Image grise 265x256x3
            y: Image du label 265x256x3
        '''
        rest = idx%20
        idx = self.index[int(idx/20)]
        
        if torch.is_tensor(idx):
            idx = idx.tolist()
        X_folder_path = os.path.join(self.root_dir, self.labels.iloc[idx, 0], "Coronal", "t1", "") #self.labels.iloc[int(idx/20), 0] à la place de "12648-10464"
        y_folder_path = os.path.join(self.root_dir, self.labels.iloc[idx, 0], "Coronal", "labels", "")
        
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
    def __init__(self, csv_path, data_path, trained_model = None, device = "cpu"):
        '''
        In
            data_path: path vers les images
            trained_model: modèle unet qui predit la segmetation
            device = "cpu" ou "cuda"
            label_idx: zone qu'on veux segmenter
        '''
        self.csv_path = csv_path
        self.data_path = data_path
        self.device = device
        if trained_model != None:
            self.model = trained_model

    def train(self, nb_epoch, learning_rate, momentum, batch_size, train_index):
        '''
        Entraîne le modèle unet avec une entrée de 256x256x3 et une sortie de 256x256x3
        In
            nb_epoch, learning_rate, momentum, batch_size: hyperparamètres
            train_index: noms des cerveaux à entrainer
        Out
            model entrainé
        '''
        self.cervo_dataset = CervoDataset(csv_file=self.csv_path, root_dir=self.data_path, index = train_index)
        self.cervo_loader = DataLoader(self.cervo_dataset, batch_size=batch_size, shuffle = True)
        
        self.model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet', in_channels=3, out_channels=3, init_features=32, pretrained=False)

        self.model.to(self.device)

        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr = learning_rate, momentum=momentum)

        self.model.train()
        
        for i_epoch in range(nb_epoch):

            start_time, train_losses = time.time(), []
            for i_batch, batch in enumerate(self.cervo_loader):
                if i_batch %100 == 0:
                    print(" [-] batch no: ", i_batch)
                images, targets = batch
                images = images.to(self.device)
                targets = targets.to(self.device)

                # ===================forward=====================
                optimizer.zero_grad()
                predictions = self.model(images)
                loss = criterion(predictions, targets)
                
                # ===================backward=====================
                loss.backward()
                optimizer.step()
                

                train_losses.append(loss.item())

            print(' [-] epoch {:4}/{:}, train loss {:.6f} in {:.2f}s'.format(
                i_epoch+1, nb_epoch, np.mean(train_losses), time.time()-start_time))

        return self.model
     
def trainTestSplit(dataLen = 7000, trainTestRatio = 0.8):
    '''
    Sépare les données en train et test. 
    In:
        datalen: nombre de cerveaux à traîter
        traintestRatio: ration entre train et test
        csv_file: path vers le csv
    Out:
        Array qui contient les noms des cerveaux pour train et test des Pass ainsi que pour les Fails
    '''
    linspace = np.arange(dataLen)
    np.random.shuffle(linspace)
    train_index = linspace[:int(dataLen*trainTestRatio)]
    test_index = linspace[int(dataLen*trainTestRatio):]
    return train_index, test_index
    
    

if __name__ == '__main__':
    '''
    Entraîne un modèle unet pour tout le cerveau.
    '''
    unet = u_net(csv_path = 'data/raw/AI_FS_QC_img/data_AI_QC.csv', data_path = 'data/raw/AI_FS_QC_img/', device = "cuda", trained_model = None)

    train_index, test_index = trainTestSplit(dataLen = 7180, trainTestRatio = 0.95)
    
    trained = unet.train(nb_epoch = 10, learning_rate = 0.01, momentum = 0.99, batch_size = 32, train_index = train_index)
    torch.save(trained.state_dict(), "models/model0")
