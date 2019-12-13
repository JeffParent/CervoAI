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
    def __init__(self, root_dir, index, label_idx, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.label_idx = label_idx
        self.index = index
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.index)*20

    def separate_label(self,image,label_idx):
        legende = np.array([[136,  34,  51],
                            [238, 204, 136],
                            [185, 119, 232],
                            [ 23, 147,  39],
                            [  0,  76, 221],
                            [119, 102, 204],
                            [ 85,  34, 136],
                            [103,  34,  69],
                            [ 45,  95, 222],
                            [ 52, 190, 140]])
        legende = legende[:,::-1]

        mask = cv2.inRange(image, legende[label_idx], legende[label_idx])
        #output = cv2.bitwise_and(image, image, mask = mask)
        #gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
        #gray[np.where(gray>0)] = 255

        return mask

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

        X_folder_path = os.path.join(self.root_dir, self.index[int(idx/20), 0], "Coronal", "t1", "") #self.index[int(idx/20), 0] à la place de "12648-10464"
        y_folder_path = os.path.join(self.root_dir, self.index[int(idx/20), 0], "Coronal", "labels", "")
        
        X = self.extract_image(X_folder_path, rest)
        X = X[:,:,:3]
        X = cv2.cvtColor(X, cv2.COLOR_BGR2GRAY)

        y = self.extract_image(y_folder_path, rest)
        y = y[:,:,:3]
        y = self.separate_label(y,self.label_idx)
        
        #print(X.shape, y.shape, np.max(X), np.max(y))
        #print(1/0)

        if self.transform is not None:
            trans = transforms.Compose([transforms.ToTensor()]+self.transform)
            X = (trans(X))
            y = (trans(y))
        else:
            X = transforms.ToTensor()(X)
            y = transforms.ToTensor()(y)

        return X, y


class u_net():
    def __init__(self, data_path, trained_model = None, device = "cuda", label_idx = 0):
        self.label_idx = label_idx
        self.data_path = data_path
        self.device = device
        if trained_model != None:
            self.model = trained_model



    def predict(self, image_index, test_index):
        #self.model.to(self.device)
        self.cervo_dataset = CervoDataset(root_dir=self.data_path, index = test_index, label_idx = self.label_idx)
        self.model.eval()
        image, label = self.cervo_dataset.__getitem__(image_index)
        image = (image.unsqueeze(0)).to(self.device)
        label = (label.unsqueeze(0)).to("cpu")
        with torch.no_grad():
            prediction = self.model(image)
        prediction.to("cpu")
        image.to("cpu")
        image = image[0].permute(1, 2, 0).numpy()
        prediction = prediction.detach()[0].permute(1, 2, 0).numpy()
        label = label[0].permute(1, 2, 0).numpy()
        return image, prediction, label 

    def score(self, prediction, label):
        good_pred = len(np.where((prediction + label) == 2)[0])
        total = good_pred + len(np.where((prediction + label) == 1)[0])
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
    print("Version 1.0.0")
    for label in range(2):
        print("Zone %s segmentation" %(label))
        trained = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet', in_channels=1, out_channels=1, init_features=32, pretrained=False)
        trained.load_state_dict(torch.load("models/model_zone_%s"%(label)))
        trained.cuda()
        unet = u_net(data_path = 'data/raw/AI_FS_QC_img/', device = "cuda", trained_model = trained, label_idx = label)

        train_index, test_index, Fail_index = trainTestSplit(dataLen = 7100, trainTestRatio = 0.9)

        X = []
        y = []
        for brain_no in range(len(test_index)):
            score = []
            for img in range(20):
                gray, prediction, label = unet.predict(brain_no*20 + img, test_index)
                score.append(unet.score(prediction, label))
            X.append(score)
            y.append(0)
        
        for brain_no in range(len(Fail_index)):
            score = []
            for img in range(20):
                gray, prediction, label = unet.predict(brain_no*20 + img, Fail_index)
                score.append(unet.score(prediction, label))
            X.append(score)
            y.append(1)
        X = np.array(X)
        y = np.array(y)
        np.save("saves/zone_%s_scores_X" %(label), X)
        np.save("saves/zone_%s_scores_y" %(label), y)
        #trained = unet.train(nb_epoch = 3, learning_rate = 0.01, momentum = 0.99, batch_size = 32, train_index = train_index)
        #torch.save(trained.state_dict(), "models/model_zone_%s" %(label))

