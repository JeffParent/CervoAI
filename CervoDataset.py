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

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.labels = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return 5 #len(self.labels)*20

    def extract_image(self, img_folder_path, idx):
        file = os.listdir(img_folder_path)[idx%20]
        filename = os.fsdecode(file)
        if filename[-3:] == "png":
            img_name = os.path.join(img_folder_path, filename)
            image = io.imread(img_name)
            return image
        
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        X_folder_path = os.path.join(self.root_dir, "12648-10464", "Coronal", "t1", "") #self.labels.iloc[int(idx/20), 0] à la place de "12648-10464"
        y_folder_path = os.path.join(self.root_dir, "12648-10464", "Coronal", "labels", "")
        
        X = self.extract_image(X_folder_path, idx)
        X = X[:,:,:3]
        y = self.extract_image(y_folder_path, idx)
        y = y[:,:,:3]

        if self.transform is not None:
            trans = transforms.Compose([transforms.ToTensor()]+self.transform)
            X = (trans(X))
            y = (trans(y))
        else:
            X = transforms.ToTensor()(X)
            y = transforms.ToTensor()(y)


        print(X.shape,y.shape)
        return X, y


class u_net():
    def __init__(self, csv_path, data_path, trained_model = None):
        self.cervo_dataset = CervoDataset(csv_file=csv_path, root_dir=data_path)
        if trained_model != None:
            self.model = trained_model

    def train(self, nb_epoch, learning_rate, momentum, batch_size):
        self.cervo_loader = DataLoader(self.cervo_dataset, batch_size=batch_size)
        
        self.model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet', in_channels=3, out_channels=3, init_features=32, pretrained=False)

        self.model.to("cpu")

        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr = learning_rate, momentum=momentum)

        self.model.train()
        
        for i_epoch in range(nb_epoch):

            start_time, train_losses = time.time(), []
            for i_batch, batch in enumerate(self.cervo_loader):
                images, targets = batch
                images = images.to("cpu")
                targets = targets.to("cpu")

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


    def predict(self, image_index):
        self.model.eval()
        image, label = self.cervo_dataset.__getitem__(image_index)
        image = (image.unsqueeze(0)).to("cpu")
        label = (label.unsqueeze(0)).to("cpu")
        with torch.no_grad():
            prediction = self.model(image)
        return image[0].permute(1, 2, 0).numpy(), prediction.detach()[0].permute(1, 2, 0).numpy(), label[0].permute(1, 2, 0).numpy()
        


if __name__ == '__main__':
        unet = u_net(csv_path = 'data/raw/AI_FS_QC_img/data_AI_QC.csv',
                      data_path = 'data/raw/AI_FS_QC_img/',
                      trained_model = None)

        trained = unet.train(nb_epoch = 1, learning_rate = 0.01, momentum = 0.99, batch_size = 1)

        
        image, prediction, label = unet.predict(image_index = 2)
        
        print(image.shape, prediction.shape, label.shape)

        plt.imshow(image)
        plt.show()
        plt.imshow(prediction)
        plt.show()
        plt.imshow(label)
        plt.show()
