import os

import torch
from skimage import io
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class CervoDataset(Dataset):


    def __init__(self, csv_file, root_dir):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.labels = pd.read_csv(csv_file)
        self.root_dir = root_dir

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_folder_path = os.path.join(self.root_dir, self.labels.iloc[idx, 0], "Coronal", "merged", "")
        print(img_folder_path)
        images = list()
        for file in os.listdir(img_folder_path):
            filename = os.fsdecode(file)
            if filename[-3:] == "png":
                print(filename)
                img_name = os.path.join(self.root_dir, self.labels.iloc[idx, 0], "Coronal", "merged", filename)
                image = io.imread(img_name)
                images.append(image)
        print(len(images))
        X = np.array(images)
        y = self.labels.iloc[idx, 1]

        return X, y


if __name__ == '__main__':

    cervo_dataset = CervoDataset(csv_file='data/raw/AI_FS_QC_img/data_AI_QC.csv', root_dir='data/raw/AI_FS_QC_img/')

    fig = plt.figure()
    print(len(cervo_dataset))

    for i in range(len(cervo_dataset)):
        sample = cervo_dataset[i]

        print(i, sample[0].shape, sample[1])

        ax = plt.subplot(1, 4, i + 1)
        plt.tight_layout()
        ax.set_title('Sample #{}'.format(i))
        ax.axis('off')
        plt.imshow(sample[0][1])
        plt.pause(0.001)

        if i == 3:
            plt.show()
            break
