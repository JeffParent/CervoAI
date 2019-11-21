# Train on GPU if one is present
import math
import numpy as np
import torch
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataset import Subset

from torchvision import transforms, utils

from poutyne.framework import Model, ModelCheckpoint, CSVLogger, Callback
from poutyne import torch_to_numpy
from poutyne.layers import Lambda

from CervoDataset import CervoDataset

cuda_device = 0
device = torch.device("cuda:%d" % cuda_device if torch.cuda.is_available() else "cpu")

# The dataset is split 80/20 for the train and validation datasets respectively.
train_split_percent = 0.8

# The MNIST dataset has 10 classes
num_classes = 2

# Training hyperparameters
batch_size = 32
learning_rate = 0.1
num_epochs = 5

cervo_dataset = CervoDataset(csv_file='data/raw/AI_FS_QC_img/data_AI_QC.csv', root_dir='data/raw/AI_FS_QC_img/', transform = transforms.ToTensor())

num_data = len(cervo_dataset)
indices = list(range(num_data))
np.random.shuffle(indices)

test_split = math.floor(train_split_percent * num_data)

test_indices = indices[test_split:]
test_dataset = Subset(cervo_dataset, test_indices)

valid_split = math.floor(train_split_percent * num_data - len(test_dataset))

valid_indices = indices[valid_split:test_split]
valid_dataset = Subset(cervo_dataset, valid_indices)

train_indices = indices[:valid_split]
train_dataset = Subset(cervo_dataset, train_indices)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

loaders = train_loader, valid_loader, test_loader

def create_convolutional_network():
    """
    This function returns the convolutional network layed out above.
    """
    return nn.Sequential(
        nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Dropout(0.25),
        Lambda(lambda x: x.flatten(1)), # Flatten layer is in Poutyne.
        nn.Linear(32*7*7, 128),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Lin
