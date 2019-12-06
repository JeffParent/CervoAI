# Train on GPU if one is present
import math
import numpy as np
import torch
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataset import Subset
from torchvision.models import resnet18, resnet101

from utils import create_balanced_sampler, create_callbacks

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
batch_size = 16
learning_rate = 0.1
num_epochs = 200

cervo_dataset = CervoDataset(csv_file='data/raw/AI_FS_QC_img/data_AI_QC.csv', root_dir='data/raw/AI_FS_QC_img/', transform=[transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

#num_data = len(cervo_dataset)
#indices = list(range(num_data))

indices = list()
for index in range(len(cervo_dataset)):
    label = cervo_dataset.labels.iloc[index, 1]
    if label == 1:
        indices.append(index)

max_number_good_seg = len(indices)
print(len(indices))
count = 0
while count != max_number_good_seg:
    indx = np.random.randint(0, len(cervo_dataset))
    if cervo_dataset[indx][1] == 1:
        continue
    indices.append(indx)
    count += 1
print(len(indices))
num_data = len(indices)
np.random.shuffle(indices)

# splitting into train, valid and test datasets
test_split = math.floor(train_split_percent * num_data)

test_indices = indices[test_split:]
test_dataset = Subset(cervo_dataset, test_indices)

valid_split = math.floor(train_split_percent * num_data - len(test_dataset))

valid_indices = indices[valid_split:test_split]
valid_dataset = Subset(cervo_dataset, valid_indices)

train_indices = indices[:valid_split]
train_dataset = Subset(cervo_dataset, train_indices)

#balanced_train_sampler = create_balanced_sampler(train_dataset)
#balanced_val_sampler = create_balanced_sampler(valid_dataset)
#balanced_test_sampler = create_balanced_sampler(test_dataset)

#train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=balanced_train_sampler)
#valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, sampler=balanced_val_sampler)
#test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, sampler=balanced_test_sampler)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


loaders = train_loader, valid_loader, test_loader


class CervoResNet(nn.Module):

    def __init__(self, pretrained=False):
        super().__init__()

        # Crée le réseau de neurone pré-entraîné
        self.model = resnet101(pretrained=pretrained)

        # Récupère le nombre de neurones avant
        # la couche de classification
        dim_before_fc = self.model.fc.in_features
        channels_after_conv_1 = self.model.conv1.out_channels

        # TODO Q2A
        # Changer la dernière fully-connected layer
        # pour avoir le bon nombre de neurones de
        # sortie
        self.model.conv1 = nn.Conv2d(in_channels=60, out_channels=channels_after_conv_1, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.model.fc = nn.Linear(dim_before_fc, 2)

        if pretrained:
            # TODO Q2A
            # Geler les paramètres qui ne font pas partie
            # de la dernière couche fc
            # Conseil: utiliser l'itérateur named_parameters()
            # et la variable requires_grad
            for name, param in self.model.named_parameters():
                if "fc" or "conv1" not in name:
                    param.requires_grad = False

    def forward(self, x):
        # TODO Q2A
        # Appeler la fonction forward du réseau
        # pré-entraîné (resnet18) de LegoNet
        return self.model.forward(x)


def poutyne_train(pytorch_module):
    """
    This function creates a Poutyne Model, sends the Model on the specified device,
    and uses the `fit_generator` method to train the neural network. At the end,
    the `evaluate_generator` is used on the test set.

    Args:
        pytorch_module (torch.nn.Module): The neural network to train.
    """
    print(pytorch_module)

    optimizer = optim.SGD(pytorch_module.parameters(), lr=learning_rate)
    loss_function = nn.CrossEntropyLoss()

    # Poutyne Model
    model = Model(pytorch_module, optimizer, loss_function, metrics=['accuracy'])

    # Send model on GPU
    model.to(device)

    # Train
    model.fit_generator(train_loader, valid_loader, epochs=num_epochs, callbacks=create_callbacks("resnet101EqualData"))

    # Test
    test_loss, test_acc = model.evaluate_generator(test_loader)
    print('Test:\n\tLoss: {}\n\tAccuracy: {}'.format(test_loss, test_acc))


conv_net = CervoResNet()
poutyne_train(conv_net)
print(create_confusion_matrix(conv_net, test_loader))
