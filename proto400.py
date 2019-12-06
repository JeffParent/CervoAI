# Train on GPU if one is present
import math
import numpy as np
import torch
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataset import Subset
from utils import create_balanced_sampler, create_confusion_matrix, create_callbacks

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
num_epochs = 50

cervo_dataset = CervoDataset(csv_file='data/raw/AI_FS_QC_img/data_AI_QC.csv', root_dir='data/raw/AI_FS_QC_img/')

#num_data = len(cervo_dataset)
#indices = list(range(num_data))
indices = list()
for index, label in enumerate(cervo_dataset.labels):
    if label == 1:
        indices.append(index)

max_number_good_seg = len(indices)
count = 0
while count != max_number_good_seg:
    indx = np.randint(0, len(cervo_dataset))
    if cervo_dataset[indx][1] == 1:
        continue
    indices.append(indx)
    count += 1
print(len(indices))
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


def create_convolutional_network():
    """
    This function returns the convolutional network layed out above.
    """
    return nn.Sequential(
        nn.Conv2d(in_channels=60, out_channels=480, kernel_size=5, padding=2),
        nn.ReLU(),
        nn.MaxPool2d(4),
        nn.Conv2d(in_channels=480, out_channels=960, kernel_size=5, padding=2),
        nn.ReLU(),
        nn.MaxPool2d(4),
        nn.Dropout(0.25),
        Lambda(lambda x: x.flatten(1)), # Flatten layer is in Poutyne.
        nn.Linear(960*16*16, 128),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(128, num_classes)
    )


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
    model.fit_generator(train_loader, valid_loader, epochs=num_epochs, callbacks=create_callbacks("2convShuffle"))

    # Test
    test_loss, test_acc = model.evaluate_generator(test_loader)
    print('Test:\n\tLoss: {}\n\tAccuracy: {}'.format(test_loss, test_acc))


conv_net = create_convolutional_network()
poutyne_train(conv_net)
print(create_confusion_matrix(conv_net, test_loader))