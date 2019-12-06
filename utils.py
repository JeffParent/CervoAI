import torch
from poutyne.framework import ModelCheckpoint, CSVLogger
from sklearn import metrics


def create_balanced_sampler(dataset):
    def make_weights_for_balanced_classes(a_dataset, n_classes):
        count = [0] * n_classes
        for i in range(len(a_dataset)):
            count[a_dataset[i][1]] += 1
        weight_per_class = [0.] * n_classes
        N = float(sum(count))
        for i in range(n_classes):
            weight_per_class[i] = N/float(count[i])
        weight = [0] * len(a_dataset)
        for idx in range(len(a_dataset)):
            weight[idx] = weight_per_class[a_dataset[idx][1]]
        return weight

    weights = make_weights_for_balanced_classes(dataset, 2)
    weights = torch.DoubleTensor(weights)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))
    return sampler


def create_callbacks(name):
    callbacks = [
            # Save the latest weights to be able to continue the optimization at the end for more epochs.
            ModelCheckpoint(name + '_last_epoch.ckpt', temporary_filename='last_epoch.ckpt.tmp'),

            # Save the weights in a new file when the current model is better than all previous models.
            ModelCheckpoint(name + '_best_epoch_{epoch}.ckpt', monitor='val_acc', mode='max', save_best_only=True, restore_best=True, verbose=True, temporary_filename='best_epoch.ckpt.tmp'),

            # Save the losses and accuracies for each epoch in a TSV.
            CSVLogger(name + '_log.tsv', separator='\t'),
        ]
    return callbacks


def create_confusion_matrix(pytorch_module, loader):
    pytorch_module.eval()
    with torch.no_grad():
        y_total = list()
        y_pred_total = list()
        for (x, y) in loader:
            # Transfer batch on GPU if needed.
            x = x.to("cuda")
            y = y.to("cuda")
            y_total.extend(y)
            y_pred = pytorch_module(x)
            y_pred_total.extend(torch.argmax(y_pred))
    print("y: {}", y_total[12])
    print("y:{}", y_pred_total[12])
    return metrics.confusion_matrix(y_total, y_pred_total)

