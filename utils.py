import torch


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
