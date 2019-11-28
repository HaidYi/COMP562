import torch
import torch.nn.functional as F
import torch.distributions as distributions
import torch.utils.data as data

import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np

class DataInfo():
    def __init__(self, name, channel, size):
        """Instantiates a DataInfo.

        Args:
            name: name of dataset.
            channel: number of image channels.
            size: height and width of an image.
        """
        self.name = name
        self.channel = channel
        self.size = size

def load(dataset):
    """Load dataset.

    Args:
        dataset: name of dataset.
    Returns:
        a torch dataset and its associated information.
    """
    if dataset == 'mnist':
        data_info = DataInfo(dataset, 1, 32)
        transform = transforms.Compose(
            [transforms.Resize(32),
             transforms.ToTensor(),
             transforms.Normalize([0.5], [0.5])])
        train_set = datasets.MNIST('./data/MNIST',
            train=True, download=True, transform=transform)

    if dataset == 'cifar10':    # 3 x 32 x 32
        data_info = DataInfo(dataset, 3, 32)
        transform = transforms.Compose(
            [transforms.Resize(32), 
             transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))])
        train_set = datasets.CIFAR10('../../data/CIFAR10', 
            train=True, download=True, transform=transform)
        #[train_split, val_split] = data.random_split(train_set, [50000, 0])

    elif dataset == 'celeba':   # 3 x 218 x 178
        data_info = DataInfo(dataset, 3, 32)
        def CelebACrop(images):
            return transforms.functional.crop(images, 40, 15, 148, 148)
        transform = transforms.Compose(
            [CelebACrop, 
             transforms.Resize(32), 
             transforms.RandomHorizontalFlip(p=0.5), 
             transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_set = datasets.ImageFolder('../../data/CelebA', 
            transform=transform)
        [train_split, val_split] = data.random_split(train_set, [150000, 12770])

    elif dataset == 'imnet32':
        data_info = DataInfo(dataset, 3, 32)
        transform = transforms.Compose(
            [transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor()])
        train_set = datasets.ImageFolder('../../data/ImageNet32/train', 
            transform=transform)
        [train_split, val_split] = data.random_split(train_set, [1250000, 31149])

    elif dataset == 'imnet64':
        data_info = DataInfo(dataset, 3, 64)
        transform = transforms.Compose(
            [transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor()])
        train_set = datasets.ImageFolder('../../data/ImageNet64/train', 
            transform=transform)
        [train_split, val_split] = data.random_split(train_set, [1250000, 31149])
    
    else:
        raise NotimplementedError('Dataset {} is not understood.'.format(dataset))

    return train_set, data_info


class InfiniteLoader():
    """A data loader that can load a dataset repeatedly."""

    def __init__(self, data_loader, num_epochs=None):
        """Constructor.
        Args:
            dataloader: A `Dataloader` object to be loaded.
            batch_size: int, the size of each batch.
            shuffle: bool, whether to shuffle the dataset after each epoch.
            drop_last: bool, whether to drop last batch if its size is less than
                `batch_size`.
            num_epochs: int or None, number of epochs to iterate over the dataset.
                If None, defaults to infinity.
        """
        self.loader = data_loader
        self.finite_iterable = iter(self.loader)
        self.counter = 0
        self.num_epochs = float('inf') if num_epochs is None else num_epochs

    def __next__(self):
        try:
            return next(self.finite_iterable)
        except StopIteration:
            self.counter += 1
            if self.counter >= self.num_epochs:
                raise StopIteration
            self.finite_iterable = iter(self.loader)
            return next(self.finite_iterable)

    def __iter__(self):
        return self