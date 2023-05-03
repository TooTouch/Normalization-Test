import os
from torchvision import datasets
from torch.utils.data import DataLoader
from .augmentation import train_augmentation, test_augmentation



def load_cifar10(datadir, img_size, mean, std):
    trainset = datasets.CIFAR10(
        root      = os.path.join(datadir,'CIFAR10'), 
        train     = True, 
        download  = True, 
        transform = train_augmentation(img_size, mean, std)
    )
    testset = datasets.CIFAR10(
        root      = os.path.join(datadir,'CIFAR10'), 
        train     = False, 
        download  = True, 
        transform = test_augmentation(img_size, mean, std)
    )
    
    return trainset, testset


def load_cifar100(datadir, img_size, mean, std):
    trainset = datasets.CIFAR100(
        root      = os.path.join(datadir,'CIFAR100'), 
        train     = True, 
        download  = True, 
        transform = train_augmentation(img_size, mean, std)
    )
    testset = datasets.CIFAR100(
        root      = os.path.join(datadir,'CIFAR100'), 
        train     = False, 
        download  = True, 
        transform = test_augmentation(img_size, mean, std)
    )
    
    return trainset, testset


def load_svhn(datadir, img_size, mean, std):
    trainset = datasets.SVHN(
        root      = os.path.join(datadir,'SVHN'), 
        split     = 'train', 
        download  = True, 
        transform = train_augmentation(img_size, mean, std)
    )
    testset = datasets.SVHN(
        root      = os.path.join(datadir,'SVHN'), 
        split     = 'test', 
        download  = True, 
        transform = test_augmentation(img_size, mean, std)
    )
    
    return trainset, testset


def load_tiny_imagenet_200(datadir, img_size, mean, std):
    trainset = datasets.ImageFolder(
        root      = os.path.join(datadir,'tiny-imagenet-200','train'),
        transform = train_augmentation(img_size, mean, std)
    )
    testset = datasets.ImageFolder(
        root      = os.path.join(datadir,'tiny-imagenet-200','val'),
        transform = test_augmentation(img_size, mean, std)
    )
    
    return trainset, testset


def create_dataloader(dataset, batch_size: int = 4, shuffle: bool = False):

    return DataLoader(
        dataset     = dataset,
        batch_size  = batch_size,
        shuffle     = shuffle,
        num_workers = 16
    )
