import os
from torchvision import datasets
from torch.utils.data import DataLoader
from .augmentation import train_augmentation, test_augmentation



def load_cifar10(datadir: str, img_size: int, mean: tuple, std: tuple, normalize: str = None):

    trainset = datasets.CIFAR10(
        root      = os.path.join(datadir,'CIFAR10'), 
        train     = True, 
        download  = True,
        transform = train_augmentation(img_size=img_size, mean=mean, std=std, normalize=normalize)
    )

    testset = datasets.CIFAR10(
        root      = os.path.join(datadir,'CIFAR10'), 
        train     = False, 
        download  = True,
        transform = test_augmentation(img_size=img_size, mean=mean, std=std, normalize=normalize)
    )
        
    return trainset, testset


def load_cifar100(datadir: str, img_size: int, mean: tuple, std: tuple, normalize: str = None):

    trainset = datasets.CIFAR100(
        root      = os.path.join(datadir,'CIFAR100'), 
        train     = True, 
        download  = True,
        transform = train_augmentation(img_size=img_size, mean=mean, std=std, normalize=normalize)
    )

    testset = datasets.CIFAR100(
        root      = os.path.join(datadir,'CIFAR100'), 
        train     = False, 
        download  = True,
        transform = test_augmentation(img_size=img_size, mean=mean, std=std, normalize=normalize)
    )

    return trainset, testset


def load_svhn(datadir: str, img_size: int, mean: tuple, std: tuple, normalize: str = None):

    trainset = datasets.SVHN(
        root      = os.path.join(datadir,'SVHN'), 
        split     = 'train', 
        download  = True,
        transform = train_augmentation(img_size=img_size, mean=mean, std=std, normalize=normalize)
    )

    testset = datasets.SVHN(
        root      = os.path.join(datadir,'SVHN'), 
        split     = 'test', 
        download  = True,
        transform = test_augmentation(img_size=img_size, mean=mean, std=std, normalize=normalize)
    )

    return trainset, testset


def load_tiny_imagenet_200(datadir: str, img_size: int, mean: tuple, std: tuple, normalize: str = None):

    trainset = datasets.ImageFolder(
        root      = os.path.join(datadir,'tiny-imagenet-200','train'),
        transform = train_augmentation(img_size=img_size, mean=mean, std=std, normalize=normalize)
    )
    
    testset = datasets.ImageFolder(
        root      = os.path.join(datadir,'tiny-imagenet-200','val'),
        transform = test_augmentation(img_size=img_size, mean=mean, std=std, normalize=normalize)
    )

    return trainset, testset


def create_dataloader(dataset, batch_size: int = 4, shuffle: bool = False):

    return DataLoader(
        dataset     = dataset,
        batch_size  = batch_size,
        shuffle     = shuffle,
        num_workers = 16
    )
