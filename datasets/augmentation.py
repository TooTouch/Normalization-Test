from torchvision import transforms
import torch

class InstanceNormalize(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, tensor):
        mu = tensor.mean(dim=[1,2], keepdims=True)
        std = tensor.std(dim=[1,2], keepdims=True)
        
        return (tensor-mu)/std


def normalize_setting(transform, mean: tuple, std: tuple, normalize: str = None):
    # set normalize
    if normalize == 'minmax':
        return transform
    elif normalize == 'instnace':
        normalize_transform = InstanceNormalize()
    elif normalize:
        normalize_transform = transforms.Normalize(mean, std)

    transform.transforms.append(normalize_transform)

    return transform


def train_augmentation(img_size: int, mean: tuple, std: tuple, normalize: str = None):
    transform = transforms.Compose([
        transforms.Pad(4),
        transforms.RandomCrop(img_size, fill=128),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    # set normalize
    return normalize_setting(transform=transform, mean=mean, std=std, normalize=normalize)


def test_augmentation(img_size: int, mean: tuple, std: tuple, normalize: str = None):
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
    ])

    # set normalize
    return normalize_setting(transform=transform, mean=mean, std=std, normalize=normalize)