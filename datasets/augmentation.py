from torchvision import transforms

def train_augmentation(mean: tuple, std: tuple):
    transform = transforms.Compose([
        transforms.Pad(4),
        transforms.RandomCrop(32, fill=128),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    return transform


def test_augmentation(mean: tuple, std: tuple):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    return transform