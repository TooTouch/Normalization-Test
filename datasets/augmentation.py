from torchvision import transforms

def train_augmentation(img_size: int, mean: tuple, std: tuple):
    transform = transforms.Compose([
        transforms.Pad(4),
        transforms.RandomCrop(img_size, fill=128),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    return transform


def test_augmentation(img_size: int, mean: tuple, std: tuple):
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    return transform