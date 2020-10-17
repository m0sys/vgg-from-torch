from torchvision import datasets, transforms
from vgg.base import BaseDataLoader
from PIL import Image


class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """

    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(
            self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class Cifar100DataLoader(BaseDataLoader):
    """
    CIFAR100 dataloader with train/val split.
    """

    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):

        trsfm = transforms.Compose([
            transforms.Resize(224, interpolation=Image.NEAREST),
            # transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        self.data_dir = data_dir
        self.dataset = datasets.CIFAR100(
            self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class DefaultCifar100DataLoader(BaseDataLoader):
    """Default size for Cifar100 images with train/split."""

    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):

        normalize = transforms.Normalize(
            mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])

        trsfm = transforms.Compose([
            ## transforms.RandomCrop(32, padding=4),
            ## transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])

        self.data_dir = data_dir
        self.dataset = datasets.CIFAR100(
            self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class DefaultCifar10DataLoader(BaseDataLoader):
    """Default size for Cifar10 images with train/split."""

    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):

        normalize = transforms.Normalize(
            mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262])

        trsfm = transforms.Compose([
            ## transforms.RandomCrop(32, padding=4),
            ## transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])

        self.data_dir = data_dir
        self.dataset = datasets.CIFAR10(
            self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
