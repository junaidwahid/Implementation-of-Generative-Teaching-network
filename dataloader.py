import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
import torch


def convert_numpy(x):
    return np.array(x)


def create_data_splits(transform):
    # Create data splits
    train = datasets.MNIST('./data', train=True, transform=transform, download=True)
    train, val = torch.utils.data.random_split(train, [50000, 10000])
    test = datasets.MNIST('./data', train=False, transform=transform, download=True)
    print('Created train, val, and test datasets.')
    return train, val, test


def get_data_loaders(mnist_mean, mnist_std, outer_loop_batch_size):
    transform = transforms.Compose([
        #     transforms.Lambda(lambda x: convert_numpy(x)),
        transforms.ToTensor(),
        transforms.Normalize((mnist_mean,), (mnist_std,)),
    ])

    train, val, test = create_data_splits(transform)

    train_loader = torch.utils.data.DataLoader(
        train, batch_size=outer_loop_batch_size, shuffle=True, drop_last=True, num_workers=1, pin_memory=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val, batch_size=outer_loop_batch_size, shuffle=True, drop_last=True, num_workers=1, pin_memory=True,
    )

    test_loader = torch.utils.data.DataLoader(
        test, batch_size=outer_loop_batch_size, shuffle=True, drop_last=True, num_workers=1, pin_memory=True,
    )
    return train_loader, val_loader, test_loader
