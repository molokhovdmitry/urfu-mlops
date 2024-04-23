import kaggle
import os
import torch
from torchvision import datasets, transforms


def init_loaders(batch_size, img_size):
    data_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    full_train_dataset = datasets.ImageFolder(
        root='data/rvf10k/train',
        transform=data_transform
    )

    train_size = int(0.8 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        full_train_dataset, [train_size, val_size]
    )

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    val_data_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    test_dataset = datasets.ImageFolder(
        root='data/rvf10k/valid',
        transform=data_transform
    )
    test_data_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )

    return train_data_loader, val_data_loader, test_data_loader


if __name__ == "__main__":
    if not os.path.exists('data'):
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(
            dataset='sachchitkunichetty/rvf10k',
            path='data',
            unzip=True)
