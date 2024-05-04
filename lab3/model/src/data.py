import kaggle
import os
import torch
from torchvision import datasets, transforms


def init_loaders(path, batch_size, img_size, random_seed=42):
    data_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    dataset = datasets.ImageFolder(
        root=path,
        transform=data_transform
    )

    # Train, val, test datasets
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [0.7, 0.15, 0.15],
        generator=torch.Generator().manual_seed(random_seed)
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
    test_data_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    return (train_dataset, val_dataset, test_dataset,
            train_data_loader, val_data_loader, test_data_loader)


if __name__ == "__main__":
    if not os.path.exists('data'):
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(
            dataset='asdasdasasdas/garbage-classification',
            path='data',
            unzip=True)
