import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_data_loaders(data_dir, batch_size=32):
    # Minimal processing for the training set
    train_transform = transforms.Compose([
        transforms.Resize((128, 128)),  # Resize to a standard size
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize
    ])
    
    # Basic transformations for the validation set
    valid_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Datasets
    train_dataset = datasets.ImageFolder(root=f"{data_dir}/train", transform=train_transform)
    valid_dataset = datasets.ImageFolder(root=f"{data_dir}/valid", transform=valid_transform)

    # Data Loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader
