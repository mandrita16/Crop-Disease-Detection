import os
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

DATA_DIR = "../data/plantvillage"
IMG_SIZE = 224
BATCH_SIZE = 32


def get_dataloaders(val_split=0.15, test_split=0.15, num_workers=2):
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225]),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225]),
    ])

    full_dataset = datasets.ImageFolder(DATA_DIR, transform=train_transform)
    class_names = full_dataset.classes

    n = len(full_dataset)
    n_val = int(n * val_split)
    n_test = int(n * test_split)
    n_train = n - n_val - n_test

    train_ds, val_ds, test_ds = random_split(full_dataset, [n_train, n_val, n_test])

    # eval sets shouldn't use train-time augmentation
    val_ds.dataset.transform = eval_transform
    test_ds.dataset.transform = eval_transform

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader, class_names
