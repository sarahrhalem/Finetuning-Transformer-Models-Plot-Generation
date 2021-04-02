import torch
from torch.utils.data import DataLoader, random_split


# Split dataset into train, val and test subsets and create dataloaders - 70% train, 20% val, 10% test
def split_datasets_and_create_dataloaders(dataset, seed=10, batch_size=10):
    train_len = int(0.7 * len(dataset))
    val_len = int(0.2 * len(dataset))
    test_len = int(len(dataset) - (train_len + val_len))

    lengths = [train_len, val_len, test_len]

    train_subset, val_subset, test_subset = random_split(dataset, lengths, torch.Generator().manual_seed(seed))

    print("Number of Training samples:", len(train_subset))
    print("Number of Validation samples:", len(val_subset))
    print("Number of Testing samples:", len(test_subset))

    train_dataloader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True
    )

    val_dataloader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False
    )

    test_dataloader = DataLoader(
        test_subset,
        batch_size=batch_size,
        shuffle=False
    )

    dataloaders = {'train_dataloader': train_dataloader, 'val_dataloader': val_dataloader,
                   'test_dataloader': test_dataloader}

    return dataloaders
