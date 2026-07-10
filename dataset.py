import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def get_transforms(input_size=128):
    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return transform

def get_dataset_splits(root_dir, val_split=0.1, test_split=0.1, input_size=128, seed=42, subset_fraction=1.0):
    transform = get_transforms(input_size)
    dataset = datasets.ImageFolder(root_dir, transform=transform)

    total_size = len(dataset)
    val_size = int(total_size * val_split)
    test_size = int(total_size * test_split)
    train_size = total_size - val_size - test_size

    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size], generator=generator
    )

    if subset_fraction < 1.0:
        sub_generator = torch.Generator().manual_seed(seed)

        train_sub_size = max(1, int(len(train_dataset) * subset_fraction))
        train_dataset, _ = random_split(train_dataset, [train_sub_size, len(train_dataset) - train_sub_size], generator=sub_generator)

        val_sub_size = max(1, int(len(val_dataset) * subset_fraction))
        val_dataset, _ = random_split(val_dataset, [val_sub_size, len(val_dataset) - val_sub_size], generator=sub_generator)

        test_sub_size = max(1, int(len(test_dataset) * subset_fraction))
        test_dataset, _ = random_split(test_dataset, [test_sub_size, len(test_dataset) - test_sub_size], generator=sub_generator)

    return train_dataset, val_dataset, test_dataset, dataset.classes

def get_dataloaders(train_dataset, val_dataset, test_dataset, batch_size=64, num_workers=4):
    # wait i think these nones are really chopped :/
    train_loader = None
    if (train_dataset != None):
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True
        )

    val_loader = None
    if (val_dataset != None):
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True
        )

    test_loader = None
    if (test_dataset != None):
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True
        )

    return train_loader, val_loader, test_loader

if __name__ == '__main__':
    import sys
    base_path = "Data for Identification of Plant Leaf Diseases Using a 9-layer Deep Convolutional Neural Network"
    no_aug_dir = os.path.join(base_path, "Plant_leave_diseases_dataset_without_augmentation")
    
    if os.path.exists(no_aug_dir):
        print("Dataset directory exists. Testing splits...")
        train_ds, val_ds, test_ds, classes = get_dataset_splits(no_aug_dir, input_size=128)
        print(f"Total classes: {len(classes)}")
        print(f"Train size: {len(train_ds)}")
        print(f"Val size: {len(val_ds)}")
        print(f"Test size: {len(test_ds)}")
        
        train_loader, _, _ = get_dataloaders(train_ds, val_ds, test_ds, batch_size=16, num_workers=0)
        img, label = next(iter(train_loader))
        print("Batch image shape:", img.shape)
        print("Batch label shape:", label.shape)
    else:
        print(f"Dataset directory not found at {no_aug_dir}")
