import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Subset

def get_transforms(input_size=128):
    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return transform

def get_canonical_test_keys(no_aug_root, val_split=0.1, test_split=0.1, seed=42):
    """
    Determines the canonical test set from the non-augmented dataset, returned as a
    frozen set of (class_name, filename) pairs.

    Both the augmented and non-augmented models must be evaluated on exactly the same
    original leaf images. Without this, independently splitting the augmented dataset
    (which is larger and has different ordering) with the same seed produces a different
    test set — meaning augmented-dataset originals that appear in our 'test set' may
    have been seen during training of the augmented model, inflating its eval score.

    By anchoring the test split to the non-augmented dataset's filenames, we guarantee
    both models are tested on the same held-out images regardless of dataset size.

    Args:
        no_aug_root:  Path to the non-augmented dataset root (ImageFolder layout).
        val_split:    Fraction of data reserved for validation (mirrors training split).
        test_split:   Fraction of data reserved for testing.
        seed:         Random seed — must match the seed used during training.

    Returns:
        A frozenset of (class_name, filename) tuples, e.g. ('Cherry___healthy', 'image (42).JPG').
    """
    dataset = datasets.ImageFolder(no_aug_root)
    total_size = len(dataset)
    val_size = int(total_size * val_split)
    test_size = int(total_size * test_split)
    train_size = total_size - val_size - test_size

    generator = torch.Generator().manual_seed(seed)
    _, _, test_subset = random_split(
        dataset, [train_size, val_size, test_size], generator=generator
    )

    test_keys = frozenset(
        (dataset.classes[dataset.samples[idx][1]], os.path.basename(dataset.samples[idx][0]))
        for idx in test_subset.indices
    )
    return test_keys


def get_dataset_splits(root_dir, val_split=0.1, test_split=0.1, input_size=128, seed=42,
                       subset_fraction=1.0, fixed_test_keys=None):
    """
    Splits a dataset into train, validation, and test subsets.

    Args:
        root_dir:         Path to the dataset root (ImageFolder layout).
        val_split:        Fraction of non-test data used for validation.
        test_split:       Fraction of data used for testing (only used when
                          fixed_test_keys is None).
        input_size:       Image resize dimension.
        seed:             Random seed for reproducibility.
        subset_fraction:  If < 1.0, randomly downsample each split by this fraction.
        fixed_test_keys:  A frozenset of (class_name, filename) pairs from
                          get_canonical_test_keys(). When provided, the test set is
                          built deterministically from those exact images and the
                          remainder is split into train/val — eliminating sampling
                          bias when comparing models trained on differently-sized
                          datasets. Falls back to a seeded random 3-way split if None.

    Returns:
        (train_dataset, val_dataset, test_dataset, class_names)
    """
    transform = get_transforms(input_size)
    dataset = datasets.ImageFolder(root_dir, transform=transform)

    if fixed_test_keys is not None:
        # Partition indices using pre-defined (class_name, filename) keys.
        test_indices = []
        train_val_indices = []
        for i, (path, label) in enumerate(dataset.samples):
            key = (dataset.classes[label], os.path.basename(path))
            if key in fixed_test_keys:
                test_indices.append(i)
            else:
                train_val_indices.append(i)

        # Split the non-test images into train and val.
        # val_split is applied to the remaining pool (same semantic meaning as before).
        n_val = int(len(train_val_indices) * val_split)
        n_train = len(train_val_indices) - n_val

        generator = torch.Generator().manual_seed(seed)
        train_val_sub = Subset(dataset, train_val_indices)
        train_sub, val_sub = random_split(train_val_sub, [n_train, n_val], generator=generator)

        train_dataset = train_sub
        val_dataset = val_sub
        test_dataset = Subset(dataset, test_indices)

    else:
        # Fallback: seeded random 3-way split (original behaviour).
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
        train_dataset, _ = random_split(
            train_dataset,
            [train_sub_size, len(train_dataset) - train_sub_size],
            generator=sub_generator
        )

        val_sub_size = max(1, int(len(val_dataset) * subset_fraction))
        val_dataset, _ = random_split(
            val_dataset,
            [val_sub_size, len(val_dataset) - val_sub_size],
            generator=sub_generator
        )

        test_sub_size = max(1, int(len(test_dataset) * subset_fraction))
        test_dataset, _ = random_split(
            test_dataset,
            [test_sub_size, len(test_dataset) - test_sub_size],
            generator=sub_generator
        )

    return train_dataset, val_dataset, test_dataset, dataset.classes


def get_dataloaders(train_dataset, val_dataset, test_dataset, batch_size=64, num_workers=4):
    train_loader = None
    if train_dataset is not None:
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True
        )

    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True
        )

    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True
        )

    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    base_path = "Data for Identification of Plant Leaf Diseases Using a 9-layer Deep Convolutional Neural Network"
    no_aug_dir = os.path.join(base_path, "Plant_leave_diseases_dataset_without_augmentation")

    if os.path.exists(no_aug_dir):
        print("Dataset directory exists. Testing splits...")
        test_keys = get_canonical_test_keys(no_aug_dir)
        train_ds, val_ds, test_ds, classes = get_dataset_splits(
            no_aug_dir, input_size=128, fixed_test_keys=test_keys
        )
        print(f"Total classes: {len(classes)}")
        print(f"Train size: {len(train_ds)}")
        print(f"Val size:   {len(val_ds)}")
        print(f"Test size:  {len(test_ds)}")

        train_loader, _, _ = get_dataloaders(train_ds, val_ds, test_ds, batch_size=16, num_workers=0)
        img, label = next(iter(train_loader))
        print("Batch image shape:", img.shape)
        print("Batch label shape:", label.shape)
    else:
        print(f"Dataset directory not found at {no_aug_dir}")
