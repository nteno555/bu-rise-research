import random
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


class BarkDiseaseDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        transform: Optional[Callable] = None,
    ) -> None:
        self.root = Path(root)
        self.transform = transform
        self.samples: list[tuple[Path, int]] = []
        self.class_names: list[str] = []
        self.class_to_idx: dict[str, int] = {}

        self._load_samples()

    def _load_samples(self) -> None:
        class_dirs = sorted(
            [d for d in self.root.iterdir() if d.is_dir()],
            key=lambda d: d.name,
        )

        idx = 0
        for class_dir in class_dirs:
            imgs = [
                p for p in class_dir.rglob("*")
                if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
            ]
            if not imgs:
                continue

            self.class_names.append(class_dir.name)
            self.class_to_idx[class_dir.name] = idx

            for img_path in sorted(imgs):
                self.samples.append((img_path, idx))

            idx += 1

        if not self.samples:
            raise RuntimeError(
                f"No images found under '{self.root}'. "
                "Run standardize_data.py first, or point to a directory with images."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        img_path, label = self.samples[index]

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as exc:
            raise RuntimeError(f"Cannot open image {img_path}: {exc}") from exc

        if self.transform:
            img = self.transform(img)

        return img, label

    def __repr__(self) -> str:
        per_class = {}
        for _, label in self.samples:
            name = self.class_names[label]
            per_class[name] = per_class.get(name, 0) + 1
        lines = [f"  {name}: {count}" for name, count in per_class.items()]
        return (
            f"BarkDiseaseDataset(root='{self.root}', "
            f"n_samples={len(self)}, classes=[\n" + "\n".join(lines) + "\n])"
        )


def get_transforms(
    image_size: int = 224,
    color_jitter_strength: float = 0.4,
    random_grayscale_p: float = 0.10,
    gaussian_blur_p: float = 0.20,
) -> tuple[transforms.Compose, transforms.Compose]:

    oversize = int(image_size * 1.12)   # e.g. 250 for 224-target

    train_transform = transforms.Compose([
        transforms.Resize(
            (oversize, oversize),
            interpolation=transforms.InterpolationMode.LANCZOS,
        ),

        transforms.RandomCrop(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),

        transforms.RandomRotation(
            degrees=20,
            interpolation=transforms.InterpolationMode.BILINEAR,
            fill=0,
        ),
        transforms.ColorJitter(
            brightness=color_jitter_strength,
            contrast=color_jitter_strength,
            saturation=color_jitter_strength,
            hue=0.1,
        ),
        transforms.RandomGrayscale(p=random_grayscale_p),
        transforms.RandomApply(
            [transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))],
            p=gaussian_blur_p,
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    val_transform = transforms.Compose([
        transforms.Resize(int(image_size * 1.15),
                          interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    return train_transform, val_transform


def get_dataloaders(
    root: str | Path,
    val_split: float = 0.20,
    batch_size: int = 32,
    image_size: int = 224,
    num_workers: int = 2,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, list[str]]:
    train_tf, val_tf = get_transforms(image_size=image_size)

    full_dataset = BarkDiseaseDataset(root=root, transform=None)
    class_names  = full_dataset.class_names
    n_total      = len(full_dataset)
    n_val        = max(1, int(n_total * val_split))
    n_train      = n_total - n_val

    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(full_dataset, [n_train, n_val], generator=generator)

    train_dataset = _TransformSubset(train_subset, transform=train_tf)
    val_dataset   = _TransformSubset(val_subset,   transform=val_tf)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=len(train_dataset) >= batch_size,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    print(f"[DataLoader] Classes  : {class_names}")
    print(f"[DataLoader] Train    : {len(train_dataset)} images | Val: {len(val_dataset)} images")
    print(f"[DataLoader] Batches  : {len(train_loader)} train | {len(val_loader)} val")

    return train_loader, val_loader, class_names


class _TransformSubset(Dataset):

    def __init__(self, subset, transform: Optional[Callable] = None) -> None:
        self.subset    = subset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.subset)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img_path, label = self.subset.dataset.samples[self.subset.indices[idx]]

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as exc:
            raise RuntimeError(f"Cannot open {img_path}: {exc}") from exc

        if self.transform:
            img = self.transform(img)

        return img, label


if __name__ == "__main__":
    import sys

    root = sys.argv[1] if len(sys.argv) > 1 else "cork_oak_dataset_standardized/Training"
    print(f"Loading dataset from: {root}")
    loader, val_loader, classes = get_dataloaders(root, batch_size=8)
    imgs, labels = next(iter(loader))
    print(f"Batch shape : {imgs.shape}   Labels: {labels.tolist()}")
    print(f"Pixel range : [{imgs.min():.2f}, {imgs.max():.2f}]")
    print("augmentation.py OK ✓")
