# EfficientNet B0

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_efficientnet(
    num_classes: int,
    dropout: float = 0.3,
    pretrained: bool = True,
) -> tuple[nn.Module, torch.device]:

    device = get_device()
    print(f"[EfficientNet] Device: {device}")

    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    backbone = models.efficientnet_b0(weights=weights)

    in_features = backbone.classifier[1].in_features  # 1280
    backbone.classifier = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, 256),
        nn.SiLU(),
        nn.Dropout(p=dropout * 0.5),
        nn.Linear(256, num_classes),
    )

    for param in backbone.features.parameters():
        param.requires_grad = False

    model = backbone.to(device)
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[EfficientNet] Parameters: {total:,} total | {trainable:,} trainable (head only)")

    return model, device


def unfreeze_last_blocks(model: nn.Module, n_blocks: int = 2) -> None:
    n_feature_blocks = len(model.features)
    for i in range(n_feature_blocks - n_blocks, n_feature_blocks):
        for param in model.features[i].parameters():
            param.requires_grad = True
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[EfficientNet] Unfroze last {n_blocks} blocks → {trainable:,} trainable params")



def train_efficientnet(
    model: nn.Module,
    device: torch.device,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_classes: int,
    epochs_phase1: int = 10,
    epochs_phase2: int = 10,
    lr_phase1: float = 1e-3,
    lr_phase2: float = 3e-4,
    save_path: Optional[str] = None,
) -> tuple[nn.Module, dict]:
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    def _run_phase(phase_epochs: int, lr: float, phase_name: str) -> None:
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr, weight_decay=1e-4,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=phase_epochs, eta_min=lr * 0.01
        )
        best_val_acc = 0.0

        for epoch in range(1, phase_epochs + 1):
            model.train()
            running_loss = 0.0
            for imgs, labels in tqdm(
                train_loader,
                desc=f"[{phase_name}] Epoch {epoch}/{phase_epochs} train",
                leave=False,
            ):
                imgs, labels = imgs.to(device), labels.to(device)
                optimizer.zero_grad()
                loss = criterion(model(imgs), labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * imgs.size(0)

            train_loss = running_loss / len(train_loader.dataset)
            history["train_loss"].append(train_loss)

            model.eval()
            val_loss, correct = 0.0, 0
            with torch.no_grad():
                for imgs, labels in val_loader:
                    imgs, labels = imgs.to(device), labels.to(device)
                    logits = model(imgs)
                    val_loss += criterion(logits, labels).item() * imgs.size(0)
                    correct += (logits.argmax(1) == labels).sum().item()

            val_loss /= len(val_loader.dataset)
            val_acc   = correct / len(val_loader.dataset)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            print(
                f"[{phase_name}] Epoch {epoch}/{phase_epochs} | "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_acc:.4f}"
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                if save_path:
                    torch.save(model.state_dict(), save_path)

            scheduler.step()

    print("\n[EfficientNet] ─── Phase 1: Training head (backbone frozen) ───")
    _run_phase(epochs_phase1, lr_phase1, "Phase1")

    print("\n[EfficientNet] ─── Phase 2: Fine-tuning last feature blocks ───")
    unfreeze_last_blocks(model, n_blocks=2)
    _run_phase(epochs_phase2, lr_phase2, "Phase2")

    return model, history


def predict_efficientnet(
    model: nn.Module,
    device: torch.device,
    loader: DataLoader,
) -> tuple[list[int], list[list[float]]]:
    model.eval()
    all_preds, all_probs = [], []
    with torch.no_grad():
        for imgs, _ in loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            probs  = torch.softmax(logits, dim=1)
            all_preds.extend(logits.argmax(1).cpu().tolist())
            all_probs.extend(probs.cpu().tolist())
    return all_preds, all_probs


def load_efficientnet(
    checkpoint_path: str,
    num_classes: int,
) -> tuple[nn.Module, torch.device]:
    model, device = build_efficientnet(num_classes, pretrained=False)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model, device
