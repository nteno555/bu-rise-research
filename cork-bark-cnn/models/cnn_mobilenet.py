#MobileNet-V3

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


def build_mobilenet(
    num_classes: int,
    dropout: float = 0.3,
    pretrained: bool = True,
) -> tuple[nn.Module, torch.device]:

    device = get_device()
    print(f"[MobileNet] Device: {device}")

    weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
    backbone = models.mobilenet_v3_small(weights=weights)

    in_features = backbone.classifier[0].in_features  # 576
    backbone.classifier = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.Hardswish(),
        nn.Dropout(p=dropout),
        nn.Linear(256, num_classes),
    )

    for param in backbone.features.parameters():
        param.requires_grad = False

    model = backbone.to(device)
    total   = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[MobileNet] Parameters: {total:,} total | {trainable:,} trainable (head only)")

    return model, device


def unfreeze_last_block(model: nn.Module) -> None:
    for i in [11, 12]:
        for param in model.features[i].parameters():
            param.requires_grad = True
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[MobileNet] Unfroze last 2 feature blocks → {trainable:,} trainable params")


def train_mobilenet(
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

# 2 phase traiing

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
                    print(f"  → Saved best model ({val_acc:.4f}) to {save_path}")

            scheduler.step()

    print("\n[MobileNet] ─── Phase 1: Training head (backbone frozen) ───")
    _run_phase(epochs_phase1, lr_phase1, "Phase1")

    print("\n[MobileNet] ─── Phase 2: Fine-tuning last feature block ───")
    unfreeze_last_block(model)
    _run_phase(epochs_phase2, lr_phase2, "Phase2")

    return model, history


def predict_mobilenet(
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


def load_mobilenet(
    checkpoint_path: str,
    num_classes: int,
    dropout: float = 0.3,
) -> tuple[nn.Module, torch.device]:
    model, device = build_mobilenet(num_classes, dropout, pretrained=False)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model, device
