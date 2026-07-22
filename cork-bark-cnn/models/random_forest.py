# Rando Forest

import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


def build_feature_extractor(device: torch.device) -> nn.Module:
    weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
    backbone = models.mobilenet_v3_small(weights=weights)

    extractor = nn.Sequential(
        backbone.features,
        backbone.avgpool,
        nn.Flatten(),
    )
    for param in extractor.parameters():
        param.requires_grad = False

    extractor = extractor.to(device)
    extractor.eval()
    print(f"[FeatureExtractor] MobileNetV3-Small (frozen) → 576-dim embeddings | device: {device}")
    return extractor


def extract_features(
    loader: DataLoader,
    device: torch.device,
    extractor: Optional[nn.Module] = None,
) -> tuple[np.ndarray, np.ndarray]:
    if extractor is None:
        extractor = build_feature_extractor(device)

    feature_loader = DataLoader(
        loader.dataset,
        batch_size=loader.batch_size or 32,
        shuffle=False,
        num_workers=loader.num_workers,
        drop_last=False,
        pin_memory=True,
    )

    all_feats, all_labels = [], []

    with torch.no_grad():
        for imgs, labels in tqdm(feature_loader, desc="Extracting features", leave=False):
            imgs = imgs.to(device)
            feats = extractor(imgs)  # (batch, 576)
            all_feats.append(feats.cpu().numpy())
            all_labels.append(labels.numpy())

    X = np.concatenate(all_feats, axis=0)
    y = np.concatenate(all_labels, axis=0)
    print(f"[FeatureExtractor] Extracted {X.shape[0]} samples × {X.shape[1]} features")
    return X, y


def train_rf(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = 300,
    max_depth: Optional[int] = None,
    class_weight: str = "balanced",
    random_state: int = 42,
) -> RandomForestClassifier:
    print(f"\n[RandomForest] Training {n_estimators} trees on {X_train.shape[0]} samples …")

    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=-1,
        oob_score=True,
    )
    rf.fit(X_train, y_train)

    print(f"[RandomForest] OOB accuracy estimate: {rf.oob_score_:.4f}")
    return rf


def predict_rf(
    rf: RandomForestClassifier,
    X: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:

    preds = rf.predict(X)
    probs = rf.predict_proba(X)
    return preds, probs


def save_rf(rf: RandomForestClassifier, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(rf, f)
    print(f"[RandomForest] Saved to {path}")


def load_rf(path: str) -> RandomForestClassifier:
    with open(path, "rb") as f:
        return pickle.load(f)
