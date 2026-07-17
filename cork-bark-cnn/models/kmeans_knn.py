# Centroid kNN

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize


@dataclass
class CentroidKNN:
    representative_X:   np.ndarray
    representative_y:   np.ndarray
    representative_idx: np.ndarray
    knn:                KNeighborsClassifier
    k_per_class:        int
    num_classes:        int
    class_names:        list[str] = field(default_factory=list)



def build_centroid_knn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    num_classes: int,
    k_per_class: int = 3,
    knn_k: int = 5,
    normalize_features: bool = True,
    random_state: int = 42,
    class_names: Optional[list[str]] = None,
) -> CentroidKNN:

    if normalize_features:
        X_train = normalize(X_train, norm="l2")

    k_total = num_classes * k_per_class
    print(f"\n[CentroidKNN] Running K-Means: k={k_total} ({num_classes} classes × {k_per_class})")

    kmeans = KMeans(n_clusters=k_total, n_init=10, random_state=random_state)
    kmeans.fit(X_train)
    centroids = kmeans.cluster_centers_

    rep_X_list, rep_y_list, rep_idx_list = [], [], []

    for c_idx, centroid in enumerate(centroids):
        dists = np.linalg.norm(X_train - centroid, axis=1)
        nearest_idx = int(np.argmin(dists))

        rep_X_list.append(X_train[nearest_idx])
        rep_y_list.append(y_train[nearest_idx])
        rep_idx_list.append(nearest_idx)

    representative_X   = np.stack(rep_X_list)    # (k_total, D)
    representative_y   = np.array(rep_y_list)    # (k_total,)
    representative_idx = np.array(rep_idx_list)  # (k_total,)

    unique_pairs = {}
    for i in range(len(representative_idx)):
        ridx = representative_idx[i]
        if ridx not in unique_pairs:
            unique_pairs[ridx] = (representative_X[i], representative_y[i])

    rep_X_dedup = np.stack([v[0] for v in unique_pairs.values()])
    rep_y_dedup = np.array([v[1] for v in unique_pairs.values()])
    rep_idx_dedup = np.array(list(unique_pairs.keys()))

    print(f"[CentroidKNN] {k_total} centroids → {len(rep_idx_dedup)} unique representatives "
          f"(after dedup)")

    for cls in range(num_classes):
        n = int((rep_y_dedup == cls).sum())
        name = class_names[cls] if class_names else str(cls)
        print(f"  Class {cls} ({name}): {n} representatives")

    actual_k = min(knn_k, len(rep_X_dedup))
    knn = KNeighborsClassifier(
        n_neighbors=actual_k,
        metric="euclidean",
        weights="distance",
        algorithm="ball_tree",
        n_jobs=-1,
    )
    knn.fit(rep_X_dedup, rep_y_dedup)
    print(f"[CentroidKNN] KNN fitted (k={actual_k}) on {len(rep_X_dedup)} representatives")

    return CentroidKNN(
        representative_X=rep_X_dedup,
        representative_y=rep_y_dedup,
        representative_idx=rep_idx_dedup,
        knn=knn,
        k_per_class=k_per_class,
        num_classes=num_classes,
        class_names=class_names or [str(i) for i in range(num_classes)],
    )


def predict_centroid_knn(
    cknn: CentroidKNN,
    X: np.ndarray,
    normalize_features: bool = True,
) -> tuple[np.ndarray, np.ndarray]:

    if normalize_features:
        X = normalize(X, norm="l2")

    preds = cknn.knn.predict(X)
    probs = cknn.knn.predict_proba(X)
    return preds, probs


def save_centroid_knn(cknn: CentroidKNN, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(cknn, f)
    print(f"[CentroidKNN] Saved to {path}")


def load_centroid_knn(path: str) -> CentroidKNN:
    with open(path, "rb") as f:
        return pickle.load(f)
