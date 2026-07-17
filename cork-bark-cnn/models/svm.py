# Simple Vector Machine (SVM)

import pickle
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline


def train_svm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    C: float = 1.0,
    gamma: str = "scale",
    kernel: str = "rbf",
    class_weight: str = "balanced",
    search_C: bool = True,
    random_state: int = 42,
) -> tuple[Pipeline, dict]:

    print(f"\n[SVM] Training SVM on {X_train.shape[0]} samples × {X_train.shape[1]} features")

    svc = SVC(
        kernel=kernel,
        C=C,
        gamma=gamma,
        class_weight=class_weight,
        probability=True,
        random_state=random_state,
        max_iter=5000,
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", svc),
    ])

    best_params = {"C": C, "gamma": gamma, "kernel": kernel}

    if search_C and len(np.unique(y_train)) > 1:
        n_splits = min(3, min(np.bincount(y_train)))
        n_splits = max(2, n_splits)

        param_grid = {
            "svc__C": [0.1, 1.0, 10.0, 100.0],
        }
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        grid_search = GridSearchCV(
            pipeline,
            param_grid,
            cv=cv,
            scoring="balanced_accuracy",
            n_jobs=-1,
            refit=True,
            verbose=1,
        )
        grid_search.fit(X_train, y_train)
        pipeline = grid_search.best_estimator_
        best_params["C"] = grid_search.best_params_["svc__C"]
        print(f"[SVM] Best C={best_params['C']:.3g} | "
              f"CV balanced-accuracy={grid_search.best_score_:.4f}")
    else:
        pipeline.fit(X_train, y_train)
        print(f"[SVM] Trained with C={C}")

    return pipeline, best_params


def predict_svm(
    pipeline: Pipeline,
    X: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:

    preds = pipeline.predict(X)
    probs = pipeline.predict_proba(X)
    return preds, probs


def save_svm(pipeline: Pipeline, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(pipeline, f)
    print(f"[SVM] Saved to {path}")


def load_svm(path: str) -> Pipeline:
    with open(path, "rb") as f:
        return pickle.load(f)
