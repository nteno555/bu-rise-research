# python train_eval.py --data cork_oak_dataset_standardized/Training
# python train_eval.py --data cork_oak_dataset_standardized/Training --models rf svm cknn
# keys: mobilenet, efficientnet, rf, svm, cknn

import argparse
import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from augmentation import get_dataloaders, get_transforms, BarkDiseaseDataset
from models.random_forest import build_feature_extractor, extract_features, train_rf, predict_rf, save_rf
from models.kmeans_knn import build_centroid_knn, predict_centroid_knn, save_centroid_knn
from models.svm import train_svm, predict_svm, save_svm


ALL_MODEL_KEYS = ["mobilenet", "efficientnet", "rf", "svm", "cknn"]
OUTPUT_DIR = Path("checkpoints")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate all bark disease models.")
    parser.add_argument(
        "--data",
        type=str,
        default="cork_oak_dataset_standardized/Training",
        help="Root data directory (class subfolders expected)",
    )
    parser.add_argument(
        "--eval-data",
        type=str,
        default=None,
        help="Optional held-out eval directory (e.g. cork_oak_dataset_standardized/Eval). "
             "If not provided, val split from --data is used.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=ALL_MODEL_KEYS,
        default=ALL_MODEL_KEYS,
        help=f"Which models to train. Choices: {ALL_MODEL_KEYS}",
    )
    parser.add_argument("--size",     type=int,   default=224,  help="Image size (default 224)")
    parser.add_argument("--batch",    type=int,   default=32,   help="Batch size (default 32)")
    parser.add_argument("--val-split",type=float, default=0.20, help="Val split fraction (default 0.20)")
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Epochs per phase for deep models. Default: 10 (phase1) + 10 (phase2). "
             "Use --smoke-test for a quick 2-epoch run.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a quick end-to-end check: 2 epochs, 10%% of data, all models.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default 42)",
    )
    return parser.parse_args()



def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def smoke_test_subset(dataset, frac: float = 0.10):
    from torch.utils.data import Subset
    import random
    n = max(2, int(len(dataset) * frac))
    indices = random.sample(range(len(dataset)), n)
    return Subset(dataset, indices)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
) -> dict:
    return {
        "accuracy":          accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1_macro":          f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted":       f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "confusion_matrix":  confusion_matrix(y_true, y_pred).tolist(),
        "report":            classification_report(
                                 y_true, y_pred,
                                 target_names=class_names,
                                 zero_division=0,
                             ),
    }


def print_comparison_table(results: dict[str, dict]) -> None:
    print("\n" + "=" * 70)
    print("  MODEL COMPARISON")
    print("=" * 70)
    header = f"{'Model':<20} {'Accuracy':>10} {'Balanced':>10} {'F1-Macro':>10} {'Train(s)':>10}"
    print(header)
    print("-" * 70)
    for name, r in results.items():
        m = r["metrics"]
        print(
            f"{name:<20} {m['accuracy']:>10.4f} {m['balanced_accuracy']:>10.4f} "
            f"{m['f1_macro']:>10.4f} {r.get('train_time', 0):>10.1f}"
        )
    print("=" * 70)


def plot_confusion_matrices(
    results: dict[str, dict],
    class_names: list[str],
    save_path: str,
) -> None:
    n_models = len(results)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 5))
    if n_models == 1:
        axes = [axes]

    for ax, (name, r) in zip(axes, results.items()):
        cm = np.array(r["metrics"]["confusion_matrix"])
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        ax.set_title(name, fontsize=11, fontweight="bold")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        tick_marks = np.arange(len(class_names))
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(class_names, rotation=30, ha="right", fontsize=8)
        ax.set_yticklabels(class_names, fontsize=8)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]),
                        ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black",
                        fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle("Confusion Matrices — Cork Oak Bark Disease Models", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"[Plot] Confusion matrices saved → {save_path}")


def plot_training_curves(
    name: str,
    history: dict,
    save_path: str,
) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(epochs, history["train_loss"], label="Train Loss", marker="o", markersize=3)
    ax1.plot(epochs, history["val_loss"],   label="Val Loss",   marker="o", markersize=3)
    ax1.axvline(x=len(history["train_loss"]) // 2, color="gray", linestyle="--", alpha=0.5,
                label="Phase boundary")
    ax1.set_title(f"{name} — Loss")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history["val_acc"], label="Val Accuracy", color="green", marker="o", markersize=3)
    ax2.set_title(f"{name} — Validation Accuracy")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy")
    ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"[Plot] Training curves saved → {save_path}")



def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    epochs_per_phase = 2 if args.smoke_test else (args.epochs or 10)
    device = get_device()
    print(f"\n{'=' * 60}")
    print(f"  Cork Oak Bark Disease — Training Pipeline")
    print(f"  Device       : {device}")
    print(f"  Data root    : {args.data}")
    print(f"  Models       : {args.models}")
    print(f"  Smoke test   : {args.smoke_test}")
    print(f"{'=' * 60}\n")

    train_loader, val_loader, class_names = get_dataloaders(
        root=args.data,
        val_split=args.val_split,
        batch_size=args.batch,
        image_size=args.size,
        seed=args.seed,
    )
    num_classes = len(class_names)
    print(f"\nClasses ({num_classes}): {class_names}\n")

    if num_classes < 2:
        raise RuntimeError(
            f"Only {num_classes} class(es) with data found. "
            "Need at least 2 classes to train classifiers. "
            "Add images to the empty class folders and re-run."
        )

    results: dict[str, dict] = {}

    need_features = any(m in args.models for m in ["rf", "svm", "cknn"])
    X_train, y_train, X_val, y_val = None, None, None, None

    if need_features:
        print("[Features] Extracting MobileNetV3 embeddings …")
        extractor = build_feature_extractor(device)
        X_train, y_train = extract_features(train_loader, device, extractor)
        X_val,   y_val   = extract_features(val_loader,   device, extractor)
        print(f"[Features] Train: {X_train.shape} | Val: {X_val.shape}\n")

    if "rf" in args.models:
        print("─" * 50)
        print("▶  Random Forest")
        t0 = time.time()
        n_trees = 50 if args.smoke_test else 300
        rf = train_rf(X_train, y_train, n_estimators=n_trees)
        preds, probs = predict_rf(rf, X_val)
        elapsed = time.time() - t0
        save_rf(rf, str(OUTPUT_DIR / "random_forest.pkl"))

        metrics = compute_metrics(y_val, preds, class_names)
        results["RandomForest"] = {"metrics": metrics, "train_time": elapsed}
        print(metrics["report"])

    if "svm" in args.models:
        print("─" * 50)
        print("▶  SVM (RBF kernel)")
        t0 = time.time()
        search = not args.smoke_test  # skip grid search in smoke test
        svm_pipeline, best_params = train_svm(X_train, y_train, search_C=search)
        preds, probs = predict_svm(svm_pipeline, X_val)
        elapsed = time.time() - t0
        save_svm(svm_pipeline, str(OUTPUT_DIR / "svm.pkl"))

        metrics = compute_metrics(y_val, preds, class_names)
        results["SVM"] = {"metrics": metrics, "train_time": elapsed}
        print(metrics["report"])

    if "cknn" in args.models:
        print("─" * 50)
        print("▶  Centroid-KNN")
        t0 = time.time()
        cknn = build_centroid_knn(
            X_train, y_train,
            num_classes=num_classes,
            k_per_class=7,
            knn_k=1,
            class_names=class_names,
        )

        print("\nREPRESENTATIVE IMAGE FILE PATHS:")
        for cls_idx, reps in cknn.representatives_by_class.items():
            cls_name = class_names[cls_idx]
            print(f"\n  Class {cls_idx} ({cls_name}):")
            for i, rep in enumerate(reps):
                row_idx = rep["original_index"]

                if hasattr(train_loader.dataset, 'get_image_path'):
                    img_path = train_loader.dataset.get_image_path(row_idx)
                elif hasattr(train_loader.dataset, 'indices'):
                    real_idx = train_loader.dataset.indices[row_idx]
                    img_path, _ = train_loader.dataset.dataset.samples[real_idx]
                else:
                    img_path, _ = train_loader.dataset.samples[row_idx]

                print(f"    [{i + 1}] Row {row_idx} -> Path: {img_path}")
        print()

        preds, probs = predict_centroid_knn(cknn, X_val)
        elapsed = time.time() - t0
        save_centroid_knn(cknn, str(OUTPUT_DIR / "centroid_knn.pkl"))

        metrics = compute_metrics(y_val, preds, class_names)
        results["CentroidKNN"] = {"metrics": metrics, "train_time": elapsed}
        print(metrics["report"])

    if "mobilenet" in args.models:
        from models.cnn_mobilenet import build_mobilenet, train_mobilenet, predict_mobilenet

        print("─" * 50)
        print("▶  MobileNetV3-Small CNN")
        t0 = time.time()
        model, device_ = build_mobilenet(num_classes=num_classes)
        model, history = train_mobilenet(
            model, device_, train_loader, val_loader,
            num_classes=num_classes,
            epochs_phase1=epochs_per_phase,
            epochs_phase2=epochs_per_phase,
            save_path=str(OUTPUT_DIR / "mobilenet_best.pt"),
        )
        preds, probs = predict_mobilenet(model, device_, val_loader)
        elapsed = time.time() - t0

        y_val_list = [label for _, label in val_loader.dataset]
        y_true = np.array(y_val_list)
        metrics = compute_metrics(y_true, np.array(preds), class_names)
        results["MobileNetV3"] = {"metrics": metrics, "train_time": elapsed}
        print(metrics["report"])

        plot_training_curves(
            "MobileNetV3",
            history,
            str(OUTPUT_DIR / "mobilenet_training_curves.png"),
        )

    if "efficientnet" in args.models:
        from models.efficientnet import build_efficientnet, train_efficientnet, predict_efficientnet

        print("─" * 50)
        print("▶  EfficientNet-B0")
        t0 = time.time()
        model, device_ = build_efficientnet(num_classes=num_classes)
        model, history = train_efficientnet(
            model, device_, train_loader, val_loader,
            num_classes=num_classes,
            epochs_phase1=epochs_per_phase,
            epochs_phase2=epochs_per_phase,
            save_path=str(OUTPUT_DIR / "efficientnet_best.pt"),
        )
        preds, probs = predict_efficientnet(model, device_, val_loader)
        elapsed = time.time() - t0

        y_val_list = [label for _, label in val_loader.dataset]
        y_true = np.array(y_val_list)
        metrics = compute_metrics(y_true, np.array(preds), class_names)
        results["EfficientNet-B0"] = {"metrics": metrics, "train_time": elapsed}
        print(metrics["report"])

        plot_training_curves(
            "EfficientNet-B0",
            history,
            str(OUTPUT_DIR / "efficientnet_training_curves.png"),
        )

    if results:
        print_comparison_table(results)

        plot_confusion_matrices(
            results,
            class_names,
            str(OUTPUT_DIR / "confusion_matrices.png"),
        )

        serialisable = {
            name: {
                "accuracy":          r["metrics"]["accuracy"],
                "balanced_accuracy": r["metrics"]["balanced_accuracy"],
                "f1_macro":          r["metrics"]["f1_macro"],
                "f1_weighted":       r["metrics"]["f1_weighted"],
                "train_time_s":      r.get("train_time", 0),
            }
            for name, r in results.items()
        }
        results_path = OUTPUT_DIR / "results.json"
        with open(results_path, "w") as f:
            json.dump(serialisable, f, indent=2)
        print(f"\n[Results] Saved to {results_path}")
        print(f"[Results] Plots saved to {OUTPUT_DIR}/")
    else:
        print("[Warning] No models were trained.")


if __name__ == "__main__":
    main()

