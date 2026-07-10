import os
import time
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from model import PlantDiseaseCNN
from dataset import get_dataset_splits, get_dataloaders

def train_model(args):
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    os.makedirs(args.save_dir, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    print(f"Loading dataset from: {args.dataset_dir}")
    train_ds, val_ds, test_ds, classes = get_dataset_splits(
        args.dataset_dir, input_size=args.input_size, seed=args.seed,
        subset_fraction=args.subset_fraction
    )
    num_classes = len(classes)
    print(f"Dataset loaded. Classes: {num_classes}")
    print(f"Train size: {len(train_ds)}, Val size: {len(val_ds)}, Test size: {len(test_ds)}")

    train_loader, val_loader, test_loader = get_dataloaders(
        train_ds, val_ds, test_ds, batch_size=args.batch_size, num_workers=args.num_workers
    )

    model = PlantDiseaseCNN(num_classes=num_classes, input_size=args.input_size)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    history = {
        'epoch': [],
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'epoch_time': []
    }

    best_val_acc = 0.0
    checkpoint_path = os.path.join(args.save_dir, f"best_model_{args.dataset_name}.pth")

    print("Starting training...")
    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()

        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [Train]"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        epoch_train_loss = train_loss / train_total
        epoch_train_acc = train_correct / train_total

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch}/{args.epochs} [Val]"):
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        epoch_val_loss = val_loss / val_total
        epoch_val_acc = val_correct / val_total

        epoch_time = time.time() - epoch_start_time

        history['epoch'].append(epoch)
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)
        history['epoch_time'].append(epoch_time)

        print(f"Epoch {epoch}/{args.epochs} Summary: "
              f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.4f} | "
              f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.4f} | "
              f"Time: {epoch_time:.1f}s")

        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': epoch_val_acc,
                'classes': classes,
                'input_size': args.input_size
            }, checkpoint_path)
            print(f"--> Saved new best model checkpoint to {checkpoint_path} with Val Acc: {epoch_val_acc:.4f}")

    history_df = pd.DataFrame(history)
    csv_path = os.path.join(args.save_dir, f"history_{args.dataset_name}.csv")
    history_df.to_csv(csv_path, index=False)
    print(f"Saved training history to {csv_path}")

    plot_training_curves(history_df, args.dataset_name, args.save_dir)
    print("Saved training curves plots.")

    return checkpoint_path

def plot_training_curves(df, dataset_name, save_dir):
    epochs = df['epoch']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, df['train_loss'], label='Train Loss', color='#1f77b4', marker='o')
    ax1.plot(epochs, df['val_loss'], label='Val Loss', color='#ff7f0e', marker='s')
    ax1.set_title(f'Loss vs. Epochs ({dataset_name})', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epochs', fontsize=12)
    ax1.set_ylabel('Cross Entropy Loss', fontsize=12)
    ax1.legend(fontsize=11)
    ax1.grid(True, linestyle='--', alpha=0.6)

    ax2.plot(epochs, df['train_acc'], label='Train Acc', color='#2ca02c', marker='o')
    ax2.plot(epochs, df['val_acc'], label='Val Acc', color='#d62728', marker='s')
    ax2.set_title(f'Accuracy vs. Epochs ({dataset_name})', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epochs', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.legend(fontsize=11)
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"curves_{dataset_name}.png"), dpi=150)
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train 9-layer CNN for plant leaf disease classification.")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Path to image dataset directory")
    parser.add_argument("--dataset_name", type=str, required=True, choices=["non_augmented", "augmented"], help="Identifier for output file naming")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--input_size", type=int, default=128, help="Image size to resize to")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of dataloader workers (default 0 for MacOS stability)")
    parser.add_argument("--save_dir", type=str, default="results", help="Directory to save checkpoints/plots")
    parser.add_argument("--subset_fraction", type=float, default=1.0, help="Fraction of dataset to use for training speed optimization")

    args = parser.parse_args()
    train_model(args)
import os
import time
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from model import PlantDiseaseCNN
from dataset import get_dataset_splits, get_dataloaders


def train_model(args):
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    os.makedirs(args.save_dir, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    print(f"Loading dataset from: {args.dataset_dir}")
    train_ds, val_ds, test_ds, classes = get_dataset_splits(
        args.dataset_dir, input_size=args.input_size, seed=args.seed,
        subset_fraction=args.subset_fraction
    )
    num_classes = len(classes)
    print(f"Dataset loaded. Classes: {num_classes}")
    print(f"Train size: {len(train_ds)}, Val size: {len(val_ds)}, Test size: {len(test_ds)}")

    train_loader, val_loader, test_loader = get_dataloaders(
        train_ds, val_ds, test_ds, batch_size=args.batch_size, num_workers=args.num_workers
    )

    model = PlantDiseaseCNN(num_classes=num_classes, input_size=args.input_size)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    history = {
        'epoch': [],
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'epoch_time': []
    }

    best_val_acc = 0.0
    checkpoint_path = os.path.join(args.save_dir, f"best_model_{args.dataset_name}.pth")

    print("Starting training...")
    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()

        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [Train]"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        epoch_train_loss = train_loss / train_total
        epoch_train_acc = train_correct / train_total

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch}/{args.epochs} [Val]"):
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        epoch_val_loss = val_loss / val_total
        epoch_val_acc = val_correct / val_total

        epoch_time = time.time() - epoch_start_time

        history['epoch'].append(epoch)
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)
        history['epoch_time'].append(epoch_time)

        print(f"Epoch {epoch}/{args.epochs} Summary: "
              f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.4f} | "
              f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.4f} | "
              f"Time: {epoch_time:.1f}s")

        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': epoch_val_acc,
                'classes': classes,
                'input_size': args.input_size
            }, checkpoint_path)
            print(f"--> Saved new best model checkpoint to {checkpoint_path} with Val Acc: {epoch_val_acc:.4f}")

    history_df = pd.DataFrame(history)
    csv_path = os.path.join(args.save_dir, f"history_{args.dataset_name}.csv")
    history_df.to_csv(csv_path, index=False)
    print(f"Saved training history to {csv_path}")

    plot_training_curves(history_df, args.dataset_name, args.save_dir)
    print("Saved training curves plots.")

    return checkpoint_path


def plot_training_curves(df, dataset_name, save_dir):
    epochs = df['epoch']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, df['train_loss'], label='Train Loss', color='#1f77b4', marker='o')
    ax1.plot(epochs, df['val_loss'], label='Val Loss', color='#ff7f0e', marker='s')
    ax1.set_title(f'Loss vs. Epochs ({dataset_name})', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epochs', fontsize=12)
    ax1.set_ylabel('Cross Entropy Loss', fontsize=12)
    ax1.legend(fontsize=11)
    ax1.grid(True, linestyle='--', alpha=0.6)

    ax2.plot(epochs, df['train_acc'], label='Train Acc', color='#2ca02c', marker='o')
    ax2.plot(epochs, df['val_acc'], label='Val Acc', color='#d62728', marker='s')
    ax2.set_title(f'Accuracy vs. Epochs ({dataset_name})', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epochs', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.legend(fontsize=11)
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"curves_{dataset_name}.png"), dpi=150)
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train 9-layer CNN for plant leaf disease classification.")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Path to image dataset directory")
    parser.add_argument("--dataset_name", type=str, required=True, choices=["non_augmented", "augmented"],
                        help="Identifier for output file naming")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--input_size", type=int, default=128, help="Image size to resize to")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="Number of dataloader workers (default 0 for MacOS stability)")
    parser.add_argument("--save_dir", type=str, default="results", help="Directory to save checkpoints/plots")
    parser.add_argument("--subset_fraction", type=float, default=1.0,
                        help="Fraction of dataset to use for training speed optimization")

    args = parser.parse_args()
    train_model(args)
