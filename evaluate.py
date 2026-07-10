import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support

from model import PlantDiseaseCNN
from dataset import get_dataset_splits, get_dataloaders

def evaluate_model(args):
    torch.manual_seed(args.seed)

    os.makedirs(args.save_dir, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    print(f"Loading checkpoint from: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    classes = checkpoint['classes']
    num_classes = len(classes)
    input_size = checkpoint['input_size']
    epoch = checkpoint['epoch']
    saved_val_acc = checkpoint['val_acc']

    print(f"Loaded model checkpoint trained for {epoch} epochs (Validation Acc: {saved_val_acc:.4f})")

    print(f"Loading dataset for evaluation from: {args.dataset_dir}")
    _, _, test_ds, eval_classes = get_dataset_splits(
        args.dataset_dir, input_size=input_size, seed=args.seed,
        subset_fraction=args.subset_fraction
    )

    if eval_classes != classes:
        print("WARNING: Dataset classes for evaluation do not match the checkpoint classes!")
        print(f"Checkpoint classes: {len(classes)}, Eval dataset classes: {len(eval_classes)}")

    loaders = get_dataloaders(None, None, test_ds, batch_size=args.batch_size, num_workers=args.num_workers)
    print(f"DEBUG - Return type: {type(loaders)}, Value: {loaders}")

    _, _, test_loader = get_dataloaders(
        None, None, test_ds, batch_size=args.batch_size, num_workers=args.num_workers
    )

    model = PlantDiseaseCNN(num_classes=num_classes, input_size=input_size)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    all_preds = []
    all_targets = []

    print("Running inference on test set...")
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    acc = accuracy_score(all_targets, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average='weighted')

    print("\n" + "="*50)
    print(f"EVALUATION RESULTS FOR {args.model_name} ON {args.eval_set_name}")
    print("="*50)
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision (weighted): {precision:.4f}")
    print(f"Recall (weighted):    {recall:.4f}")
    print(f"F1-score (weighted):  {f1:.4f}")
    print("="*50)

    report = classification_report(all_targets, all_preds, target_names=classes)
    report_path = os.path.join(args.save_dir, f"report_{args.model_name}_on_{args.eval_set_name}.txt")
    with open(report_path, "w") as f:
        f.write(f"Evaluation of {args.model_name} on {args.eval_set_name}\n")
        f.write(f"Checkpoint source: {args.checkpoint}\n")
        f.write(f"Timestamp: {os.popen('date').read().strip()}\n")
        f.write("="*50 + "\n")
        f.write(f"Overall Accuracy:  {acc:.4f}\n")
        f.write(f"Overall Precision (weighted): {precision:.4f}\n")
        f.write(f"Overall Recall (weighted):    {recall:.4f}\n")
        f.write(f"Overall F1-score (weighted):  {f1:.4f}\n")
        f.write("="*50 + "\n\n")
        f.write(report)
    print(f"Saved classification report to {report_path}")

    cm = confusion_matrix(all_targets, all_preds)
    plot_confusion_matrix(cm, classes, args.model_name, args.eval_set_name, args.save_dir)
    print("Saved confusion matrix plot.")

    return acc, f1

def plot_confusion_matrix(cm, classes, model_name, eval_set_name, save_dir):
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm) # handle division by zero if any class has 0 images

    fig, ax = plt.subplots(figsize=(18, 16))

    im = ax.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=f'Normalized Confusion Matrix\n{model_name} on {eval_set_name}',
           ylabel='True disease class',
           xlabel='Predicted disease class')

    plt.setp(ax.get_xticklabels(), rotation=90, ha="right", rotation_mode="anchor", fontsize=9)
    plt.setp(ax.get_yticklabels(), fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"cm_{model_name}_on_{eval_set_name}.png"), dpi=150)
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate a trained 9-layer CNN model.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint .pth file")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Path to evaluation image dataset directory")
    parser.add_argument("--model_name", type=str, required=True, help="Name of model (e.g. non_augmented or augmented)")
    parser.add_argument("--eval_set_name", type=str, required=True, help="Name of evaluation set (e.g. non_augmented_test)")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of dataloader workers")
    parser.add_argument("--save_dir", type=str, default="results", help="Directory to save evaluation results")
    parser.add_argument("--subset_fraction", type=float, default=1.0, help="Fraction of dataset to use for evaluation")
    
    args = parser.parse_args()
    evaluate_model(args)
