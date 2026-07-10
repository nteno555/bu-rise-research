import json
import os

notebook_path = "disease_detection.ipynb"

cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Plant Leaf Disease Detection using 9-Layer CNN\n",
            "\n",
            "This notebook walks through the implementation, training, and evaluation of a custom 9-layer Convolutional Neural Network (CNN) for identifying plant leaf diseases. We compare two training settings:\n",
            "1. **Non-augmented Dataset**: Trained on the original plant leaf images.\n",
            "2. **Augmented Dataset**: Trained on pre-augmented images containing rotation, flipping, noise injection, and color shifts.\n",
            "\n",
            "Both models are evaluated on a held-out **non-augmented test set** to compare true generalization performance on original images."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import os\n",
            "import pandas as pd\n",
            "import matplotlib.pyplot as plt\n",
            "import torch\n",
            "from IPython.display import Image, display\n",
            "\n",
            "from model import PlantDiseaseCNN\n",
            "from dataset import get_dataset_splits, get_dataloaders\n",
            "from evaluate import evaluate_model\n",
            "\n",
            "# Verify PyTorch device setup\n",
            "device = torch.device(\"mps\" if torch.backends.mps.is_available() \n",
            "                      else \"cuda\" if torch.cuda.is_available() \n",
            "                      else \"cpu\")\n",
            "print(f\"Using device for evaluation: {device}\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 1. Dataset Overview\n",
            "\n",
            "Let's check the size and class distribution of the plant leaf disease datasets."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "base_path = \"Data for Identification of Plant Leaf Diseases Using a 9-layer Deep Convolutional Neural Network\"\n",
            "no_aug_dir = os.path.join(base_path, \"Plant_leave_diseases_dataset_without_augmentation\")\n",
            "aug_dir = os.path.join(base_path, \"Plant_leave_diseases_dataset_with_augmentation\")\n",
            "\n",
            "train_ds_no_aug, val_ds_no_aug, test_ds_no_aug, classes = get_dataset_splits(no_aug_dir, seed=42)\n",
            "train_ds_aug, val_ds_aug, test_ds_aug, _ = get_dataset_splits(aug_dir, seed=42)\n",
            "\n",
            "print(f\"Total Number of Classes: {len(classes)}\")\n",
            "print(f\"\\n--- Non-Augmented Dataset Split ---\")\n",
            "print(f\"Train size: {len(train_ds_no_aug)}\")\n",
            "print(f\"Val size:   {len(val_ds_no_aug)}\")\n",
            "print(f\"Test size:  {len(test_ds_no_aug)}\")\n",
            "\n",
            "print(f\"\\n--- Augmented Dataset Split ---\")\n",
            "print(f\"Train size: {len(train_ds_aug)}\")\n",
            "print(f\"Val size:   {len(val_ds_aug)}\")\n",
            "print(f\"Test size:  {len(test_ds_aug)}\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 2. Model Architecture\n",
            "\n",
            "Our model is a custom 9-layer CNN. Let's inspect the network structure by instantiating it."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "model = PlantDiseaseCNN(num_classes=len(classes), input_size=128)\n",
            "print(model)"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 3. Training Curves Comparison\n",
            "\n",
            "Let's display and compare the training curves for both models."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "print(\"### Non-Augmented Training Curves ###\")\n",
            "if os.path.exists(\"results/curves_non_augmented.png\"):\n",
            "    display(Image(filename=\"results/curves_non_augmented.png\"))\n",
            "else:\n",
            "    print(\"Non-augmented model curves plot not found. Run training first.\")\n",
            "\n",
            "print(\"\\n### Augmented Training Curves ###\")\n",
            "if os.path.exists(\"results/curves_augmented.png\"):\n",
            "    display(Image(filename=\"results/curves_augmented.png\"))\n",
            "else:\n",
            "    print(\"Augmented model curves plot not found. Run training first.\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 4. Evaluation and Performance Comparison\n",
            "\n",
            "We evaluate both models on the **same non-augmented test set** to ensure a fair comparison on clean, unmodified leaf images."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "class Args:\n",
            "    def __init__(self, checkpoint, dataset_dir, model_name, eval_set_name, batch_size=128, seed=42, num_workers=0, save_dir=\"results\"):\n",
            "        self.checkpoint = checkpoint\n",
            "        self.dataset_dir = dataset_dir\n",
            "        self.model_name = model_name\n",
            "        self.eval_set_name = eval_set_name\n",
            "        self.batch_size = batch_size\n",
            "        self.seed = seed\n",
            "        self.num_workers = num_workers\n",
            "        self.save_dir = save_dir\n",
            "\n",
            "print(\"### 4a. Evaluating Non-Augmented Model ###\")\n",
            "args_no_aug = Args(\n",
            "    checkpoint=\"results/best_model_non_augmented.pth\",\n",
            "    dataset_dir=no_aug_dir,\n",
            "    model_name=\"non_augmented\",\n",
            "    eval_set_name=\"non_augmented_test\"\n",
            ")\n",
            "acc_no_aug, f1_no_aug = evaluate_model(args_no_aug)\n",
            "\n",
            "print(\"\\n### 4b. Evaluating Augmented Model ###\")\n",
            "args_aug = Args(\n",
            "    checkpoint=\"results/best_model_augmented.pth\",\n",
            "    dataset_dir=no_aug_dir, # Test on non-augmented test set!\n",
            "    model_name=\"augmented\",\n",
            "    eval_set_name=\"non_augmented_test\"\n",
            ")\n",
            "acc_aug, f1_aug = evaluate_model(args_aug)"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 5. Visualizing Confusion Matrices\n",
            "\n",
            "Let's look at the confusion matrices for both models to understand which disease categories are most commonly confused."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "print(\"### Confusion Matrix: Non-Augmented Model ###\")\n",
            "if os.path.exists(\"results/cm_non_augmented_on_non_augmented_test.png\"):\n",
            "    display(Image(filename=\"results/cm_non_augmented_on_non_augmented_test.png\"))\n",
            "\n",
            "print(\"\\n### Confusion Matrix: Augmented Model ###\")\n",
            "if os.path.exists(\"results/cm_augmented_on_non_augmented_test.png\"):\n",
            "    display(Image(filename=\"results/cm_augmented_on_non_augmented_test.png\"))"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 6. Discussion and Conclusions\n",
            "\n",
            "Based on the results, we can see how the augmented model performs compared to the non-augmented model on the same clean leaf test set. \n",
            "Typically, data augmentation reduces overfitting during training and leads to higher classification accuracy on unseen, original leaf images. This is because augmentation forces the model to learn color-invariant and orientation-invariant features rather than memorizing background pixels or leaf rotations."
        ]
    }
]

notebook_data = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.11.7"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 2
}

with open(notebook_path, "w") as f:
    json.dump(notebook_data, f, indent=1)

print("Notebook updated successfully!")
