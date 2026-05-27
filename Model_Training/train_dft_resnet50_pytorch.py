import os
import json
import copy
import time
import random
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms
from torchvision.models import resnet50, ResNet50_Weights

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
)


# Split Image into RGB Channels
# # DFT is applied to each channel separately.

# Apply 2D DFT to Each Channel
# # Converts image from spatial domain into frequency domain.

# Apply fftshift
# # Moves low-frequency components to the centre.

# Calculate Magnitude Spectrum
# # Measures the strength of each frequency component.

# Apply Logarithmic Scaling
# # Compresses large frequency values for better visualisation.

# Normalize to 0–255
# # Converts magnitude values into image intensity range.

# Merge Transformed Channels
# # Produces 3-channel DFT image for ResNet50.

# Save as PNG
# # Stores the converted frequency-domain dataset.

# Input to ResNet50
# # Uses the DFT image as model input.

# Binary Classification Output
# # Predicts fake or real.



# ============================================================
# Configuration
# ============================================================

DATA_PATH = r"C:\Users\kcwong6\Downloads\DeepFakeFace_binary\DeepFakeFace_fft"
OUTPUT_DIR = r"C:\Users\kcwong6\Downloads\DeepFakeFace_binary\F1\resnet50_dft_report"

IMG_SIZE = 256
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
RANDOM_SEED = 42

PATIENCE = 10
LR_PATIENCE = 5
LR_FACTOR = 0.2
MIN_LR = 1e-6

NUM_CLASSES = 2

# Important for Windows
NUM_WORKERS = 0


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# Data
# ============================================================

def build_dataloaders(data_path):
    train_dir = os.path.join(data_path, "train")
    val_dir = os.path.join(data_path, "val")
    test_dir = os.path.join(data_path, "test")

    # For DFT RGB images:
    # Your DFT images are already 3-channel PNG images.
    # Therefore, DO NOT use Grayscale here.
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.RandomResizedCrop(
            size=IMG_SIZE,
            scale=(0.9, 1.0),
            ratio=(0.95, 1.05)
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=eval_transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=eval_transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    return train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader


# ============================================================
# Model
# ============================================================

def build_dft_resnet50_model(device):
    # Load ImageNet-pretrained ResNet50.
    # Although the input is DFT images, pretrained ResNet50 still provides
    # useful low-level filters and a strong CNN backbone.
    weights = ResNet50_Weights.IMAGENET1K_V2
    model = resnet50(weights=weights)

    # Freeze first 80% of parameters and fine-tune last 20%.
    # This reduces training cost and avoids overfitting.
    params = list(model.parameters())
    freeze_until = int(len(params) * 0.8)

    for param in params[:freeze_until]:
        param.requires_grad = False

    for param in params[freeze_until:]:
        param.requires_grad = True

    # Replace original classifier with a binary classifier.
    in_features = model.fc.in_features

    model.fc = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.ReLU(inplace=True),
        nn.BatchNorm1d(512),
        nn.Dropout(0.5),

        nn.Linear(512, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),

        nn.Linear(256, NUM_CLASSES)
    )

    return model.to(device)


# ============================================================
# Training helpers
# ============================================================

def get_lr(optimizer):
    return optimizer.param_groups[0]["lr"]


def run_one_epoch(model, loader, criterion, device, optimizer=None):
    # If optimizer is provided, this is training mode.
    # If optimizer is None, this is validation mode.
    is_train = optimizer is not None

    if is_train:
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    all_labels = []
    all_preds = []
    all_probs = []

    for images, labels in tqdm(loader):
        # Move images and labels to GPU/CPU device.
        images = images.to(device)
        labels = labels.to(device)

        # Clear old gradients before backpropagation.
        if is_train:
            optimizer.zero_grad()

        # Enable gradient calculation only during training.
        with torch.set_grad_enabled(is_train):
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backpropagation and weight update only happen in training mode.
            if is_train:
                loss.backward()
                optimizer.step()

        # Convert logits into probabilities.
        probs = torch.softmax(outputs, dim=1)

        # Choose the class with the highest probability.
        preds = torch.argmax(probs, dim=1)

        # Accumulate total loss.
        running_loss += loss.item() * images.size(0)

        # Store results for metric calculation.
        all_labels.extend(labels.detach().cpu().numpy())
        all_preds.extend(preds.detach().cpu().numpy())

        # Store probability of real class.
        # Since fake = 0 and real = 1, probs[:, 1] is real probability.
        all_probs.extend(probs[:, 1].detach().cpu().numpy())

    # Calculate average loss and accuracy.
    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)

    # Calculate AUC using probability scores, not only predicted labels.
    try:
        epoch_auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        epoch_auc = 0.0

    return epoch_loss, epoch_acc, epoch_auc


# ============================================================
# Plot functions
# ============================================================

def plot_training_history(history_df, output_dir):
    plt.figure(figsize=(8, 6))
    plt.plot(history_df["epoch"], history_df["train_acc"], label="Training Accuracy")
    plt.plot(history_df["epoch"], history_df["val_acc"], label="Validation Accuracy")
    plt.title("DFT-ResNet50 Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "accuracy_curve.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.plot(history_df["epoch"], history_df["train_loss"], label="Training Loss")
    plt.plot(history_df["epoch"], history_df["val_loss"], label="Validation Loss")
    plt.title("DFT-ResNet50 Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "loss_curve.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.plot(history_df["epoch"], history_df["train_auc"], label="Training ROC-AUC")
    plt.plot(history_df["epoch"], history_df["val_auc"], label="Validation ROC-AUC")
    plt.title("DFT-ResNet50 ROC-AUC")
    plt.xlabel("Epoch")
    plt.ylabel("ROC-AUC")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "auc_curve.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.plot(history_df["epoch"], history_df["train_ap"], label="Training AP")
    plt.plot(history_df["epoch"], history_df["val_ap"], label="Validation AP")
    plt.title("DFT-ResNet50 Average Precision")
    plt.xlabel("Epoch")
    plt.ylabel("Average Precision")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "average_precision_curve.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.plot(history_df["epoch"], history_df["lr"], label="Learning Rate")
    plt.title("Learning Rate Schedule")
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "learning_rate_curve.png"), dpi=300)
    plt.close()


def plot_confusion_matrix(cm, class_names, output_dir):
    plt.figure(figsize=(7, 6))
    plt.imshow(cm, interpolation="nearest")
    plt.title("DFT-ResNet50 Confusion Matrix")
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                format(cm[i, j], "d"),
                horizontalalignment="center",
                verticalalignment="center"
            )

    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=300)
    plt.close()


def plot_roc_curve(y_true, y_prob, output_dir):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_score = roc_auc_score(y_true, y_prob)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC Curve, AUC = {auc_score:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random Classifier")
    plt.title("DFT-ResNet50 ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "roc_curve.png"), dpi=300)
    plt.close()


def plot_precision_recall_curve(y_true, y_prob, output_dir):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap_score = average_precision_score(y_true, y_prob)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f"AP = {ap_score:.4f}")
    plt.title("DFT-ResNet50 Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "precision_recall_curve.png"), dpi=300)
    plt.close()


# ============================================================
# Test evaluation
# ============================================================

def evaluate_on_test(model, loader, class_names, device, output_dir):
    model.eval()

    all_labels = []
    all_preds = []
    all_probs = []

    inference_start = time.time()

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Testing"):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

    inference_time = time.time() - inference_start
    avg_time_per_image = inference_time / len(loader.dataset)

    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="weighted", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.0

    try:
        ap = average_precision_score(all_labels, all_probs)
    except ValueError:
        ap = 0.0

    cm = confusion_matrix(all_labels, all_preds)

    report_text = classification_report(
        all_labels,
        all_preds,
        target_names=class_names,
        digits=4,
        zero_division=0
    )

    report_dict = classification_report(
        all_labels,
        all_preds,
        target_names=class_names,
        digits=4,
        zero_division=0,
        output_dict=True
    )

    print("\n========== DFT-RESNET50 TEST RESULT ==========")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"ROC-AUC:   {auc:.4f}")
    print(f"AP:        {ap:.4f}")
    print(f"Total inference time: {inference_time:.2f} sec")
    print(f"Avg time per image:   {avg_time_per_image * 1000:.4f} ms")

    print("\nClassification Report:")
    print(report_text)

    print("Confusion Matrix:")
    print(cm)

    metrics_summary = {
        "model": "DFT-ResNet50",
        "accuracy": acc,
        "precision_weighted": precision,
        "recall_weighted": recall,
        "f1_weighted": f1,
        "roc_auc": auc,
        "average_precision": ap,
        "total_inference_time_sec": inference_time,
        "avg_time_per_image_ms": avg_time_per_image * 1000,
        "class_names": class_names,
    }

    with open(os.path.join(output_dir, "test_metrics.json"), "w") as f:
        json.dump(metrics_summary, f, indent=4)

    with open(os.path.join(output_dir, "classification_report.txt"), "w") as f:
        f.write(report_text)

    pd.DataFrame(report_dict).transpose().to_csv(
        os.path.join(output_dir, "classification_report.csv")
    )

    pred_df = pd.DataFrame({
        "true_label": all_labels,
        "pred_label": all_preds,
        "prob_real_class_1": all_probs,
        "true_class": [class_names[i] for i in all_labels],
        "pred_class": [class_names[i] for i in all_preds],
    })

    pred_df.to_csv(os.path.join(output_dir, "test_predictions.csv"), index=False)

    plot_confusion_matrix(cm, class_names, output_dir)
    plot_roc_curve(all_labels, all_probs, output_dir)
    plot_precision_recall_curve(all_labels, all_probs, output_dir)

    return metrics_summary


# ============================================================
# Main training function
# ============================================================

def main():
    set_seed(RANDOM_SEED)

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader = build_dataloaders(DATA_PATH)

    class_names = train_dataset.classes
    class_to_idx = train_dataset.class_to_idx

    print(f"[INFO] Classes: {class_names}")
    print(f"[INFO] Class mapping: {class_to_idx}")
    print(f"[INFO] Train images: {len(train_dataset)}")
    print(f"[INFO] Val images:   {len(val_dataset)}")
    print(f"[INFO] Test images:  {len(test_dataset)}")

    model = build_dft_resnet50_model(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=LR_FACTOR,
        patience=LR_PATIENCE,
        min_lr=MIN_LR
    )

    history = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "train_auc": [],
        "train_ap": [],
        "val_loss": [],
        "val_acc": [],
        "val_auc": [],
        "val_ap": [],
        "lr": [],
    }

    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_acc = 0.0
    best_epoch = 0
    early_stop_counter = 0

    best_model_path = os.path.join(OUTPUT_DIR, "best_dft_resnet50.pth")
    last_model_path = os.path.join(OUTPUT_DIR, "last_dft_resnet50.pth")

    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS}")
        print("-" * 50)

        train_loss, train_acc, train_auc, train_ap = run_one_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer
        )

        val_loss, val_acc, val_auc, val_ap = run_one_epoch(
            model,
            val_loader,
            criterion,
            device,
            optimizer=None
        )

        scheduler.step(val_loss)

        current_lr = get_lr(optimizer)

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["train_auc"].append(train_auc)
        history["train_ap"].append(train_ap)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_auc"].append(val_auc)
        history["val_ap"].append(val_ap)
        history["lr"].append(current_lr)

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Train AUC: {train_auc:.4f} | Train AP: {train_ap:.4f}")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f} | Val AUC:   {val_auc:.4f} | Val AP:   {val_ap:.4f}")
        print(f"LR: {current_lr:.8f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_model_wts = copy.deepcopy(model.state_dict())

            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "val_loss": val_loss,
                "val_auc": val_auc,
                "val_ap": val_ap,
                "class_to_idx": class_to_idx,
                "class_names": class_names,
            }, best_model_path)

            print(f"[INFO] Best model saved: {best_model_path}")
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            print(f"[INFO] Early stop counter: {early_stop_counter}/{PATIENCE}")

        if early_stop_counter >= PATIENCE:
            print("[INFO] Early stopping triggered.")
            break

    total_time = time.time() - start_time

    print(f"\n[INFO] Training completed in {total_time / 60:.2f} minutes.")
    print(f"[INFO] Best epoch: {best_epoch}")
    print(f"[INFO] Best val accuracy: {best_val_acc:.4f}")

    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "class_to_idx": class_to_idx,
        "class_names": class_names,
    }, last_model_path)

    model.load_state_dict(best_model_wts)

    history_df = pd.DataFrame(history)
    history_csv_path = os.path.join(OUTPUT_DIR, "training_history.csv")
    history_df.to_csv(history_csv_path, index=False)

    plot_training_history(history_df, OUTPUT_DIR)

    final_model_path = os.path.join(OUTPUT_DIR, "dft_resnet50_final_state_dict.pth")
    torch.save(model.state_dict(), final_model_path)

    evaluate_on_test(
        model=model,
        loader=test_loader,
        class_names=class_names,
        device=device,
        output_dir=OUTPUT_DIR
    )

    print("\n[DONE] Frequency-domain / DFT-ResNet50 training completed.")
    print(f"[DONE] Outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()