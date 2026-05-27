import os
import json
import copy
import time
import random
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import transforms
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


# ============================================================
# Configuration
# ============================================================

RGB_ROOT = r"C:\Users\kcwong6\Downloads\DeepFakeFace_binary"
DFT_ROOT = r"C:\Users\kcwong6\Downloads\DeepFakeFace_binary\DeepFakeFace_fft"

OUTPUT_DIR = r"C:\Users\kcwong6\Downloads\DeepFakeFace_binary\SF1\spatial_frequency_fusion_report"

IMG_SIZE = 256
BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
RANDOM_SEED = 42

PATIENCE = 10
LR_PATIENCE = 5
LR_FACTOR = 0.2
MIN_LR = 1e-6

NUM_CLASSES = 2
NUM_WORKERS = 0  # safer for Windows

CLASS_NAMES = ["fake", "real"]
CLASS_TO_IDX = {"fake": 0, "real": 1}

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)


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
# Dataset
# ============================================================

class PairedRGBDFTDataset(torch.utils.data.Dataset):
    def __init__(self, rgb_root, dft_root, split, transform_rgb=None, transform_dft=None):
        self.rgb_root = Path(rgb_root) / split
        self.dft_root = Path(dft_root) / split
        self.transform_rgb = transform_rgb
        self.transform_dft = transform_dft
        self.samples = []

        for class_name in CLASS_NAMES:
            rgb_class_dir = self.rgb_root / class_name
            dft_class_dir = self.dft_root / class_name
            label = CLASS_TO_IDX[class_name]

            rgb_paths = []
            for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]:
                rgb_paths.extend(rgb_class_dir.glob(ext))

            rgb_paths = sorted(rgb_paths)

            for rgb_path in rgb_paths:
                stem = rgb_path.stem

                possible_dft_paths = [
                    dft_class_dir / f"{stem}.png",
                    dft_class_dir / f"{stem}.jpg",
                    dft_class_dir / f"{stem}.jpeg",
                    dft_class_dir / f"{stem}.bmp",
                    dft_class_dir / f"{stem}.webp",
                ]

                dft_path = None
                for p in possible_dft_paths:
                    if p.exists():
                        dft_path = p
                        break

                if dft_path is not None:
                    self.samples.append((rgb_path, dft_path, label))
                else:
                    print(f"[WARNING] Missing DFT pair for: {rgb_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rgb_path, dft_path, label = self.samples[idx]

        rgb_img = Image.open(rgb_path).convert("RGB")
        dft_img = Image.open(dft_path).convert("RGB")

        if self.transform_rgb:
            rgb_img = self.transform_rgb(rgb_img)

        if self.transform_dft:
            dft_img = self.transform_dft(dft_img)

        return rgb_img, dft_img, label


# ============================================================
# Transforms
# ============================================================

def build_transforms():
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

    return train_transform, eval_transform


def build_dataloaders():
    train_transform, eval_transform = build_transforms()

    train_dataset = PairedRGBDFTDataset(
        RGB_ROOT,
        DFT_ROOT,
        split="train",
        transform_rgb=train_transform,
        transform_dft=train_transform
    )

    val_dataset = PairedRGBDFTDataset(
        RGB_ROOT,
        DFT_ROOT,
        split="val",
        transform_rgb=eval_transform,
        transform_dft=eval_transform
    )

    test_dataset = PairedRGBDFTDataset(
        RGB_ROOT,
        DFT_ROOT,
        split="test",
        transform_rgb=eval_transform,
        transform_dft=eval_transform
    )

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
# Dual-branch spatial-frequency fusion model
# ============================================================

class SpatialFrequencyFusionResNet50(nn.Module):
    def __init__(self, num_classes=2, freeze_backbone=True):
        super().__init__()

        weights = ResNet50_Weights.IMAGENET1K_V2

        self.spatial_branch = resnet50(weights=weights)
        self.frequency_branch = resnet50(weights=weights)

        spatial_in_features = self.spatial_branch.fc.in_features
        frequency_in_features = self.frequency_branch.fc.in_features

        # Remove original ResNet classifiers
        self.spatial_branch.fc = nn.Identity()
        self.frequency_branch.fc = nn.Identity()

        if freeze_backbone:
            for param in self.spatial_branch.parameters():
                param.requires_grad = False
            for param in self.frequency_branch.parameters():
                param.requires_grad = False

            # Fine-tune only layer4 for both branches
            for param in self.spatial_branch.layer4.parameters():
                param.requires_grad = True
            for param in self.frequency_branch.layer4.parameters():
                param.requires_grad = True

        fused_dim = spatial_in_features + frequency_in_features

        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),

            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),

            nn.Linear(256, num_classes)
        )

    def forward(self, rgb_img, dft_img):
        spatial_features = self.spatial_branch(rgb_img)
        frequency_features = self.frequency_branch(dft_img)

        fused_features = torch.cat(
            [spatial_features, frequency_features],
            dim=1
        )

        logits = self.classifier(fused_features)
        return logits


# ============================================================
# Training helpers
# ============================================================

def get_lr(optimizer):
    return optimizer.param_groups[0]["lr"]


def run_one_epoch(model, loader, criterion, device, optimizer=None):
    is_train = optimizer is not None

    model.train() if is_train else model.eval()

    running_loss = 0.0
    all_labels = []
    all_preds = []
    all_probs = []

    loop = tqdm(loader, desc="Train" if is_train else "Val", leave=False)

    for rgb_imgs, dft_imgs, labels in loop:
        rgb_imgs = rgb_imgs.to(device)
        dft_imgs = dft_imgs.to(device)
        labels = labels.to(device)

        if is_train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            outputs = model(rgb_imgs, dft_imgs)
            loss = criterion(outputs, labels)

            if is_train:
                loss.backward()
                optimizer.step()

        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)

        running_loss += loss.item() * rgb_imgs.size(0)

        all_labels.extend(labels.detach().cpu().numpy())
        all_preds.extend(preds.detach().cpu().numpy())
        all_probs.extend(probs[:, 1].detach().cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)

    try:
        epoch_auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        epoch_auc = 0.0

    try:
        epoch_ap = average_precision_score(all_labels, all_probs)
    except ValueError:
        epoch_ap = 0.0

    return epoch_loss, epoch_acc, epoch_auc, epoch_ap


# ============================================================
# Plot functions
# ============================================================

def plot_training_history(history_df, output_dir):
    plt.figure(figsize=(8, 6))
    plt.plot(history_df["epoch"], history_df["train_acc"], label="Training Accuracy")
    plt.plot(history_df["epoch"], history_df["val_acc"], label="Validation Accuracy")
    plt.title("Spatial-Frequency Fusion Accuracy")
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
    plt.title("Spatial-Frequency Fusion Loss")
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
    plt.title("Spatial-Frequency Fusion ROC-AUC")
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
    plt.title("Spatial-Frequency Fusion Average Precision")
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


def plot_confusion_matrix(cm, output_dir):
    plt.figure(figsize=(7, 6))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Spatial-Frequency Fusion Confusion Matrix")
    plt.colorbar()

    tick_marks = np.arange(len(CLASS_NAMES))
    plt.xticks(tick_marks, CLASS_NAMES)
    plt.yticks(tick_marks, CLASS_NAMES)

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
    plt.title("Spatial-Frequency Fusion ROC Curve")
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
    plt.title("Spatial-Frequency Fusion Precision-Recall Curve")
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

def evaluate_on_test(model, loader, device, output_dir):
    model.eval()

    all_labels = []
    all_preds = []
    all_probs = []

    inference_start = time.time()

    with torch.no_grad():
        for rgb_imgs, dft_imgs, labels in tqdm(loader, desc="Testing"):
            rgb_imgs = rgb_imgs.to(device)
            dft_imgs = dft_imgs.to(device)
            labels = labels.to(device)

            outputs = model(rgb_imgs, dft_imgs)
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
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0
    )

    print("\n========== SPATIAL-FREQUENCY FUSION TEST RESULT ==========")
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
        "model": "Spatial-Frequency Fusion ResNet50",
        "accuracy": acc,
        "precision_weighted": precision,
        "recall_weighted": recall,
        "f1_weighted": f1,
        "roc_auc": auc,
        "average_precision": ap,
        "total_inference_time_sec": inference_time,
        "avg_time_per_image_ms": avg_time_per_image * 1000,
        "class_names": CLASS_NAMES,
    }

    with open(os.path.join(output_dir, "test_metrics.json"), "w") as f:
        json.dump(metrics_summary, f, indent=4)

    with open(os.path.join(output_dir, "classification_report.txt"), "w") as f:
        f.write(report_text)

    pd.DataFrame({
        "true_label": all_labels,
        "pred_label": all_preds,
        "prob_real_class_1": all_probs,
        "true_class": [CLASS_NAMES[i] for i in all_labels],
        "pred_class": [CLASS_NAMES[i] for i in all_preds],
    }).to_csv(os.path.join(output_dir, "test_predictions.csv"), index=False)

    pd.DataFrame(
        classification_report(
            all_labels,
            all_preds,
            target_names=CLASS_NAMES,
            digits=4,
            zero_division=0,
            output_dict=True
        )
    ).transpose().to_csv(os.path.join(output_dir, "classification_report.csv"))

    plot_confusion_matrix(cm, output_dir)
    plot_roc_curve(all_labels, all_probs, output_dir)
    plot_precision_recall_curve(all_labels, all_probs, output_dir)

    return metrics_summary


# ============================================================
# Main
# ============================================================

def main():
    set_seed(RANDOM_SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader = build_dataloaders()

    print(f"[INFO] Train pairs: {len(train_dataset)}")
    print(f"[INFO] Val pairs:   {len(val_dataset)}")
    print(f"[INFO] Test pairs:  {len(test_dataset)}")

    model = SpatialFrequencyFusionResNet50(
        num_classes=NUM_CLASSES,
        freeze_backbone=True
    ).to(device)

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

    best_model_path = os.path.join(OUTPUT_DIR, "best_spatial_frequency_fusion.pth")
    last_model_path = os.path.join(OUTPUT_DIR, "last_spatial_frequency_fusion.pth")

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
                "val_acc": val_acc,
                "val_loss": val_loss,
                "val_auc": val_auc,
                "val_ap": val_ap,
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
    }, last_model_path)

    model.load_state_dict(best_model_wts)

    history_df = pd.DataFrame(history)
    history_df.to_csv(os.path.join(OUTPUT_DIR, "training_history.csv"), index=False)

    plot_training_history(history_df, OUTPUT_DIR)

    torch.save(
        model.state_dict(),
        os.path.join(OUTPUT_DIR, "spatial_frequency_fusion_final_state_dict.pth")
    )

    evaluate_on_test(model, test_loader, device, OUTPUT_DIR)

    print("\n[DONE] Spatial-frequency fusion training completed.")
    print(f"[DONE] Outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()