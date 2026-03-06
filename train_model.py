#!/usr/bin/env python3
"""
Train a cloud detection model and export to ONNX.

Uses PyTorch with MobileNetV3-Small (pretrained on ImageNet) fine-tuned
for binary classification (Clear/Cloudy). Exports directly to ONNX format
compatible with the mlCloudDetect detector.

Requirements (install on training machine):
    pip install torch torchvision onnx onnxscript

Usage:
    python train_model.py                          # Train from ./images/
    python train_model.py --data ./sorted_images   # Custom data directory
    python train_model.py --epochs 25 --lr 0.0005  # Custom hyperparameters

Data directory should be in ImageFolder format:
    images/
    +-- clear/
    |   +-- img001.jpg
    |   +-- img002.jpg
    +-- cloudy/
        +-- img001.jpg
        +-- img002.jpg

Use sort_images.py to create this structure from training capture data.
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms


class ModelWithSoftmax(nn.Module):
    """Wraps a model to apply softmax on output for ONNX export."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.model(x), dim=1)


def train(
    data_dir: str = "./images",
    model_out: str = "model.onnx",
    labels_out: str = "labels.txt",
    img_size: int = 224,
    batch_size: int = 32,
    epochs: int = 15,
    lr: float = 0.001,
    patience: int = 5,
    freeze: bool = True,
) -> None:
    """Train the model and export to ONNX.

    Args:
        data_dir: Path to ImageFolder dataset.
        model_out: Output path for ONNX model.
        labels_out: Output path for labels file.
        img_size: Input image size (square).
        batch_size: Training batch size.
        epochs: Maximum training epochs.
        lr: Initial learning rate.
        patience: Early stopping patience (epochs without improvement).
        freeze: Freeze backbone, only train classifier (recommended for <1000 images).
    """
    # --- Transforms ---
    # Normalize to [-1, 1] to match detector preprocessing:
    #   detector does: (pixel / 127.5) - 1.0
    #   ToTensor maps [0,255] -> [0,1], then Normalize([0.5]*3, [0.5]*3)
    #   does: (x - 0.5) / 0.5 = 2x - 1, mapping [0,1] -> [-1,1]
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    # --- Dataset ---
    dataset = datasets.ImageFolder(data_dir, transform=train_transform)
    class_names = dataset.classes
    num_classes = len(class_names)

    print(f"Dataset: {len(dataset)} images, {num_classes} classes: {class_names}")

    if num_classes < 2:
        print("Error: Need at least 2 classes. Check your data directory structure.")
        sys.exit(1)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    # Apply validation transform (no augmentation) to validation set
    val_ds.dataset = datasets.ImageFolder(data_dir, transform=val_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    print(f"Split: {train_size} train, {val_size} validation")

    # --- Model ---
    model = models.mobilenet_v3_small(weights="IMAGENET1K_V1")
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)

    # Freeze backbone to prevent overfitting on small datasets
    if freeze:
        for param in model.features.parameters():
            param.requires_grad = False
        trainable = model.classifier.parameters()
        print("Backbone frozen, training classifier only")
    else:
        trainable = model.parameters()
        print("Training all layers")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Device: {device}")

    # --- Training ---
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(trainable, lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    best_acc = 0.0
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(epochs):
        # Train
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            train_correct += (out.argmax(dim=1) == labels).sum().item()
            train_total += labels.size(0)

        # Validate
        model.eval()
        val_correct = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                preds = model(imgs).argmax(dim=1)
                val_correct += (preds == labels).sum().item()

        scheduler.step()

        train_acc = train_correct / train_total * 100
        val_acc = val_correct / val_size * 100
        avg_loss = running_loss / len(train_loader)

        print(
            f"Epoch {epoch + 1:2d}/{epochs} | "
            f"Loss: {avg_loss:.4f} | "
            f"Train: {train_acc:.1f}% | "
            f"Val: {val_acc:.1f}%",
            end="",
        )

        # Track best model
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = model.state_dict().copy()
            epochs_without_improvement = 0
            print(" *best*")
        else:
            epochs_without_improvement += 1
            print()

        # Early stopping
        if epochs_without_improvement >= patience:
            print(f"Early stopping after {patience} epochs without improvement")
            break

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"\nBest validation accuracy: {best_acc:.1f}%")

    # --- Export to ONNX ---
    # Wrap with softmax so ONNX model outputs probabilities (matching
    # what the detector expects for confidence values).
    export_model = ModelWithSoftmax(model)
    export_model.eval()
    export_model = export_model.to(device)

    dummy = torch.randn(1, 3, img_size, img_size).to(device)
    torch.onnx.export(
        export_model,
        dummy,
        model_out,
        input_names=["input"],
        output_names=["output"],
    )
    print(f"Saved ONNX model: {model_out}")

    # --- Generate labels.txt ---
    # Format matches what detector._load_labels() expects: "0 Clear\n1 Cloudy\n"
    with open(labels_out, "w") as f:
        for idx, name in enumerate(class_names):
            # Capitalize class names (ImageFolder uses directory names which
            # are typically lowercase)
            f.write(f"{idx} {name.capitalize()}\n")

    print(f"Saved labels: {labels_out}")
    print(f"Class mapping: { {name: idx for idx, name in enumerate(class_names)} }")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train cloud detection model and export to ONNX",
    )
    parser.add_argument(
        "--data", default="./images",
        help="Path to ImageFolder dataset (default: ./images)",
    )
    parser.add_argument(
        "--output", default="model.onnx",
        help="Output ONNX model path (default: model.onnx)",
    )
    parser.add_argument(
        "--labels", default="labels.txt",
        help="Output labels file path (default: labels.txt)",
    )
    parser.add_argument(
        "--epochs", type=int, default=15,
        help="Max training epochs (default: 15)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Batch size (default: 32)",
    )
    parser.add_argument(
        "--lr", type=float, default=0.001,
        help="Learning rate (default: 0.001)",
    )
    parser.add_argument(
        "--patience", type=int, default=5,
        help="Early stopping patience (default: 5)",
    )
    parser.add_argument(
        "--unfreeze", action="store_true",
        help="Train all layers instead of just classifier (use with >1000 images)",
    )
    args = parser.parse_args()

    train(
        data_dir=args.data,
        model_out=args.output,
        labels_out=args.labels,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        freeze=not args.unfreeze,
    )


if __name__ == "__main__":
    main()
