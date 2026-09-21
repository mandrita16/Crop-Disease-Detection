import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from tqdm import tqdm
import json
import os

from dataset import get_dataloaders

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 15
LR = 1e-3
MODEL_PATH = "../models/best_model.pth"


def build_model(num_classes):
    weights = EfficientNet_B0_Weights.DEFAULT
    model = efficientnet_b0(weights=weights)

    # Freeze backbone initially
    for param in model.features.parameters():
        param.requires_grad = False

    # Replace classifier head
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    return model.to(DEVICE)


def train():
    train_loader, val_loader, test_loader, class_names = get_dataloaders()
    num_classes = len(class_names)
    print(f"Classes ({num_classes}): {class_names}")

    model = build_model(num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=LR)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    best_val_loss = float("inf")

    for epoch in range(EPOCHS):
        # unfreeze backbone after a few epochs of head-only training
        if epoch == 3:
            for param in model.features.parameters():
                param.requires_grad = True
            optimizer = Adam(model.parameters(), lr=LR / 10)

        model.train()
        train_loss, correct, total = 0.0, 0, 0

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [train]")
        for images, labels in loop:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            loop.set_postfix(loss=loss.item())

        train_loss /= total
        train_acc = correct / total

        # validation
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total
        scheduler.step(val_loss)

        print(f"Epoch {epoch+1}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs("../models", exist_ok=True)
            torch.save(model.state_dict(), MODEL_PATH)
            with open("../models/class_names.json", "w") as f:
                json.dump(class_names, f)
            print(f"  -> saved new best model (val_loss={val_loss:.4f})")

    print("Training complete. Best model saved to", MODEL_PATH)


if __name__ == "__main__":
    train()
