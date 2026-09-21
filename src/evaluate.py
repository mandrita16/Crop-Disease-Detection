import torch
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

from dataset import get_dataloaders
from train import build_model, DEVICE, MODEL_PATH


def evaluate():
    _, _, test_loader, class_names = get_dataloaders()

    model = build_model(len(class_names))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    print(classification_report(all_labels, all_preds, target_names=class_names))

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=False, cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix - Crop Disease Detection")
    plt.tight_layout()
    plt.savefig("../static/confusion_matrix.png")
    print("Confusion matrix saved to static/confusion_matrix.png")


if __name__ == "__main__":
    evaluate()
