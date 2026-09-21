import pandas as pd
import numpy as np
import torch
import xgboost as xgb
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from PIL import Image
from torchvision import transforms
import json
import random
import os

from train import build_model, DEVICE, MODEL_PATH
from fusion import FusionModel

IMG_SIZE = 224


def get_image_probs_for_class(model, class_names, target_class, sample_dir, n_samples=200):
    """
    Runs the trained image model on real sample images from each class
    to get realistic probability distributions (not synthetic guesses).
    """
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    class_dir = os.path.join(sample_dir, target_class)
    image_files = [f for f in os.listdir(class_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    sampled_files = random.sample(image_files, min(n_samples, len(image_files)))

    probs_list = []
    with torch.no_grad():
        for fname in sampled_files:
            img = Image.open(os.path.join(class_dir, fname)).convert("RGB")
            tensor = transform(img).unsqueeze(0).to(DEVICE)
            outputs = model(tensor)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
            probs_list.append(probs)

    return np.array(probs_list)


def train_fusion():
    with open("../models/class_names.json") as f:
        class_names = json.load(f)

    image_model = build_model(len(class_names))
    image_model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    image_model.eval()

    tabular_model = xgb.XGBClassifier()
    tabular_model.load_model("../models/tabular_xgb_model.json")
    tabular_scaler = joblib.load("../models/tabular_scaler.pkl")

    paired_df = pd.read_csv("../data/tabular/paired_fusion_dataset.csv")

    all_image_probs = []
    all_tabular_probs = []
    all_labels = []

    fusion_le = LabelEncoder()
    fusion_le.fit(paired_df["true_final_label"])

    for disease_class in paired_df["disease_class"].unique():
        subset = paired_df[paired_df["disease_class"] == disease_class]
        n = len(subset)

        img_probs = get_image_probs_for_class(
            image_model, class_names, disease_class,
            sample_dir="../data/plantvillage", n_samples=n
        )
        # align lengths (image sampling with replacement if short)
        if len(img_probs) < n:
            idx = np.random.choice(len(img_probs), n, replace=True)
            img_probs = img_probs[idx]

        tab_features = subset[["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]].values
        tab_scaled = tabular_scaler.transform(tab_features)
        tab_probs = tabular_model.predict_proba(tab_scaled)

        all_image_probs.append(img_probs)
        all_tabular_probs.append(tab_probs)
        all_labels.extend(subset["true_final_label"].values)

    X_image = np.vstack(all_image_probs)
    X_tabular = np.vstack(all_tabular_probs)
    y = fusion_le.transform(all_labels)

    fusion_model = FusionModel()
    fusion_model.fit(X_image, X_tabular, y)
    fusion_model.save("../models/fusion_meta_model.pkl")
    joblib.dump(fusion_le, "../models/fusion_label_encoder.pkl")

    preds = fusion_model.meta_model.predict(np.hstack([X_image, X_tabular]))
    print(classification_report(y, preds, target_names=fusion_le.classes_))
    print("Fusion model saved.")


if __name__ == "__main__":
    train_fusion()
