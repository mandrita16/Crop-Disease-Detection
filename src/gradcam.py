import torch
import json
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from train import build_model, DEVICE, MODEL_PATH

IMG_SIZE = 224


def load_model_and_classes():
    with open("../models/class_names.json") as f:
        class_names = json.load(f)
    model = build_model(len(class_names))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model, class_names


def predict_with_gradcam(image_path, output_path="../static/gradcam_output.jpg"):
    model, class_names = load_model_and_classes()

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225]),
    ])

    pil_img = Image.open(image_path).convert("RGB")
    input_tensor = transform(pil_img).unsqueeze(0).to(DEVICE)

    # Prediction
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        conf, pred_idx = torch.max(probs, 1)
        predicted_class = class_names[pred_idx.item()]
        confidence = conf.item()
        all_probs = probs.cpu().numpy()[0].tolist()

    # Grad-CAM - target the last conv block of EfficientNet-B0
    target_layers = [model.features[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)
    grayscale_cam = cam(input_tensor=input_tensor)[0]

    rgb_img = np.array(pil_img.resize((IMG_SIZE, IMG_SIZE))) / 255.0
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    cv2.imwrite(output_path, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))

    return {
        "predicted_class": predicted_class,
        "confidence": round(confidence, 4),
        "all_probs": all_probs,
        "class_names": class_names,
        "gradcam_path": output_path,
    }


if __name__ == "__main__":
    result = predict_with_gradcam("../data/plantvillage/Tomato___Late_blight/sample.JPG")
    print(result)
