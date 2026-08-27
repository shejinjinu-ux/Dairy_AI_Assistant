import argparse
import json
from pathlib import Path

import torch
import timm
from PIL import Image
from torchvision import transforms


ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT / "models" / "cattle_breed" / "Indian_bovine_finetuned_model.pth"
CLASS_PATH = ROOT / "models" / "cattle_breed" / "classes.json"

IMAGE_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_classes():
    with open(CLASS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def build_model(num_classes):
    model = timm.create_model(
        "convnext_tiny",
        pretrained=False,
        num_classes=num_classes,
        drop_path_rate=0.2,
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
    )

    state_dict = checkpoint["model_state_dict"]

    model.load_state_dict(
        state_dict,
        strict=True,
    )

    model = model.to(DEVICE)
    model.eval()

    return model


def predict(image_path):
    classes = load_classes()

    model = build_model(len(classes))

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)

    confidence, index = probabilities.max(dim=1)

    top_values, top_indices = probabilities.topk(
        min(5, len(classes)),
        dim=1,
    )

    top_predictions = []

    for value, idx in zip(top_values[0], top_indices[0]):
        top_predictions.append({
            "breed": classes[idx.item()],
            "confidence": float(value.item()),
        })

    return {
        "prediction": classes[index.item()],
        "confidence": float(confidence.item()),
        "top_5": top_predictions,
        "device": str(DEVICE),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image",
        required=True,
        help="Path to cow/buffalo image",
    )

    args = parser.parse_args()

    result = predict(args.image)

    print(json.dumps(result, indent=2))