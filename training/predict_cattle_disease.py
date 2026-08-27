from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from torchvision.models import EfficientNet_B3_Weights, efficientnet_b3

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True)
    args = parser.parse_args()
    model_path = ROOT / 'models' / 'cattle_disease' / 'cattle_disease_efficientnet_b3.pth'
    classes = json.loads((ROOT / 'models' / 'cattle_disease' / 'class_names.json').read_text(encoding='utf-8'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(model_path, map_location=device)
    model = efficientnet_b3(weights=None)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, len(classes))
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device).eval()
    weights = EfficientNet_B3_Weights.DEFAULT
    image = weights.transforms()(Image.open(args.image).convert('RGB')).unsqueeze(0).to(device)
    with torch.no_grad():
        probabilities = torch.softmax(model(image), dim=1)[0].cpu().tolist()
    prediction = max(range(len(classes)), key=probabilities.__getitem__)
    print(json.dumps({'image': str(Path(args.image)), 'predicted_class': classes[prediction], 'confidence': probabilities[prediction], 'probabilities': dict(zip(classes, probabilities))}))


if __name__ == '__main__':
    main()