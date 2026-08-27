from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import EfficientNet_B3_Weights, efficientnet_b3
from torchvision.transforms import v2

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / 'datasets' / 'extracted' / 'cattle_disease'
OUT = ROOT / 'models' / 'cattle_disease'
RESULTS = ROOT / 'training' / 'results'
CLASSES = ['FMD', 'IBK', 'LSD', 'Normal']
SEED = 42
IMAGE_SIZE = 300


def seed_everything():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def image_files():
    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tif', '.tiff', '.webp'}
    records = []
    skipped = []
    for label, class_name in enumerate(CLASSES):
        for path in sorted((DATASET / class_name).rglob('*')):
            if not path.is_file() or path.suffix.lower() not in extensions:
                continue
            try:
                with Image.open(path) as image:
                    image.verify()
                records.append((path, label))
            except (OSError, UnidentifiedImageError, ValueError) as exc:
                skipped.append({'path': path.relative_to(ROOT).as_posix(), 'error': f'{type(exc).__name__}: {exc}'})
    return records, skipped


class DiseaseDataset(Dataset):
    def __init__(self, records, transform):
        self.records = records
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        path, label = self.records[index]
        try:
            image = Image.open(path).convert('RGB')
            return self.transform(image), label, path.name
        except (OSError, UnidentifiedImageError, ValueError):
            return None


def collate(batch):
    valid = [item for item in batch if item is not None]
    if not valid:
        return None
    images, labels, names = zip(*valid)
    return torch.stack(images), torch.tensor(labels), names


def run_epoch(model, loader, criterion, optimizer, scaler, device, training):
    model.train(training)
    total_loss = 0.0
    actual, predicted = [], []
    for batch in loader:
        if batch is None:
            continue
        images, labels, _ = batch
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', enabled=True):
            output = model(images)
            loss = criterion(output, labels)
        if training:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        total_loss += loss.item() * labels.size(0)
        actual.extend(labels.cpu().tolist())
        predicted.extend(output.argmax(1).detach().cpu().tolist())
    return total_loss / max(1, len(actual)), accuracy_score(actual, predicted), actual, predicted


def evaluate(model, loader, device):
    model.eval()
    actual, predicted, names, probabilities = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue
            images, labels, batch_names = batch
            with torch.amp.autocast('cuda'):
                output = model(images.to(device, non_blocking=True))
            actual.extend(labels.tolist())
            predicted.extend(output.argmax(1).cpu().tolist())
            names.extend(batch_names)
            probabilities.extend(torch.softmax(output, 1).cpu().tolist())
    return actual, predicted, names, probabilities


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--stage1-epochs', type=int, default=3)
    parser.add_argument('--stage2-epochs', type=int, default=8)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit('CUDA is unavailable; refusing CPU training.')
    seed_everything()
    device = torch.device('cuda')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    records, skipped = image_files()
    labels = np.array([label for _, label in records])
    train_records, holdout_records = train_test_split(records, test_size=0.30, random_state=SEED, stratify=labels)
    holdout_labels = np.array([label for _, label in holdout_records])
    val_records, test_records = train_test_split(holdout_records, test_size=0.50, random_state=SEED, stratify=holdout_labels)
    weights = EfficientNet_B3_Weights.DEFAULT
    train_transform = v2.Compose([v2.RandomResizedCrop((IMAGE_SIZE, IMAGE_SIZE), antialias=True), v2.RandomHorizontalFlip(), v2.RandomRotation(8), v2.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15), v2.ToImage(), v2.ToDtype(torch.float32, scale=True), v2.Normalize(weights.transforms().mean, weights.transforms().std)])
    eval_transform = v2.Compose([v2.Resize(320, antialias=True), v2.CenterCrop(IMAGE_SIZE), v2.ToImage(), v2.ToDtype(torch.float32, scale=True), v2.Normalize(weights.transforms().mean, weights.transforms().std)])
    loaders = {name: DataLoader(DiseaseDataset(data, train_transform if name == 'train' else eval_transform), batch_size=args.batch_size, shuffle=name == 'train', num_workers=0, pin_memory=True, collate_fn=collate) for name, data in [('train', train_records), ('val', val_records), ('test', test_records)]}
    counts = np.bincount([label for _, label in train_records], minlength=len(CLASSES))
    class_weights = torch.tensor(len(train_records) / (len(CLASSES) * counts), dtype=torch.float32, device=device)
    model = efficientnet_b3(weights=weights)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(CLASSES))
    model.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    scaler = torch.amp.GradScaler('cuda')
    best_state, best_val, patience = None, float('inf'), 0
    history = []
    for parameter in model.features.parameters():
        parameter.requires_grad = False
    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=1)
    for stage, epochs in [('head', args.stage1_epochs)]:
        for epoch in range(epochs):
            train_loss, train_acc, _, _ = run_epoch(model, loaders['train'], criterion, optimizer, scaler, device, True)
            val_loss, val_acc, _, _ = run_epoch(model, loaders['val'], criterion, optimizer, scaler, device, False)
            scheduler.step(val_loss)
            history.append({'stage': stage, 'epoch': epoch + 1, 'train_loss': train_loss, 'train_accuracy': train_acc, 'val_loss': val_loss, 'val_accuracy': val_acc})
            print(history[-1])
            if val_loss < best_val:
                best_val, patience, best_state = val_loss, 0, copy.deepcopy(model.state_dict())
            else:
                patience += 1
            if patience >= 3:
                break
    for block in list(model.features.children())[-4:]:
        for parameter in block.parameters():
            parameter.requires_grad = True
    optimizer = torch.optim.AdamW(filter(lambda parameter: parameter.requires_grad, model.parameters()), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=1)
    patience = 0
    for epoch in range(args.stage2_epochs):
        train_loss, train_acc, _, _ = run_epoch(model, loaders['train'], criterion, optimizer, scaler, device, True)
        val_loss, val_acc, _, _ = run_epoch(model, loaders['val'], criterion, optimizer, scaler, device, False)
        scheduler.step(val_loss)
        history.append({'stage': 'fine_tune', 'epoch': epoch + 1, 'train_loss': train_loss, 'train_accuracy': train_acc, 'val_loss': val_loss, 'val_accuracy': val_acc})
        print(history[-1])
        if val_loss < best_val:
            best_val, patience, best_state = val_loss, 0, copy.deepcopy(model.state_dict())
        else:
            patience += 1
        if patience >= 3:
            break
    model.load_state_dict(best_state)
    OUT.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    model_path = OUT / 'cattle_disease_efficientnet_b3.pth'
    torch.save({'architecture': 'efficientnet_b3', 'state_dict': model.state_dict(), 'classes': CLASSES}, model_path)
    (OUT / 'class_names.json').write_text(json.dumps(CLASSES, indent=2) + '\n', encoding='utf-8')
    split_path = RESULTS / 'cattle_disease_split.json'
    split_path.write_text(json.dumps({'seed': SEED, 'train': [p.relative_to(ROOT).as_posix() for p, _ in train_records], 'validation': [p.relative_to(ROOT).as_posix() for p, _ in val_records], 'test': [p.relative_to(ROOT).as_posix() for p, _ in test_records], 'skipped': skipped}, indent=2) + '\n', encoding='utf-8')
    metadata = {'architecture': 'efficientnet_b3', 'pytorch_version': torch.__version__, 'torchvision_version': __import__('torchvision').__version__, 'classes': CLASSES, 'image_size': IMAGE_SIZE, 'normalization': {'mean': list(weights.transforms().mean), 'std': list(weights.transforms().std)}, 'seed': SEED, 'train_images': len(train_records), 'validation_images': len(val_records), 'test_images': len(test_records), 'skipped_corrupted_images': skipped, 'history': history}
    (OUT / 'model_metadata.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
    actual, predicted, names, probabilities = evaluate(model, loaders['test'], device)
    cm = confusion_matrix(actual, predicted, labels=list(range(len(CLASSES))))
    report = classification_report(actual, predicted, labels=list(range(len(CLASSES))), target_names=CLASSES, output_dict=True, zero_division=0)
    fig, axis = plt.subplots(figsize=(7, 6)); axis.imshow(cm, cmap='Blues'); axis.set(xticks=range(4), yticks=range(4), xticklabels=CLASSES, yticklabels=CLASSES, xlabel='Predicted', ylabel='Actual', title='Cattle Disease Confusion Matrix');
    for row in range(4):
        for col in range(4): axis.text(col, row, cm[row, col], ha='center', va='center')
    fig.tight_layout(); fig.savefig(RESULTS / 'cattle_disease_confusion_matrix.png', dpi=150); plt.close(fig)
    metrics = {'accuracy': accuracy_score(actual, predicted), 'precision_macro': precision_score(actual, predicted, average='macro', zero_division=0), 'recall_macro': recall_score(actual, predicted, average='macro', zero_division=0), 'f1_macro': f1_score(actual, predicted, average='macro', zero_division=0), 'f1_weighted': f1_score(actual, predicted, average='weighted', zero_division=0)}
    lines = ['# Cattle Disease Image Model Report', '', f'- Model: `{model_path.relative_to(ROOT).as_posix()}`', f'- GPU: `{torch.cuda.get_device_name(0)}`', f'- Split: 70% train ({len(train_records)}), 15% validation ({len(val_records)}), 15% test ({len(test_records)})', f'- Skipped unreadable images: {len(skipped)}', '', '## Metrics', ''] + [f'- {key}: {value:.6f}' for key, value in metrics.items()] + ['', '## Classification Report', '', '```text', classification_report(actual, predicted, target_names=CLASSES, zero_division=0), '```', '## Confusion Matrix', '', '| Actual / Predicted | ' + ' | '.join(CLASSES) + ' |', '|---|' + '---|' * 4]
    lines += [f'| {CLASSES[row]} | ' + ' | '.join(str(value) for value in cm[row]) + ' |' for row in range(4)]
    lines += ['', '## Prediction Smoke Test', '']
    for name, prediction, probability in list(zip(names, predicted, probabilities))[:10]: lines.append(f'- `{name}` -> **{CLASSES[prediction]}** ({max(probability):.6f})')
    (RESULTS / 'cattle_disease_report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps({'model': str(model_path), 'metrics': metrics, 'smoke_test_predictions': min(10, len(names))}, indent=2))


if __name__ == '__main__':
    main()