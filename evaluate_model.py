import argparse
import json
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def get_model(arch, num_classes):
    if arch == "mobilenetv3_small":
        m = models.mobilenet_v3_small()
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, num_classes)
    elif arch == "efficientnet_b0":
        m = models.efficientnet_b0()
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    else:
        raise ValueError("unsupported arch")
    return m

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="artifacts/model_final.pt")
    ap.add_argument("--class_map", default="artifacts/class_map.json")
    ap.add_argument("--data_dir", required=True, help="Path to test data directory")
    ap.add_argument("--arch", default="efficientnet_b0", choices=["mobilenetv3_small", "efficientnet_b0"])
    ap.add_argument("--img_size", type=int, default=320)
    ap.add_argument("--batch_size", type=int, default=32)
    args = ap.parse_args()

    # Load class map
    with open(args.class_map, "r") as f:
        class_map = json.load(f)
    classes = [class_map["idx_to_class"][str(i)] for i in range(len(class_map["idx_to_class"]))]

    # Data transforms
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    test_tf = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        normalize
    ])

    # Load test dataset
    test_ds = datasets.ImageFolder(args.data_dir, transform=test_tf)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(args.arch, len(classes))
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()

    # Evaluate
    preds, gts = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            p = logits.argmax(1)
            preds.extend(p.cpu().tolist())
            gts.extend(y.cpu().tolist())

    # Metrics
    acc = accuracy_score(gts, preds)
    print(f"Test Accuracy: {acc:.4f}")

    # Classification report
    report = classification_report(gts, preds, target_names=classes)
    print("\nClassification Report:")
    print(report)

    # Confusion matrix
    cm = confusion_matrix(gts, preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('artifacts/confusion_matrix.png')
    plt.show()

if __name__ == "__main__":
    main()