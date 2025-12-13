import argparse, json, os, random, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from PIL import Image, ImageFile
from sklearn.metrics import confusion_matrix, classification_report
import os
import shutil
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

def set_seed(s=42):
    """Sets seeds for reproducibility."""
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def get_model(arch, num_classes):
    """Initializes a pre-trained model and modifies the classification head."""
    if arch == "mobilenetv3_small":
        m = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        # Assuming the final layer is at index 3 for the small version
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, num_classes)
    elif arch == "efficientnet_b0":
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        # Assuming the final layer is at index 1 for EfficientNet's classifier
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    else:
        raise ValueError(f"unsupported arch: {arch}")
    return m

# --- Custom Dataset Class with Error Handling ---
# We wrap the ImageFolder dataset to catch the PIL/Pillow loading error 
# and safely skip corrupted files during training/validation.
class SafeImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        try:
            return super().__getitem__(index)
        except OSError as e:
            # Catch file corruption errors (like truncation)
            print(f"\n[Data Load Error] Skipping file index {index} due to: {e}")
            # Try to get the next valid image instead of crashing
            # This is a simple fix; production code might be more sophisticated
            # Find a valid index that is not the current one
            new_index = (index + 1) % len(self)
            return self.__getitem__(new_index) 

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="Path to the root data directory (containing species folders).")
    ap.add_argument("--out_dir", default="artifacts", help="Directory to save model checkpoints and artifacts.")
    ap.add_argument("--arch", default="efficientnet_b0", choices=["mobilenetv3_small","efficientnet_b0"], help="Model architecture to use.")
    ap.add_argument("--batch_size", type=int, default=64, help="Batch size for training.")
    ap.add_argument("--epochs", type=int, default=10, help="Total epochs for head training.")
    ap.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    ap.add_argument("--img_size", type=int, default=320, help="Input image size (width and height).")
    # Increased num_workers for faster local data loading
    ap.add_argument("--num_workers", type=int, default=8, help="Number of data loading processes. Use 0 for debugging.")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed()

    # --- DEVICE SETUP: Auto-detect GPU ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # --- Data Augmentation and Normalization ---
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_tf = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(args.img_size, scale=(0.8,1.0)),
        transforms.ColorJitter(0.1,0.1,0.1,0.05),
        transforms.ToTensor(),
        normalize
    ])
    val_tf = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        normalize
    ])


    # Split dataset into train and val
    train_dir = os.path.join(args.data_dir, "train")
    val_dir   = os.path.join(args.data_dir, "val")
    test_dir  = os.path.join(args.data_dir, "test")

    train_ds = SafeImageFolder(train_dir, transform=train_tf)
    val_ds   = SafeImageFolder(val_dir, transform=val_tf)
    test_ds  = SafeImageFolder(test_dir, transform=val_tf)

    classes = train_ds.classes
    class_to_idx = train_ds.class_to_idx

    val_dir = "data/processed/val"

    for cls in os.listdir(val_dir):
        cls_path = os.path.join(val_dir, cls)
        if os.path.isdir(cls_path) and len(os.listdir(cls_path)) == 0:
            print("Removing empty folder:", cls_path)
            shutil.rmtree(cls_path)

    with open(os.path.join(args.out_dir, "class_map.json"), "w") as f:
        json.dump({"idx_to_class": {i: c for i, c in enumerate(classes)},
                    "class_to_idx": class_to_idx}, f, indent=2)
 


    # Use the increased num_workers argument
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # --- Model Setup ---
    model = get_model(args.arch, num_classes=len(classes))
    model.to(device)

    # --- 1. Train Head (Transfer Learning) ---
    for n,p in model.named_parameters():
        p.requires_grad = False
    if "mobilenet" in args.arch:
        for p in model.classifier.parameters():
            p.requires_grad = True
    else:
        for p in model.classifier.parameters():
            p.requires_grad = True

    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    ce = nn.CrossEntropyLoss()

    best_acc = 0.0
    for epoch in range(args.epochs):
        model.train()
        tot, correct = 0,0
        for x,y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [head]"):
            x,y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = ce(logits, y)
            loss.backward()
            opt.step()
        # val
        model.eval()
        vt, vc, preds, gts = 0,0,[],[]
        with torch.no_grad():
            for x,y in val_loader:
                x,y = x.to(device), y.to(device)
                logits = model(x)
                p = logits.argmax(1)
                preds.extend(p.cpu().tolist()); gts.extend(y.cpu().tolist())
                vt += y.size(0)
                vc += (p==y).sum().item()
        acc = vc / vt
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), os.path.join(args.out_dir, "best_head.pt"))
        print(f"Val acc (head, Epoch {epoch+1}): {acc:.4f}")

    # --- 2. Fine-tune Backbone ---
    # Unfreeze the last block of the feature extractor and classifier
    for n,p in model.named_parameters():
        p.requires_grad = False
    if "mobilenet" in args.arch:
        for p in model.features[-1].parameters():
            p.requires_grad = True
        for p in model.classifier.parameters():
            p.requires_grad = True
    else:
        # For EfficientNet, unfreeze the last block of features and the classifier
        for p in model.features[-1].parameters():
            p.requires_grad = True
        for p in model.classifier.parameters():
            p.requires_grad = True

    # Use a lower LR for fine-tuning
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr*0.5)
    
    # Run fewer epochs for fine-tuning
    finetune_epochs = max(2, args.epochs // 2)
    for epoch in range(finetune_epochs):
        model.train()
        for x,y in tqdm(train_loader, desc=f"Finetune {epoch+1}/{finetune_epochs}"):
            x,y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = ce(logits, y)
            loss.backward()
            opt.step()
            
    # --- Final Validation and Save ---
    model.eval()
    preds, gts = [], []
    with torch.no_grad():
        for x,y in val_loader:
            x,y = x.to(device), y.to(device)
            p = model(x).argmax(1)
            preds.extend(p.cpu().tolist()); gts.extend(y.cpu().tolist())
            
    from sklearn.metrics import accuracy_score
    acc = accuracy_score(gts, preds)
    print(f"\nFinal val acc: {acc:.4f}")
    torch.save(model.state_dict(), os.path.join(args.out_dir, "model_final.pt"))
    
    # --- Export ONNX ---
    # Note: Moving to CPU before ONNX export is good practice
    dummy = torch.randn(1,3,args.img_size,args.img_size, device="cpu") 
    model.cpu() 
    model.eval()
    onnx_path = os.path.join(args.out_dir, "model_fp32.onnx")
    torch.onnx.export(model, dummy, onnx_path, input_names=["input"], output_names=["logits"], opset_version=17, dynamic_axes={"input":{0:"batch"}, "logits":{0:"batch"}})
    print("Exported:", onnx_path)
        
    """ --- Quantize to INT8 (Activated) ---
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        int8_path = os.path.join(args.out_dir, "model_int8.onnx")
        # Dynamic quantization is suitable for model inference on CPU
        quantize_dynamic(model_input=onnx_path, model_output=int8_path, weight_type=QuantType.QInt8)
        print("Quantized to INT8:", int8_path)
    except Exception as e:
        print(f"Quantization failed (check onnxruntime installation): {e}")
"""
if __name__ == "__main__":
    main()