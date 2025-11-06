
import argparse, json, os, random, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm

def set_seed(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def get_model(arch, num_classes):
    if arch == "mobilenetv3_small":
        m = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, num_classes)
    elif arch == "efficientnet_b0":
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    else:
        raise ValueError("unsupported arch")
    return m

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_dir", default="artifacts")
    ap.add_argument("--arch", default="mobilenetv3_small", choices=["mobilenetv3_small","efficientnet_b0"])
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--img_size", type=int, default=320)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed()

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

    full = datasets.ImageFolder(args.data_dir)
    classes = full.classes
    with open(os.path.join(args.out_dir, "class_map.json"), "w") as f:
        json.dump({"idx_to_class": {i:c for i,c in enumerate(classes)},
                   "class_to_idx": full.class_to_idx}, f, indent=2)

    # simple split
    indices = list(range(len(full)))
    random.shuffle(indices)
    split = int(0.8 * len(indices))
    train_idx, val_idx = indices[:split], indices[split:]
    train_ds = torch.utils.data.Subset(datasets.ImageFolder(args.data_dir, transform=train_tf), train_idx)
    val_ds   = torch.utils.data.Subset(datasets.ImageFolder(args.data_dir, transform=val_tf),   val_idx)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    device = torch.device("cpu")
    model = get_model(args.arch, num_classes=len(classes))
    model.to(device)

    # freeze backbone, train head first
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
        print("Val acc (head):", acc)

    # unfreeze last block for fine-tune
    for n,p in model.named_parameters():
        p.requires_grad = False
    if "mobilenet" in args.arch:
        for p in model.features[-1].parameters():
            p.requires_grad = True
        for p in model.classifier.parameters():
            p.requires_grad = True
    else:
        for p in model.features[-1].parameters():
            p.requires_grad = True
        for p in model.classifier.parameters():
            p.requires_grad = True

    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr*0.5)
    for epoch in range(max(2, args.epochs//2)):
        model.train()
        for x,y in tqdm(train_loader, desc=f"Finetune {epoch+1}"):
            x,y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = ce(logits, y)
            loss.backward()
            opt.step()
    # final val
    model.eval()
    preds, gts = [], []
    with torch.no_grad():
        for x,y in val_loader:
            x,y = x.to(device), y.to(device)
            p = model(x).argmax(1)
            preds.extend(p.cpu().tolist()); gts.extend(y.cpu().tolist())
    from sklearn.metrics import accuracy_score
    acc = accuracy_score(gts, preds)
    print("Final val acc:", acc)
    torch.save(model.state_dict(), os.path.join(args.out_dir, "model_final.pt"))

    # Export ONNX
    dummy = torch.randn(1,3,args.img_size,args.img_size, device=device)
    model.cpu()
    model.eval()
    onnx_path = os.path.join(args.out_dir, "model_fp32.onnx")
    torch.onnx.export(model, dummy, onnx_path, input_names=["input"], output_names=["logits"], opset_version=17, dynamic_axes={"input":{0:"batch"}, "logits":{0:"batch"}})
    print("Exported:", onnx_path)

    # Quantize to INT8
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        int8_path = os.path.join(args.out_dir, "model_int8.onnx")
        quantize_dynamic(model_input=onnx_path, model_output=int8_path, weight_type=QuantType.QInt8)
        print("Quantized:", int8_path)
    except Exception as e:
        print("Quantization failed:", e)

if __name__ == "__main__":
    main()
