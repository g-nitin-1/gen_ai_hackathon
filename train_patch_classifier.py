"""
Train a simple patch classifier (signature/stamp/background) on patch_data.

Usage:
  python3 train_patch_classifier.py --epochs 15 --device 0 --arch resnet18
"""

import argparse
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


def get_loaders(data_root: Path, batch_size=32):
    train_tfms = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(5),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    val_tfms = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    train_ds = datasets.ImageFolder(data_root / "train", transform=train_tfms)
    val_ds = datasets.ImageFolder(data_root / "val", transform=val_tfms)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, val_loader, train_ds.classes


def train(model, train_loader, val_loader, device, epochs, lr=1e-3):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    best_acc = 0.0
    best_state = None
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0
        correct = 0
        loss_sum = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * labels.size(0)
            preds = out.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        train_acc = correct / total if total else 0

        model.eval()
        val_total = 0
        val_correct = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                out = model(imgs)
                preds = out.argmax(1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
        val_acc = val_correct / val_total if val_total else 0
        print(f"Epoch {epoch}: train_acc={train_acc:.3f} val_acc={val_acc:.3f}")
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = model.state_dict()
    return best_state, best_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="patch_data", help="Path to patch_data directory")
    parser.add_argument("--device", type=str, default="0", help="Device, e.g., 0 or cpu")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument(
        "--arch",
        type=str,
        default="resnet18",
        choices=["resnet18", "resnet34", "resnet50"],
        help="Backbone for patch classifier",
    )
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.device}" if args.device != "cpu" and torch.cuda.is_available() else "cpu")
    train_loader, val_loader, classes = get_loaders(Path(args.data), batch_size=args.batch)
    print("Classes:", classes)

    if args.arch == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    elif args.arch == "resnet34":
        model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
    else:
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model.to(device)

    best_state, best_acc = train(model, train_loader, val_loader, device, epochs=args.epochs)
    out_dir = Path("patch_runs")
    out_dir.mkdir(exist_ok=True)
    if best_state:
        torch.save({"state_dict": best_state, "classes": classes}, out_dir / "patch_classifier.pt")
        print("Saved best model to", out_dir / "patch_classifier.pt", "val_acc", best_acc)
    else:
        print("No best state saved (empty dataset?)")


if __name__ == "__main__":
    main()
