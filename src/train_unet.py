import argparse
import os

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split


# ── U-Net architecture ──────────────────────────────────────────────
class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        self.enc1 = DoubleConv(in_channels, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(512, 1024)

        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = DoubleConv(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = DoubleConv(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = DoubleConv(128, 64)

        self.final = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        d4 = self.dec4(torch.cat([self.up4(b), e4], 1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        return self.final(d1)


# ── Dataset ──────────────────────────────────────────────────────────
class SegmentationDataset(Dataset):
    def __init__(self, img_dir, mask_dir, mode="binary"):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.mode = mode
        img_names = set(os.listdir(img_dir))
        mask_names = set(os.listdir(mask_dir))
        self.names = sorted(img_names & mask_names)

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        img = cv2.imread(os.path.join(self.img_dir, name), cv2.IMREAD_COLOR)
        mask = cv2.imread(os.path.join(self.mask_dir, name), cv2.IMREAD_GRAYSCALE)

        img = img.astype(np.float32) / 255.0
        img = torch.tensor(img.transpose(2, 0, 1), dtype=torch.float32)

        if self.mode == "binary":
            mask = (mask > 127).astype(np.float32)
            mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
        else:
            mask = torch.tensor(mask.astype(np.int64), dtype=torch.long)

        return img, mask


# ── Compute class weights from dataset ───────────────────────────────
def compute_class_weights(mask_dir: str, num_classes: int) -> torch.Tensor:
    counts = np.zeros(num_classes, dtype=np.float64)
    for fname in os.listdir(mask_dir):
        m = cv2.imread(os.path.join(mask_dir, fname), cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        for c in range(num_classes):
            counts[c] += np.sum(m == c)
    total = counts.sum()
    if total == 0:
        return torch.ones(num_classes)
    freq = counts / total
    # inverse frequency, clamped
    weights = 1.0 / (freq + 1e-6)
    weights = weights / weights.sum() * num_classes
    print(f"Class pixel counts: {counts.astype(int).tolist()}")
    print(f"Class weights:      {[f'{w:.2f}' for w in weights]}")
    return torch.tensor(weights, dtype=torch.float32)


# ── Training ─────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Train U-Net for vegetation segmentation")
    parser.add_argument("--mode", choices=["binary", "multiclass"], default="binary")
    parser.add_argument("--img-dir", default="dataset/images")
    parser.add_argument("--mask-dir", default="dataset/masks")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-classes", type=int, default=5)
    parser.add_argument("--val-split", type=float, default=0.15)
    args = parser.parse_args()

    num_classes = 1 if args.mode == "binary" else args.num_classes
    save_path = (
        "models/unet_vegetation.pth" if args.mode == "binary" else "models/unet_landcover_multiclass.pth"
    )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    dataset = SegmentationDataset(args.img_dir, args.mask_dir, args.mode)
    if len(dataset) == 0:
        raise ValueError("No image-mask pairs found in dataset directories.")

    if len(dataset) >= 2 and args.val_split > 0:
        val_size = max(1, int(len(dataset) * args.val_split))
        train_size = len(dataset) - val_size
        train_data, val_data = random_split(dataset, [train_size, val_size])
        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=0)
    else:
        train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
        val_loader = None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_channels=3, out_channels=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    if args.mode == "binary":
        criterion = nn.BCEWithLogitsLoss()
    else:
        weights = compute_class_weights(args.mask_dir, num_classes).to(device)
        criterion = nn.CrossEntropyLoss(weight=weights)

    best_val_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            logits = model(imgs)
            loss = criterion(logits, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)

        train_loss = running_loss / len(train_loader.dataset)

        val_loss = None
        if val_loader is not None:
            model.eval()
            val_running = 0.0
            with torch.no_grad():
                for imgs, masks in val_loader:
                    imgs, masks = imgs.to(device), masks.to(device)
                    logits = model(imgs)
                    val_running += criterion(logits, masks).item() * imgs.size(0)
            val_loss = val_running / len(val_loader.dataset)

        tag = f"Epoch {epoch}/{args.epochs}  train_loss={train_loss:.4f}"
        if val_loss is not None:
            tag += f"  val_loss={val_loss:.4f}"
        print(tag)

        save_loss = val_loss if val_loss is not None else train_loss
        if save_loss < best_val_loss:
            best_val_loss = save_loss
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "mode": args.mode,
                    "num_classes": num_classes,
                },
                save_path,
            )
            print(f"  ✓ Saved best model to {save_path}")

    print("Training complete.")


if __name__ == "__main__":
    main()