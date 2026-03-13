import argparse
import os

import cv2
import numpy as np
import torch
import torch.nn as nn


COLOR_MAP = {
    0: (120, 120, 120),  # built_up  – grey
    1: (34, 139, 34),    # tree      – dark green
    2: (144, 238, 144),  # grassland – light green
    3: (30, 144, 255),   # water     – blue
    4: (210, 180, 140),  # wasteland – tan
}


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


def to_colored_mask(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, color in COLOR_MAP.items():
        colored[mask == class_id] = color
    return colored


def collect_inputs(path: str):
    if os.path.isdir(path):
        return [
            os.path.join(path, name)
            for name in sorted(os.listdir(path))
            if name.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
    return [path]


def main():
    parser = argparse.ArgumentParser(description="Run U-Net inference for binary or multiclass masks")
    parser.add_argument("--input", default="dataset/images", help="Image file or image directory")
    parser.add_argument("--checkpoint", default="models/unet_vegetation.pth")
    parser.add_argument("--output", default="maps/predictions")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(args.checkpoint, map_location=device)
    mode = checkpoint.get("mode", "binary")
    num_classes = checkpoint.get("num_classes", 1)

    model = UNet(in_channels=3, out_channels=num_classes).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    input_paths = collect_inputs(args.input)
    for image_path in input_paths:
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            print(f"Skipping unreadable: {image_path}")
            continue

        img_tensor = torch.tensor(
            (image.astype(np.float32) / 255.0).transpose(2, 0, 1),
            dtype=torch.float32,
        ).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(img_tensor)

        file_stem = os.path.splitext(os.path.basename(image_path))[0]
        if mode == "binary":
            prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
            mask = (prob > args.threshold).astype(np.uint8) * 255
            color = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
            color[:, :, 1] = mask

            cv2.imwrite(os.path.join(args.output, f"{file_stem}_mask.png"), mask)
            cv2.imwrite(os.path.join(args.output, f"{file_stem}_overlay.png"), color)
        else:
            class_mask = torch.argmax(logits, dim=1)[0].cpu().numpy().astype(np.uint8)

            # Debug: print predicted class distribution
            for cid in range(num_classes):
                pct = np.sum(class_mask == cid) / class_mask.size * 100
                print(f"  {file_stem} class {cid}: {pct:.1f}%")

            color = to_colored_mask(class_mask)
            cv2.imwrite(os.path.join(args.output, f"{file_stem}_mask.png"), class_mask)
            cv2.imwrite(os.path.join(args.output, f"{file_stem}_overlay.png"), color)

    print(f"Saved predictions for {len(input_paths)} image(s) in {args.output}")


if __name__ == "__main__":
    main()