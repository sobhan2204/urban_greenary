import argparse
import os

import cv2
import numpy as np
import rasterio


CLASS_NAMES = {
    0: "built_up",
    1: "tree",
    2: "grassland",
    3: "water",
    4: "wasteland",
}


def normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    mn, mx = arr.min(), arr.max()
    if mx - mn < 1e-6:
        return np.zeros_like(arr, dtype=np.uint8)
    return ((arr - mn) / (mx - mn) * 255).astype(np.uint8)


def compute_indices(red: np.ndarray, green: np.ndarray, nir: np.ndarray):
    ndvi = (nir - red) / (nir + red + 1e-8)
    ndwi = (green - nir) / (green + nir + 1e-8)
    return ndvi, ndwi


def build_binary_mask(ndvi: np.ndarray, threshold: float) -> np.ndarray:
    return (ndvi >= threshold).astype(np.uint8)


def build_multiclass_mask(
    ndvi: np.ndarray,
    ndwi: np.ndarray,
    red_u8: np.ndarray,
    green_u8: np.ndarray,
    blue_u8: np.ndarray,
) -> np.ndarray:
    brightness = (
        red_u8.astype(np.float32)
        + green_u8.astype(np.float32)
        + blue_u8.astype(np.float32)
    ) / (255.0 * 3.0)
    mask = np.zeros(ndvi.shape, dtype=np.uint8)

    # --- relaxed & reordered thresholds for Delhi Sentinel-2 ---
    water = (ndwi > 0.05) & (ndvi < 0.15)
    tree = ndvi >= 0.40
    grassland = (ndvi >= 0.20) & (ndvi < 0.40)
    wasteland = (ndvi < 0.20) & (brightness > 0.45) & (~water)
    built_up = ~(water | tree | grassland | wasteland)

    mask[built_up] = 0
    mask[tree] = 1
    mask[grassland] = 2
    mask[water] = 3
    mask[wasteland] = 4

    # Debug: print class distribution so you can verify
    total = mask.size
    for cid, cname in CLASS_NAMES.items():
        pct = np.sum(mask == cid) / total * 100
        print(f"  {cname}: {pct:.1f}%")

    return mask


def save_patches(
    rgb: np.ndarray,
    mask: np.ndarray,
    out_images: str,
    out_masks: str,
    patch_size: int,
    step: int,
) -> int:
    os.makedirs(out_images, exist_ok=True)
    os.makedirs(out_masks, exist_ok=True)
    h, w = mask.shape[:2]
    count = 0
    for y in range(0, h - patch_size + 1, step):
        for x in range(0, w - patch_size + 1, step):
            img_patch = rgb[y : y + patch_size, x : x + patch_size]
            mask_patch = mask[y : y + patch_size, x : x + patch_size]

            # skip patches that are >95 % single class (uninformative)
            vals, cnts = np.unique(mask_patch, return_counts=True)
            if cnts.max() / mask_patch.size > 0.95 and len(vals) == 1:
                continue

            cv2.imwrite(os.path.join(out_images, f"img_{count}.png"), img_patch)
            cv2.imwrite(os.path.join(out_masks, f"img_{count}.png"), mask_patch)
            count += 1

    # If too few diverse patches, save all patches without filtering
    if count == 0:
        print("  Warning: all patches were uniform, saving all without filter")
        count = 0
        for y in range(0, h - patch_size + 1, step):
            for x in range(0, w - patch_size + 1, step):
                img_patch = rgb[y : y + patch_size, x : x + patch_size]
                mask_patch = mask[y : y + patch_size, x : x + patch_size]
                cv2.imwrite(os.path.join(out_images, f"img_{count}.png"), img_patch)
                cv2.imwrite(os.path.join(out_masks, f"img_{count}.png"), mask_patch)
                count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description="Prepare segmentation dataset from Sentinel image")
    parser.add_argument("--image", default="data/delhi_satellite.tif", help="Input raster path")
    parser.add_argument("--mode", choices=["binary", "multiclass"], default="binary")
    parser.add_argument("--out-images", default="dataset/images")
    parser.add_argument("--out-masks", default="dataset/masks")
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--step", type=int, default=256)
    parser.add_argument("--ndvi-threshold", type=float, default=0.3)
    args = parser.parse_args()

    with rasterio.open(args.image) as src:
        red = src.read(1).astype(np.float32)
        green = src.read(2).astype(np.float32)
        blue = src.read(3).astype(np.float32)
        nir = src.read(4).astype(np.float32)

    # Sentinel-2 values can be 0-10000; normalise to 0-1 for index calc
    if red.max() > 100:
        scale = 10000.0
        red_s, green_s, blue_s, nir_s = red / scale, green / scale, blue / scale, nir / scale
    else:
        red_s, green_s, blue_s, nir_s = red, green, blue, nir

    ndvi, ndwi = compute_indices(red_s, green_s, nir_s)

    print(f"NDVI range: {ndvi.min():.3f} .. {ndvi.max():.3f}")
    print(f"NDWI range: {ndwi.min():.3f} .. {ndwi.max():.3f}")

    red_u8 = normalize_to_uint8(red)
    green_u8 = normalize_to_uint8(green)
    blue_u8 = normalize_to_uint8(blue)
    rgb = np.stack([red_u8, green_u8, blue_u8], axis=2)

    if args.mode == "binary":
        mask = build_binary_mask(ndvi, args.ndvi_threshold) * 255
    else:
        mask = build_multiclass_mask(ndvi, ndwi, red_u8, green_u8, blue_u8)

    saved = save_patches(
        rgb=rgb,
        mask=mask,
        out_images=args.out_images,
        out_masks=args.out_masks,
        patch_size=args.patch_size,
        step=args.step,
    )

    print(f"Mode: {args.mode}")
    if args.mode == "multiclass":
        print(f"Classes: {CLASS_NAMES}")
    print(f"Dataset created: {saved} patches")


if __name__ == "__main__":
    main()