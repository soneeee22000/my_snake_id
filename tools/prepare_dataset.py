import os
import shutil
import random
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import csv

stats = []  

# =========================
# CONFIG
# =========================
SOURCE_DIR = "data/images"
OUTPUT_DIR = "data/processed"

IMAGE_SIZE = (256, 256)  # resize to 256×256
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

AUGMENTATIONS_PER_IMAGE = 2   # how many augmented versions per original image


# ==================================================
# Helper: remove corrupted images
# ==================================================
def remove_corrupted_images(folder):
    removed = 0
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            img = Image.open(file_path)
            img.verify()
        except Exception:
            os.remove(file_path)
            removed += 1
    return removed


# ==================================================
# Strong augmentation (Option B)
# ==================================================
def strong_augment(img):
    # random rotation
    angle = random.uniform(-25, 25)
    img = img.rotate(angle, expand=True)

    # random brightness
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(random.uniform(0.6, 1.4))

    # random contrast
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(random.uniform(0.6, 1.4))

    # random color jitter
    enhancer = ImageEnhance.Color(img)
    img = enhancer.enhance(random.uniform(0.6, 1.4))

    # gaussian blur
    if random.random() < 0.3:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.2)))

    # random noise
    if random.random() < 0.3:
        arr = np.array(img)
        noise = np.random.normal(0, 12, arr.shape).astype(np.int16)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)

    # random crop (slight)
    if random.random() < 0.5:
        w, h = img.size
        crop_pct = random.uniform(0.05, 0.15)
        dx = int(w * crop_pct)
        dy = int(h * crop_pct)
        img = img.crop((dx, dy, w - dx, h - dy))

    return img

# ==================================================
# Process one species folder
# ==================================================
def process_species_folder(species_path, species_name):
    print(f"\nProcessing species: {species_name}")

    removed = remove_corrupted_images(species_path)
    if removed > 0:
        print(f"  Removed {removed} corrupted files")

    images = [
        os.path.join(species_path, f)
        for f in os.listdir(species_path)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    if len(images) == 0:
        print("No valid images, skipping")
        return

    # ---- count raw images BEFORE shuffle ----
    raw_count = len(images)

    random.shuffle(images)

    # split counts
    train_end = int(len(images) * TRAIN_SPLIT)
    val_end = train_end + int(len(images) * VAL_SPLIT)

    splits = {
        "train": images[:train_end],
        "val": images[train_end:val_end],
        "test": images[val_end:]
    }

    # create output dirs
    for split, imgs in splits.items():
        split_dir = os.path.join(OUTPUT_DIR, split, species_name)
        os.makedirs(split_dir, exist_ok=True)

        for img_path in imgs:
            try:
                img = Image.open(img_path).convert("RGB")

                # resize original
                base_img = img.resize(IMAGE_SIZE)
                filename = os.path.basename(img_path)
                out_path = os.path.join(split_dir, filename)
                base_img.save(out_path, "JPEG", quality=90)

                # -------------------------
                # APPLY AUGMENTATION (train only)
                # -------------------------
                if split == "train":
                    for i in range(AUGMENTATIONS_PER_IMAGE):
                        aug = strong_augment(img)
                        aug = aug.resize(IMAGE_SIZE)

                        aug_filename = filename.replace(".jpg", "") + f"_aug{i}.jpg"
                        aug_path = os.path.join(split_dir, aug_filename)
                        aug.save(aug_path, "JPEG", quality=90)

            except Exception:
                print(f"Failed processing: {img_path}")

    # ---- count processed images ----
    processed_count = 0
    for split, imgs in splits.items():
        if split == "train":
            processed_count += len(imgs) * (1 + AUGMENTATIONS_PER_IMAGE)
        else:
            processed_count += len(imgs)

    # save statistics into the global list
    stats.append([species_name, raw_count, processed_count])


# ==================================================
# MAIN
# ==================================================
def main():
    species_list = [
        d for d in os.listdir(SOURCE_DIR)
        if os.path.isdir(os.path.join(SOURCE_DIR, d))
    ]

    print(f"🔍 Found {len(species_list)} species folders.\n")

    for species in species_list:
        species_path = os.path.join(SOURCE_DIR, species)
        process_species_folder(species_path, species)

    print("\n Dataset preparation complete!")
    print(f"Output folders created in: {OUTPUT_DIR}")
    print(" - train/")
    print(" - val/")
    print(" - test/")

# write the CSV file
csv_path = os.path.join(OUTPUT_DIR, "dataset_summary.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Species", "Raw Images", "Processed Images"])
    writer.writerows(stats)

print(f"\nCSV summary saved to: {csv_path}")


if __name__ == "__main__":
    main()
