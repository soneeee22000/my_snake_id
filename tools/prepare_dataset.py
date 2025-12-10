import os
import shutil
import random
from PIL import Image

# =========================
# CONFIG
# =========================
SOURCE_DIR = "data/images"
OUTPUT_DIR = "data/processed"

IMAGE_SIZE = (256, 256)      # resize to 256×256
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15


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
# Process one species folder
# ==================================================
def process_species_folder(species_path, species_name):
    print(f"\n📌 Processing species: {species_name}")

    # clean corrupted images
    removed = remove_corrupted_images(species_path)
    if removed > 0:
        print(f"   ⚠️ Removed {removed} corrupted files")

    images = [
        os.path.join(species_path, f)
        for f in os.listdir(species_path)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    if len(images) == 0:
        print("   ❌ No valid images, skipping")
        return

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
                img = img.resize(IMAGE_SIZE)

                filename = os.path.basename(img_path)
                out_path = os.path.join(split_dir, filename)

                img.save(out_path, "JPEG", quality=90)
            except Exception:
                print(f"   ⚠️ Failed processing: {img_path}")


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

    print("\n🎉 Dataset preparation complete!")
    print(f"Output folders created in: {OUTPUT_DIR}")
    print(" - train/")
    print(" - val/")
    print(" - test/")


if __name__ == "__main__":
    main()
