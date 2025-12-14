import os
import shutil
import random

# Set random seed for reproducibility
random.seed(42)

# Paths
data_dir = "data"
test_dir = os.path.join(data_dir, "test")
num_test_images_per_class = 10  # Adjust as needed

# Create test directory
os.makedirs(test_dir, exist_ok=True)

# Get list of species (subfolders in data)
species = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d)) and d != "test"]

for species_name in species:
    species_path = os.path.join(data_dir, species_name)
    test_species_path = os.path.join(test_dir, species_name)
    
    # Create species subfolder in test
    os.makedirs(test_species_path, exist_ok=True)
    
    # Get list of image files
    images = [f for f in os.listdir(species_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    
    # Shuffle and select test images
    random.shuffle(images)
    test_images = images[:num_test_images_per_class]
    
    # Copy selected images to test folder
    for img in test_images:
        src = os.path.join(species_path, img)
        dst = os.path.join(test_species_path, img)
        shutil.copy2(src, dst)
        print(f"Copied {src} to {dst}")

print(f"Test set created with {num_test_images_per_class} images per species in {test_dir}")