import os
import shutil
from sklearn.model_selection import train_test_split

# Full standford dataset
SOURCE_DIR = "full_breeds_dataset"
# Directory that I want to save the split data to
TARGET_DIR = "data"

breeds = {
    "chihuahua": "n02085620-Chihuahua",
    "corgi": "n02113023-Pembroke",
    "german_shepherd": "n02106662-German_shepherd",
    "golden_retriever": "n02099601-Golden_retriever",
    "great_dane": "n02109047-Great_Dane",
    "husky": "n02110185-Siberian_husky",
    "pomeranian": "n02112018-Pomeranian",
    "pug": "n02110958-Pug",
    "saint_bernard": "n02109525-Saint_Bernard"
}

splits = ["train", "valid", "test"]
ratios = [0.7, 0.15, 0.15]

if sum(ratios) != 1.0:
    raise ValueError("Ratios must sum to 1.0")

for breed_name, folder in breeds.items():
    img_dir = os.path.join(SOURCE_DIR + "/images/Images/" + folder)

    # Only retrieve valid image files
    imgs = [
        f for f in os.listdir(img_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    # Make sure that there are images to process
    if not imgs:
        print(f"Warning: No images found for {breed_name} in {img_dir}")
        continue

    # Split images according to ratios
    train_ratio, valid_ratio, test_ratio = ratios

    train_imgs, temp = train_test_split(imgs, test_size=(1 - train_ratio), random_state=42)
    valid_relative = valid_ratio / (valid_ratio + test_ratio)
    valid_imgs, test_imgs = train_test_split(temp, test_size=(1 - valid_relative), random_state=42)

    # Create directories if they don't already exist (they should)
    for split in splits:
        split_dir = os.path.join(TARGET_DIR, split, breed_name)
        os.makedirs(split_dir, exist_ok=True)

    # Copy images
    for img in train_imgs:
        shutil.copy(os.path.join(img_dir, img), os.path.join(TARGET_DIR, "train", breed_name))

    for img in valid_imgs:
        shutil.copy(os.path.join(img_dir, img), os.path.join(TARGET_DIR, "valid", breed_name))

    for img in test_imgs:
        shutil.copy(os.path.join(img_dir, img), os.path.join(TARGET_DIR, "test", breed_name))

print("All images moved to the data directory successfully.")
