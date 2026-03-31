"""
setup_dataset.py
================
Helper script to organize the PlantVillage dataset into
the correct train/test folder structure.

USAGE:
  1. Download PlantVillage from Kaggle:
     https://www.kaggle.com/datasets/emmarex/plantdisease

  2. Extract and note the path to the PlantVillage folder.

  3. Run:
     python setup_dataset.py --source /path/to/PlantVillage

  This will create:
    dataset/
    ├── train/  (80% images)
    └── test/   (20% images)
"""

import os
import shutil
import argparse
import random
from pathlib import Path

# ─── Classes to use ───
TARGET_CLASSES = {
    'Tomato___healthy'      : 'Tomato_healthy',
    'Tomato___Early_blight' : 'Tomato_Early_blight',
    'Tomato___Late_blight'  : 'Tomato_Late_blight',
}

TRAIN_SPLIT = 0.80  # 80% train, 20% test
MAX_PER_CLASS = 500  # Limit per class (so it runs fast on laptops)


def setup_dataset(source_dir, dest_dir='dataset', seed=42):
    """
    Organize images from PlantVillage into train/test splits.

    Args:
        source_dir : Path to extracted PlantVillage dataset folder
        dest_dir   : Destination folder (default: 'dataset/')
        seed       : Random seed for reproducibility
    """
    random.seed(seed)
    source_path = Path(source_dir)
    dest_path   = Path(dest_dir)

    if not source_path.exists():
        print(f"❌ Source directory not found: {source_dir}")
        return

    total_copied = 0
    print(f"\n🗂️  Setting up dataset from: {source_dir}")
    print(f"📁 Destination: {dest_dir}/")
    print("-" * 50)

    for source_class, dest_class in TARGET_CLASSES.items():
        # Try to find the class folder (case-insensitive matching)
        class_folder = None
        for folder in source_path.iterdir():
            if folder.is_dir() and source_class.lower() in folder.name.lower():
                class_folder = folder
                break

        if class_folder is None:
            print(f"⚠️  Class folder not found: {source_class}")
            print(f"   Available folders: {[f.name for f in source_path.iterdir() if f.is_dir()][:5]}")
            continue

        # Get all image files
        images = [f for f in class_folder.iterdir()
                  if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}]

        if not images:
            print(f"⚠️  No images found in: {class_folder.name}")
            continue

        # Limit number of images per class (laptop-friendly)
        if len(images) > MAX_PER_CLASS:
            images = random.sample(images, MAX_PER_CLASS)

        # Shuffle and split
        random.shuffle(images)
        split_idx    = int(len(images) * TRAIN_SPLIT)
        train_images = images[:split_idx]
        test_images  = images[split_idx:]

        # Create destination folders
        train_dir = dest_path / 'train' / dest_class
        test_dir  = dest_path / 'test'  / dest_class
        train_dir.mkdir(parents=True, exist_ok=True)
        test_dir.mkdir(parents=True, exist_ok=True)

        # Copy images
        for img in train_images:
            shutil.copy2(img, train_dir / img.name)
        for img in test_images:
            shutil.copy2(img, test_dir / img.name)

        total_copied += len(images)
        print(f"✅ {dest_class:<30} | Train: {len(train_images):>4} | Test: {len(test_images):>3}")

    print("-" * 50)
    print(f"✅ Dataset setup complete! Total images: {total_copied}")
    print(f"\nFolder structure created:")
    print(f"  {dest_dir}/")
    print(f"  ├── train/")
    for cls in TARGET_CLASSES.values():
        print(f"  │   ├── {cls}/")
    print(f"  └── test/")
    for cls in TARGET_CLASSES.values():
        print(f"      ├── {cls}/")


def verify_dataset(dest_dir='dataset'):
    """Print dataset statistics after setup."""
    print("\n📊 Dataset Verification:")
    print("-" * 50)
    dest_path = Path(dest_dir)

    for split in ['train', 'test']:
        split_path = dest_path / split
        if not split_path.exists():
            print(f"⚠️  {split}/ folder not found")
            continue

        print(f"\n  {split.upper()}/")
        total = 0
        for cls_folder in sorted(split_path.iterdir()):
            if cls_folder.is_dir():
                count = len(list(cls_folder.glob('*.jpg'))) + \
                        len(list(cls_folder.glob('*.jpeg'))) + \
                        len(list(cls_folder.glob('*.png')))
                print(f"    {cls_folder.name:<32} : {count:>4} images")
                total += count
        print(f"    {'TOTAL':<32} : {total:>4} images")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Setup PlantVillage dataset')
    parser.add_argument('--source', type=str, required=True,
                        help='Path to extracted PlantVillage dataset folder')
    parser.add_argument('--dest',   type=str, default='dataset',
                        help='Destination folder (default: dataset/)')
    parser.add_argument('--max',    type=int, default=500,
                        help='Max images per class (default: 500)')
    args = parser.parse_args()

    MAX_PER_CLASS = args.max
    setup_dataset(args.source, args.dest)
    verify_dataset(args.dest)
