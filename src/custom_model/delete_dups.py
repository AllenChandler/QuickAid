import os
import hashlib
from pathlib import Path
from PIL import Image
import imagehash



def compute_hash(file_path):
    """Compute a perceptual hash for an image."""
    with Image.open(file_path) as img:
        return str(imagehash.average_hash(img))


def find_and_remove_duplicates(train_dir, valid_dir, test_dir, output_dir):
    """Find duplicates across train, valid, and test directories and remove them from train."""
    os.makedirs(output_dir, exist_ok=True)

    # Compute hashes for validation and test sets
    print("Computing hashes for validation and test sets...")
    valid_hashes = {
        compute_hash(os.path.join(root, file))
        for root, _, files in os.walk(valid_dir)
        for file in files if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
    }
    test_hashes = {
        compute_hash(os.path.join(root, file))
        for root, _, files in os.walk(test_dir)
        for file in files if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
    }
    all_hashes = valid_hashes.union(test_hashes)

    # Check train set for duplicates
    print("Checking for duplicates in the train set...")
    duplicate_count = 0
    for root, _, files in os.walk(train_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                file_path = os.path.join(root, file)
                file_hash = compute_hash(file_path)
                if file_hash in all_hashes:
                    duplicate_count += 1
                    # Move duplicate to output directory, preserving subfolder structure
                    rel_path = os.path.relpath(file_path, train_dir)
                    duplicate_output_path = os.path.join(output_dir, rel_path)
                    os.makedirs(os.path.dirname(duplicate_output_path), exist_ok=True)
                    os.rename(file_path, duplicate_output_path)

    print(f"Duplicate removal complete. {duplicate_count} duplicates moved to {output_dir}.")


# Paths to data directories
train_dir = "J:/projects/QuickAid/data/images/train"
valid_dir = "J:/projects/QuickAid/data/images/valid"
test_dir = "J:/projects/QuickAid/data/images/test"
output_dir = "J:/projects/QuickAid/duplicates"



for directory in [train_dir, valid_dir, test_dir]:
    print(f"Contents of {directory}:")
    for root, _, files in os.walk(directory):
        for file in files:
            print(os.path.join(root, file))

if __name__ == "__main__":
    find_and_remove_duplicates(train_dir, valid_dir, test_dir, output_dir)
