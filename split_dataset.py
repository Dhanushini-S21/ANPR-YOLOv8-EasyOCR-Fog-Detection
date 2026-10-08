import os
import shutil
import random

IMAGE_ROOTS = [
    "data/images",
    "data/images 01"
]

LABEL_ROOT = "data/labels"
OUTPUT_ROOT = "data/plate_dataset"

TRAIN_RATIO = 0.8
SEED = 42

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def find_images():
    images = []

    for image_root in IMAGE_ROOTS:

        if not os.path.exists(image_root):
            print(f"Warning: folder not found: {image_root}")
            continue

        for root, _, files in os.walk(image_root):
            for file in files:
                if file.lower().endswith(IMAGE_EXTENSIONS):
                    images.append(os.path.join(root, file))

    return images


def main():

    random.seed(SEED)

    images = find_images()

    print(f"Images found: {len(images)}")

    pairs = []

    for image_path in images:

        filename = os.path.basename(image_path)
        stem = os.path.splitext(filename)[0]

        label_path = os.path.join(
            LABEL_ROOT,
            stem + ".txt"
        )

        if os.path.exists(label_path):
            pairs.append((image_path, label_path))

    print(f"Images with labels: {len(pairs)}")

    if len(pairs) < 2:
        print("ERROR: Not enough image-label pairs.")
        return

    random.shuffle(pairs)

    split_index = int(len(pairs) * TRAIN_RATIO)

    # Make sure both sets contain at least one image
    split_index = max(1, min(split_index, len(pairs) - 1))

    train_pairs = pairs[:split_index]
    val_pairs = pairs[split_index:]

    # Remove old dataset
    if os.path.exists(OUTPUT_ROOT):
        shutil.rmtree(OUTPUT_ROOT)

    # Create directories
    directories = [
        os.path.join(OUTPUT_ROOT, "images", "train"),
        os.path.join(OUTPUT_ROOT, "images", "val"),
        os.path.join(OUTPUT_ROOT, "labels", "train"),
        os.path.join(OUTPUT_ROOT, "labels", "val"),
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)

    def copy_pairs(data, split):

        for image_path, label_path in data:

            image_filename = os.path.basename(image_path)
            label_filename = os.path.basename(label_path)

            destination_image = os.path.join(
                OUTPUT_ROOT,
                "images",
                split,
                image_filename
            )

            destination_label = os.path.join(
                OUTPUT_ROOT,
                "labels",
                split,
                label_filename
            )

            shutil.copy2(
                image_path,
                destination_image
            )

            shutil.copy2(
                label_path,
                destination_label
            )

    copy_pairs(train_pairs, "train")
    copy_pairs(val_pairs, "val")

    print()
    print("======================================")
    print("DATASET PREPARATION COMPLETE")
    print("======================================")
    print(f"Total images found     : {len(images)}")
    print(f"Images with labels     : {len(pairs)}")
    print(f"Training images        : {len(train_pairs)}")
    print(f"Validation images      : {len(val_pairs)}")
    print("======================================")


if __name__ == "__main__":
    main()