import random
import shutil
from pathlib import Path
from tqdm import tqdm

# =========================
# CONFIG
# =========================

RAW_ROOT = Path("./DeepFakeFace")
OUTPUT_ROOT = Path("./DeepFakeFace_binary")

FAKE_DIRS = [
    RAW_ROOT / "inpainting",
    RAW_ROOT / "insight",
    RAW_ROOT / "text2img",
]

REAL_DIRS = [
    RAW_ROOT / "wiki",
]

SEED = 42

# Use balanced dataset:
# 30,000 real images + 30,000 fake images
MAX_REAL = 30000
MAX_FAKE = 30000

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# =========================
# FUNCTIONS
# =========================

def collect_images(folders):
    images = []

    for folder in folders:
        if not folder.exists():
            print(f"[WARNING] Folder not found: {folder}")
            continue

        for path in folder.rglob("*"):
            if path.suffix.lower() in IMAGE_EXTENSIONS:
                images.append(path)

    return images


def split_images(images):
    random.shuffle(images)

    n = len(images)
    train_end = int(n * TRAIN_RATIO)
    val_end = train_end + int(n * VAL_RATIO)

    train = images[:train_end]
    val = images[train_end:val_end]
    test = images[val_end:]

    return train, val, test


def copy_images(images, split_name, class_name):
    output_dir = OUTPUT_ROOT / split_name / class_name
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, src in enumerate(tqdm(images, desc=f"Copying {split_name}/{class_name}")):
        # Add index prefix to avoid duplicate filenames
        dst = output_dir / f"{class_name}_{i:06d}{src.suffix.lower()}"
        shutil.copy2(src, dst)


def main():
    random.seed(SEED)

    print("[INFO] Collecting fake images...")
    fake_images = collect_images(FAKE_DIRS)

    print("[INFO] Collecting real images...")
    real_images = collect_images(REAL_DIRS)

    print(f"[INFO] Total fake images found: {len(fake_images)}")
    print(f"[INFO] Total real images found: {len(real_images)}")

    if len(fake_images) == 0 or len(real_images) == 0:
        raise RuntimeError("No images found. Please check your folder paths.")

    # Balance classes
    random.shuffle(fake_images)
    random.shuffle(real_images)

    fake_images = fake_images[:MAX_FAKE]
    real_images = real_images[:MAX_REAL]

    print(f"[INFO] Using fake images: {len(fake_images)}")
    print(f"[INFO] Using real images: {len(real_images)}")

    fake_train, fake_val, fake_test = split_images(fake_images)
    real_train, real_val, real_test = split_images(real_images)

    print("\n[INFO] Final split:")
    print(f"Train fake: {len(fake_train)}")
    print(f"Val fake:   {len(fake_val)}")
    print(f"Test fake:  {len(fake_test)}")
    print(f"Train real: {len(real_train)}")
    print(f"Val real:   {len(real_val)}")
    print(f"Test real:  {len(real_test)}")

    copy_images(fake_train, "train", "fake")
    copy_images(fake_val, "val", "fake")
    copy_images(fake_test, "test", "fake")

    copy_images(real_train, "train", "real")
    copy_images(real_val, "val", "real")
    copy_images(real_test, "test", "real")

    print("\n[DONE] Dataset created at:")
    print(OUTPUT_ROOT.resolve())


if __name__ == "__main__":
    main()