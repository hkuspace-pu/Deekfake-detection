import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def apply_fourier_transform(input_dir, output_dir, size=(256, 256)):
    """
    Convert original RGB images into DFT magnitude spectrum images.

    This function is used to create the frequency-domain dataset.
    The output images will later be used as input for DFT-ResNet50.
    """

    # Convert string paths into Path objects.
    # Path makes folder and file handling easier.
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    # Create output directory if it does not exist.
    # Example: DeepFakeFace_fft/train/fake
    output_dir.mkdir(parents=True, exist_ok=True)

    # Recursively collect all supported image files in the input folder.
    image_paths = [
        p for p in input_dir.rglob("*")
        if p.suffix.lower() in IMAGE_EXTENSIONS
    ]

    print(f"[INFO] Processing: {input_dir}")
    print(f"[INFO] Output to:   {output_dir}")
    print(f"[INFO] Images found: {len(image_paths)}")

    for img_path in tqdm(image_paths):

        # Save all converted outputs as PNG files.
        output_path = output_dir / f"{img_path.stem}.png"

        # Skip image if it has already been converted.
        # This allows the script to resume if it was interrupted.
        if output_path.exists():
            continue

        try:
            # Read image as a colour image.
            # This keeps the image as 3-channel BGR format for ResNet50 compatibility.
            img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)

            if img is None:
                print(f"[WARNING] Cannot read image: {img_path}")
                continue

            # Resize all images to a fixed size.
            # This makes the generated DFT images consistent.
            img = cv2.resize(img, size)

            # Split the colour image into B, G, R channels.
            # 2D DFT will be applied to each channel separately.
            channels = cv2.split(img)

            dft_channels = []

            for ch in channels:
                # Convert channel to float32 because cv2.dft requires float input.
                ch = np.float32(ch)

                # Apply 2D Discrete Fourier Transform.
                # The output contains two parts: real and imaginary components.
                fourier = cv2.dft(
                    ch,
                    flags=cv2.DFT_COMPLEX_OUTPUT
                )

                # Shift the low-frequency component to the centre.
                # This makes the magnitude spectrum easier to visualise.
                fourier_shift = np.fft.fftshift(fourier)

                # Calculate magnitude spectrum:
                # magnitude = sqrt(real^2 + imaginary^2)
                # This represents the strength of each frequency component.
                magnitude = cv2.magnitude(
                    fourier_shift[:, :, 0],
                    fourier_shift[:, :, 1]
                )

                # Apply logarithmic scaling.
                # Raw magnitude values can be very large, so log scaling
                # makes the spectrum easier to display and learn from.
                magnitude = np.log1p(magnitude)

                # Normalize the magnitude values to 0–255.
                # This converts the spectrum into standard image intensity range.
                magnitude = cv2.normalize(
                    magnitude,
                    None,
                    0,
                    255,
                    cv2.NORM_MINMAX
                )

                # Convert to unsigned 8-bit integer so it can be saved as PNG.
                magnitude = np.uint8(magnitude)

                # Store the transformed channel.
                dft_channels.append(magnitude)

            # Merge the transformed channels back into a 3-channel image.
            # This is useful because ResNet50 expects 3-channel input.
            dft_img = cv2.merge(dft_channels)

            # Save the DFT magnitude spectrum image.
            cv2.imwrite(str(output_path), dft_img)

        except Exception as e:
            print(f"[ERROR] Failed: {img_path}")
            print(e)
            continue


def convert_full_dataset(src_root, out_root):
    """
    Convert the whole DeepFakeFace dataset into frequency-domain format.

    The original folder structure is preserved:
    train/fake, train/real, val/fake, val/real, test/fake, test/real
    """

    src_root = Path(src_root)
    out_root = Path(out_root)

    # Dataset splits used in model training.
    splits = ["train", "val", "test"]

    # Binary classification classes.
    classes = ["fake", "real"]

    for split in splits:
        for cls in classes:
            # Example input:
            # DeepFakeFace_binary/train/fake
            input_dir = src_root / split / cls

            # Example output:
            # DeepFakeFace_fft/train/fake
            output_dir = out_root / split / cls

            # Convert all images in this class folder.
            apply_fourier_transform(input_dir, output_dir)


if __name__ == "__main__":
    # Original spatial-domain dataset.
    src_root = r"C:\Users\kcwong6\Downloads\DeepFakeFace_binary"

    # Output frequency-domain dataset.
    out_root = r"C:\Users\kcwong6\Downloads\DeepFakeFace_binary\DeepFakeFace_fft"

    # Start conversion.
    convert_full_dataset(src_root, out_root)