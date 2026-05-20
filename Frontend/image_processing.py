import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from config import DEVICE


def generate_dft_image(pil_image, img_size):
    img_rgb = np.array(pil_image.convert("RGB"))
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    img_bgr = cv2.resize(img_bgr, (img_size, img_size))

    channels = cv2.split(img_bgr)
    dft_channels = []

    for ch in channels:
        ch = np.float32(ch)
        dft = cv2.dft(ch, flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        magnitude = cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])
        magnitude = np.log1p(magnitude)
        magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
        magnitude = np.uint8(magnitude)
        dft_channels.append(magnitude)

    dft_bgr = cv2.merge(dft_channels)
    dft_rgb = cv2.cvtColor(dft_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(dft_rgb)


def get_preprocess_config(model_type):
    if model_type == "Stage 1 Baseline 2D CNN":
        return {"img_size": 224, "mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]}

    if model_type == "FFT ResNet50 Stage 2":
        return {"img_size": 224, "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}

    return {"img_size": 256, "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}


def get_transform(model_type):
    cfg = get_preprocess_config(model_type)
    return transforms.Compose([
        transforms.Resize((cfg["img_size"], cfg["img_size"])),
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg["mean"], std=cfg["std"]),
    ])


def preprocess_image(pil_image, model_type):
    image = pil_image.convert("RGB")
    tensor = get_transform(model_type)(image)
    return tensor.unsqueeze(0).to(DEVICE)


def compute_benford_dct_score(pil_image):
    img_gray = np.array(pil_image.convert("L")).astype(np.float32)

    h, w = img_gray.shape
    if h % 2 != 0:
        img_gray = img_gray[:-1, :]
    if w % 2 != 0:
        img_gray = img_gray[:, :-1]

    dct = cv2.dct(img_gray)
    coeffs = np.abs(dct.flatten())
    coeffs = coeffs[coeffs > 0]

    if len(coeffs) == 0:
        return None, None, None

    first_digits = np.floor(coeffs / (10 ** np.floor(np.log10(coeffs)))).astype(int)
    first_digits = first_digits[(first_digits >= 1) & (first_digits <= 9)]

    if len(first_digits) == 0:
        return None, None, None

    counts = np.bincount(first_digits, minlength=10)[1:10]
    actual_dist = counts / np.sum(counts)

    digits = np.arange(1, 10)
    benford_dist = np.log10(1 + 1 / digits)

    benford_div = np.sum((actual_dist - benford_dist) ** 2 / benford_dist)

    return benford_div, actual_dist, benford_dist
