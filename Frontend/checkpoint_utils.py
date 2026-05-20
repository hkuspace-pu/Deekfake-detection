from pathlib import Path
import torch
import streamlit as st

from config import DEVICE, NUM_CLASSES
from model_architectures import (
    Baseline2DCNN,
    SpatialFrequencyFusionResNet50,
    build_custom_resnet50_model,
    build_fft_stage2_resnet50_model,
)


def extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    return checkpoint


def guess_model_type_from_state_dict(state_dict):
    keys = list(state_dict.keys())

    if any(k.startswith("spatial_branch.") for k in keys):
        return "Spatial-Frequency Fusion"

    if any(k.startswith("features.") for k in keys):
        return "Stage 1 Baseline 2D CNN"

    if "fc.weight" in keys and "fc.bias" in keys and any(k.startswith("layer4.") for k in keys):
        return "FFT ResNet50 Stage 2"

    if any(k.startswith("fc.0.") for k in keys) and any(k.startswith("layer4.") for k in keys):
        return "Custom ResNet50"

    return "Unknown"


@st.cache_data(show_spinner=False)
def scan_model_folder(folder_path):
    folder = Path(folder_path)

    if not folder.exists() or not folder.is_dir():
        return []

    records = []
    for pth_path in sorted(folder.glob("*.pth")):
        try:
            checkpoint = torch.load(str(pth_path), map_location="cpu")
            state_dict = extract_state_dict(checkpoint)
            detected_type = guess_model_type_from_state_dict(state_dict)
            records.append({
                "name": pth_path.name,
                "path": str(pth_path),
                "type": detected_type,
                "size_mb": pth_path.stat().st_size / (1024 * 1024),
                "status": "OK",
            })
        except Exception as e:
            records.append({
                "name": pth_path.name,
                "path": str(pth_path),
                "type": "Error",
                "size_mb": pth_path.stat().st_size / (1024 * 1024),
                "status": str(e),
            })
    return records


@st.cache_resource
def load_model_from_path(model_path, detected_type):
    checkpoint = torch.load(model_path, map_location=DEVICE)
    state_dict = extract_state_dict(checkpoint)

    if detected_type == "Stage 1 Baseline 2D CNN":
        model = Baseline2DCNN(num_classes=NUM_CLASSES)
    elif detected_type == "FFT ResNet50 Stage 2":
        model = build_fft_stage2_resnet50_model()
    elif detected_type == "Spatial-Frequency Fusion":
        model = SpatialFrequencyFusionResNet50(num_classes=NUM_CLASSES)
    elif detected_type == "Custom ResNet50":
        model = build_custom_resnet50_model()
    else:
        raise RuntimeError(f"Unknown model architecture: {detected_type}")

    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model


def resolve_classifier_mode(record):
    detected_type = record["type"]
    name = record["name"].lower()

    if detected_type != "Custom ResNet50":
        return detected_type

    if "dft" in name or "fft" in name or "frequency" in name:
        return "Frequency-domain / DFT-ResNet50"

    return "Spatial-domain ResNet50"
