import torch

from image_processing import generate_dft_image, get_preprocess_config, preprocess_image


def predict_single_input(model, pil_image, model_type):
    input_tensor = preprocess_image(pil_image, model_type)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)[0]

    fake_prob = probs[0].item()
    real_prob = probs[1].item()
    return fake_prob, real_prob


def predict_fusion(model, original_image):
    cfg = get_preprocess_config("Spatial-Frequency Fusion")
    dft_image = generate_dft_image(original_image, img_size=cfg["img_size"])

    rgb_tensor = preprocess_image(original_image, "Spatial-Frequency Fusion")
    dft_tensor = preprocess_image(dft_image, "Spatial-Frequency Fusion")

    with torch.no_grad():
        outputs = model(rgb_tensor, dft_tensor)
        probs = torch.softmax(outputs, dim=1)[0]

    fake_prob = probs[0].item()
    real_prob = probs[1].item()
    return fake_prob, real_prob, dft_image


def interpret_result(real_prob):
    if real_prob >= 0.90:
        return "Most likely real", "Low fake risk"
    if real_prob >= 0.70:
        return "Likely real", "Moderate-low fake risk"
    if real_prob >= 0.55:
        return "Maybe real", "Uncertain"
    if real_prob >= 0.45:
        return "Uncertain", "Borderline"
    if real_prob >= 0.30:
        return "Maybe fake", "Moderate fake risk"
    if real_prob >= 0.10:
        return "Likely fake", "High fake risk"
    return "Most likely fake", "Very high fake risk"


def interpret_benford_score(benford_div):
    if benford_div is None:
        return "Unavailable", "Could not calculate Benford score."

    if benford_div < 0.005:
        return "Low forensic anomaly", "The image follows Benford distribution closely."
    if benford_div < 0.015:
        return "Moderate forensic anomaly", "The image slightly deviates from Benford distribution."

    return "High forensic anomaly", "The image strongly deviates from Benford distribution and may be suspicious."


def combined_interpretation(real_prob, benford_div):
    if benford_div is None:
        return "Benford analysis was unavailable. Please rely on the model result only."

    if real_prob >= 0.70 and benford_div >= 0.015:
        return "The model leans toward real, but Benford analysis shows high forensic anomaly. Manual review is recommended."

    if real_prob < 0.30 and benford_div >= 0.015:
        return "Both the model prediction and Benford analysis suggest suspicious characteristics."

    if real_prob >= 0.70 and benford_div < 0.005:
        return "Both the model prediction and Benford analysis support a likely real result."

    if real_prob < 0.30 and benford_div < 0.005:
        return "The model predicts fake, but Benford analysis does not show strong anomaly. Further manual inspection is recommended."

    return "The result is mixed or uncertain. Additional analysis is recommended."
