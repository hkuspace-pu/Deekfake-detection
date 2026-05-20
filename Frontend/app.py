import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from PIL import Image
import streamlit as st

from config import DEFAULT_MODEL_FOLDER
from checkpoint_utils import load_model_from_path, resolve_classifier_mode, scan_model_folder
from image_processing import compute_benford_dct_score, generate_dft_image, get_preprocess_config
from inference import (
    combined_interpretation,
    interpret_benford_score,
    interpret_result,
    predict_fusion,
    predict_single_input,
)
from ui_components import (
    show_benford_chart,
    show_benford_result,
    show_prediction_result,
    show_sidebar_logo,
)


st.set_page_config(
    page_title="Folder Model DeepFake Detector",
    page_icon="🧠",
    layout="wide",
)

st.title("DeepFake Image Detection System")
st.write(
    "Step 1: Enter the folder path containing your `.pth` files. "
    "Step 2: Select which classifier to use. "
    "Step 3: Upload an image for detection."
)

show_sidebar_logo()

st.sidebar.header("Model Folder")

model_folder = st.sidebar.text_input(
    "Enter folder path containing .pth files",
    value=DEFAULT_MODEL_FOLDER,
)

refresh = st.sidebar.button("Refresh model list")

if refresh:
    st.cache_data.clear()
    st.cache_resource.clear()

model_records = scan_model_folder(model_folder)

selected_model_record = None

if len(model_records) == 0:
    st.sidebar.warning("No `.pth` files found in this folder, or the path is invalid.")
else:
    model_options = [
        f"{record['name']} | {record['type']} | {record['size_mb']:.1f} MB"
        for record in model_records
    ]

    selected_option = st.sidebar.selectbox("Select classifier model", model_options)
    selected_index = model_options.index(selected_option)
    selected_model_record = model_records[selected_index]

    st.sidebar.success("Model selected")
    st.sidebar.write(f"**File:** {selected_model_record['name']}")
    st.sidebar.write(f"**Detected architecture:** {selected_model_record['type']}")
    st.sidebar.write(f"**Size:** {selected_model_record['size_mb']:.1f} MB")

enable_benford = st.sidebar.checkbox("Enable Benford DCT forensic check", value=True)
show_benford_graph = st.sidebar.checkbox("Show Benford distribution graph", value=False)

uploaded_image = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
)


if selected_model_record is None:
    st.info("Please enter a valid model folder path containing `.pth` files.")

elif selected_model_record["type"] == "Error":
    st.error("This selected `.pth` file cannot be scanned.")
    st.write(selected_model_record["status"])

elif uploaded_image is None:
    st.success("Model selected. Now upload an image for detection.")

else:
    try:
        detected_type = selected_model_record["type"]
        classifier_mode = resolve_classifier_mode(selected_model_record)

        model_path = selected_model_record["path"]
        model = load_model_from_path(model_path, detected_type)

        original_image = Image.open(uploaded_image).convert("RGB")

        st.info(f"Selected file: **{selected_model_record['name']}**")
        st.info(f"Detected architecture: **{detected_type}**")
        st.info(f"Classifier mode: **{classifier_mode}**")

        if classifier_mode == "Stage 1 Baseline 2D CNN":
            display_image = original_image
            fake_prob, real_prob = predict_single_input(
                model,
                original_image,
                "Stage 1 Baseline 2D CNN",
            )

        elif classifier_mode == "Spatial-domain ResNet50":
            display_image = original_image
            fake_prob, real_prob = predict_single_input(
                model,
                original_image,
                "Spatial-domain ResNet50",
            )

        elif classifier_mode in ["Frequency-domain / DFT-ResNet50", "FFT ResNet50 Stage 2"]:
            cfg = get_preprocess_config(classifier_mode)
            display_image = generate_dft_image(original_image, img_size=cfg["img_size"])
            fake_prob, real_prob = predict_single_input(model, display_image, classifier_mode)

        elif classifier_mode == "Spatial-Frequency Fusion":
            fake_prob, real_prob, display_image = predict_fusion(model, original_image)

        else:
            raise ValueError(f"Unknown classifier mode: {classifier_mode}")

        label, risk = interpret_result(real_prob)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Uploaded Image")
            st.image(original_image, use_container_width=True)

        with col2:
            if classifier_mode in ["Stage 1 Baseline 2D CNN", "Spatial-domain ResNet50"]:
                st.subheader("Model Input")
            elif classifier_mode == "Spatial-Frequency Fusion":
                st.subheader("Generated DFT Image for Frequency Branch")
            else:
                st.subheader("Generated DFT Image")

            st.image(display_image, use_container_width=True)

        show_prediction_result(real_prob, fake_prob, label, risk)

        if enable_benford:
            benford_div, actual_dist, benford_dist = compute_benford_dct_score(original_image)
            benford_label, benford_explanation = interpret_benford_score(benford_div)

            show_benford_result(benford_div, benford_label, benford_explanation)

            st.write("### Combined Interpretation")
            st.write(combined_interpretation(real_prob, benford_div))

            if show_benford_graph:
                show_benford_chart(actual_dist, benford_dist)

        st.warning(
            "This result is generated by a machine learning model and should not be treated as absolute forensic proof."
        )

    except RuntimeError as e:
        st.error("Model loading or prediction failed.")
        st.write(
            "Most likely reason: this `.pth` architecture is not supported, "
            "or the selected classifier mode does not match the model."
        )
        st.exception(e)

    except Exception as e:
        st.error("Unexpected error occurred.")
        st.exception(e)
