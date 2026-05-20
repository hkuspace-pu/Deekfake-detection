# DeepFake Streamlit Modular App

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Structure

- `app.py` — main Streamlit UI
- `config.py` — global settings
- `model_architectures.py` — model classes
- `checkpoint_utils.py` — `.pth` scanning, detection, loading
- `image_processing.py` — DFT generation, preprocessing, Benford DCT
- `inference.py` — prediction and interpretation
- `ui_components.py` — UI components

## Supported `.pth`

Dataset - [ArtiFact: Real and Fake Image Dataset](https://www.kaggle.com/datasets/awsaf49/artifact-dataset)
- `best_rgb_2dcnn_stage1.pth` 
- `best_fft_resnet50_stage2.pth`
- `best_fft_resnet50_stage3_finetuned.pth`

Dataset - [DeepFakeFace](https://huggingface.co/datasets/OpenRL/DeepFakeFace/tree/main)
- `spatial_resnet50_final_state_dict.pth`
- `dft_resnet50_final_state_dict.pth`
- `spatial_frequency_fusion_final_state_dict.pth`
