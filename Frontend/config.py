import torch

NUM_CLASSES = 2
CLASS_NAMES = ["fake", "real"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_MODEL_FOLDER = r"C:\Users\kcwong6\Downloads\models"
