import torch
import torch.nn as nn
from torchvision.models import resnet50
from config import NUM_CLASSES


class Baseline2DCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def build_custom_resnet50_model():
    model = resnet50(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.ReLU(inplace=True),
        nn.BatchNorm1d(512),
        nn.Dropout(0.5),
        nn.Linear(512, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(256, NUM_CLASSES),
    )
    return model


def build_fft_stage2_resnet50_model():
    model = resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model


class SpatialFrequencyFusionResNet50(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.spatial_branch = resnet50(weights=None)
        self.frequency_branch = resnet50(weights=None)

        spatial_in_features = self.spatial_branch.fc.in_features
        frequency_in_features = self.frequency_branch.fc.in_features

        self.spatial_branch.fc = nn.Identity()
        self.frequency_branch.fc = nn.Identity()

        self.classifier = nn.Sequential(
            nn.Linear(spatial_in_features + frequency_in_features, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, rgb_img, dft_img):
        spatial_features = self.spatial_branch(rgb_img)
        frequency_features = self.frequency_branch(dft_img)
        fused_features = torch.cat([spatial_features, frequency_features], dim=1)
        return self.classifier(fused_features)
