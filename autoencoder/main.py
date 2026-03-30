from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from src.utils import load_train_paths, CableDataset
from src.utils import DATA_PATH
from src.models import AnomalyDetectionModel

AE = AnomalyDetectionModel(model_type="ae", use_clahe=True, use_blur=True)
# # AE.train_model(epochs=80, lr=1e-3, save_path="autoencoder/src/model/autoencoder_cable_v4.pth")

AE.load_model("autoencoder/src/models/trained/autoencoder_cable_dropout.pth")
_, threshold = AE.evaluate_model("ssim")
AE.threshold = threshold  # Ustawiamy optymalny próg

# # AE.evaluate_model("l1_mean")
# # AE.evaluate_model("l1_top1")
# # AE.evaluate_model("mse_mean")
# # AE.evaluate_model_top_k()

test_image = "./data/test/cable_swap/005.png"

# Generujemy wykres
AE.visualize_defect(test_image)

AE.evaluate_segmentation_iou(percentile=0.95)

# ============================

PADIM = AnomalyDetectionModel(model_type="padim", use_clahe=False, use_blur=False)

# Trening (czyli fit Gaussa)
PADIM.train_model()

# Image-level
PADIM.evaluate_model()

# Pixel-level
PADIM.evaluate_segmentation_iou(percentile=0.95)

# Wizualizacja tej samej próbki
PADIM.visualize_defect(test_image)