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
from src.model import ConvolutionalAutoencoder
from src.model import AnomalyDetectionModel

ADM = AnomalyDetectionModel()
# ADM.train_model(epochs=80, lr=1e-3, save_path="autoencoder/src/model/autoencoder_cable_dropout.pth")

ADM.load_model("autoencoder/src/model/autoencoder_cable_dropout.pth")
# ADM.evaluate_model("ssim")
# ADM.evaluate_model("l1_mean")
# ADM.evaluate_model("l1_top1")
# ADM.evaluate_model("mse_mean")
# ADM.evaluate_model_top_k()


test_image = "./data/test/bent_wire/011.png"

# Generujemy wykres
ADM.visualize_defect(test_image)