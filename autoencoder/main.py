from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import numpy as np
from src.utils import load_train_paths, CableDataset
from src.utils import DATA_PATH
from src.model import ConvolutionalAutoencoder
from src.model import AnomalyDetectionModel

ADM = AnomalyDetectionModel()
# ADM.train_model(epochs=20, lr=1e-3, save_path="autoencoder_cable.pth")

ADM.load_model("/home/miloush7/Documents/agh/cable_defect_detection/autoencoder/src/model/autoencoder_cable.pth")
ADM.evaluate_model()