import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import torchvision.transforms as T
from scipy.ndimage import gaussian_filter
import numpy as np
import cv2
from PIL import Image
import os

class PaDiM_v2(nn.Module):
    def __init__(self, n_features=100, device='cpu'):
        super().__init__()

        self.n_features = n_features
        self.device = device

        self.model = self.load_backbone()
        self.model.to(self.device)
        self.model.eval()

        self.features = {}
        self.register_hooks()

        self.threshold = None
        self.mean = None
        self.cov_inv = None
        self.idx = None 

    def load_backbone(self):
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        return model
    
    def save_parameters(self, path):
        torch.save({
            "mean": self.mean,
            "cov_inv": self.cov_inv,
            "idx": self.idx,
        }, path)

    def load_parameters(self, path):
        data = torch.load(path, map_location=self.device)
        self.mean = data["mean"].to(self.device)
        self.cov_inv = data["cov_inv"].to(self.device)
        self.idx = data["idx"]

    def hook(self, idx):
        def fn(module, input, output):
            self.features[idx] = output
        return fn

    def register_hooks(self):
        self.model.layer1.register_forward_hook(self.hook('layer1'))
        self.model.layer2.register_forward_hook(self.hook('layer2'))
        self.model.layer3.register_forward_hook(self.hook('layer3'))

    def extract_features(self, x):
        self.features = {}
        _ = self.model(x)

        f1 = self.features['layer1']
        f2 = self.features['layer2']
        f3 = self.features['layer3']

        f2 = F.interpolate(f2, size=f1.shape[-2:], mode="bilinear", align_corners=False)
        f3 = F.interpolate(f3, size=f1.shape[-2:], mode="bilinear", align_corners=False)

        x = torch.cat([f1, f2, f3], dim=1)

        return x

    def fit(self, dataloader):
        embeddings = []

        for x in dataloader:
            x = x.to(self.device)

            with torch.no_grad():
                feat = self.extract_features(x)
                B, C, H, W = feat.shape

                feat = feat.reshape(B, C, H * W)

                embeddings.append(feat)

        embeddings = torch.cat(embeddings, dim=0)  # (N, C, HW)

        # losowy wybór feature'ów
        C = embeddings.shape[1]
        self.idx = torch.randperm(C)[:self.n_features]
        embeddings = embeddings[:, self.idx, :] # wybranie kanałow (C)

        N, C, HW = embeddings.shape

        self.mean = []
        self.cov_inv = []
        eps = 1e-6

        for i in range(HW):
            patch = embeddings[:, :, i].detach().cpu()  # [N,C]

            mu = patch.mean(dim=0)
            x = patch - mu

            cov = (x.T @ x) / (N - 1)
            cov = cov + eps * torch.eye(C)

            cov_inv = torch.linalg.pinv(cov)

            self.mean.append(mu)
            self.cov_inv.append(cov_inv)

        self.mean = torch.stack(self.mean).to(self.device)
        self.cov_inv = torch.stack(self.cov_inv).to(self.device)

    def predict(self, x):
        x = x.to(self.device)

        with torch.no_grad():
            feat = self.extract_features(x)

        feat = feat[:, self.idx, :, :]  # [B,C,H,W]
        B, C, H, W = feat.shape

        scores = torch.zeros((B, H, W), device=self.device)

        for i in range(H * W):
            h = i // W
            w = i % W

            f = feat[:, :, h, w]  # [B,C]
            mu = self.mean[i]
            cov_inv = self.cov_inv[i]

            diff = f - mu
            left = diff @ cov_inv
            dist = (left * diff).sum(dim=1)

            scores[:, h, w] = dist

        scores = scores.unsqueeze(1)
        scores = F.interpolate(scores, size=x.shape[-2:], mode='bilinear', align_corners=False)
        scores = scores.squeeze(1)

        return scores.detach().cpu().numpy()

def remove_small_objects(mask, min_area=1000):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    
    output = np.zeros_like(mask)
    
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            output[labels == i] = 1
    return output

def predict(image: np.ndarray) -> np.ndarray:
    """
    Args:
        image: tablica NumPy, kształt (H, W, 3), dtype uint8, RGB

    Returns:
        Maska binarna, kształt (H, W), dtype uint8, wartości 0 lub 255
        255 = wada, 0 = brak wady
    """
    model = PaDiM_v2(n_features=100)
    model.load_parameters("/app/ingested_program/padim_resnet_100_padim_v2.pth") 
    model.eval()
    transform = T.Compose([T.Resize((256, 256)),
                        T.ToTensor()])
    
    image = Image.fromarray(image)
    img = transform(image).unsqueeze(0)
    scores = model.predict(img).squeeze(0)
    scores = cv2.resize(scores, (1024, 1024), interpolation=cv2.INTER_NEAREST)

    th = scores.mean() + 2.5 * scores.std()
    mask = (scores > th).astype(np.uint8) * 255
    mask = gaussian_filter(mask,sigma=4)
    kernel = np.ones((7,7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = remove_small_objects(mask, 5000)

    return mask