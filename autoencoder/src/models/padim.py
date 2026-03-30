import torchvision.models as models
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from src.utils.config import IMG_SIZE


class PaDiMModel:
    def __init__(self, device, feature_dim=100):

        self.device = device
        self.feature_dim = feature_dim

        backbone = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1
        )

        self.layer1 = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1
        )
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3

        for p in self.layer1.parameters():
            p.requires_grad = False

        for p in self.layer2.parameters():
            p.requires_grad = False

        for p in self.layer3.parameters():
            p.requires_grad = False

        self.mean = None
        self.inv_cov = None
        self.selected_idx = None

        torch.manual_seed(42)

    # --------------------------------------------------
    # FEATURE EXTRACTION
    # --------------------------------------------------

    def extract_features(self, loader):

        feats = []

        with torch.no_grad():
            for imgs in loader:
                imgs = imgs.to(self.device)

                f1 = self.layer1(imgs)
                f2 = self.layer2(f1)
                f3 = self.layer3(f2)

                target_size = f1.shape[-2:]

                f2 = F.interpolate(f2, size=target_size,
                                   mode="bilinear",
                                   align_corners=False)
                f3 = F.interpolate(f3, size=target_size,
                                   mode="bilinear",
                                   align_corners=False)

                emb = torch.cat([f1, f2, f3], dim=1)
                feats.append(emb.cpu())

        return torch.cat(feats, dim=0)

    # --------------------------------------------------
    # FIT
    # --------------------------------------------------

    def fit(self, train_loader):

        print("Training PaDiM...")

        feats = self.extract_features(train_loader)
        N, C, H, W = feats.shape

        # losowy wybór kanałów
        idx = torch.randperm(C)[:self.feature_dim]
        self.selected_idx = idx

        feats = feats[:, idx, :, :]
        feats = feats.permute(0, 2, 3, 1).numpy().astype(np.float32)

        self.mean = np.zeros((H, W, self.feature_dim), dtype=np.float32)
        self.inv_cov = np.zeros(
            (H, W, self.feature_dim, self.feature_dim),
            dtype=np.float32
        )

        identity = np.eye(self.feature_dim, dtype=np.float32) * 0.01

        for h in range(H):
            for w in range(W):

                vec = feats[:, h, w, :]  # (N, D)

                mean = np.mean(vec, axis=0)
                cov = np.cov(vec, rowvar=False).astype(np.float32)

                cov += identity

                self.mean[h, w] = mean
                self.inv_cov[h, w] = np.linalg.inv(cov)

        print("PaDiM fitted")

    # --------------------------------------------------
    # ANOMALY MAP
    # --------------------------------------------------

    def anomaly_map(self, img_tensor):

        with torch.no_grad():

            f1 = self.layer1(img_tensor)
            f2 = self.layer2(f1)
            f3 = self.layer3(f2)

            target_size = f1.shape[-2:]

            f2 = F.interpolate(f2, size=target_size,
                               mode="bilinear",
                               align_corners=False)
            f3 = F.interpolate(f3, size=target_size,
                               mode="bilinear",
                               align_corners=False)

            emb = torch.cat([f1, f2, f3], dim=1)

        emb = emb[0, self.selected_idx].cpu().numpy().astype(np.float32)

        C, H, W = emb.shape
        amap = np.zeros((H, W), dtype=np.float32)

        for h in range(H):
            for w in range(W):

                diff = emb[:, h, w] - self.mean[h, w]

                # stabilna wersja Mahalanobisa
                dist = diff @ self.inv_cov[h, w] @ diff
                dist = np.maximum(dist, 0)  # zabezpieczenie
                amap[h, w] = np.sqrt(dist)

        # resize do obrazu wejściowego
        amap = cv2.resize(amap, (IMG_SIZE, IMG_SIZE))

        # normalizacja
        amap = (amap - amap.min()) / (
            amap.max() - amap.min() + 1e-8
        )

        return amap

    # --------------------------------------------------
    # IMAGE SCORE (lepszy niż max)
    # --------------------------------------------------

    def image_score(self, anomaly_map):

        flat = anomaly_map.flatten()

        # stabilniejszy niż max
        score = np.mean(np.sort(flat)[-50:])

        return score