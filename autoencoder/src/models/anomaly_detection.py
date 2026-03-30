from pathlib import Path
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
from pytorch_msssim import ssim
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, precision_score, recall_score, confusion_matrix
from src.models.padim import PaDiMModel
from src.utils.preprocessing import get_test_transform
from src.utils.config import IMG_SIZE
from src.utils import load_train_paths, load_test_paths, CableDataset
from src.utils import DATA_PATH
from .ae import ConvolutionalAutoencoder
from .padim import PaDiMModel

class AnomalyDetectionModel:
    def __init__(self, model_type="ae", use_clahe=True, use_blur=True):

        self.model_type = model_type
        self.train_paths = load_train_paths(DATA_PATH)
        self.test_img_paths, self.test_labels = load_test_paths(DATA_PATH)

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        print(f'Używane urządzenie: {self.device}')

        if model_type == "padim":

            train_transform = get_test_transform(
                IMG_SIZE,
                use_clahe=False,
                use_blur=False,
                imagenet_norm=True
            )

            test_transform = get_test_transform(
                IMG_SIZE,
                use_clahe=False,
                use_blur=False,
                imagenet_norm=True
            )

            self.model = PaDiMModel(self.device)

        elif model_type == "ae":

            train_transform = get_test_transform(
                IMG_SIZE,
                use_clahe=use_clahe,
                use_blur=use_blur,
                imagenet_norm=False
            )

            test_transform = get_test_transform(
                IMG_SIZE,
                use_clahe=use_clahe,
                use_blur=use_blur,
                imagenet_norm=False
            )

            self.model = ConvolutionalAutoencoder().to(self.device)

        else:
            raise ValueError("model_type must be 'ae' or 'padim'")

        self.train_dataset = CableDataset(
            self.train_paths,
            transform=train_transform
        )

        self.test_dataset = CableDataset(
            self.test_img_paths,
            transform=test_transform
        )

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=32,
            shuffle=(model_type == "ae")
        )

        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=32,
            shuffle=False
        )

        self.threshold = 0.5
        self.use_clahe = use_clahe
        self.use_blur = use_blur

    def train_model(self, epochs=20, lr=1e-3, save_path="new_cable.pth"):

        if self.model_type == "padim":
            self.model.fit(self.train_loader)
            return
        torch.manual_seed(42)  # Dla powtarzalności wyników

        # loss_fn = nn.MSELoss()
        # Wykorzystanie SSIM jako funkcji straty
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        train_losses = []

        for epoch in range(epochs):
            self.model.train()
            epoch_loss = []
            
            for imgs in self.train_loader:
                imgs = imgs.to(self.device)
                
                # Forward pass
                x_hat = self.model(imgs)
                # loss = loss_fn(x_hat, imgs)
                loss = 1 - ssim(x_hat, imgs, data_range=1.0, size_average=True)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss.append(loss.item())
                
            avg_loss = np.mean(epoch_loss)
            train_losses.append(avg_loss)

            print(f'Epoka {epoch+1:2d}/{epochs} | Train Loss (1 - SSIM): {avg_loss:.5f}')

        self.save_model(save_path)

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        print(f"Wagi modelu zostały pomyślnie zapisane do: {path}")
    
    def load_model(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
        self.model.eval()
        print(f"Model został pomyślnie załadowany z: {path}")
    
    def evaluate_model(self, mode="ssim"):
        if self.model_type == "padim":

            image_scores = []

            for imgs in self.test_loader:
                imgs = imgs.to(self.device)

                for i in range(imgs.size(0)):

                    amap = self.model.anomaly_map(
                        imgs[i].unsqueeze(0)
                    )

                    flat = amap.flatten()
                    score = np.mean(np.sort(flat)[-50:])
                    image_scores.append(score)

            image_scores = np.array(image_scores)
            true_labels = np.array(self.test_labels)

            # ROC AUC
            auc = roc_auc_score(true_labels, image_scores)
            print(f"ROC AUC (PaDiM): {auc:.4f}")

            fpr, tpr, thresholds = roc_curve(true_labels, image_scores)
            optimal_idx = np.argmax(tpr - fpr)
            optimal_threshold = thresholds[optimal_idx]

            print(f"Optymalny próg (PaDiM): {optimal_threshold:.5f}")

            self.threshold = optimal_threshold

            return auc, optimal_threshold
        else:
            # Autoencoder
            self.model.eval()
            image_scores = []

            print(f"\nRozpoczynam ewaluację ({mode})...")

            with torch.no_grad():
                for imgs in self.test_loader:
                    imgs = imgs.to(self.device)
                    x_hat = self.model(imgs)

                    if mode == "ssim":
                        scores = 1 - ssim(x_hat, imgs, data_range=1.0, size_average=False)

                    elif mode == "l1_mean":
                        loss_map = torch.abs(x_hat - imgs)
                        scores = loss_map.mean(dim=(1,2,3))

                    elif mode == "l1_top1":
                        loss_map = torch.abs(x_hat - imgs)
                        loss_map = loss_map.mean(dim=1)
                        loss_flat = loss_map.view(loss_map.size(0), -1)
                        scores = torch.quantile(loss_flat, 0.99, dim=1)

                    elif mode == "mse_mean":
                        loss_map = (x_hat - imgs) ** 2
                        scores = loss_map.mean(dim=(1,2,3))

                    else:
                        raise ValueError("Nieznany tryb ewaluacji")

                    image_scores.extend(scores.cpu().numpy())

            image_scores = np.array(image_scores)
            true_labels = np.array(self.test_labels)

            # --- METRYKI ---
            auc = roc_auc_score(true_labels, image_scores)
            print(f"\nROC AUC: {auc:.4f}")

            fpr, tpr, thresholds = roc_curve(true_labels, image_scores)
            optimal_idx = np.argmax(tpr - fpr)
            optimal_threshold = thresholds[optimal_idx]
            print(f"Optymalny próg: {optimal_threshold:.5f}")

            predictions = (image_scores >= optimal_threshold).astype(int)

            acc = accuracy_score(true_labels, predictions)
            prec = precision_score(true_labels, predictions, zero_division=0)
            rec = recall_score(true_labels, predictions, zero_division=0)
            cm = confusion_matrix(true_labels, predictions)

            print(f"Accuracy:  {acc:.4f}")
            print(f"Precision: {prec:.4f}")
            print(f"Recall:    {rec:.4f}")

            print("\nMacierz pomyłek:")
            print("                Przewidziane OK | Przewidziane WADY")
            print(f"Rzeczywiste OK   |      {cm[0,0]:3d}       |       {cm[0,1]:3d}")
            print(f"Rzeczywiste WADY |      {cm[1,0]:3d}       |       {cm[1,1]:3d}")

            return auc, optimal_threshold
    
    def get_error_mask_from_tensor(self, img_tensor, percentile=0.95):
        with torch.no_grad():
            reconstruction = self.model(img_tensor)

        error_map = torch.abs(reconstruction - img_tensor)
        error_map = error_map.mean(dim=1).squeeze(0)

        # normalizacja
        error_map = (error_map - error_map.min()) / (error_map.max() - error_map.min() + 1e-8)

        # próg percentylowy
        threshold = torch.quantile(error_map.view(-1), percentile)
        mask = (error_map > threshold).float()

        return mask.cpu().numpy()

    def evaluate_segmentation_iou(self, percentile=0.95):

        ious = []

        if self.model_type == "padim":

            transform = get_test_transform(
                IMG_SIZE,
                use_clahe=False,
                use_blur=False,
                imagenet_norm=True
            )

            for img_path in self.test_img_paths:

                if Path(img_path).parent.name == "good":
                    continue

                img = Image.open(img_path).convert("RGB")
                img_tensor = transform(img).unsqueeze(0).to(self.device)

                amap = self.model.anomaly_map(img_tensor)

                threshold = np.percentile(amap, percentile * 100)
                pred_mask = (amap > threshold).astype(np.float32)

                gt_path = str(img_path) \
                    .replace("test", "ground_truth") \
                    .replace(".png", "_mask.png")

                if not Path(gt_path).exists():
                    continue

                gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
                gt_mask = cv2.resize(gt_mask, (IMG_SIZE, IMG_SIZE))
                gt_mask = (gt_mask > 0).astype(np.float32)

                intersection = np.sum(pred_mask * gt_mask)
                union = np.sum(pred_mask) + np.sum(gt_mask) - intersection

                if union > 0:
                    ious.append(intersection / union)

            mean_iou = np.mean(ious) if len(ious) > 0 else 0.0
            print(f"Mean IoU (PaDiM): {mean_iou:.4f}")
            return mean_iou

        # AUTOENCODER
        else:

            transform = transforms.Compose([
                transforms.Resize((IMG_SIZE, IMG_SIZE)),
                transforms.ToTensor(),
            ])

            for img_path in self.test_img_paths:

                if Path(img_path).parent.name == "good":
                    continue

                img = Image.open(img_path).convert("RGB")
                img_tensor = transform(img).unsqueeze(0).to(self.device)

                pred_mask = self.get_error_mask_from_tensor(
                    img_tensor,
                    percentile
                )

                gt_path = str(img_path) \
                    .replace("test", "ground_truth") \
                    .replace(".png", "_mask.png")

                if not Path(gt_path).exists():
                    continue

                gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
                gt_mask = cv2.resize(
                    gt_mask,
                    (pred_mask.shape[1], pred_mask.shape[0])
                )
                gt_mask = (gt_mask > 0).astype(np.float32)

                intersection = np.sum(pred_mask * gt_mask)
                union = np.sum(pred_mask) + np.sum(gt_mask) - intersection

                if union > 0:
                    ious.append(intersection / union)

            mean_iou = np.mean(ious) if len(ious) > 0 else 0.0
            print(f"Mean IoU (AE): {mean_iou:.4f}")
            return mean_iou
    
    def visualize_defect(self, image_path):

        if self.model_type == "padim":

            transform = get_test_transform(
                IMG_SIZE,
                use_clahe=False,
                use_blur=False,
                imagenet_norm=True
            )

        else:
            transform = get_test_transform(
                IMG_SIZE,
                use_clahe=self.use_clahe,
                use_blur=self.use_blur,
                imagenet_norm=False
            )

        img = Image.open(image_path).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(self.device)

        # =====================================================
        # AUTOENCODER
        # =====================================================
        if self.model_type == "ae":

            self.model.eval()

            with torch.no_grad():
                reconstruction = self.model(img_tensor)

            img_np = img_tensor.squeeze().cpu().numpy().transpose(1, 2, 0)
            rec_np = reconstruction.squeeze().cpu().numpy().transpose(1, 2, 0)

            diff_map = np.abs(img_np - rec_np)
            anomaly_map = np.mean(diff_map, axis=-1)
            anomaly_map = (anomaly_map - anomaly_map.min()) / (
                anomaly_map.max() - anomaly_map.min() + 1e-8
            )
            anomaly_map = anomaly_map ** 0.5

            score = 1 - ssim(
                reconstruction, img_tensor,
                data_range=1.0,
                size_average=False
            )
            score = score.item()
            prediction = "WADA" if score >= self.threshold else "OK"

            fig, axes = plt.subplots(1, 4, figsize=(18, 5))

            axes[0].imshow(img_np)
            axes[0].set_title("Oryginał (AE input)")
            axes[0].axis("off")

            axes[1].imshow(rec_np)
            axes[1].set_title("Rekonstrukcja AE")
            axes[1].axis("off")

            im = axes[2].imshow(anomaly_map, cmap="jet")
            axes[2].set_title("Heatmapa anomalii")
            axes[2].axis("off")
            fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

            axes[3].imshow(img_np)
            axes[3].imshow(anomaly_map, cmap="jet", alpha=0.5)
            axes[3].set_title("Overlay")
            axes[3].axis("off")

            plt.suptitle(
                f"AE | Score: {score:.4f} | Predykcja: {prediction}",
                fontsize=14
            )
            plt.tight_layout()
            plt.show()

            return

        # =====================================================
        # PADIM
        # =====================================================
        elif self.model_type == "padim":

            anomaly_map = self.model.anomaly_map(img_tensor)

            img_np = img.resize((IMG_SIZE, IMG_SIZE))
            img_np = np.array(img_np)

            score = np.mean(np.sort(anomaly_map.flatten())[-50:])
            prediction = "WADA" if score >= self.threshold else "OK"

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            axes[0].imshow(img_np)
            axes[0].set_title("Oryginał")
            axes[0].axis("off")

            im = axes[1].imshow(anomaly_map, cmap="jet")
            axes[1].set_title("Mapa anomalii (PaDiM)")
            axes[1].axis("off")
            fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

            axes[2].imshow(img_np)
            axes[2].imshow(anomaly_map, cmap="jet", alpha=0.5)
            axes[2].set_title("Overlay")
            axes[2].axis("off")

            plt.suptitle(
                f"PaDiM | Score: {score:.4f} | Predykcja: {prediction}",
                fontsize=14
            )
            plt.tight_layout()
            plt.show()

            return