from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
from pytorch_msssim import ssim
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, precision_score, recall_score, confusion_matrix
from src.utils.config import IMG_SIZE
from src.utils import load_train_paths, load_test_paths, CableDataset
from src.utils import DATA_PATH
from src.model import ConvolutionalAutoencoder

class AnomalyDetectionModel:
    def __init__(self):
        self.train_paths = load_train_paths(DATA_PATH)
        self.train_dataset = CableDataset(self.train_paths)
        self.train_loader = DataLoader(self.train_dataset, batch_size=32, shuffle=True)

        test_img_paths, self.test_labels = load_test_paths(DATA_PATH)
        self.test_dataset = CableDataset(test_img_paths)
        self.test_loader = DataLoader(self.test_dataset, batch_size=32, shuffle=False)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Używane urządzenie: {self.device}')

        # Inicjalizacja architektury sieci
        self.ae = ConvolutionalAutoencoder().to(self.device)

        self.threshold = 0.5

    def train_model(self, epochs=20, lr=1e-3, save_path="autoencoder_cable.pth"):
        torch.manual_seed(42)  # Dla powtarzalności wyników

        # loss_fn = nn.MSELoss()
        # Wykorzystanie SSIM jako funkcji straty
        optimizer = optim.Adam(self.ae.parameters(), lr=lr)
        train_losses = []

        for epoch in range(epochs):
            self.ae.train()
            epoch_loss = []
            
            for imgs in self.train_loader:
                imgs = imgs.to(self.device)
                
                # Forward pass
                x_hat = self.ae(imgs)
                # loss = loss_fn(x_hat, imgs)
                loss = 1 - ssim(x_hat, imgs, data_range=1.0, size_average=True)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss.append(loss.item())
                
            avg_loss = np.mean(epoch_loss)
            train_losses.append(avg_loss)

            # print(f'Epoka {epoch+1:2d}/{epochs} | Train Loss (MSE): {avg_loss:.5f}')
            print(f'Epoka {epoch+1:2d}/{epochs} | Train Loss (1 - SSIM): {avg_loss:.5f}')

        self.save_model(save_path)

    def save_model(self, path):
        torch.save(self.ae.state_dict(), path)
        print(f"Wagi modelu zostały pomyślnie zapisane do: {path}")
    
    def load_model(self, path):
        self.ae.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
        self.ae.eval()
        print(f"Model został pomyślnie załadowany z: {path}")
    
    def evaluate_model(self, mode="ssim"):
        self.ae.eval()
        image_scores = []

        print(f"\nRozpoczynam ewaluację ({mode})...")

        with torch.no_grad():
            for imgs in self.test_loader:
                imgs = imgs.to(self.device)
                x_hat = self.ae(imgs)

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
        
    def evaluate_model_top_k(self):
        self.ae.eval()
        image_scores = []
        
        print("\nRozpoczynam ewaluację na zbiorze testowym (Top 1% najgorszych pikseli)...")
        with torch.no_grad():
            for imgs in self.test_loader:
                imgs = imgs.to(self.device)
                x_hat = self.ae(imgs)
                
                # Liczymy błąd piksel po pikselu
                loss_map = F.mse_loss(x_hat, imgs, reduction='none')
                
                # --- NOWA LOGIKA: Skupiamy się na defekcie ---
                # 1. Uśredniamy błąd po kanałach RGB (zostaje mapa 2D)
                loss_map = loss_map.mean(dim=1)
                
                # 2. Spłaszczamy mapę do wektora 1D dla każdego obrazka [Batch, 65536]
                loss_flat = loss_map.view(loss_map.size(0), -1)
                
                # 3. Ustalamy, że interesuje nas 1% najbardziej odstających pikseli
                k = int(loss_flat.size(1) * 0.01)
                
                # 4. Wyciągamy z wektora 'k' najwyższych wartości błędów
                top_k_errors, _ = torch.topk(loss_flat, k=k, dim=1)
                
                # 5. Wynikiem dla obrazka jest średnia z TYCH NAJGORSZYCH pikseli
                scores = top_k_errors.mean(dim=1).cpu().numpy()
                # ---------------------------------------------
                
                image_scores.extend(scores)
                
        image_scores = np.array(image_scores)
        true_labels = np.array(self.test_labels)
        
        # --- WYLICZANIE METRYK ---
        print("\n--- METRYKI (Top 1% najgorszych pikseli) ---")
        auc = roc_auc_score(true_labels, image_scores)
        print(f"ROC AUC:   {auc:.4f}")
        
        fpr, tpr, thresholds = roc_curve(true_labels, image_scores)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        print(f"Optymalny próg odcięcia: {optimal_threshold:.5f}")
        
        predictions = (image_scores >= optimal_threshold).astype(int)
        
        acc = accuracy_score(true_labels, predictions)
        prec = precision_score(true_labels, predictions, zero_division=0)
        rec = recall_score(true_labels, predictions, zero_division=0)
        cm = confusion_matrix(true_labels, predictions)
        
        print(f"Accuracy:  {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall:    {rec:.4f}")
        print("\nMacierz pomyłek (Confusion Matrix):")
        print("                Przewidziane OK | Przewidziane WADY")
        print(f"Rzeczywiste OK   |      {cm[0, 0]:3d}       |       {cm[0, 1]:3d}")
        print(f"Rzeczywiste WADY |      {cm[1, 0]:3d}       |       {cm[1, 1]:3d}")
        
        return auc, optimal_threshold
    
    def visualize_defect(self, image_path):

        self.ae.eval()
        transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
        ])
        
        # Wczytanie obrazu i dodanie wymiaru batcha: [1, Channels, Height, Width]
        img = Image.open(image_path).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(self.device)
        
        # Predykcja bez liczenia gradientów
        with torch.no_grad():
            reconstruction = self.ae(img_tensor)
            
        # Konwersja tensorów do tablic NumPy i zmiana kolejności wymiarów pod matplotlib
        # z [C, H, W] na [H, W, C]
        img_np = img_tensor.squeeze().cpu().numpy().transpose(1, 2, 0)
        rec_np = reconstruction.squeeze().cpu().numpy().transpose(1, 2, 0)

        # Obliczenie mapy anomalii (L1)
        diff_map = np.abs(img_np - rec_np)
        anomaly_map = np.mean(diff_map, axis=-1)

        # Normalizacja mapy anomalii do zakresu [0, 1]
        anomaly_map = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min() + 1e-8)

        # Wzmocnienie kontrastu
        anomaly_map = anomaly_map ** 0.5

        fig, axes = plt.subplots(1, 4, figsize=(18, 5))

        axes[0].imshow(img_np)
        axes[0].set_title("Oryginał")
        axes[0].axis('off')

        axes[1].imshow(rec_np)
        axes[1].set_title("Rekonstrukcja AE")
        axes[1].axis('off')

        im = axes[2].imshow(anomaly_map, cmap='jet')
        axes[2].set_title("Heatmapa błędu (Anomalie)")
        axes[2].axis('off')
        fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

        axes[3].imshow(img_np)
        axes[3].imshow(anomaly_map, cmap='jet', alpha=0.5)
        axes[3].set_title("Overlay Anomalii")
        axes[3].axis('off')
        
        score = 1 - ssim(reconstruction, img_tensor, data_range=1.0, size_average=False)
        score = score.item()

        prediction = "WADA" if score >= self.threshold else "OK"

        plt.suptitle(f"Score: {score:.4f} | Predykcja: {prediction}", fontsize=14)
        plt.tight_layout()  
        plt.show()