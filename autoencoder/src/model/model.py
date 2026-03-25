from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, precision_score, recall_score, confusion_matrix
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

    def train_model(self, epochs=20, lr=1e-3, save_path="autoencoder_cable.pth"):
        torch.manual_seed(42)  # Dla powtarzalności wyników

        loss_fn = nn.MSELoss()
        optimizer = optim.Adam(self.ae.parameters(), lr=lr)
        train_losses = []

        for epoch in range(epochs):
            self.ae.train()
            epoch_loss = []
            
            for imgs in self.train_loader:
                imgs = imgs.to(self.device)
                
                # Forward pass
                x_hat = self.ae(imgs)
                loss = loss_fn(x_hat, imgs)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss.append(loss.item())
                
            avg_loss = np.mean(epoch_loss)
            train_losses.append(avg_loss)

            print(f'Epoka {epoch+1:2d}/{epochs} | Train Loss (MSE): {avg_loss:.5f}')

        self.save_model(save_path)

    def save_model(self, path):
        torch.save(self.ae.state_dict(), path)
        print(f"Wagi modelu zostały pomyślnie zapisane do: {path}")
    
    def load_model(self, path):
        self.ae.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
        self.ae.eval()
        print(f"Model został pomyślnie załadowany z: {path}")
    
    def evaluate_model(self):
            self.ae.eval()
            image_scores = []
            
            print("\nRozpoczynam ewaluację na zbiorze testowym...")
            with torch.no_grad():
                for imgs in self.test_loader:
                    imgs = imgs.to(self.device)
                    x_hat = self.ae(imgs)
                    
                    # Liczymy błąd piksel po pikselu (bez uśredniania całego batcha)
                    loss_map = F.mse_loss(x_hat, imgs, reduction='none')
                    
                    # loss_map ma kształt: [Batch, Channels, Height, Width]
                    # Uśredniamy wymiary (1, 2, 3), żeby uzyskać jeden wynik błędu na pojedyncze zdjęcie
                    scores = loss_map.mean(dim=(1, 2, 3)).cpu().numpy()
                    image_scores.extend(scores)
                    
            # Konwersja do tablic numpy dla scikit-learn
            image_scores = np.array(image_scores)
            true_labels = np.array(self.test_labels)
            
            # --- WYLICZANIE METRYK (Zgodnie z wymaganiami projektu) ---
            
            # 1. Pole pod krzywą ROC (AUC ROC) - najważniejsza metryka w detekcji anomalii
            auc = roc_auc_score(true_labels, image_scores)
            print(f"ROC AUC:   {auc:.4f}")
            
            # 2. Szukanie optymalnego progu (threshold) za pomocą statystyki Youdena
            fpr, tpr, thresholds = roc_curve(true_labels, image_scores)
            optimal_idx = np.argmax(tpr - fpr)
            optimal_threshold = thresholds[optimal_idx]
            print(f"Optymalny próg odcięcia: {optimal_threshold:.5f}")
            
            # 3. Klasyfikacja binarna z użyciem znalezionego progu
            # Jeśli błąd zdjęcia jest >= threshold, to przypisujemy klasę 1 (wada), w przeciwnym razie 0 (ok)
            predictions = (image_scores >= optimal_threshold).astype(int)
            
            # 4. Obliczenie Accuracy, Precision, Recall i Macierzy Pomyłek
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