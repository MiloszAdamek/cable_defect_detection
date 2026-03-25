import torch.nn as nn

class ConvolutionalAutoencoder(nn.Module):
    def __init__(self):
        super(ConvolutionalAutoencoder, self).__init__()
        
        # Enkoder: redukcja wymiarów przestrzennych, zwiększanie liczby kanałów
        self.encoder = nn.Sequential(
            # Wejście: [Batch, 3, 256, 256]
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            # Wymiar: [Batch, 16, 128, 128]
            
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            # Wymiar: [Batch, 32, 64, 64]
            
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            # Wymiar: [Batch, 64, 32, 32]
            
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True)
            # Przestrzeń ukryta (bottleneck): [Batch, 128, 16, 16]
        )
        
        # Dekoder: powiększanie wymiarów przestrzennych, redukcja liczby kanałów
        self.decoder = nn.Sequential(
            # Wejście: [Batch, 128, 16, 16]
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            # Wymiar: [Batch, 64, 32, 32]
            
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            # Wymiar: [Batch, 32, 64, 64]
            
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            # Wymiar: [Batch, 16, 128, 128]
            
            nn.ConvTranspose2d(in_channels=16, out_channels=3, kernel_size=3, stride=2, padding=1, output_padding=1),
            # Funkcja aktywacji Sigmoid gwarantuje, że wartości pikseli na wyjściu 
            # będą w przedziale [0, 1], co zgadza się z formatem transforms.ToTensor()
            nn.Sigmoid() 
            # Wyjście: [Batch, 3, 256, 256]
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded