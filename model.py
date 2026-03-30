import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import cv2
import pickle

IMG_SIZE = 256
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

_model_loaded = False
_padim_data = None
_layer1 = None
_layer2 = None
_layer3 = None
_transform = None

def _load_model():

    global _model_loaded, _padim_data, _layer1, _layer2, _layer3, _transform
    
    if _model_loaded:
        return
    
    with open("padim_cable.pkl", 'rb') as f:
        _padim_data = pickle.load(f)
    
    # Backbone ResNet18
    backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    
    _layer1 = nn.Sequential(
        backbone.conv1, backbone.bn1, backbone.relu,
        backbone.maxpool, backbone.layer1
    ).to(DEVICE).eval()
    
    _layer2 = backbone.layer2.to(DEVICE).eval()
    _layer3 = backbone.layer3.to(DEVICE).eval()
    
    # Transform
    _transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    _model_loaded = True


def predict(image):
    """PaDiM anomaly segmentation.

    Args:
        image: numpy array of shape (H, W, 3), uint8 RGB image.

    Returns:
        Binary mask as numpy array of shape (H, W), uint8 with values 0 or 255.
    """
    global _padim_data, _layer1, _layer2, _layer3, _transform
    
    _load_model()
    
    # Preprocess
    img = Image.fromarray(image)
    img_tensor = _transform(img).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        f1 = _layer1(img_tensor)
        f2 = _layer2(f1)
        f3 = _layer3(f2)
        
        f2 = F.interpolate(f2, size=f1.shape[-2:], mode="bilinear", align_corners=False)
        f3 = F.interpolate(f3, size=f1.shape[-2:], mode="bilinear", align_corners=False)
        
        emb = torch.cat([f1, f2, f3], dim=1)
    
    emb = emb[0, _padim_data['selected_idx']].cpu().numpy().astype(np.float32)
    
    # Anomaly map
    C, H, W = emb.shape
    amap = np.zeros((H, W), dtype=np.float32)
    
    for h in range(H):
        for w in range(W):
            diff = emb[:, h, w] - _padim_data['mean'][h, w]
            dist = diff @ _padim_data['inv_cov'][h, w] @ diff
            amap[h, w] = np.sqrt(np.maximum(dist, 0))
    
    # Resize to original size
    original_h, original_w = image.shape[:2]
    amap = cv2.resize(amap, (original_w, original_h))
    
    # Normalize
    amap = (amap - amap.min()) / (amap.max() - amap.min() + 1e-8)
    
    # Threshold (percentyl 95)
    threshold = np.percentile(amap, 95)
    mask = (amap > threshold).astype(np.uint8) * 255
    
    return mask