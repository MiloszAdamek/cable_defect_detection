from pathlib import Path

IMG_EXTENSIONS = [".png", ".jpg", ".jpeg"]


def get_image_paths(directory):
    directory = Path(directory)
    image_paths = []

    for ext in IMG_EXTENSIONS:
        image_paths.extend(directory.rglob(f"*{ext}"))

    return sorted(image_paths)

def load_train_paths(data_path):
    train_dir = Path(data_path) / "train" / "good"
    return get_image_paths(train_dir)

def load_test_paths(data_path):
    test_dir = Path(data_path) / "test"
    
    image_paths = []
    labels = []

    for class_dir in test_dir.iterdir():
        if not class_dir.is_dir():
            continue

        paths = get_image_paths(class_dir)

        image_paths.extend(paths)

        if class_dir.name == "good":
            labels.extend([0] * len(paths))
        else:
            labels.extend([1] * len(paths))

    return image_paths, labels

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


class CableDataset(Dataset):
    def __init__(self, image_paths, image_size=256):
        self.image_paths = image_paths

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        return image