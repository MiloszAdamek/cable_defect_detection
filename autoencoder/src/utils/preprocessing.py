import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

class CLAHETransform:

    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, img):
        img_np = np.array(img)

        lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(
            clipLimit=self.clip_limit,
            tileGridSize=self.tile_grid_size
        )
        l = clahe.apply(l)

        lab = cv2.merge([l, a, b])
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

        return Image.fromarray(result)


class GaussianBlur:

    def __init__(self, kernel_size=3):
        self.kernel_size = kernel_size

    def __call__(self, img):
        img_np = np.array(img)
        blurred = cv2.GaussianBlur(img_np, (self.kernel_size, self.kernel_size), 0)
        return Image.fromarray(blurred)


class CircularMask:

    def __init__(self, margin=0.05):
        self.margin = margin

    def __call__(self, img):
        img_np = np.array(img)
        h, w = img_np.shape[:2]

        # Maska kołowa
        center = (w // 2, h // 2)
        radius = int(min(h, w) * (0.5 - self.margin))
        Y, X = np.ogrid[:h, :w]
        dist = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
        mask = (dist <= radius).astype(np.uint8)

        result = img_np * mask[:, :, np.newaxis]
        return Image.fromarray(result)


def get_train_transform(image_size=256, use_clahe=True, use_blur=True):

    steps = [
        transforms.Resize((image_size, image_size)),
    ]

    # Preprocessing
    if use_blur:
        steps.append(GaussianBlur(kernel_size=3))
    if use_clahe:
        steps.append(CLAHETransform(clip_limit=2.0))

    steps.extend([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.05, contrast=0.05),
    ])

    # Tensor + normalizacja
    steps.extend([
        transforms.ToTensor(),
        # Normalizacja do zakresu [0,1] jest domyślna w ToTensor
    ])

    return transforms.Compose(steps)


def get_test_transform(image_size=256,
                       use_clahe=True,
                       use_blur=True,
                       imagenet_norm=False):
    steps = [
        transforms.Resize((image_size, image_size)),
    ]

    # Preprocessing
    if use_blur:
        steps.append(GaussianBlur(kernel_size=3))

    if use_clahe:
        steps.append(CLAHETransform(clip_limit=2.0))

    steps.append(transforms.ToTensor())

    # ImageNet normalization ResNet / PaDiM)
    if imagenet_norm:
        steps.append(
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        )

    return transforms.Compose(steps)
