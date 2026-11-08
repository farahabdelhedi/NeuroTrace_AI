
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
TRAINING_DIR = BASE_DIR / "datasets_preprocessed" / "Training"

transform = transforms.ToTensor()

dataset = datasets.ImageFolder(
    TRAINING_DIR,
    transform=transform
)

loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=False
)

mean = torch.zeros(3)
std = torch.zeros(3)
total_batches = 0

for images, _ in loader:
    mean += images.mean(dim=[0, 2, 3])
    std += images.std(dim=[0, 2, 3])
    total_batches += 1

mean /= total_batches
std /= total_batches

print("Moyenne RGB :", mean)
print("Écart-type RGB :", std)
