
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TRAINING_DIR = os.path.join(
    BASE_DIR, "datasets_preprocessed", "Training"
)

TESTING_DIR = os.path.join(
    BASE_DIR, "datasets_preprocessed", "Testing"
)

# Transformation des images
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# Chargement des datasets
train_dataset = datasets.ImageFolder(
    TRAINING_DIR,
    transform=transform
)

test_dataset = datasets.ImageFolder(
    TESTING_DIR,
    transform=transform
)

# Création des DataLoaders
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)

print("Classes :", train_dataset.classes)
print("Nombre d'images Training :", len(train_dataset))
print("Nombre d'images Testing :", len(test_dataset))

# Vérifier un batch
images, labels = next(iter(train_loader))

print("Dimensions des images :", images.shape)
print("Dimensions des labels :", labels.shape)
