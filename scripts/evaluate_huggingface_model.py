
import torch
import torch.nn as nn
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from pathlib import Path

# ==========================================
# 1. CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

TESTING_DIR = BASE_DIR / "datasets" / "Testing"

REPO_ID = "Abuzaid01/brain-tumor-resnet50-classifier"

CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Appareil utilisé :", DEVICE)

# ==========================================
# 2. PRETRAITEMENT
# ==========================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ==========================================
# 3. CHARGEMENT DU DATASET
# ==========================================

test_dataset = datasets.ImageFolder(
    root=TESTING_DIR,
    transform=transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)

print("\nClasses détectées :", test_dataset.classes)
print("Nombre total d'images :", len(test_dataset))

assert test_dataset.classes == CLASSES, (
    "Attention : l'ordre des classes est différent !"
)

# ==========================================
# 4. CHARGEMENT DU MODELE
# ==========================================

print("\nChargement du modèle...")

model_path = hf_hub_download(
    repo_id=REPO_ID,
    filename="model.safetensors"
)

model = models.resnet50(weights=None)

model.fc = nn.Sequential(
    nn.Linear(2048, 512),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512, 4)
)

model.load_state_dict(load_file(model_path))
model.to(DEVICE)
model.eval()

print("Modèle chargé avec succès !")

# ==========================================
# 5. EVALUATION
# ==========================================

# Lignes = classes réelles
# Colonnes = classes prédites

confusion_matrix = torch.zeros(4, 4, dtype=torch.int64)

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)
        predictions = outputs.argmax(dim=1)

        for real, predicted in zip(labels, predictions):
            confusion_matrix[real.item(), predicted.item()] += 1

# ==========================================
# 6. CALCUL DES METRIQUES
# ==========================================

total = confusion_matrix.sum().item()
correct = confusion_matrix.diag().sum().item()

accuracy = correct / total

print("\n========== ACCURACY GLOBALE ==========")
print(f"Images correctement classées : {correct}/{total}")
print(f"Accuracy : {accuracy * 100:.2f}%")

print("\n========== MATRICE DE CONFUSION ==========")

print("Réel \\ Prédit", end="\t")

for class_name in CLASSES:
    print(class_name, end="\t")

print()

for i, class_name in enumerate(CLASSES):
    print(class_name, end="\t")

    for j in range(4):
        print(confusion_matrix[i, j].item(), end="\t")

    print()

print("\n========== METRIQUES PAR CLASSE ==========")

for i, class_name in enumerate(CLASSES):

    true_positive = confusion_matrix[i, i].item()

    false_positive = (
        confusion_matrix[:, i].sum().item() - true_positive
    )

    false_negative = (
        confusion_matrix[i, :].sum().item() - true_positive
    )

    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive > 0 else 0
    )

    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative > 0 else 0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0 else 0
    )

    print(f"\nClasse : {class_name}")
    print(f"Precision : {precision * 100:.2f}%")
    print(f"Recall    : {recall * 100:.2f}%")
    print(f"F1-score  : {f1 * 100:.2f}%")

print("\nÉvaluation terminée !")
