
import torch
import torch.nn as nn
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from PIL import Image
from torchvision import models, transforms, datasets
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file


# ==========================================
# 1. CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent
TESTING_DIR = BASE_DIR / "datasets" / "Testing"
CSV_PATH = BASE_DIR / "misclassified_images.csv"

OUTPUT_DIR = BASE_DIR / "gradcam_results"
OUTPUT_DIR.mkdir(exist_ok=True)

REPO_ID = "Abuzaid01/brain-tumor-resnet50-classifier"

CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Prétraitement expérimental utilisé lors de l'évaluation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ==========================================
# 2. CHARGEMENT DU MODELE
# ==========================================

print("Chargement du modèle...")

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

print("Modèle chargé !")
print("Appareil utilisé :", DEVICE)


# ==========================================
# 3. FONCTION GRAD-CAM
# ==========================================

def generate_gradcam(image_path, target_class=None):

    if target_class is not None and target_class not in CLASSES:
        raise ValueError(
            f"Classe inconnue : {target_class}. "
            f"Classes disponibles : {CLASSES}"
        )

    image = Image.open(image_path).convert("RGB")
    image_resized = image.resize((224, 224))

    input_tensor = transform(image).unsqueeze(0).to(DEVICE)

    activations = {}
    gradients = {}

    target_layer = model.layer4[-1]

    def forward_hook(module, inputs, output):
        activations["value"] = output
        output.register_hook(
            lambda grad: gradients.update({"value": grad})
        )

    handle = target_layer.register_forward_hook(forward_hook)

    try:
        model.zero_grad(set_to_none=True)

        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)

        predicted_index = outputs.argmax(dim=1).item()
        predicted_class = CLASSES[predicted_index]

        # Classe à expliquer
        if target_class is None:
            target_index = predicted_index
        else:
            target_index = CLASSES.index(target_class)

        target_score = outputs[0, target_index]

        # Calcul des gradients pour la classe choisie
        target_score.backward()

        # Récupération des activations et gradients
        feature_maps = activations["value"]
        grads = gradients["value"]

        # Calcul des poids Grad-CAM
        weights = grads.mean(dim=(2, 3), keepdim=True)

        # Construction de la carte
        cam = (weights * feature_maps).sum(dim=1)
        cam = torch.relu(cam)

        cam = cam[0].detach().cpu().numpy()

        cam = cam - cam.min()

        if cam.max() > 0:
            cam = cam / cam.max()

        cam = cv2.resize(cam, (224, 224))

        # Création de la heatmap
        heatmap = np.uint8(255 * cam)

        heatmap = cv2.applyColorMap(
            heatmap,
            cv2.COLORMAP_JET
        )

        heatmap = cv2.cvtColor(
            heatmap,
            cv2.COLOR_BGR2RGB
        )

        original = np.array(image_resized)

        # Superposition de la heatmap sur l'image originale
        overlay = cv2.addWeighted(
            original,
            0.6,
            heatmap,
            0.4,
            0
        )

        confidence = probabilities[0, target_index].item() * 100

        return (
            original,
            heatmap,
            overlay,
            predicted_class,
            CLASSES[target_index],
            confidence
        )

    finally:
        handle.remove()


# ==========================================
# 4. SELECTION DES IMAGES
# ==========================================

print("\nRecherche des images...")

if not CSV_PATH.exists():
    raise FileNotFoundError(f"Fichier introuvable : {CSV_PATH}")

# Charger les erreurs de classification
errors_df = pd.read_csv(CSV_PATH)

wrong_rows = errors_df[
    (errors_df["real_class"] == "glioma") &
    (errors_df["predicted_class"] == "meningioma")
]

if wrong_rows.empty:
    raise RuntimeError(
        "Aucune erreur glioma -> meningioma trouvée."
    )

wrong_image_path = Path(wrong_rows.iloc[0]["image_path"])

if not wrong_image_path.exists():
    raise FileNotFoundError(
        f"Image mal classée introuvable : {wrong_image_path}"
    )

# Recherche d'une image glioma correctement classée
dataset = datasets.ImageFolder(TESTING_DIR)

correct_image_path = None

for image_path, label in dataset.samples:

    if dataset.classes[label] != "glioma":
        continue

    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        prediction = model(input_tensor).argmax(dim=1).item()

    if prediction == CLASSES.index("glioma"):
        correct_image_path = image_path
        break

if correct_image_path is None:
    raise RuntimeError(
        "Aucune image glioma correctement classée trouvée."
    )

print("Image mal classée :", wrong_image_path)
print("Image correctement classée :", correct_image_path)


# ==========================================
# 5. GENERATION DES QUATRE EXPLICATIONS
# ==========================================

# On utilise l'image glioma mal classée
image_path = wrong_image_path

print("\nGénération des cartes Grad-CAM...")

results = {}

for class_name in CLASSES:

    print(f"Analyse de la classe : {class_name}")

    results[class_name] = generate_gradcam(
        image_path,
        target_class=class_name
    )


# ==========================================
# 6. AFFICHAGE DES QUATRE CARTES
# ==========================================

original = results["glioma"][0]

fig, axes = plt.subplots(
    1,
    5,
    figsize=(22, 5)
)

# Image originale
axes[0].imshow(original)
axes[0].set_title("Image IRM originale")

# Une carte Grad-CAM pour chaque classe
titles = {
    "glioma": "Grad-CAM : Glioma",
    "meningioma": "Grad-CAM : Meningioma",
    "notumor": "Grad-CAM : No Tumor",
    "pituitary": "Grad-CAM : Pituitary"
}

for i, class_name in enumerate(CLASSES):

    overlay = results[class_name][2]

    axes[i + 1].imshow(overlay)
    axes[i + 1].set_title(titles[class_name])

# Masquer les axes
for ax in axes:
    ax.axis("off")

plt.suptitle(
    "NeuroTrace - Comparaison Grad-CAM des quatre classes",
    fontsize=15
)

plt.tight_layout()

# Enregistrer la comparaison
output_path = OUTPUT_DIR / "gradcam_four_classes.png"

plt.savefig(
    output_path,
    dpi=150,
    bbox_inches="tight"
)

plt.show()


# ==========================================
# 7. AFFICHAGE DES RESULTATS
# ==========================================

print("\n========== RESULTATS DE L'ANALYSE ==========")

print("Classe réelle : glioma")
print("Classe prédite :", results["glioma"][3])

print("\nScores Softmax :")

for class_name in CLASSES:

    confidence = results[class_name][5]

    print(
        f"{class_name} : {confidence:.2f}%"
    )

print("\nImage enregistrée :", output_path)

print("\nGénération des quatre cartes Grad-CAM terminée !")
