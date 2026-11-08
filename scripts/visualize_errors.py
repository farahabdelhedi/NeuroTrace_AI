
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

CSV_PATH = BASE_DIR / "misclassified_images.csv"

# Charger les erreurs
df = pd.read_csv(CSV_PATH)

# Sélectionner les erreurs les plus intéressantes :
# 6 gliomes prédits meningioma
# 6 gliomes prédits notumor
glioma_meningioma = df[
    (df["real_class"] == "glioma") &
    (df["predicted_class"] == "meningioma")
].nlargest(6, "confidence")

glioma_notumor = df[
    (df["real_class"] == "glioma") &
    (df["predicted_class"] == "notumor")
].nlargest(6, "confidence")

selected = pd.concat([
    glioma_meningioma,
    glioma_notumor
])

# Affichage des images
fig, axes = plt.subplots(3, 4, figsize=(16, 12))
axes = axes.flatten()

for ax, (_, row) in zip(axes, selected.iterrows()):

    image = Image.open(row["image_path"]).convert("RGB")

    ax.imshow(image)
    ax.set_title(
        f"Réelle : {row['real_class']}\n"
        f"Prédite : {row['predicted_class']}\n"
        f"Confiance : {row['confidence']}%",
        fontsize=9
    )
    ax.axis("off")

# Masquer les cases inutilisées
for ax in axes[len(selected):]:
    ax.axis("off")

plt.suptitle(
    "Analyse des erreurs de classification - NeuroTrace",
    fontsize=15
)

plt.tight_layout()
plt.show()
