
from huggingface_hub import HfApi, hf_hub_download

REPO_ID = "Abuzaid01/brain-tumor-resnet50-classifier"

# 1. Afficher les fichiers disponibles sur Hugging Face
api = HfApi()
files = api.list_repo_files(repo_id=REPO_ID)

print("Fichiers disponibles :")
for file in files:
    print("-", file)

# 2. Télécharger le README
readme_path = hf_hub_download(
    repo_id=REPO_ID,
    filename="README.md"
)

# 3. Chercher les informations sur le prétraitement
with open(readme_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

keywords = [
    "transform",
    "normalize",
    "normalization",
    "mean",
    "std",
    "256",
    "224",
    "class_to_idx",
    "preprocess"
]

print("\nInformations importantes trouvées dans le README :")

for i, line in enumerate(lines):
    if any(word in line.lower() for word in keywords):
        print(f"Ligne {i + 1} : {line.strip()}")
