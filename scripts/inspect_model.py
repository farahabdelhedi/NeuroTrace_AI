
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

REPO_ID = "Abuzaid01/brain-tumor-resnet50-classifier"

print("Téléchargement du modèle...")

model_path = hf_hub_download(
    repo_id=REPO_ID,
    filename="model.safetensors"
)

print("Modèle téléchargé :", model_path)

state_dict = load_file(model_path)

print("\nNombre de tenseurs :", len(state_dict))
print("\nStructure du modèle :")

for name, tensor in state_dict.items():
    print(name, tuple(tensor.shape))
print("\n=== COUCHE DE CLASSIFICATION ===")

for name, tensor in state_dict.items():
    if name.startswith("fc."):
        print(name, tuple(tensor.shape))

from huggingface_hub import hf_hub_download

readme_path = hf_hub_download(
    repo_id="Abuzaid01/brain-tumor-resnet50-classifier",
    filename="README.md"
)

with open(readme_path, "r", encoding="utf-8") as file:
    print(file.read())
print("\n=== STRUCTURE DU CLASSIFICATEUR ===")

for name, module in model.named_modules():
    if name.startswith("fc"):
        print(name, ":", module)