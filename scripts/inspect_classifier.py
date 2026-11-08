
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download

REPO_ID = "Abuzaid01/brain-tumor-resnet50-classifier"

model_path = hf_hub_download(
    repo_id=REPO_ID,
    filename="model.safetensors"
)

state_dict = load_file(model_path)

print("=== Types des tenseurs ===")

types = {}

for name, tensor in state_dict.items():
    dtype = str(tensor.dtype)
    types[dtype] = types.get(dtype, 0) + 1
    print(name, "->", dtype)

print("\n=== Résumé ===")
for dtype, count in types.items():
    print(dtype, ":", count, "tenseurs")

from torchvision import models

model = models.resnet50(weights=None)

print("\n=== Première couche convolutionnelle ===")
print(model.conv1)

print("\n=== Dimensions des poids ===")
print(model.conv1.weight.shape)

