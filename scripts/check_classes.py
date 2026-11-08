
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TRAINING_DIR = os.path.join(
    BASE_DIR, "datasets_preprocessed", "Training"
)

classes = sorted([
    name for name in os.listdir(TRAINING_DIR)
    if os.path.isdir(os.path.join(TRAINING_DIR, name))
])

print("Ordre des classes utilisé par PyTorch :")

for index, class_name in enumerate(classes):
    print(f"{index} -> {class_name}")
