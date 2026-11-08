
"""
NeuroTrace - Configuration du modèle IA
"""

# Identifiant du modèle Hugging Face
MODEL_REPO_ID = "Abuzaid01/brain-tumor-resnet50-classifier"

# Fichier contenant les poids du modèle
MODEL_FILENAME = "model.safetensors"

# Ordre des classes utilisé par le modèle
CLASSES = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary"
]

# Architecture
MODEL_ARCHITECTURE = "ResNet50"

# Taille d'entrée
IMAGE_SIZE = (224, 224)

# Normalisation ImageNet
NORMALIZATION_MEAN = [0.485, 0.456, 0.406]
NORMALIZATION_STD = [0.229, 0.224, 0.225]
