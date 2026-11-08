
import os
from PIL import Image

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

folder = os.path.join(
    BASE_DIR,
    "datasets_preprocessed",
    "Training",
    "glioma"
)

filename = os.listdir(folder)[0]
image_path = os.path.join(folder, filename)

img = Image.open(image_path)

print("Nom :", filename)
print("Dimensions :", img.size)
print("Mode :", img.mode)
print("Nombre de canaux :", len(img.getbands()))
