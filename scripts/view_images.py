
import os
import matplotlib.pyplot as plt
from PIL import Image

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

original_dir = os.path.join(
    BASE_DIR, "datasets", "Training", "glioma"
)

processed_dir = os.path.join(
    BASE_DIR, "datasets_preprocessed", "Training", "glioma"
)

filename = os.listdir(original_dir)[0]

original = Image.open(os.path.join(original_dir, filename))
processed = Image.open(os.path.join(processed_dir, filename))

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(original)
plt.title("Image originale")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(processed)
plt.title("Image après prétraitement")
plt.axis("off")

plt.tight_layout()
plt.show()
