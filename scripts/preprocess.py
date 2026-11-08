
import os
import cv2
import imutils

# Dossiers source et destination
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TRAINING_DIR = os.path.join(BASE_DIR, "datasets", "Training")
TESTING_DIR = os.path.join(BASE_DIR, "datasets", "Testing")

OUTPUT_DIR = os.path.join(BASE_DIR, "datasets_preprocessed")

IMG_SIZE = 256


def crop_img(img):
    """Recadrer la région cérébrale."""

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    thresh = cv2.threshold(
        gray, 45, 255, cv2.THRESH_BINARY
    )[1]

    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)

    cnts = cv2.findContours(
        thresh.copy(),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    cnts = imutils.grab_contours(cnts)

    if not cnts:
        return img

    c = max(cnts, key=cv2.contourArea)

    extLeft = c[c[:, :, 0].argmin()][0]
    extRight = c[c[:, :, 0].argmax()][0]
    extTop = c[c[:, :, 1].argmin()][0]
    extBot = c[c[:, :, 1].argmax()][0]

    x1 = max(0, extLeft[0])
    x2 = min(img.shape[1], extRight[0] + 1)
    y1 = max(0, extTop[1])
    y2 = min(img.shape[0], extBot[1] + 1)

    return img[y1:y2, x1:x2]


def preprocess_dataset(input_dir, split):
    for class_name in os.listdir(input_dir):

        class_path = os.path.join(input_dir, class_name)

        if not os.path.isdir(class_path):
            continue

        output_class = os.path.join(
            OUTPUT_DIR, split, class_name
        )

        os.makedirs(output_class, exist_ok=True)

        images = os.listdir(class_path)

        for filename in images:
            image_path = os.path.join(class_path, filename)

            img = cv2.imread(image_path)

            if img is None:
                print("Image illisible :", image_path)
                continue

            cropped = crop_img(img)

            resized = cv2.resize(
                cropped, (IMG_SIZE, IMG_SIZE)
            )

            output_path = os.path.join(output_class, filename)

            cv2.imwrite(output_path, resized)

        print(f"{class_name} : {len(images)} images traitées")


if __name__ == "__main__":

    print("Prétraitement du dataset...")

    preprocess_dataset(TRAINING_DIR, "Training")
    preprocess_dataset(TESTING_DIR, "Testing")

    print("Prétraitement terminé !")
