import os
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

# Load the model
model_path = r"d:\New folder\PROJECTS\MajorProject\alzheimers-disease-prediction\Project Code\models\efficientnet_model.keras"
model = tf.keras.models.load_model(model_path)

# Pick an image from Sample_Images
sample_image_dir = r"d:\New folder\PROJECTS\MajorProject\alzheimers-disease-prediction\Sample_Images"
sample_images = [f for f in os.listdir(sample_image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
if not sample_images:
    print("No sample images found.")
    exit(1)

test_image_path = os.path.join(sample_image_dir, sample_images[0])

def crop_brain(image_np):
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        return image_np[y:y+h, x:x+w]
    return image_np

# Read original
img_pil = Image.open(test_image_path).convert("RGB")
img_np_original = np.array(img_pil)

# Crop
img_np_cropped = crop_brain(img_np_original)

# Preprocess Original
img_orig_resized = cv2.resize(img_np_original, (224, 224))
img_orig_expanded = np.expand_dims(img_orig_resized, axis=0)
img_orig_preprocessed = tf.keras.applications.efficientnet.preprocess_input(img_orig_expanded)

# Preprocess Cropped
img_crop_resized = cv2.resize(img_np_cropped, (224, 224))
img_crop_expanded = np.expand_dims(img_crop_resized, axis=0)
img_crop_preprocessed = tf.keras.applications.efficientnet.preprocess_input(img_crop_expanded)

# Predict
pred_orig = model.predict(img_orig_preprocessed, verbose=0)[0]
pred_crop = model.predict(img_crop_preprocessed, verbose=0)[0]

print(f"Testing on {sample_images[0]}")
print("Original Prediction:", np.argmax(pred_orig), "Confidence:", np.max(pred_orig))
print("Cropped Prediction:", np.argmax(pred_crop), "Confidence:", np.max(pred_crop))
