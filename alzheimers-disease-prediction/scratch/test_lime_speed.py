import time
import os
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from lime import lime_image

# Load the model
model_path = r"d:\New folder\PROJECTS\MajorProject\alzheimers-disease-prediction\Project Code\models\efficientnet_model.keras"
model = tf.keras.models.load_model(model_path)

# Pick an image from Sample_Images
sample_image_dir = r"d:\New folder\PROJECTS\MajorProject\alzheimers-disease-prediction\Sample_Images"
sample_images = [f for f in os.listdir(sample_image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
test_image_path = os.path.join(sample_image_dir, sample_images[0])

# Read and resize
img_pil = Image.open(test_image_path).convert("RGB").resize((224, 224))
img_np = np.array(img_pil)

# LIME Setup
explainer = lime_image.LimeImageExplainer(random_state=42)
def predict_fn(images):
    processed = tf.keras.applications.efficientnet.preprocess_input(images.copy())
    return model.predict(processed, verbose=0)

def test_speed(num_samples, batch_size):
    start = time.time()
    explainer.explain_instance(
        img_np.astype('double'), 
        predict_fn, 
        top_labels=1, 
        hide_color=0, 
        num_samples=num_samples,
        batch_size=batch_size
    )
    end = time.time()
    print(f"num_samples={num_samples}, batch_size={batch_size} took {end-start:.2f} seconds")

# Warm up
test_speed(10, 10)

# Tests
test_speed(1000, 10) # default batch size is 10
test_speed(1000, 64)
test_speed(500, 64)
test_speed(500, 100)
