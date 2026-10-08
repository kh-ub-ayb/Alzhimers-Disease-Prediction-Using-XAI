from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
import os
from lime import lime_image
from skimage.segmentation import mark_boundaries

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

import os
# ===============================
# LOAD MODELS (Tolerate missing models due to Kaggle RAM limits)
# ===============================
efficientnet_model = tf.keras.models.load_model("models/efficientnet_model.keras") if os.path.exists("models/efficientnet_model.keras") else None
cnn_model = tf.keras.models.load_model("models/cnn_model.keras") if os.path.exists("models/cnn_model.keras") else None
resnet_model = tf.keras.models.load_model("models/resnet50_model.keras") if os.path.exists("models/resnet50_model.keras") else None

# ===============================
# LABEL MAP (SAME AS GRADIO)
# ===============================
label_map = {
    "0": "NonDemented",
    "1": "VeryMildDemented",
    "2": "MildDemented",
    "3": "ModerateDemented"
}

# ===============================
# GRAD-CAM (SAME AS GRADIO)
# ===============================
def make_gradcam_heatmap(img_array, model, model_name):
    try:
        if model_name == "efficientnet":
            inner_model_name = "efficientnetb0"
            last_conv_layer_name = "top_conv"
        elif model_name == "resnet":
            inner_model_name = "resnet50"
            last_conv_layer_name = "conv5_block3_out"
        elif model_name == "cnn":
            inner_model_name = None
            last_conv_layer_name = "conv2d_2"
        else:
            return None

        if inner_model_name:
            # Extract nested base model
            inner_model = model.get_layer(inner_model_name)
            
            # Model that outputs the last conv layer AND the inner model's final output
            inner_grad_model = tf.keras.models.Model(
                inputs=[inner_model.inputs], 
                outputs=[inner_model.get_layer(last_conv_layer_name).output, inner_model.output]
            )
            
            # Build the classifier head (everything after the inner model)
            classifier_input = tf.keras.Input(shape=inner_model.output.shape[1:])
            x = classifier_input
            
            inner_idx = 0
            for i, layer in enumerate(model.layers):
                if layer.name == inner_model_name:
                    inner_idx = i
                    break
                    
            for layer in model.layers[inner_idx+1:]:
                x = layer(x)
            classifier_model = tf.keras.models.Model(classifier_input, x)
            
            # Compute gradients
            with tf.GradientTape() as tape:
                conv_outputs, inner_preds = inner_grad_model(img_array)
                tape.watch(conv_outputs)
                predictions = classifier_model(inner_preds)
                pred_index = int(tf.argmax(predictions[0]))
                loss = predictions[:, pred_index]
                
        else:
            # Handle non-nested flat models (like the CNN)
            grad_model = tf.keras.models.Model(
                inputs=model.inputs,
                outputs=[model.get_layer(last_conv_layer_name).output, model.output]
            )
            with tf.GradientTape() as tape:
                conv_outputs, predictions = grad_model(img_array)
                pred_index = int(tf.argmax(predictions[0]))
                loss = predictions[:, pred_index]

        grads = tape.gradient(loss, conv_outputs)

        if grads is None:
            return None

        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]

        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        heatmap = tf.maximum(heatmap, 0)
        max_val = tf.reduce_max(heatmap)

        if max_val == 0:
            return None

        heatmap /= max_val
        return heatmap.numpy()

    except Exception as e:
        print("GradCAM Error:", e)
        return None


def overlay_heatmap(heatmap, image, alpha=0.4, colormap=cv2.COLORMAP_JET):
    # Resize the float heatmap FIRST to get smooth interpolation
    heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    
    # Mask the heatmap to 0 outside the brain. 0 maps to dark blue in JET colormap.
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    heatmap_resized[mask == 0] = 0.0
    
    # Convert to uint8 and apply colormap
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, colormap)
    
    superimposed_img = cv2.addWeighted(heatmap_color, alpha, image, 1 - alpha, 0)
    return heatmap_color, superimposed_img

# ===============================
# BRAIN CROPPING
# ===============================
def crop_brain_contour(image_np):
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        return image_np[y:y+h, x:x+w]
    return image_np

# ===============================
# ROUTES
# ===============================
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        file = request.files["image"]
        model_choice = request.form["model"]

        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # LOAD AND CROP ORIGINAL IMAGE
        raw_pil = Image.open(filepath).convert("RGB")
        raw_np = np.array(raw_pil)
        
        # Crop the black background
        cropped_np = crop_brain_contour(raw_np)
        
        # Resize to model input size
        original_image = Image.fromarray(cropped_np).resize((224, 224))
        img_array_raw = np.array(original_image)
        img_array_raw = np.expand_dims(img_array_raw, axis=0)

        # COPY FOR MODEL
        img_array = img_array_raw.copy()

        # ===============================
        # MODEL-SPECIFIC PREPROCESSING
        # ===============================
        if model_choice == "efficientnet":
            model = efficientnet_model
            img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)

        elif model_choice == "cnn":
            model = cnn_model
            img_array = img_array / 255.0

        elif model_choice == "resnet":
            model = resnet_model
            img_array = tf.keras.applications.resnet50.preprocess_input(img_array)

        else:
            return "Invalid Model"

        # ===============================
        # PREDICTION
        # ===============================
        if model is None:
            return f"<b>Error:</b> You selected '{model_choice}', but that model was not found in the 'models' folder. Because Kaggle crashed, we only have the EfficientNet model available. Please click 'Go Back', select 'EfficientNet', and try again!"

        preds = model.predict(img_array)[0]

        class_idx = int(np.argmax(preds))
        confidence = preds[class_idx]
        result = label_map[str(class_idx)]

        print("Preds:", preds)
        print("Predicted:", result)

        # ===============================
        # SAVE GRAPH
        # ===============================
        classes = list(label_map.values())
        plt.figure()
        plt.bar(classes, preds)
        plt.xticks(rotation=30)
        plt.title("Prediction Confidence")

        graph_path = os.path.join("static/uploads", "graph.png")
        plt.savefig(graph_path)
        plt.close()

        # ===============================
        # GRAD-CAM
        # ===============================
        heatmap = make_gradcam_heatmap(img_array_raw, model, model_choice)

        if heatmap is not None:
            raw_heatmap, gradcam_image = overlay_heatmap(heatmap, np.array(original_image))
            raw_heatmap_path = os.path.join("static/uploads", "raw_heatmap.jpg")
            gradcam_path = os.path.join("static/uploads", "gradcam.jpg")
            cv2.imwrite(raw_heatmap_path, raw_heatmap)
            cv2.imwrite(gradcam_path, gradcam_image)
        else:
            raw_heatmap_path = filepath
            gradcam_path = filepath

        # ===============================
        # LIME
        # ===============================
        lime_path = None
        lime_chart_path = None
        try:
            explainer = lime_image.LimeImageExplainer(random_state=42)
            
            def predict_fn(images):
                # Apply model-specific preprocessing
                if model_choice == "efficientnet":
                    processed = tf.keras.applications.efficientnet.preprocess_input(images.copy())
                elif model_choice == "resnet":
                    processed = tf.keras.applications.resnet50.preprocess_input(images.copy())
                elif model_choice == "cnn":
                    processed = images.copy() / 255.0
                else:
                    processed = images.copy()
                return model.predict(processed, verbose=0)
            
            explanation = explainer.explain_instance(
                img_array_raw[0].astype('double'), 
                predict_fn, 
                top_labels=1, 
                hide_color=0, 
                num_samples=100
            )
            
            temp, mask = explanation.get_image_and_mask(
                explanation.top_labels[0], 
                positive_only=False, 
                num_features=5, 
                hide_rest=False
            )
            
            img_boundry = mark_boundaries(temp / 255.0, mask)
            
            # Save the LIME image
            img_boundry_uint8 = (img_boundry * 255).astype(np.uint8)
            img_boundry_bgr = cv2.cvtColor(img_boundry_uint8, cv2.COLOR_RGB2BGR)
            
            lime_path_full = os.path.join("static/uploads", "lime.jpg")
            cv2.imwrite(lime_path_full, img_boundry_bgr)
            lime_path = lime_path_full
            
            # LIME Bar Chart
            label = explanation.top_labels[0]
            exp = explanation.local_exp[label]
            exp.sort(key=lambda x: abs(x[1]), reverse=True)
            top_features = exp[:5]
            
            features = [f"Region {x[0]}" for x in top_features]
            weights = [x[1] for x in top_features]
            
            plt.figure(figsize=(8, 4))
            colors = ['#22c55e' if w > 0 else '#ef4444' for w in weights] # Tailwind green/red
            plt.barh(features[::-1], weights[::-1], color=colors[::-1])
            plt.title("LIME Explanation (Positive/Negative Regions)", color="white", fontsize=16, fontweight='bold')
            
            # Make the chart transparent for glassmorphism
            plt.gca().set_facecolor((0, 0, 0, 0))
            plt.gcf().patch.set_facecolor((0, 0, 0, 0))
            plt.gca().tick_params(colors='white', labelsize=14)
            for spine in plt.gca().spines.values():
                spine.set_color('white')
                spine.set_linewidth(1.5)
                
            plt.tight_layout()
            
            lime_chart_path_full = os.path.join("static/uploads", "lime_chart.png")
            plt.savefig(lime_chart_path_full, transparent=True)
            plt.close()
            lime_chart_path = lime_chart_path_full
            
        except Exception as e:
            print("LIME Error:", e)

        return render_template(
            "result.html",
            prediction=result,
            confidence=round(confidence * 100, 2),
            image=filepath,
            raw_heatmap=raw_heatmap_path,
            gradcam=gradcam_path,
            lime=lime_path,
            lime_chart=lime_chart_path,
            graph=graph_path
        )

    except Exception as e:
        return f"Error: {str(e)}"

# ===============================
if __name__ == "__main__":
    app.run(debug=True)