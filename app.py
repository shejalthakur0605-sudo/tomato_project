import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import cv2
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# -----------------------------
# Upload Folder
# -----------------------------
UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# -----------------------------
# Load Model
# -----------------------------
print("Loading model...")
model = tf.keras.models.load_model("tomato_disease_model.h5")
print("Model loaded successfully!")

# -----------------------------
# Disease Information
# -----------------------------
disease_info = {
    "Early Blight": {
        "symptoms": "Brown spots with concentric rings on older leaves.",
        "causes": "Fungal pathogen Alternaria solani.",
        "treatment": "Remove infected leaves and apply fungicide.",
        "prevention": "Avoid overhead watering and rotate crops."
    },
    "Healthy": {
        "symptoms": "Leaves are green and healthy.",
        "causes": "No disease.",
        "treatment": "No treatment required.",
        "prevention": "Maintain proper care."
    }
}

# -----------------------------
# Prediction Function
# -----------------------------
def predict_disease(filepath):
    try:
        print("Reading image:", filepath)

        img = cv2.imread(filepath)

        if img is None:
            print("Image read failed")
            return "Error", {}, 0, "Low", "Invalid image"

        img = cv2.resize(img, (224, 224))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)

        print("Running prediction...")
        prediction = model(img, training=False).numpy()

        pred = prediction[0][0]

        healthy_prob = (pred * 0.9 + 0.05) * 100
        blight_prob = ((1 - pred) * 0.9 + 0.05) * 100

        probabilities = {
            "Healthy": round(healthy_prob, 2),
            "Early Blight": round(blight_prob, 2)
        }

        if pred >= 0.5:
            disease = "Healthy"
            confidence = healthy_prob
        else:
            disease = "Early Blight"
            confidence = blight_prob

        confidence = min(round(confidence, 2), 95)

        if confidence < 60:
            severity = "Low"
        elif confidence < 85:
            severity = "Moderate"
        else:
            severity = "High"

        recommendation = (
            "No disease detected."
            if disease == "Healthy"
            else "Apply fungicide and remove infected leaves."
        )

        return disease, probabilities, confidence, severity, recommendation

    except Exception as e:
        print("Prediction error:", e)
        return "Error", {}, 0, "Low", "Prediction failed"


# -----------------------------
# Routes
# -----------------------------
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "image" not in request.files:
            return "No file uploaded"

        file = request.files["image"]

        if file.filename == "":
            return "No selected file"

        filename = secure_filename(file.filename)

        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        disease, probabilities, confidence, severity, recommendation = predict_disease(filepath)
        info = disease_info.get(disease, {})

        return render_template(
            "result.html",
            prediction=disease,
            confidence=confidence,
            severity=severity,
            probabilities=probabilities,
            recommendation=recommendation,
            info=info,
            image_path="uploads/" + filename
        )

    except Exception as e:
        print("Upload error:", e)
        return "Server error during prediction"


@app.route("/about")
def about():
    return render_template("about.html")


# -----------------------------
# Run App
# -----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
