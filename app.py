from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import cv2
import os

app = Flask(__name__)

# -----------------------------
# Load Model
# -----------------------------
print("Loading model...")
model = tf.keras.models.load_model("tomato_disease_model.h5")
print("Model loaded successfully!")

# -----------------------------
# Upload Folder
# -----------------------------
UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

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
        img = cv2.imread(filepath)

        if img is None:
            return "Error", {}, 0, "Low", "Invalid image"

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)

        prediction = model.predict(img)[0][0]

        healthy_prob = (prediction * 0.9 + 0.05) * 100
        blight_prob = ((1 - prediction) * 0.9 + 0.05) * 100

        probabilities = {
            "Healthy": round(healthy_prob, 2),
            "Early Blight": round(blight_prob, 2)
        }

        if prediction >= 0.5:
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

        if disease == "Healthy":
            recommendation = "No disease detected."
        else:
            recommendation = "Apply fungicide and remove infected leaves."

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
    if "image" not in request.files:
        return "No file uploaded"

    file = request.files["image"]

    if file.filename == "":
        return "No selected file"

    filename = file.filename
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


@app.route("/sample/<type>")
def sample(type):
    if type == "healthy":
        filename = "healthy.jpg"
    else:
        filename = "early_blight.jpg"

    filepath = os.path.join("static/images", filename)

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
        image_path="images/" + filename
    )


@app.route("/about")
def about():
    return render_template("about.html")


# -----------------------------
# Run App
# -----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
