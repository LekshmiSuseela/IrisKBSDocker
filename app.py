from flask import Flask, request, jsonify
import joblib
import numpy as np
import mlflow.pyfunc
import os

app = Flask(__name__)

# Load the model (replace with MLflow model if using registry)
MODEL_PATH = "model.joblib"

try:
    model_path = os.path.join(os.getcwd(), "model.joblib")
    model = joblib.load(model_path)
    print(f"✅ Model loaded successfully from {model_path}")
except Exception as e:
    print(f"⚠️ Failed to load model: {e}")
    model = None

@app.route('/')
def home():
    return jsonify({"message": "IRIS Prediction API is running!"})

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy"}), 200

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    data = request.get_json()
    if not data or "features" not in data:
        return jsonify({"error": "Please provide input features"}), 400

    features = np.array(data["features"]).reshape(1, -1)
    prediction = model.predict(features).tolist()

    return jsonify({"prediction": prediction})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
