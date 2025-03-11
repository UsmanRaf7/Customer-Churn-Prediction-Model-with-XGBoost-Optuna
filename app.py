from flask import Flask, request, jsonify
import joblib
import numpy as np

# Load the trained model
model = joblib.load("churn_model.pkl")

# Initialize Flask app
app = Flask(__name__)

@app.route("/")
def home():
    return "Churn Prediction API is Running!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get JSON data from request
        data = request.json
        features = np.array(data["features"]).reshape(1, -1)  # Convert to 2D array

        # Make prediction
        prediction = model.predict(features)
        probability = model.predict_proba(features)[:, 1]  # Get probability for class 1

        return jsonify({
            "prediction": int(prediction[0]),
            "churn_probability": float(probability[0])
        })
    
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)