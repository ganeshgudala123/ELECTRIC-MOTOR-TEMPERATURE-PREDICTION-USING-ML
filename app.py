from flask import Flask, request, jsonify, send_file
import pickle
import numpy as np

app = Flask(__name__)

# Load model once
model = pickle.load(open("model.pkl", "rb"))

# Serve HTML file
@app.route("/")
def home():
    return send_file("index.html")

# Prediction API
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        features = np.array([[
            float(data["ambient"]),
            float(data["coolant"]),
            float(data["u_d"]),
            float(data["u_q"]),
            float(data["motor_speed"]),
            float(data["torque"])
        ]])

        prediction = model.predict(features)

        return jsonify({
            "prediction": round(float(prediction[0]), 2)
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        })

if __name__ == "__main__":
    app.run(debug=True)
