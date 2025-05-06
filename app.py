from flask import Flask, request, jsonify
import numpy as np
import joblib

app = Flask(__name__)

# Model ve LabelEncoder'ı yükle
model = joblib.load("heart_rate_emotion_model1.pkl")
le = joblib.load("label_encoder1.pkl")

@app.route("/tahmin", methods=["POST"])
def tahmin():
    try:
        data = request.get_json()
        bpm = data.get("bpm")

        if bpm is None:
            return jsonify({"error": "BPM verisi eksik"}), 400

        input_data = np.array([[float(bpm)]])
        prediction = model.predict(input_data)
        predicted_emotion = le.inverse_transform(prediction)[0]

        return jsonify({"duygu": predicted_emotion})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
