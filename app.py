from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
import pickle

# Flask uygulaması
app = Flask(__name__)

# -------------------------------
# Model ve yardımcı dosyaları yükle
model = tf.keras.models.load_model("emotion_recognition_modelv9.h5")

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

emotion_classes = list(encoder.classes_)

# -------------------------------
# API endpoint: /predict
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json.get("data")
        if data is None:
            return jsonify({"error": "Veri bulunamadı"}), 400

        if len(data) != 600:
            return jsonify({"error": f"600 veri bekleniyor, ama {len(data)} veri geldi"}), 400

        # Normalizasyon
        normalized = scaler.transform(np.array(data).reshape(-1, 1)).flatten()

        # Modele uygun giriş oluştur
        X_input = np.array(normalized).reshape((1, 600, 1))

        # Tahmin yap
        pred = model.predict(X_input, verbose=0)
        predicted_label = np.argmax(pred)
        predicted_emotion = emotion_classes[predicted_label]

        return jsonify({"emotion": predicted_emotion})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -------------------------------
# Ana dosya olarak çalıştırıldığında
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
