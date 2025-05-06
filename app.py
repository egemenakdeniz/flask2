from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder

# Flask uygulaması
app = Flask(__name__)

# CSV dosyasından etiketleri al
df = pd.read_csv("heart_rate_emotion_dataset.csv")
selected_classes = ["happy", "sad", "neutral"]
df_filtered = df[df["Emotion"].isin(selected_classes)]

le = LabelEncoder()
le.fit(df_filtered["Emotion"])

# Modeli yükle
model = load_model("yepyenimodel.keras")

@app.route("/tahmin", methods=["POST"])
def tahmin():
    try:
        data = request.get_json()
        bpm = data.get("bpm")

        if bpm is None:
            return jsonify({"error": "BPM verisi eksik"}), 400

        input_data = np.array([[float(bpm)]])
        prediction = model.predict(input_data)

        predicted_index = np.argmax(prediction)
        predicted_emotion = le.inverse_transform([predicted_index])[0]

        return jsonify({"duygu": predicted_emotion})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
