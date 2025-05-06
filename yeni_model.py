import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

# Veriyi yükle
df = pd.read_csv("heart_rate_emotion_dataset.csv")

# Sadece 3 sınıfı kullan
selected_classes = ["happy", "sad", "neutral"]
df_filtered = df[df["Emotion"].isin(selected_classes)]

# Özellik ve hedef
X = df_filtered[["HeartRate"]]
y = df_filtered["Emotion"]

# Etiketleri sayısal hale getir
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Eğitim ve test verisine ayır
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Modeli oluştur ve eğit
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Tahmin yap ve performansı yazdır
y_pred = model.predict(X_test)
print("Sınıflar:", le.classes_)
print("\nPerformans:\n")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# 🔍 Dışarıdan BPM girerek tahmin
bpm_input = float(input("\nBir BPM değeri girin: "))
prediction = model.predict([[bpm_input]])
predicted_emotion = le.inverse_transform(prediction)[0]
print(f"Tahmin edilen duygu: {predicted_emotion}")
