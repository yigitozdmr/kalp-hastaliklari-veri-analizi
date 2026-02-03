import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Veri setini oku
df = pd.read_csv("data/heart (1).csv")

# Özellikler ve hedef değişken
X = df.drop("target", axis=1)   # target sütun adını kontrol et
y = df["target"]

# Eğitim / test bölme
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Başarım
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Model doğruluğu: {acc:.2f}")

# Modeli kaydet
joblib.dump(model, "model/heart_model.joblib")
print("Model kaydedildi: model/heart_model.joblib")
