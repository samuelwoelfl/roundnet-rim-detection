import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelEncoder
import tensorflowjs as tfjs

# Dateien laden
with open("data/sensor_data-2.json", "r") as f:
    sensor_data = json.load(f)

with open("data/labels-2.json", "r") as f:
    labels = json.load(f)

# JSON in DataFrame umwandeln
sensor_df = pd.DataFrame(sensor_data)
sensor_df["timestamp"] = pd.to_datetime(sensor_df["timestamp"])  # Timestamp umwandeln

label_df = pd.DataFrame(labels)
label_df["timestamp"] = pd.to_datetime(label_df["timestamp"])  # Timestamp umwandeln

# 🔹 Gesamte Bewegung berechnen (Euklidische Norm der x, y, z-Werte)
sensor_df["magnitude"] = np.sqrt(sensor_df["x"]**2 + sensor_df["y"]**2 + sensor_df["z"]**2)

# Standardmäßig alle Werte auf "Unknown" setzen
sensor_df["label"] = "Clean"

# Labels rückwärts zuweisen (bis zum vorherigen Label)
for i in range(len(label_df) - 1, 0 , -1):  # Von hinten nach vorne durchgehen
    current_time = label_df.loc[i, "timestamp"]
    prev_time = label_df.loc[i - 1, "timestamp"]
    label = label_df.loc[i, "label"]
    
    sensor_df.loc[(sensor_df["timestamp"] >= prev_time) & (sensor_df["timestamp"] < current_time), "label"] = label

# 🔹 Label-Encoding ("Clean" = 0, "Not Clean" = 1)
encoder = LabelEncoder()
sensor_df["label"] = encoder.fit_transform(sensor_df["label"])

# 🔹 Features für CNN vorbereiten (Fenster von 1 Sekunde)
window_size = 20  # Fenstergröße (z.B. 1 Sekunde, wenn 20 Hz)
X, y = [], []

for i in range(len(sensor_df) - window_size):
    X.append(sensor_df.iloc[i:i+window_size][["x", "y", "z"]].values)
    y.append(sensor_df.iloc[i+window_size]["label"])

X = np.array(X)
y = np.array(y)

# 🔹 Train-Test-Split (80% Training, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 🔹 CNN-Modell definieren
model = keras.Sequential([
    layers.Conv1D(filters=32, kernel_size=3, activation="relu", input_shape=(window_size, 3)),
    layers.MaxPooling1D(pool_size=2),
    layers.Conv1D(filters=64, kernel_size=3, activation="relu"),
    layers.MaxPooling1D(pool_size=2),
    layers.Flatten(),
    layers.Dense(64, activation="relu"),
    layers.Dense(1, activation="sigmoid")  # Binäre Klassifikation (0 = Clean, 1 = Not Clean)
])

# 🔹 Modell kompilieren
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# 🔹 Modell trainieren
early_stopping = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping])

# 🔹 Modell testen
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5).astype(int)  # Threshold für binäre Klassifikation

model.save("model.keras")
tfjs.converters.save_keras_model(model, "tfjs_model")

# 🔹 Visualisierung der Vorhersagen
plt.figure(figsize=(12, 6))
plt.plot(y_test[:100], label="Echte Labels", color="green", marker="o", linestyle="None")
plt.plot(y_pred[:100], label="Vorhersagen", color="red", marker="x", linestyle="None")
plt.xlabel("Test-Samples")
plt.ylabel("Label (0 = Clean, 1 = Not Clean)")
plt.title("Vergleich: Vorhersagen vs. Echte Labels")
plt.legend()
plt.show()

