import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.stats import kurtosis, skew
from scipy.fft import fft
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# 🔹 Daten laden
with open("data/sensor_data-2.json", "r") as f:
    sensor_data = json.load(f)

with open("data/labels-2.json", "r") as f:
    labels = json.load(f)

# 🔹 JSON in DataFrame umwandeln
sensor_df = pd.DataFrame(sensor_data)
sensor_df["timestamp"] = pd.to_datetime(sensor_df["timestamp"])

label_df = pd.DataFrame(labels)
label_df["timestamp"] = pd.to_datetime(label_df["timestamp"])

# Standardmäßig alle Werte auf "Unknown" setzen
sensor_df["label"] = "Unknown"

# Labels rückwärts zuweisen (bis zum vorherigen Label)
for i in range(len(label_df) - 1, 0 , -1):  # Von hinten nach vorne durchgehen
    current_time = label_df.loc[i, "timestamp"]
    prev_time = label_df.loc[i - 1, "timestamp"]
    label = label_df.loc[i, "label"]
    
    sensor_df.loc[(sensor_df["timestamp"] >= prev_time) & (sensor_df["timestamp"] < current_time), "label"] = label

# 🔹 Label Encoding (Clean = 0, Not Clean = 1)
sensor_df["label"] = sensor_df["label"].map({"Clean": 0, "Not Clean": 1, "Unknown": np.nan})
sensor_df.dropna(inplace=True)  # Entferne "Unknown"

# 🔹 Feature Extraction
window_size = 20  # Anzahl der Messwerte pro Fenster
features = []
labels = []

def compute_fft(signal):
    """Berechnet die dominante Frequenz und die spektrale Energie."""
    fft_vals = np.abs(fft(signal))
    fft_freqs = np.fft.fftfreq(len(signal))
    dominant_freq = fft_freqs[np.argmax(fft_vals)]
    spectral_energy = np.sum(fft_vals**2)
    return dominant_freq, spectral_energy

for i in range(0, len(sensor_df) - window_size, window_size):  
    window = sensor_df.iloc[i : i + window_size]
    label = window["label"].mode()[0]  

    # FFT-Features berechnen
    dom_freq_x, spec_energy_x = compute_fft(window["x"].values)
    dom_freq_y, spec_energy_y = compute_fft(window["y"].values)
    dom_freq_z, spec_energy_z = compute_fft(window["z"].values)

    feature_vector = [
        window["x"].mean(), window["y"].mean(), window["z"].mean(),
        window["x"].std(), window["y"].std(), window["z"].std(),
        window["x"].max() - window["x"].min(),
        window["y"].max() - window["y"].min(),
        window["z"].max() - window["z"].min(),
        np.sqrt((window["x"]**2 + window["y"]**2 + window["z"]**2).sum()),
        dom_freq_x, spec_energy_x,
        dom_freq_y, spec_energy_y,
        dom_freq_z, spec_energy_z
    ]
    
    features.append(feature_vector)
    labels.append(label)

X = np.array(features)
y = np.array(labels)

# 🔹 Train-/Test-Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 🔹 CNN-Datenform (Samples, Timesteps, Features)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# 🔹 Labels für Keras One-Hot-Encoding
y_train = to_categorical(y_train, num_classes=2)
y_test = to_categorical(y_test, num_classes=2)

# 🔹 CNN-Modell erstellen
model = Sequential([
    Conv1D(32, kernel_size=3, activation="relu", input_shape=(X_train.shape[1], 1)),
    MaxPooling1D(pool_size=2),
    Conv1D(64, kernel_size=3, activation="relu"),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(64, activation="relu"),
    Dropout(0.5),
    Dense(2, activation="softmax")
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# 🔹 Training
model.fit(X_train, y_train, epochs=30, batch_size=8, validation_data=(X_test, y_test))

# 🔹 Modellbewertung
loss, acc = model.evaluate(X_test, y_test)
print(f"Testgenauigkeit: {acc:.2%}")

# 🔹 Vorhersagen auf Testdaten
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(y_test, axis=1)

# 🔹 Visualisierung der Vorhersagen
plt.figure(figsize=(10, 4))
plt.plot(y_test_classes[:50], label="Echte Labels", marker="o")
plt.plot(y_pred_classes[:50], label="Vorhersagen", linestyle="dashed", marker="x")
plt.legend()
plt.xlabel("Datenpunkt")
plt.ylabel("Label (0=Clean, 1=Not Clean)")
plt.title("Vergleich: Wahre Labels vs. Vorhersagen")
plt.show()