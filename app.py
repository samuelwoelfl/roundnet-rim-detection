from flask import Flask, request, jsonify, render_template
import numpy as np
import torch
import torch.nn as nn
from scipy.fft import rfft


def pad_or_truncate(arr, length):
        pad_size = max(0, int(length - len(arr)))  # Immer >= 0
        return np.pad(arr, (0, pad_size)) if pad_size > 0 else arr[:length]


app = Flask(__name__)

# 🧠 Trainiertes Modell laden
class VibrationClassifier(nn.Module):
    def __init__(self, input_size):
        super(VibrationClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(0.3)

        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(0.3)

        self.fc3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.dropout3 = nn.Dropout(0.3)

        self.fc4 = nn.Linear(32, 3)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = self.relu(self.bn3(self.fc3(x)))
        x = self.dropout3(x)
        x = self.fc4(x)
        return x

# Modell initialisieren und laden
checkpoint = torch.load("final_model.pth")
input_size = checkpoint["input_size"]  # Gespeicherte FFT-Größe abrufen

# Modell mit der richtigen Größe initialisieren
model = VibrationClassifier(input_size=input_size)
model.load_state_dict(checkpoint["model_state"])
model.eval()

def process_gyroscope_data(gyro_data):
    x_vals = np.array(gyro_data["x"])
    y_vals = np.array(gyro_data["y"])
    z_vals = np.array(gyro_data["z"])

    # Check, ob die Listen leer sind
    if len(x_vals) == 0 or len(y_vals) == 0 or len(z_vals) == 0:
        return None  # Falls keine Daten, gib None zurück

    # Gesamtbeschleunigung berechnen
    a_total = np.sqrt(x_vals**2 + y_vals**2 + z_vals**2)

    # FFT auf Gesamtbeschleunigung anwenden
    freq_total = np.abs(rfft(a_total))

    # Padding oder Trunkieren auf richtige Länge
    max_fft_length = input_size  # Kein /3 mehr, da nur ein Signal

    freq_total = pad_or_truncate(freq_total, max_fft_length)

    # Features = Nur die FFT der Gesamtbeschleunigung
    return torch.tensor(freq_total, dtype=torch.float32).unsqueeze(0)  # Batch-Dimension


# 🎯 API-Endpoint für Echtzeit-Vorhersage
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    processed_data = process_gyroscope_data(data)

    if processed_data is None:
        return jsonify({"error": "Keine Gyroskop-Daten empfangen"}), 400  # HTTP 400 = Bad Request

    output = model(processed_data)
    # ✅ Softmax auf die Vorhersage anwenden, um Wahrscheinlichkeiten zu bekommen
    probabilities = torch.softmax(output, dim=1).squeeze().detach().numpy()
    max_prob = np.max(probabilities)
    prediction = np.argmax(probabilities)  # 0 oder 1

    # ✅ Wenn die höchste Wahrscheinlichkeit < 60%, kein sicheres Ergebnis
    if max_prob < 0.8:
        print("Kein Muster erkannt")
        return jsonify({"prediction": "unclassified"})

    print(f"Vorhersage: {prediction} mit {max_prob:.2f} Wahrscheinlichkeit")
    return jsonify({"prediction": int(prediction), "confidence": float(max_prob)})

# 📄 UI-Startseite
@app.route("/")
def index():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=6010)