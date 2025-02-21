import json
import numpy as np
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from scipy.fft import rfft, rfftfreq
import matplotlib.pyplot as plt

# JSON-Datei laden
def load_data(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    return data

data = load_data("labeled_hits.json")

# Entfernt die Zeitzone aus dem Timestamp
def parse_timestamp(timestamp):
    return datetime.datetime.strptime(timestamp[:-1], "%Y-%m-%dT%H:%M:%S.%f")


# Padding oder Trunkieren auf max_fft_length
def pad_or_truncate(arr, length):
    if len(arr) < length:
        return np.pad(arr, (0, length - len(arr)))
    return arr[:length]


# Daten extrahieren und vorbereiten
def process_data(data):
    X = []  # Features
    y = []  # Labels
    max_fft_length = 0

    # 1. Maximale FFT-Länge finden
    for entry in data:
        num_samples = len(entry["data"])
        fft_length = len(rfft(np.zeros(num_samples)))  # FFT-Länge basierend auf Anzahl der Samples
        max_fft_length = max(max_fft_length, fft_length)

    # 2. Daten pro Sample verarbeiten
    for entry in data:
        label = entry["label"]
        
        # Beschleunigungswerte extrahieren
        x_vals = np.array([d["x"] for d in entry["data"]])
        y_vals = np.array([d["y"] for d in entry["data"]])
        z_vals = np.array([d["z"] for d in entry["data"]])

        # Gesamtbeschleunigung berechnen
        a_total = np.sqrt(x_vals**2 + y_vals**2 + z_vals**2)

        # FFT anwenden (nur auf a_total)
        freq_total = np.abs(rfft(a_total))        

        freq_total = pad_or_truncate(freq_total, max_fft_length)

        # Features speichern (nur Gesamt-FFT)
        X.append(freq_total)
        y.append(label)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)

X, y = process_data(data)








# 💡 Modell mit Dropout & BatchNorm
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

        self.fc4 = nn.Linear(32, 3)  # Ändere die Ausgabeschicht auf 3 für 3 Klassen

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


# 📌 Datenaugmentation: Zufälliges Rauschen hinzufügen
def augment_data(X, noise_factor=0.02):
    noise = np.random.normal(0, noise_factor, X.shape)
    return X + noise


# 📊 Daten vorbereiten
X_augmented = augment_data(X)  # Augmentation nur fürs Training
X_tensor = torch.tensor(X_augmented, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)

dataset = TensorDataset(X_tensor, y_tensor)

# 🏋️‍♂️ Trainings- & Testsplit
train_size = int(0.8 * len(dataset))  # 80% Training, 20% Test
val_size = int(0.1 * len(dataset))    # 10% Validation
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 🏋️‍♂️ Training mit Validation
def train_model(model, train_loader, val_loader, epochs=500):
    model.train()
    best_val_loss = float("inf")

    for epoch in range(epochs):
        total_loss = 0
        model.train()

        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # 📊 Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                predicted = torch.argmax(outputs, dim=1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = correct / total

        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}, Val Acc={val_accuracy:.4f}")

        # Speichere bestes Modell
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({"input_size": input_size, "model_state": model.state_dict()}, "best_model.pth")


# 🚀 Starte Training
input_size = X.shape[1]
model = VibrationClassifier(input_size)

# class_weights = torch.tensor([1.0, 1.0, 2.0])  # Gewichtung für jede Klasse (z.B. Label 0 mit höherem Gewicht)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

train_model(model, train_loader, val_loader, epochs=500)

# 🔥 Speichere finales Modell
torch.save({"input_size": input_size, "model_state": model.state_dict()}, "final_model.pth")


# 📊 Modell nach dem Training testen
def test_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            predictions.extend(predicted.numpy())
            true_labels.extend(labels.numpy())
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    print(f"🔥 Test Accuracy: {accuracy * 100:.2f}%")
    return predictions, true_labels


# 📌 Test-Loader wird erst NACH dem Training ausgeführt!
predictions, true_labels = test_model(model, test_loader)

# Visualisierung
plt.figure(figsize=(10, 5))
plt.plot(true_labels, label="True Labels", linestyle="None", marker="o", markersize=10)
plt.plot(predictions, label="Predictions", linestyle="None", marker="x", markersize=10)
plt.xlabel("Test Samples")
plt.ylabel("Label")
plt.legend()
plt.title("Vergleich der Vorhersagen mit den echten Labels")
plt.show()