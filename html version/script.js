let model;
const windowSize = 20; // Gleiche Fenstergröße wie beim Training
let sensorData = [];

async function loadModel() {
    try {
        model = await tf.loadLayersModel('tfjs_model/model.json'); // Stelle sicher, dass der Pfad richtig ist
        document.getElementById("status").innerText = "Modell geladen!";
    } catch (error) {
        console.error("Fehler beim Laden des Modells:", error);
        document.getElementById("status").innerText = "Fehler beim Laden des Modells!";
    }
}

async function predict() {
    if (sensorData.length >= windowSize) {
        let input = tf.tensor([sensorData.slice(-windowSize)]); // Letzte Werte nehmen
        let prediction = model.predict(input);
        let result = await prediction.data();
        
        document.getElementById("prediction").innerText = result[0] > 0.5 
            ? "Vorhersage: Not Clean 🚨" 
            : "Vorhersage: Clean ✅";
    }
}

// 🔹 Bewegungssensor aktivieren (Gyroskop / Beschleunigungssensor)
window.addEventListener("devicemotion", (event) => {
    let x = event.accelerationIncludingGravity.x || 0;
    let y = event.accelerationIncludingGravity.y || 0;
    let z = event.accelerationIncludingGravity.z || 0;

    let magnitude = Math.sqrt(x*x + y*y + z*z);
    
    sensorData.push([x, y, z]);
    if (sensorData.length > windowSize) {
        sensorData.shift(); // Älteste Werte entfernen
    }
    
    predict(); // Kontinuierlich Vorhersagen machen
});

// 🔹 Modell laden
loadModel();