import json
import numpy as np
import matplotlib.pyplot as plt
import math
from datetime import datetime, timedelta

# Hilfsfunktion zum Umwandeln von ISO-Timestamps
def parse_timestamp(ts):
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))
  
# Hilfsfunktion: Finde den Index des nächstgelegenen Zeitstempels
def find_nearest_index(timestamp_list, target_timestamp):
    return min(range(len(timestamp_list)), key=lambda i: abs(timestamp_list[i] - target_timestamp))

# Sensor-Daten laden
with open("data/sensor_data-2.json", "r") as file:
    sensor_data = json.load(file)

# Labels laden
with open("data/labels-2.json", "r") as file:
    labels_data = json.load(file)

# Daten extrahieren
sensor_timestamps = [parse_timestamp(entry["timestamp"]) for entry in sensor_data]
x_vals = np.array([entry["x"] for entry in sensor_data])
y_vals = np.array([entry["y"] for entry in sensor_data])
z_vals = np.array([entry["z"] for entry in sensor_data])

# Labels extrahieren
labels_timestamps = [parse_timestamp(entry["timestamp"]) for entry in labels_data]
labels_values = [1 if entry["label"] == "Not Clean" else 0 for entry in labels_data]

# Gesamte Beschleunigung berechnen
acceleration = np.sqrt(x_vals**2 + y_vals**2 + z_vals**2)



def identify_segments(sensor_data, acc_data, movement_threshold, stillness_max, stillness_min_duration, min_segment_length, new_hit_threshold, recent_movement_window, extra_start_points, min_peak_threshold):
  segments = []
  current_segment = []  # Immer eine Liste
  in_motion = False
  low_movement_counter = 0
  start_index = None

  # Segmentierung der Sensor-Daten
  for i in range(len(acc_data)):
    if acc_data[i] > movement_threshold:  
        if not in_motion:
            start_index = max(0, i - extra_start_points)  
            current_segment = sensor_data[start_index:i]  
        in_motion = True
        low_movement_counter = 0  

    elif in_motion:  
        if acc_data[i] < stillness_max:
            low_movement_counter += 1
        else:
            low_movement_counter = 0  
            
        if low_movement_counter >= stillness_min_duration:
            in_motion = False
            if len(current_segment) >= min_segment_length:  
                segment_acc = [np.sqrt(e["x"]**2 + e["y"]**2 + e["z"]**2) for e in current_segment]
                if max(segment_acc) >= min_peak_threshold:
                    segments.append(current_segment)  

    if in_motion:  
        if len(current_segment) > recent_movement_window:  
            recent_values = acc_data[i - recent_movement_window:i]  
            if np.mean(recent_values) < movement_threshold and acc_data[i] > new_hit_threshold:
                if len(current_segment) >= min_segment_length:  
                    segment_acc = [np.sqrt(e["x"]**2 + e["y"]**2 + e["z"]**2) for e in current_segment]
                    if max(segment_acc) >= min_peak_threshold:
                        segments.append(current_segment)
                start_index = max(0, i - extra_start_points)  
                current_segment = sensor_data[start_index:i]  

        current_segment.append(sensor_data[i])

  return segments


def clean_segments(segments, labels_data, min_segment_length):
  segments_cleaned = []
  
  for i, segment in enumerate(segments):
    segment_start_timestamp = parse_timestamp(segment[0]["timestamp"])
    segment_end_timestamp = parse_timestamp(segment[-1]["timestamp"])
    
    for label in labels_data:
      label_timestamp = parse_timestamp(label["timestamp"])
      if label_timestamp >= segment_start_timestamp - timedelta(milliseconds=100) and label_timestamp <= segment_end_timestamp + timedelta(milliseconds=100): # if label is in segment with a bit range
        # label_found = True
        segment_timestamps = [parse_timestamp(entry["timestamp"]) for entry in segment]
        closest_index_within_segment = min(range(len(segment_timestamps)), key=lambda i: abs(segment_timestamps[i] - label_timestamp))
        segment_new_end_index = closest_index_within_segment - 10
                
        if segment_new_end_index > 0:
          segment = segment[:segment_new_end_index]
        else:
          segment = segment[1:1]
                  
    if len(segment) > min_segment_length:
      segments_cleaned.append(segment)
          
  return segments_cleaned


def label_segments(segments, labels_data):
  segments_labeled = []  # Speichert Segmente mit Labels
  for i, label in enumerate(labels_data):
    label_timestamp = parse_timestamp(label["timestamp"])
    label_value = 1 if label["label"] == "Clean" else 2 if label["label"] == "Not Clean" else 0
    
    if i == 0:
      label_before_timestamp = parse_timestamp(sensor_data[0]["timestamp"])
    else:
      label_before_timestamp = parse_timestamp(labels_data[i - 1]["timestamp"])
    
    segments_for_label = []
    for segment in reversed(segments):
      segment_start_timestamp = parse_timestamp(segment[0]["timestamp"])
      segment_end_timestamp = parse_timestamp(segment[-1]["timestamp"])
      if segment_start_timestamp >= label_before_timestamp and segment_end_timestamp <= label_timestamp:
        segments_for_label.append(segment)
        break
    
    if len(segments_for_label) > 1:
      segment = segments_for_label[-1]
    
    segments_labeled.append({"label": label_value, "data": segment})
  
  return segments_labeled


def add_quiet_segments(segments, sensor_data):
  for i, segment in enumerate(segments[:]):
    
    segment_start_timestamp = parse_timestamp(segment['data'][0]["timestamp"])
  
    if i == 0:
      new_segment_start = parse_timestamp(sensor_data[0]["timestamp"]) 
    else:
      new_segment_start = parse_timestamp(segments[i - 1]['data'][-1]["timestamp"]) + timedelta(milliseconds=1500)
      
    new_segment_end = segment_start_timestamp - timedelta(milliseconds=200)
    
    closest_index_start = min(range(len(sensor_data)), key=lambda i: abs(parse_timestamp(sensor_data[i]['timestamp']) - new_segment_start))
    closest_index_end = min(range(len(sensor_data)), key=lambda i: abs(parse_timestamp(sensor_data[i]['timestamp']) - new_segment_end))
    
    segment = sensor_data[closest_index_start:closest_index_end]
        
    if new_segment_end > new_segment_start and len(segment) < 2000:
      segments.append({
        'label': 0,
        'data': segment
      })

    
  return segments
    



def remove_segments(segments, indices_to_remove):
  segments_removed = [segment for i, segment in enumerate(segments) if i not in indices_to_remove]
  segments_with_content = [segment for segment in segments_removed if len(segment['data']) > 0]
  return segments_with_content







min_segment_length = 50


segments = identify_segments(
  sensor_data = sensor_data,
  acc_data = acceleration,
  movement_threshold = 0.5,
  stillness_max = 0.5,
  stillness_min_duration = 50,
  min_segment_length = min_segment_length,
  new_hit_threshold = 2.5,
  recent_movement_window = 20,
  extra_start_points = 3,
  min_peak_threshold = 1.5
)

print(f'=========== {len(segments)} segments done ==========')


segments_cleaned = clean_segments(
  segments = segments,
  labels_data = labels_data,
  min_segment_length = min_segment_length,
)

print(f'=========== {len(segments_cleaned)} segments left after cleaning ==========')


segments_labeled = label_segments(
  segments = segments_cleaned,
  labels_data = labels_data,
)

print(f'=========== {len(segments_labeled)} segments gelabelt ==========')


segments_with_quiet = add_quiet_segments(
  segments = segments_labeled,
  sensor_data = sensor_data
)

print(f'=========== {len(segments_with_quiet)} segments mit quiet segments ==========')


segments_labeled_clean = remove_segments(
  segments = segments_with_quiet,
  indices_to_remove = [29, 40, 42, 52, 66]
)

print(f'=========== {len(segments_labeled_clean)} segments immer noch gelabelt ==========')





# Treffer mit Labels als JSON speichern
with open("labeled_hits.json", "w") as outfile:
    json.dump(segments_labeled_clean, outfile, indent=4)

# Visualisierung
plt.figure(figsize=(12, 6))
plt.plot(acceleration, label="Beschleunigung", color="blue")

for segment in segments_labeled_clean:
    indices = [sensor_timestamps.index(parse_timestamp(entry["timestamp"])) for entry in segment["data"]]
    color = "red" if segment["label"] == 2 else "green" if segment["label"] == 1 else "gray"
    plt.axvspan(indices[0], indices[-1], color=color, alpha=0.3)

# Labels als vertikale Linien einzeichnen
for i, label_ts in enumerate(labels_timestamps):
    nearest_index = find_nearest_index(sensor_timestamps, label_ts)  # Nächstgelegenen Index suchen
    label_color = "red" if labels_values[i] == 1 else "green"  # Rot für "Not Clean", Grün für "Clean"
    plt.axvline(x=nearest_index, color=label_color, linestyle="--", alpha=0.7)
    plt.text(nearest_index, max(acceleration), f"{'Not Clean' if labels_values[i] else 'Clean'}", 
             rotation=90, verticalalignment='top', fontsize=10, color="black")

plt.legend()
plt.show()

print(f"{len(segments_labeled)} Treffer wurden isoliert und gelabelt gespeichert.")