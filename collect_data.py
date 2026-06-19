# ============================================================
# collect_data.py  —  v2  (Multi-Employee Edition)
# Records real fatigue data from the webcam + keyboard
#
# Usage:
#   python collect_data.py
#
# Output:
#   real_fatigue_data_<employee_id>_<timestamp>.csv
#   Each row: 137 fused features  +  label (0/1/2)
# ============================================================

import cv2
import numpy as np
import csv
import time
import os
import datetime
import threading
import collections
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, Model
from pynput import keyboard as kb   # pip install pynput

# -----------------------------------------------------------
# CNN — kept EXACTLY the same as before (weights unchanged)
# -----------------------------------------------------------
def load_feature_extractor():
    inp = layers.Input(shape=(64, 64, 3), name='input')
    x   = layers.Conv2D(32, (3,3), activation='relu',
                        padding='same', name='conv1_1')(inp)
    x   = layers.BatchNormalization(name='bn1')(x)
    x   = layers.MaxPooling2D((2,2), name='pool1')(x)
    x   = layers.Dropout(0.25, name='drop1')(x)
    x   = layers.Conv2D(64, (3,3), activation='relu',
                        padding='same', name='conv2_1')(x)
    x   = layers.BatchNormalization(name='bn2')(x)
    x   = layers.MaxPooling2D((2,2), name='pool2')(x)
    x   = layers.Dropout(0.25, name='drop2')(x)
    x   = layers.Conv2D(128, (3,3), activation='relu',
                        padding='same', name='conv3_1')(x)
    x   = layers.BatchNormalization(name='bn3')(x)
    x   = layers.MaxPooling2D((2,2), name='pool3')(x)
    x   = layers.Dropout(0.3, name='drop3')(x)
    x   = layers.Flatten(name='flatten')(x)
    x   = layers.Dense(128, activation='relu',
                       kernel_regularizer=regularizers.l2(0.0001),
                       name='fc1')(x)
    x   = layers.BatchNormalization(name='bn_fc')(x)
    x   = layers.Dropout(0.4, name='drop_fc')(x)
    out = layers.Dense(1, activation='sigmoid', name='output')(x)

    full_cnn = Model(inputs=inp, outputs=out, name='full_cnn')
    full_cnn.load_weights('cnn_full_weights.weights.h5')

    feature_extractor = Model(
        inputs=full_cnn.input,
        outputs=full_cnn.get_layer('fc1').output,
        name='eye_feature_extractor'
    )
    return feature_extractor


# -----------------------------------------------------------
# OpenCV cascades
# -----------------------------------------------------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_eye.xml'
)

# -----------------------------------------------------------
# Silent keyboard monitor — 30-second rolling WPM window
# -----------------------------------------------------------
class KeyboardMonitor:
    """
    Background thread that listens to keypresses silently.
    WPM = (characters typed in last 30 s) / 5 / (30/60)
    Keypresses older than 30 s are automatically expired.
    """
    WINDOW = 30.0   # seconds

    def __init__(self):
        self._lock       = threading.Lock()
        self._timestamps = collections.deque()
        self._listener   = kb.Listener(on_press=self._on_press,
                                       suppress=False)

    def start(self):
        self._listener.start()

    def stop(self):
        self._listener.stop()

    def _on_press(self, key):
        now = time.time()
        with self._lock:
            self._timestamps.append(now)

    def _expire(self):
        cutoff = time.time() - self.WINDOW
        with self._lock:
            while self._timestamps and self._timestamps[0] < cutoff:
                self._timestamps.popleft()

    def get_features(self):
        """Returns (wpm, avg_delay, error_rate) as np.float32 array."""
        self._expire()
        with self._lock:
            n   = len(self._timestamps)
            ts  = list(self._timestamps)

        if n < 2:
            return np.array([0.0, 2.0, 0.05], dtype=np.float32)

        wpm       = float(n) / 5.0 / (self.WINDOW / 60.0)
        wpm       = min(wpm, 120.0)

        delays    = [ts[i+1] - ts[i] for i in range(len(ts)-1)]
        avg_delay = float(np.mean(delays))
        avg_delay = min(avg_delay, 2.0)

        return np.array([wpm, avg_delay, 0.05], dtype=np.float32)


# -----------------------------------------------------------
# Feature helpers  (identical logic to app.py)
# -----------------------------------------------------------
def extract_eye_features(feature_extractor, frame):
    resized = cv2.resize(frame, (64, 64))
    normed  = resized.astype(np.float32) / 255.0
    normed  = np.expand_dims(normed, axis=0)
    return feature_extractor.predict(normed, verbose=0)[0]


def compute_ear_from_box(eye_box):
    x, y, w, h = eye_box
    return round(h / w, 4) if w != 0 else 0.0


def extract_facial_features(frame):
    gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    default = {
        'avg_ear': 0.0, 'blink_detected': 0,
        'mouth_ratio': 0.0, 'yawn_detected': 0,
        'gaze_ratio': 0.0, 'face_detected': 0
    }
    faces = face_cascade.detectMultiScale(
        gray, 1.1, 5, minSize=(80, 80)
    )
    if len(faces) == 0:
        return default

    fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
    face_gray      = gray[fy:fy+fh, fx:fx+fw]
    eyes = sorted(
        eye_cascade.detectMultiScale(
            face_gray, 1.1, 5, minSize=(20, 20)
        ),
        key=lambda e: e[0]
    )

    left_ear = right_ear = 0.0
    if len(eyes) >= 2:
        right_ear = compute_ear_from_box(eyes[0])
        left_ear  = compute_ear_from_box(eyes[1])
    elif len(eyes) == 1:
        left_ear = right_ear = compute_ear_from_box(eyes[0])

    avg_ear     = round((left_ear + right_ear) / 2, 4)
    mouth_roi   = face_gray[int(fh * 0.65):fh, :]
    edges       = cv2.Canny(mouth_roi, 50, 150)
    mouth_ratio = round(float(edges.mean() / 255.0), 4)

    gaze_ratio = 0.0
    if len(eyes) >= 1:
        ex, ey, ew, eh = eyes[0]
        eye_roi        = face_gray[ey:ey+eh, ex:ex+ew]
        _, thresh      = cv2.threshold(
            eye_roi, 50, 255, cv2.THRESH_BINARY_INV
        )
        M = cv2.moments(thresh)
        if M['m00'] != 0:
            cx         = M['m10'] / M['m00']
            gaze_ratio = round(
                float(np.clip((cx - ew/2) / (ew/2), -1, 1)), 4
            )

    return {
        'avg_ear'       : avg_ear,
        'blink_detected': 1 if avg_ear < 0.20 else 0,
        'mouth_ratio'   : mouth_ratio,
        'yawn_detected' : 1 if mouth_ratio > 0.10 else 0,
        'gaze_ratio'    : gaze_ratio,
        'face_detected' : 1
    }


TYPING_RANGES = {
    'wpm'       : (0.0, 120.0),
    'avg_delay' : (0.0, 2.0),
    'error_rate': (0.0, 1.0)
}

def normalize_typing(vec):
    mins = np.array([r[0] for r in TYPING_RANGES.values()],
                    dtype=np.float32)
    maxs = np.array([r[1] for r in TYPING_RANGES.values()],
                    dtype=np.float32)
    return np.clip((vec - mins) / (maxs - mins), 0.0, 1.0)


def fuse(eye_feat, facial_dict, typing_vec):
    facial_vec = np.array([
        facial_dict['avg_ear'],
        facial_dict['blink_detected'],
        facial_dict['mouth_ratio'],
        facial_dict['yawn_detected'],
        facial_dict['gaze_ratio'],
        facial_dict['face_detected']
    ], dtype=np.float32)
    return np.concatenate([
        eye_feat.flatten(),
        facial_vec,
        normalize_typing(typing_vec)
    ]).astype(np.float32)


# -----------------------------------------------------------
# Record one session
# -----------------------------------------------------------
LABEL_NAMES = {
    0: 'LOW FATIGUE',
    1: 'MEDIUM FATIGUE',
    2: 'HIGH FATIGUE'
}
INSTRUCTIONS = {
    0: 'Sit normally. Eyes fully open. Type naturally.',
    1: 'Act slightly tired. Eyes drooping. Type slowly.',
    2: 'Act very tired. Yawn. Almost close eyes.'
}

def record_session(label, duration_seconds,
                   feature_extractor, keyboard_monitor, csv_writer):
    print(f"\n{'='*52}")
    print(f"  Recording : {LABEL_NAMES[label]}")
    print(f"  Action    : {INSTRUCTIONS[label]}")
    print(f"  Duration  : {duration_seconds} seconds")
    print(f"  Starting in 3 seconds — get ready...")
    time.sleep(3)
    print("  *** RECORDING NOW — type freely, behave naturally ***")

    cap         = cv2.VideoCapture(0)
    start_time  = time.time()
    frame_count = 0

    while time.time() - start_time < duration_seconds:
        ret, frame = cap.read()
        if not ret:
            break

        typing_vec  = keyboard_monitor.get_features()
        eye_feat    = extract_eye_features(feature_extractor, frame)
        facial_dict = extract_facial_features(frame)
        fused       = fuse(eye_feat, facial_dict, typing_vec)

        row = list(fused) + [label]
        csv_writer.writerow(row)
        frame_count += 1

        remaining = int(duration_seconds - (time.time() - start_time))
        wpm       = typing_vec[0]

        cv2.putText(frame,
                    f"{LABEL_NAMES[label]}  —  {remaining}s left",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 0), 2)
        cv2.putText(frame,
                    INSTRUCTIONS[label],
                    (10, 75), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 0), 1)
        cv2.putText(frame,
                    f"Live WPM: {wpm:.1f}",
                    (10, 110), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 200, 255), 2)
        cv2.imshow('Recording — press Q to skip', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"  Recorded {frame_count} frames for label {label}")
    return frame_count


# -----------------------------------------------------------
# Main
# -----------------------------------------------------------
if __name__ == '__main__':
    print("=" * 52)
    print("  REAL FATIGUE DATA COLLECTION — v2 Multi-Employee")
    print("=" * 52)
    print()
    print("  This records 3 sessions × 60 seconds each:")
    print("    Session 1 → Low Fatigue    (sit normally)")
    print("    Session 2 → Medium Fatigue (act slightly tired)")
    print("    Session 3 → High Fatigue   (act very tired)")
    print()
    print("  Keyboard is monitored SILENTLY — just type normally.")
    print("  WPM is calculated from a 30-second rolling window.")
    print("  Typing in any app (Notepad, browser, etc.) is captured.")
    print()

    employee_id = input("  Enter your Employee ID (e.g. emp_001): ").strip()
    if not employee_id:
        employee_id = "emp_unknown"

    input("\n  Press Enter when ready to start recording...\n")

    print("  Loading CNN feature extractor...")
    feature_extractor = load_feature_extractor()
    print("  CNN loaded ✓")

    print("  Starting keyboard monitor...")
    km = KeyboardMonitor()
    km.start()
    print("  Keyboard monitor running ✓")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    CSV_FILE  = f'real_fatigue_data_{employee_id}_{timestamp}.csv'
    header    = [f'feat_{i}' for i in range(137)] + ['label']

    with open(CSV_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)

        total_frames = 0
        for label in [0, 1, 2]:
            frames = record_session(
                label=label,
                duration_seconds=60,
                feature_extractor=feature_extractor,
                keyboard_monitor=km,
                csv_writer=writer
            )
            total_frames += frames
            if label < 2:
                print(f"\n  Rest for 10 seconds — relax...\n")
                time.sleep(10)

    km.stop()
    print(f"\n{'='*52}")
    print(f"  Done! {total_frames} frames recorded.")
    print(f"  File saved → {CSV_FILE}")
    print(f"  Upload this CSV to Colab to retrain the LSTM.")
    print("=" * 52)