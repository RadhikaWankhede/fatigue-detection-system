# ============================================================
# app.py  —  v2  Multi-Employee Fatigue Monitoring System
# ============================================================
# Features:
#   • Multi-employee login + per-employee SQLite history
#   • Real LSTM trained on actual employee data
#   • Silent background keyboard monitoring (pynput)
#   • Popup break reminders when High fatigue detected
#   • Company dashboard with charts + workload analysis
#   • High-fatigue early-warning system
# ============================================================

import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, Model
import time
import sqlite3
import threading
import collections
import datetime
from collections import deque
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pynput import keyboard as kb   # pip install pynput

# -----------------------------------------------------------
# Page config
# -----------------------------------------------------------
st.set_page_config(
    page_title="Employee Fatigue Monitor",
    page_icon="🏢",
    layout="wide"
)

# -----------------------------------------------------------
# SQLite — database setup
# -----------------------------------------------------------
DB_PATH = 'fatigue_records.db'

def init_db():
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS employees (
            employee_id TEXT PRIMARY KEY,
            name        TEXT,
            department  TEXT,
            created_at  TEXT
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS fatigue_logs (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            employee_id TEXT,
            timestamp   TEXT,
            fatigue     INTEGER,   -- 0/1/2
            wpm         REAL,
            ear         REAL,
            yawn        INTEGER
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS break_logs (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            employee_id TEXT,
            timestamp   TEXT,
            duration_s  INTEGER
        )
    """)
    con.commit()
    con.close()

init_db()

def log_reading(employee_id, fatigue, wpm, ear, yawn):
    con = sqlite3.connect(DB_PATH)
    con.execute(
        "INSERT INTO fatigue_logs (employee_id,timestamp,fatigue,wpm,ear,yawn) "
        "VALUES (?,?,?,?,?,?)",
        (employee_id,
         datetime.datetime.now().isoformat(timespec='seconds'),
         int(fatigue), float(wpm), float(ear), int(yawn))
    )
    con.commit()
    con.close()

def log_break(employee_id, duration_s):
    con = sqlite3.connect(DB_PATH)
    con.execute(
        "INSERT INTO break_logs (employee_id,timestamp,duration_s) VALUES (?,?,?)",
        (employee_id,
         datetime.datetime.now().isoformat(timespec='seconds'),
         duration_s)
    )
    con.commit()
    con.close()

def get_employees():
    con = sqlite3.connect(DB_PATH)
    df  = pd.read_sql("SELECT * FROM employees", con)
    con.close()
    return df

def add_employee(eid, name, dept):
    con = sqlite3.connect(DB_PATH)
    con.execute(
        "INSERT OR REPLACE INTO employees VALUES (?,?,?,?)",
        (eid, name, dept, datetime.datetime.now().isoformat())
    )
    con.commit()
    con.close()

def get_fatigue_history(employee_id=None, hours=8):
    con  = sqlite3.connect(DB_PATH)
    since = (datetime.datetime.now() -
             datetime.timedelta(hours=hours)).isoformat()
    if employee_id:
        df = pd.read_sql(
            "SELECT * FROM fatigue_logs WHERE employee_id=? AND timestamp>=? "
            "ORDER BY timestamp",
            con, params=(employee_id, since)
        )
    else:
        df = pd.read_sql(
            "SELECT * FROM fatigue_logs WHERE timestamp>=? ORDER BY timestamp",
            con, params=(since,)
        )
    con.close()
    return df


# -----------------------------------------------------------
# Silent keyboard monitor — 30-second rolling WPM window
# -----------------------------------------------------------
class KeyboardMonitor:
    WINDOW = 30.0

    def __init__(self):
        self._lock       = threading.Lock()
        self._timestamps = collections.deque()
        self._listener   = None
        self._running    = False

    def start(self):
        if self._running:
            return
        self._listener = kb.Listener(on_press=self._on_press, suppress=False)
        self._listener.start()
        self._running = True

    def stop(self):
        if self._listener:
            self._listener.stop()
        self._running = False

    def _on_press(self, key):
        with self._lock:
            self._timestamps.append(time.time())

    def _expire(self):
        cutoff = time.time() - self.WINDOW
        with self._lock:
            while self._timestamps and self._timestamps[0] < cutoff:
                self._timestamps.popleft()

    def get_features(self):
        self._expire()
        with self._lock:
            n  = len(self._timestamps)
            ts = list(self._timestamps)
        if n < 2:
            return np.array([0.0, 2.0, 0.05], dtype=np.float32)
        wpm       = min(float(n) / 5.0 / (self.WINDOW / 60.0), 120.0)
        delays    = [ts[i+1] - ts[i] for i in range(len(ts)-1)]
        avg_delay = min(float(np.mean(delays)), 2.0)
        return np.array([wpm, avg_delay, 0.05], dtype=np.float32)

    def get_wpm(self):
        return float(self.get_features()[0])


# -----------------------------------------------------------
# Load models
# -----------------------------------------------------------
@st.cache_resource
def load_models():
    # CNN — weights unchanged
    inp = layers.Input(shape=(64, 64, 3), name='input')
    x   = layers.Conv2D(32, (3,3), activation='relu', padding='same', name='conv1_1')(inp)
    x   = layers.BatchNormalization(name='bn1')(x)
    x   = layers.MaxPooling2D((2,2), name='pool1')(x)
    x   = layers.Dropout(0.25, name='drop1')(x)
    x   = layers.Conv2D(64, (3,3), activation='relu', padding='same', name='conv2_1')(x)
    x   = layers.BatchNormalization(name='bn2')(x)
    x   = layers.MaxPooling2D((2,2), name='pool2')(x)
    x   = layers.Dropout(0.25, name='drop2')(x)
    x   = layers.Conv2D(128, (3,3), activation='relu', padding='same', name='conv3_1')(x)
    x   = layers.BatchNormalization(name='bn3')(x)
    x   = layers.MaxPooling2D((2,2), name='pool3')(x)
    x   = layers.Dropout(0.3, name='drop3')(x)
    x   = layers.Flatten(name='flatten')(x)
    x   = layers.Dense(128, activation='relu',
                       kernel_regularizer=regularizers.l2(0.0001), name='fc1')(x)
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

    # LSTM — trained on real employee data
    lstm_inp = layers.Input(shape=(10, 137), name='lstm_input')
    y = layers.LSTM(128, return_sequences=True,
                    kernel_regularizer=regularizers.l2(0.001), name='lstm_1')(lstm_inp)
    y = layers.Dropout(0.3, name='drop_lstm1')(y)
    y = layers.LSTM(64, return_sequences=False,
                    kernel_regularizer=regularizers.l2(0.001), name='lstm_2')(y)
    y = layers.Dropout(0.3, name='drop_lstm2')(y)
    y = layers.Dense(64, activation='relu',
                     kernel_regularizer=regularizers.l2(0.001), name='fc1')(y)
    y = layers.Dropout(0.3, name='drop_fc')(y)
    lstm_out = layers.Dense(3, activation='softmax', name='output')(y)
    lstm_model = Model(inputs=lstm_inp, outputs=lstm_out, name='lstm_model')
    lstm_model.load_weights('lstm_weights_v2.weights.h5')

    return feature_extractor, lstm_model


@st.cache_resource
def load_cascades():
    return (
        cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        ),
        cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )
    )


# -----------------------------------------------------------
# Feature extraction (unchanged from original)
# -----------------------------------------------------------
IMG_SIZE   = (64, 64)
TIME_STEPS = 10
CLASS_NAMES  = ['Low Fatigue', 'Medium Fatigue', 'High Fatigue']
CLASS_COLORS = ['#00c853', '#ff6d00', '#d50000']
CLASS_EMOJI  = ['😊', '😐', '😴']
CLASS_BG     = ['#0a2e1a', '#2e1a00', '#2e0000']

TYPING_RANGES = {
    'wpm'       : (0.0, 120.0),
    'avg_delay' : (0.0, 2.0),
    'error_rate': (0.0, 1.0)
}

def normalize_typing(vec):
    mins = np.array([r[0] for r in TYPING_RANGES.values()], dtype=np.float32)
    maxs = np.array([r[1] for r in TYPING_RANGES.values()], dtype=np.float32)
    return np.clip((vec - mins) / (maxs - mins), 0.0, 1.0)

def extract_eye_features(extractor, image_array):
    if image_array.ndim == 3:
        image_array = np.expand_dims(image_array, axis=0)
    return extractor.predict(image_array, verbose=0)[0]

def compute_ear_from_box(eye_box):
    x, y, w, h = eye_box
    return round(h / w, 4) if w != 0 else 0.0

def extract_facial_features(frame_bgr, face_cas, eye_cas):
    gray    = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    default = {
        'avg_ear': 0.0, 'blink_detected': 0,
        'mouth_ratio': 0.0, 'yawn_detected': 0,
        'gaze_ratio': 0.0, 'face_detected': 0
    }
    faces = face_cas.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))
    if len(faces) == 0:
        return default

    fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
    face_gray      = gray[fy:fy+fh, fx:fx+fw]
    eyes = sorted(
        eye_cas.detectMultiScale(face_gray, 1.1, 5, minSize=(20, 20)),
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
        _, thresh      = cv2.threshold(eye_roi, 50, 255, cv2.THRESH_BINARY_INV)
        M              = cv2.moments(thresh)
        if M['m00'] != 0:
            cx         = M['m10'] / M['m00']
            gaze_ratio = round(float(np.clip((cx - ew/2) / (ew/2), -1, 1)), 4)

    return {
        'avg_ear'       : avg_ear,
        'blink_detected': 1 if avg_ear < 0.20 else 0,
        'mouth_ratio'   : mouth_ratio,
        'yawn_detected' : 1 if mouth_ratio > 0.05 else 0,
        'gaze_ratio'    : gaze_ratio,
        'face_detected' : 1
    }

def fuse_features(eye_feat, facial_dict, typing_vec):
    facial_vec = np.array([
        facial_dict['avg_ear'],     facial_dict['blink_detected'],
        facial_dict['mouth_ratio'], facial_dict['yawn_detected'],
        facial_dict['gaze_ratio'],  facial_dict['face_detected']
    ], dtype=np.float32)
    return np.concatenate([
        eye_feat.flatten(),
        facial_vec,
        normalize_typing(typing_vec)
    ]).astype(np.float32)

def rule_based_fatigue(facial_dict, wpm):
    score = 0
    ear   = facial_dict['avg_ear']
    if ear < 0:       score += 3
    elif ear < 0.24:  score += 2
    if facial_dict['mouth_ratio'] > 0.05:  score += 2
    elif facial_dict['mouth_ratio'] > 0.03: score += 1
    if facial_dict['blink_detected']:       score += 1
    if 0 < wpm < 15:  score += 2
    elif 0 < wpm < 25: score += 1
    if score >= 5:    return 2
    elif score >= 3:  return 1
    return 0


# -----------------------------------------------------------
# Session state defaults
# -----------------------------------------------------------
defaults = {
    'employee_id'     : None,
    'employee_name'   : '',
    'page'            : 'login',      # login | monitor | dashboard
    'sequence_buffer' : deque(maxlen=TIME_STEPS),
    'prediction'      : None,
    'confidences'     : None,
    'frame_count'     : 0,
    'last_facial'     : None,
    'fatigue_history' : [],
    'high_fatigue_streak': 0,
    'break_popup'     : False,
    'break_start'     : None,
    'kb_monitor'      : None,
    'session_start'   : None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ============================================================
# PAGE: LOGIN / REGISTER
# ============================================================
def page_login():
    st.title("🏢 Employee Fatigue Monitoring System")
    st.caption("Sign in to start your monitoring session")
    st.divider()

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("👤 Sign In")
        emp_id   = st.text_input("Employee ID", placeholder="emp_001")
        emp_name = st.text_input("Full Name",    placeholder="Aarav Shah")
        dept     = st.selectbox("Department",
                                ["Engineering", "Design", "HR",
                                 "Finance", "Operations", "Sales"])

        if st.button("▶️ Start Monitoring Session", type="primary"):
            if emp_id.strip():
                add_employee(emp_id.strip(), emp_name.strip(), dept)
                st.session_state.employee_id   = emp_id.strip()
                st.session_state.employee_name = emp_name.strip() or emp_id.strip()
                st.session_state.page          = 'monitor'
                st.session_state.session_start = time.time()
                # Start keyboard monitor
                km = KeyboardMonitor()
                km.start()
                st.session_state.kb_monitor = km
                st.rerun()
            else:
                st.error("Please enter an Employee ID.")

    with col2:
        st.subheader("📊 Dashboard")
        st.caption("View company-wide fatigue analytics")
        if st.button("🏠 Open Company Dashboard"):
            st.session_state.page = 'dashboard'
            st.rerun()

        st.divider()
        st.subheader("📋 Registered Employees")
        df = get_employees()
        if len(df):
            st.dataframe(df[['employee_id','name','department']],
                         use_container_width=True, hide_index=True)
        else:
            st.info("No employees registered yet.")


# ============================================================
# PAGE: MONITORING (per-employee)
# ============================================================
def page_monitor():
    feature_extractor, lstm_model = load_models()
    face_cascade, eye_cascade     = load_cascades()

    eid  = st.session_state.employee_id
    name = st.session_state.employee_name
    km   = st.session_state.kb_monitor

    # ---------- Break popup ----------
    if st.session_state.break_popup:

        # ── Play alert sound via Web Audio API (no external file needed) ──
        import streamlit.components.v1 as components
        components.html("""
        <script>
        (function() {
            const ctx = new (window.AudioContext || window.webkitAudioContext)();

            function beep(freq, startTime, duration, vol=0.6) {
                const osc  = ctx.createOscillator();
                const gain = ctx.createGain();
                osc.connect(gain);
                gain.connect(ctx.destination);

                osc.type      = 'sine';
                osc.frequency.setValueAtTime(freq, startTime);

                gain.gain.setValueAtTime(0, startTime);
                gain.gain.linearRampToValueAtTime(vol, startTime + 0.05);
                gain.gain.setValueAtTime(vol, startTime + duration - 0.08);
                gain.gain.linearRampToValueAtTime(0, startTime + duration);

                osc.start(startTime);
                osc.stop(startTime + duration);
            }

            // Three rising tones — urgent but not jarring
            const t = ctx.currentTime;
            beep(520, t + 0.0,  0.18);
            beep(660, t + 0.22, 0.18);
            beep(800, t + 0.44, 0.35);

            // Repeat once after 1.2 seconds
            beep(520, t + 1.2,  0.18);
            beep(660, t + 1.42, 0.18);
            beep(800, t + 1.64, 0.35);
        })();
        </script>
        """, height=0)
        # ─────────────────────────────────────────────────────────

        with st.container():
            st.error("## 🚨 HIGH FATIGUE ALERT — Time to take a break!")
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                if st.button("✅ Starting 5-min break now"):
                    st.session_state.break_start = time.time()
                    st.session_state.break_popup  = False
                    log_break(eid, 300)
                    st.rerun()
            with col_b:
                if st.button("⏭️ Skip break (not recommended)"):
                    st.session_state.break_popup        = False
                    st.session_state.high_fatigue_streak = 0
                    st.rerun()
            with col_c:
                if st.button("🚪 End session"):
                    _end_session()
        return

    # ---------- Break countdown ----------
    if st.session_state.break_start:
        elapsed = time.time() - st.session_state.break_start
        remain  = max(0, 300 - elapsed)
        st.info(f"☕ Break in progress — {int(remain // 60)}m {int(remain % 60)}s remaining. "
                f"Relax and step away from the screen.")
        if remain == 0:
            st.session_state.break_start = None
            st.success("✅ Break over. Welcome back!")
            time.sleep(2)
            st.rerun()
        time.sleep(1)
        st.rerun()
        return

    # ---------- Header ----------
    col_h1, col_h2, col_h3 = st.columns([3, 1, 1])
    with col_h1:
        st.title(f"🧠 Monitoring: {name} ({eid})")
        if st.session_state.session_start:
            elapsed = int(time.time() - st.session_state.session_start)
            st.caption(f"Session duration: {elapsed // 3600:02d}h "
                       f"{(elapsed % 3600)//60:02d}m {elapsed % 60:02d}s")
    with col_h2:
        if st.button("📊 Dashboard"):
            st.session_state.page = 'dashboard'
            st.rerun()
    with col_h3:
        if st.button("🚪 End Session"):
            _end_session()

    st.divider()

    run_detection = st.toggle("▶️ Enable Detection", value=False)

    # Main layout
    col1, col2 = st.columns([3, 2])
    with col1:
        st.subheader("📷 Live Webcam")
        camera_placeholder = st.empty()
    with col2:
        st.subheader("🎯 Fatigue Level")
        prediction_placeholder = st.empty()
        st.subheader("📊 Confidence")
        confidence_placeholder = st.empty()

    st.divider()
    col3, col4, col5 = st.columns(3)
    with col3:
        st.subheader("👁️ Eye Features")
        eye_placeholder = st.empty()
    with col4:
        st.subheader("😮 Mouth & Gaze")
        mouth_placeholder = st.empty()
    with col5:
        st.subheader("⌨️ Keyboard (live)")
        typing_placeholder = st.empty()

    if not run_detection:
        prediction_placeholder.markdown(
            "<div style='text-align:center;padding:24px;border-radius:16px;"
            "background:#1a1a1a;border:1px solid #333'>"
            "<div style='font-size:48px'>💤</div>"
            "<div style='color:#666;margin-top:8px'>Detection paused</div></div>",
            unsafe_allow_html=True
        )
        camera_placeholder.info("Toggle 'Enable Detection' to start")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("❌ Cannot open webcam. Check camera permissions.")
        return

    stop = st.button("⏹️ Stop Detection", type="primary")
    log_interval   = 5    # seconds between DB writes
    last_log_time  = 0
    HIGH_THRESHOLD = 15   # consecutive frames before popup

    while not stop:
        ret, frame = cap.read()
        if not ret:
            st.error("❌ Webcam read failed.")
            break

        st.session_state.frame_count += 1

        # Typing features from real keyboard
        typing_vec = km.get_features()
        wpm        = float(typing_vec[0])

        # If not typing, send neutral values so LSTM relies on eyes/face only
        if wpm < 5.0:
            typing_vec_for_lstm = np.array([40.0, 0.3, 0.05], dtype=np.float32)
        else:
            typing_vec_for_lstm = typing_vec

        # Vision features
        frame_resized = cv2.resize(frame, IMG_SIZE)
        frame_norm    = frame_resized.astype(np.float32) / 255.0
        eye_feat      = extract_eye_features(feature_extractor, frame_norm)
        facial_dict   = extract_facial_features(frame, face_cascade, eye_cascade)
        fused = fuse_features(eye_feat, facial_dict, typing_vec_for_lstm)

        st.session_state.sequence_buffer.append(fused)
        st.session_state.last_facial = facial_dict

        # Predict
        if len(st.session_state.sequence_buffer) == TIME_STEPS:
            seq       = np.expand_dims(np.array(st.session_state.sequence_buffer), 0)
            probs     = lstm_model.predict(seq, verbose=0)[0]


            # ADD THESE DEBUG LINES TEMPORARILY
            print(f"Probs: Low={probs[0]:.3f} Med={probs[1]:.3f} High={probs[2]:.3f}")
            print(f"EAR: {facial_dict['avg_ear']:.4f}  WPM: {typing_vec[0]:.1f}  Mouth: {facial_dict['mouth_ratio']:.4f}")
            print(f"Seq mean: {seq.mean():.4f}  Seq max: {seq.max():.4f}")


            lstm_pred = int(np.argmax(probs))
            rule_pred = rule_based_fatigue(facial_dict, wpm)
            final_pred = max(lstm_pred, rule_pred)

            st.session_state.prediction  = final_pred
            st.session_state.confidences = probs
            st.session_state.fatigue_history.append(final_pred)
            if len(st.session_state.fatigue_history) > 100:
                st.session_state.fatigue_history.pop(0)

            # High-fatigue streak counter
            if final_pred == 2:
                st.session_state.high_fatigue_streak += 1
            else:
                st.session_state.high_fatigue_streak = 0

            # Trigger break popup
            if st.session_state.high_fatigue_streak >= HIGH_THRESHOLD:
                st.session_state.break_popup        = True
                st.session_state.high_fatigue_streak = 0
                cap.release()
                st.rerun()

            # Write to DB every log_interval seconds
            now = time.time()
            if now - last_log_time >= log_interval:
                log_reading(eid, final_pred, wpm,
                            facial_dict['avg_ear'], facial_dict['yawn_detected'])
                last_log_time = now

        # Draw on frame
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))
        pred  = st.session_state.prediction
        for (fx, fy, fw, fh) in faces:
            color = (0, 255, 0)
            if pred == 1: color = (0, 165, 255)
            elif pred == 2: color = (0, 0, 255)
            cv2.rectangle(frame, (fx, fy), (fx+fw, fy+fh), color, 2)
            if pred is not None:
                cv2.putText(frame, CLASS_NAMES[pred],
                            (fx, fy - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, color, 2)

        # WPM overlay
        cv2.putText(frame, f"WPM: {wpm:.1f}",
                    (10, frame.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

        camera_placeholder.image(
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
            channels="RGB", use_container_width=True
        )

        # Prediction card
        if pred is not None:
            idx   = pred
            color = CLASS_COLORS[idx]
            bg    = CLASS_BG[idx]
            prediction_placeholder.markdown(
                f"<div style='text-align:center;padding:24px;border-radius:16px;"
                f"background:{bg};border:2px solid {color};margin-bottom:8px'>"
                f"<div style='font-size:56px'>{CLASS_EMOJI[idx]}</div>"
                f"<div style='color:{color};font-size:24px;font-weight:bold;margin-top:8px'>"
                f"{CLASS_NAMES[idx]}</div>"
                f"<div style='color:#888;font-size:13px;margin-top:6px'>"
                f"Frame #{st.session_state.frame_count} • "
                f"Streak: {st.session_state.high_fatigue_streak}/{HIGH_THRESHOLD}</div>"
                f"</div>",
                unsafe_allow_html=True
            )
            conf_text = ""
            for i in range(3):
                bar   = int(st.session_state.confidences[i] * 20)
                conf_text += (
                    f"{CLASS_EMOJI[i]} **{CLASS_NAMES[i]}**\n"
                    f"`{'█'*bar}{'░'*(20-bar)}` "
                    f"`{st.session_state.confidences[i]*100:.1f}%`\n\n"
                )
            confidence_placeholder.markdown(conf_text)

        # Eye / mouth displays
        if st.session_state.last_facial:
            f = st.session_state.last_facial
            ear_status = ("🔴 Closed" if f['avg_ear'] < 0 else
                          "🟡 Drooping" if f['avg_ear'] < 0.25 else "🟢 Open")
            eye_placeholder.markdown(
                f"**EAR:** `{f['avg_ear']:.3f}`\n\n"
                f"**Status:** {ear_status}\n\n"
                f"**Blink:** `{'⚡ Yes' if f['blink_detected'] else 'No'}`\n\n"
                f"**Face:** `{'✅' if f['face_detected'] else '❌'}`"
            )
            gaze_status = ("👈 Left" if f['gaze_ratio'] < -0.2 else
                           "👉 Right" if f['gaze_ratio'] > 0.2 else "👁️ Center")
            mouth_placeholder.markdown(
                f"**Mouth ratio:** `{f['mouth_ratio']:.3f}`\n\n"
                f"**Yawn:** `{'🥱 Yes' if f['yawn_detected'] else 'No'}`\n\n"
                f"**Gaze:** `{gaze_status}`"
            )

        wpm_status = ("🔴 Very slow" if wpm < 20 else
                      "🟡 Slow" if wpm < 35 else "🟢 Normal")
        typing_placeholder.markdown(
            f"**WPM (live):** `{wpm:.1f}`\n\n"
            f"**Status:** {wpm_status}\n\n"
            f"**Source:** Silent keyboard monitor\n\n"
            f"**Window:** Last 30 seconds"
        )

        time.sleep(0.05)

    cap.release()


def _end_session():
    if st.session_state.kb_monitor:
        st.session_state.kb_monitor.stop()
    st.session_state.employee_id    = None
    st.session_state.employee_name  = ''
    st.session_state.page           = 'login'
    st.session_state.sequence_buffer = deque(maxlen=TIME_STEPS)
    st.session_state.prediction     = None
    st.session_state.confidences    = None
    st.session_state.frame_count    = 0
    st.session_state.last_facial    = None
    st.session_state.fatigue_history = []
    st.session_state.high_fatigue_streak = 0
    st.session_state.break_popup    = False
    st.session_state.break_start    = None
    st.session_state.kb_monitor     = None
    st.session_state.session_start  = None
    st.rerun()


# ============================================================
# PAGE: COMPANY DASHBOARD
# ============================================================
def page_dashboard():
    st.title("🏢 Company Fatigue Dashboard")
    col_back, _ = st.columns([1, 5])
    with col_back:
        if st.button("← Back"):
            st.session_state.page = 'login'
            st.rerun()
    st.divider()

    hours = st.slider("Show data for last N hours", 1, 24, 8)
    df    = get_fatigue_history(hours=hours)

    if df.empty:
        st.info("No data recorded yet. Start a monitoring session first.")
        return

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['fatigue_label'] = df['fatigue'].map(
        {0: 'Low', 1: 'Medium', 2: 'High'}
    )

    # ---- KPI row ----
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Readings", len(df))
    high_pct = round((df['fatigue'] == 2).mean() * 100, 1)
    k2.metric("High Fatigue %", f"{high_pct}%",
              delta=None,
              delta_color="inverse")
    avg_wpm = round(df['wpm'].mean(), 1)
    k3.metric("Avg WPM (all employees)", avg_wpm)
    unique_emps = df['employee_id'].nunique()
    k4.metric("Employees Monitored", unique_emps)

    st.divider()

    # ---- Fatigue over time ----
    st.subheader("📈 Fatigue Level Over Time")
    fig_time = px.line(
        df, x='timestamp', y='fatigue', color='employee_id',
        labels={'fatigue': 'Fatigue Level (0=Low, 2=High)',
                'timestamp': 'Time'},
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    fig_time.update_yaxes(tickvals=[0, 1, 2],
                          ticktext=['Low', 'Medium', 'High'])
    st.plotly_chart(fig_time, use_container_width=True)

    col_a, col_b = st.columns(2)

    with col_a:
        # ---- Fatigue distribution per employee ----
        st.subheader("📊 Fatigue Distribution per Employee")
        dist = (df.groupby(['employee_id', 'fatigue_label'])
                  .size().reset_index(name='count'))
        fig_dist = px.bar(
            dist, x='employee_id', y='count', color='fatigue_label',
            color_discrete_map={'Low':'#00c853','Medium':'#ff6d00','High':'#d50000'},
            barmode='stack'
        )
        st.plotly_chart(fig_dist, use_container_width=True)

    with col_b:
        # ---- WPM vs fatigue scatter ----
        st.subheader("⌨️ Typing Speed vs Fatigue")
        fig_scatter = px.scatter(
            df, x='wpm', y='fatigue', color='employee_id',
            labels={'wpm': 'Words per Minute', 'fatigue': 'Fatigue Level'},
            opacity=0.5
        )
        fig_scatter.update_yaxes(tickvals=[0, 1, 2],
                                 ticktext=['Low', 'Medium', 'High'])
        st.plotly_chart(fig_scatter, use_container_width=True)

    # ---- Per-employee workload summary ----
    st.subheader("👥 Per-Employee Workload Summary")
    summary = df.groupby('employee_id').agg(
        readings    = ('fatigue', 'count'),
        avg_fatigue = ('fatigue', 'mean'),
        high_pct    = ('fatigue', lambda x: round((x == 2).mean()*100, 1)),
        avg_wpm     = ('wpm', lambda x: round(x.mean(), 1)),
        yawns       = ('yawn', 'sum'),
    ).reset_index()
    summary['avg_fatigue'] = summary['avg_fatigue'].round(2)
    summary.columns = [
        'Employee', 'Readings', 'Avg Fatigue (0-2)',
        'High Fatigue %', 'Avg WPM', 'Yawn Events'
    ]
    # Flag high-risk employees
    summary['Risk'] = summary['High Fatigue %'].apply(
        lambda v: '🔴 HIGH' if v > 90 else ('🟡 MEDIUM' if v > 40 else '🟢 OK')
    )
    st.dataframe(summary, use_container_width=True, hide_index=True)

    # ---- Early warning section ----
    high_risk = summary[summary['High Fatigue %'] > 90]
    if not high_risk.empty:
        st.error(
            "⚠️ **Early Warning** — The following employees have HIGH fatigue "
            "rates above 90% in the selected window:\n\n" +
            ", ".join(high_risk['Employee'].tolist()) +
            "\n\nConsider scheduling mandatory breaks or workload rebalancing."
        )


# ============================================================
# Router
# ============================================================
page = st.session_state.page

if page == 'login':
    page_login()
elif page == 'monitor':
    page_monitor()
elif page == 'dashboard':
    page_dashboard()