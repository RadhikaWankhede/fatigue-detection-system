# Detecting Cognitive Load and Mental Fatigue Using Micro-Behavior

A real-time fatigue detection system that combines eye state classification (CNN), facial feature analysis (OpenCV), and typing behavior analysis to classify cognitive fatigue into **Low**, **Medium**, and **High** levels using an LSTM network.

## Features

- Real-time eye-state detection using a custom CNN
- Facial feature extraction using OpenCV (EAR, gaze ratio, mouth opening, blink detection)
- Keystroke dynamics monitoring (WPM, inter-key delay, error rate)
- Multimodal feature fusion into a 137-dimensional feature vector
- Temporal fatigue classification using LSTM
- Rule-based fatigue verification for increased robustness
- Multi-employee login and monitoring system
- Automated break reminders for prolonged fatigue
- SQLite-based fatigue history logging
- Company-wide analytics dashboard with fatigue trends and workload insights
- Streamlit-based real-time web application

## Architecture

```text
Webcam Frame → CNN (Eye Features, 128-dim)
            → OpenCV (Facial Features, 6-dim)

Keyboard Input → Typing Features (3-dim)

                ↓
        Feature Fusion (137-dim)
                ↓
      Sliding Window (10 Frames)
                ↓
              LSTM
                ↓
   Low / Medium / High Fatigue
```

## Tech Stack

- **Programming Language:** Python 3.10
- **Deep Learning:** TensorFlow, Keras
- **Computer Vision:** OpenCV
- **Web Framework:** Streamlit
- **Training Environment:** Google Colab

## Results

| Model | Validation Accuracy |
|---------|--------------------|
| CNN (Eye State Detection) | 87.81%*|
| LSTM (Fatigue Classification) | 99.18%* |

\* Initial version trained on simulated data. The final model was retrained using real collected sessions.

## Installation

```bash
git clone https://github.com/yourusername/multimodal-fatigue-detection.git
cd multimodal-fatigue-detection

python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt
```

Download the trained model weights and place them in the project root directory.

## Usage

```bash
streamlit run app.py
```

## Dataset

The CNN model was trained on a supervised eye-state dataset containing open-eye and closed-eye images. The LSTM model was trained using real webcam sessions collected through the data collection pipeline.

## Future Enhancements

- Data collection from a larger and more diverse participant pool
- MediaPipe integration for improved facial landmark precision
- Background keyboard monitoring using pynput
- Fatigue alert and notification system
- Enterprise dashboard for workload monitoring and analytics

## Project Structure

```text
multimodal-fatigue-detection/
│
├── app.py
├── collect_data.py
├── train_cnn.py
├── train_lstm.py
├── requirements.txt
├── models/
├── data/
├── screenshots/
└── README.md
```

## Pre-trained Weights

Place `cnn_full_weights.weights.h5` and `lstm_real_weights.weights.h5` in the project root.

## Author

**Radhika Rajendra Wankhede**  
Department of Computer Science and Engineering  
MIT World Peace University, Pune

