# Multimodal Cognitive Load and Mental Fatigue Detection

A real-time fatigue detection system that combines eye state classification (CNN), facial feature analysis (OpenCV), and typing behavior to classify cognitive fatigue into Low, Medium, and High levels using an LSTM network.


## Features

- Real-time webcam-based eye state detection using a custom CNN
- Facial feature extraction (Eye Aspect Ratio, mouth ratio, gaze direction) using OpenCV
- Typing behavior analysis (WPM, inter-key delay, error rate)
- Multimodal feature fusion into a 137-dimensional vector
- Temporal fatigue classification using LSTM
- Live web dashboard built with Streamlit

## Architecture

Webcam Frame → CNN (eye features, 128-dim)

→ OpenCV (facial features, 6-dim)

Keyboard    → Typing behavior (3-dim)

↓

Feature Fusion (137-dim)

↓

Sliding Window (10 frames)

↓

LSTM

↓

Low / Medium / High Fatigue





## Tech Stack

- **Deep Learning:** TensorFlow, Keras
- **Computer Vision:** OpenCV
- **Web App:** Streamlit
- **Training:** Google Colab
- **Language:** Python 3.10

## Results

| Model | Validation Accuracy |
|-------|---------------------|
| CNN (eye state) | 84.4% |
| LSTM (fatigue classification) | 96.5%* |

\* Initial version trained on simulated data; retrained on real collected sessions for production use.

## Installation

```bash
git clone https://github.com/yourusername/multimodal-fatigue-detection.git
cd multimodal-fatigue-detection
python -m venv venv
venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

Download trained model weights from [link] and place in the project root.

## Usage

```bash
streamlit run app.py
```

## Dataset

The CNN was trained on a supervised eye state dataset (open/closed eyes, merged from 4 to 2 classes). The LSTM was trained on real webcam sessions collected using `collect_data.py`.

## Project Structure

- Real data collection from multiple subjects
- MediaPipe integration for improved facial landmark precision
- Background keyboard monitoring using pynput
- Alert system for high fatigue detection
- Company dashboard for workload monitoring

## Author

Radhika Rajendra Wankhede
Department of Computer Science and Engineering
MIT World Peace University, Pune

## License

No license has been specified for this project.
