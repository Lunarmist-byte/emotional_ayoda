import cv2
import numpy as np
from flask import Flask, Response, render_template, jsonify
from collections import deque
import os

app = Flask(__name__)

# Initialize webcam with fallback to secondary camera
camera = cv2.VideoCapture(0)
if not camera.isOpened():
    camera = cv2.VideoCapture(1)  # Try secondary camera
    if not camera.isOpened():
        print("Error: Could not open any camera")
        exit(1)

# Initialize Haar Cascades with verified paths
cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
if not os.path.exists(cascade_path):
    print(f"Error: Haar cascade file not found at {cascade_path}")
    exit(1)
face_cascade = cv2.CascadeClassifier(cascade_path)
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

# Emotion history for temporal smoothing
emotion_history = deque(maxlen=10)
current_emotion = "neutral"
emotion_confidence = 0.0

def preprocess_frame(frame):
    """Preprocess frame for robust detection under varying conditions."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    # Apply adaptive thresholding to handle lighting variations
    gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 11, 2)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)  # Additional noise reduction
    return gray

def extract_facial_features(frame, x, y, w, h):
    """Extract facial features with enhanced robustness."""
    gray = preprocess_frame(frame)
    face_roi = gray[y:y+h, x:x+w]
    
    # Fine-tuned cascade parameters for better detection
    eyes = eye_cascade.detectMultiScale(face_roi, scaleFactor=1.15, minNeighbors=5, minSize=(20, 20))
    smiles = smile_cascade.detectMultiScale(face_roi, scaleFactor=1.5, minNeighbors=15, minSize=(25, 15))
    
    # Initialize default values for robustness
    smile_intensity = 0.3
    eye_openness = 0.3
    eyebrow_height = 0.5
    mouth_curvature = 0.5
    face_aspect_ratio = w / h if h > 0 else 1.0
    
    # Smile detection with normalized intensity
    if len(smiles) > 0:
        sx, sy, sw, sh = smiles[0]
        smile_intensity = min(1.0, (sw * sh) / (w * h) * 5.0)
        mouth_curvature = min(1.0, sw / (sh * 1.3))
        mouth_position = sy / h
        if mouth_position > 0.65:
            mouth_curvature *= 0.85
    
    # Eye detection with refined metrics
    if len(eyes) >= 2:
        eye1, eye2 = sorted(eyes[:2], key=lambda e: e[0])  # Sort by x-coordinate
        ey1, ey2 = eye1[1], eye2[1]
        eye_openness = min(1.0, (eye1[3] + eye2[3]) / (h * 0.4))
        eyebrow_height = min(1.0, max(0.0, (y - min(ey1, ey2)) / (h * 0.35)))
    elif len(eyes) == 1:
        eye_openness = min(1.0, eyes[0][3] / (h * 0.2))
        eyebrow_height = min(1.0, max(0.0, (y - eyes[0][1]) / (h * 0.4)))
    
    # Adjust for face aspect ratio
    if 0.8 < face_aspect_ratio < 1.2:
        smile_intensity *= 1.1
        eye_openness *= 1.1
    
    return [smile_intensity, eye_openness, eyebrow_height, mouth_curvature, face_aspect_ratio]

def detect_emotion(features):
    """Rule-based emotion detection with confidence scores."""
    smile_intensity, eye_openness, eyebrow_height, mouth_curvature, face_aspect_ratio = features
    
    # Dynamic thresholds based on feature reliability
    smile_threshold = 0.55 if smile_intensity > 0.35 else 0.45
    eye_threshold = 0.65 if eye_openness > 0.35 else 0.45
    brow_threshold = 0.55 if eyebrow_height > 0.35 else 0.45
    
    # Emotion rules with confidence
    emotions = {
        "happiness": 0.0,
        "sadness": 0.0,
        "surprise": 0.0,
        "anger": 0.0,
        "fear": 0.0,
        "disgust": 0.0,
        "neutral": 0.0
    }
    
    if smile_intensity > smile_threshold and mouth_curvature > 0.5:
        emotions["happiness"] = 0.9 * smile_intensity + 0.1 * mouth_curvature
    if smile_intensity < 0.3 and eye_openness < 0.4 and eyebrow_height > brow_threshold:
        emotions["sadness"] = 0.7 * (1 - smile_intensity) + 0.3 * eyebrow_height
    if eye_openness > eye_threshold and eyebrow_height > brow_threshold:
        emotions["surprise"] = 0.8 * eye_openness + 0.2 * eyebrow_height
    if smile_intensity < 0.3 and eyebrow_height > brow_threshold and mouth_curvature < 0.4:
        emotions["anger"] = 0.6 * (1 - smile_intensity) + 0.4 * eyebrow_height
    if eye_openness > eye_threshold and smile_intensity < 0.3 and eyebrow_height > brow_threshold:
        emotions["fear"] = 0.7 * eye_openness + 0.3 * (1 - smile_intensity)
    if smile_intensity < 0.3 and mouth_curvature < 0.3 and eyebrow_height > 0.5:
        emotions["disgust"] = 0.6 * (1 - mouth_curvature) + 0.4 * eyebrow_height
    emotions["neutral"] = 0.5 if max(emotions.values()) < 0.4 else 0.0
    
    # Return emotion with highest confidence
    emotion = max(emotions, key=emotions.get)
    confidence = emotions[emotion]
    return emotion, confidence

def generate_frames():
    global current_emotion, emotion_confidence
    while True:
        success, frame = camera.read()
        if not success:
            continue
        
        frame = cv2.resize(frame, (640, 480))
        gray = preprocess_frame(frame)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=5, minSize=(30, 30))
        
        emotion = "neutral"
        confidence = 0.0
        for (x, y, w, h) in faces:
            features = extract_facial_features(frame, x, y, w, h)
            emotion, confidence = detect_emotion(features)
            
            # Temporal smoothing with weighted voting
            emotion_history.append((emotion, confidence))
            if len(emotion_history) >= 5:
                emotion_counts = {}
                total_weight = 0.0
                for i, (e, c) in enumerate(emotion_history):
                    weight = 1.0 + (i / len(emotion_history)) * 0.7
                    emotion_counts[e] = emotion_counts.get(e, 0) + weight * c
                    total_weight += weight
                emotion = max(emotion_counts, key=emotion_counts.get)
                confidence = emotion_counts[emotion] / total_weight
            
            # Draw bounding box and emotion label
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f'{emotion.upper()} ({confidence:.2f})', (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        current_emotion = emotion
        emotion_confidence = confidence
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def status():
    return jsonify({'emotion': current_emotion, 'confidence': emotion_confidence})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)