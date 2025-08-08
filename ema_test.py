import cv2
import numpy as np
from flask import Flask, Response, render_template, jsonify
from collections import deque
import os
import requests
import time
import base64

app = Flask(__name__)

# Initialize webcam with fallback
camera = cv2.VideoCapture(0)
if not camera.isOpened():
    camera = cv2.VideoCapture(1)
    if not camera.isOpened():
        print("Error: Could not open any camera")
        exit(1)

# Initialize Haar Cascades
cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
if not os.path.exists(cascade_path):
    print(f"Error: Haar cascade file not found at {cascade_path}")
    exit(1)
face_cascade = cv2.CascadeClassifier(cascade_path)
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

# Emotion history for temporal smoothing
emotion_history = deque(maxlen=12)
current_emotion = "neutral"
emotion_confidence = 0.0

# Spotify API configuration
SPOTIFY_CLIENT_ID = 'YOUR_SPOTIFY_CLIENT_ID'  # Replace with your Spotify Client ID
SPOTIFY_CLIENT_SECRET = 'YOUR_SPOTIFY_CLIENT_SECRET'  # Replace with your Spotify Client Secret
SPOTIFY_ACCESS_TOKEN = None
TOKEN_EXPIRY = 0

def get_spotify_token():
    """Fetch or refresh Spotify access token."""
    global SPOTIFY_ACCESS_TOKEN, TOKEN_EXPIRY
    if time.time() > TOKEN_EXPIRY:
        try:
            auth_string = f"{SPOTIFY_CLIENT_ID}:{SPOTIFY_CLIENT_SECRET}"
            auth_bytes = auth_string.encode('utf-8')
            auth_base64 = base64.b64encode(auth_bytes).decode('utf-8')
            response = requests.post(
                'https://accounts.spotify.com/api/token',
                data={'grant_type': 'client_credentials'},
                headers={'Authorization': f'Basic {auth_base64}'}
            )
            response.raise_for_status()
            token_data = response.json()
            SPOTIFY_ACCESS_TOKEN = token_data['access_token']
            TOKEN_EXPIRY = time.time() + token_data['expires_in'] - 60
        except requests.RequestException as e:
            print(f"Error fetching Spotify token: {e}")
            SPOTIFY_ACCESS_TOKEN = None
    return SPOTIFY_ACCESS_TOKEN

@app.route('/get_spotify_token')
def get_token():
    token = get_spotify_token()
    if token:
        return jsonify({'access_token': token})
    return jsonify({'error': 'Failed to fetch Spotify token'}), 500

def preprocess_frame(frame):
    """Preprocess frame for robust detection."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 11, 2)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    return gray

def extract_facial_features(frame, x, y, w, h):
    """Extract facial features with high precision."""
    gray = preprocess_frame(frame)
    face_roi = gray[y:y+h, x:x+w]
    
    eyes = eye_cascade.detectMultiScale(face_roi, scaleFactor=1.1, minNeighbors=6, minSize=(25, 25))
    smiles = smile_cascade.detectMultiScale(face_roi, scaleFactor=1.4, minNeighbors=20, minSize=(30, 15))
    
    smile_intensity = 0.3
    eye_openness = 0.3
    eyebrow_height = 0.5
    mouth_curvature = 0.5
    mouth_width_ratio = 0.5
    face_aspect_ratio = w / h if h > 0 else 1.0
    
    if len(smiles) > 0:
        sx, sy, sw, sh = smiles[0]
        smile_intensity = min(1.0, (sw * sh) / (w * h) * 6.0)
        mouth_curvature = min(1.0, sw / (sh * 1.2))
        mouth_width_ratio = sw / w if w > 0 else 0.5
        mouth_position = sy / h
        if mouth_position > 0.6:
            mouth_curvature *= 0.9
    
    if len(eyes) >= 2:
        eye1, eye2 = sorted(eyes[:2], key=lambda e: e[0])
        ey1, ey2 = eye1[1], eye2[1]
        eye_openness = min(1.0, (eye1[3] + eye2[3]) / (h * 0.35))
        eyebrow_height = min(1.0, max(0.0, (y - min(ey1, ey2)) / (h * 0.3)))
    elif len(eyes) == 1:
        eye_openness = min(1.0, eyes[0][3] / (h * 0.25))
        eyebrow_height = min(1.0, max(0.0, (y - eyes[0][1]) / (h * 0.35)))
    
    if 0.75 < face_aspect_ratio < 1.25:
        smile_intensity = min(1.0, smile_intensity * 1.15)
        eye_openness = min(1.0, eye_openness * 1.15)
    
    return [smile_intensity, eye_openness, eyebrow_height, mouth_curvature, mouth_width_ratio, face_aspect_ratio]

def detect_emotion(features):
    """Rule-based emotion detection with high precision."""
    smile_intensity, eye_openness, eyebrow_height, mouth_curvature, mouth_width_ratio, face_aspect_ratio = features
    
    smile_threshold = 0.6 if smile_intensity > 0.4 else 0.5
    eye_threshold = 0.7 if eye_openness > 0.4 else 0.5
    brow_threshold = 0.6 if eyebrow_height > 0.4 else 0.5
    mouth_threshold = 0.6 if mouth_width_ratio > 0.4 else 0.5
    
    emotions = {
        "happiness": 0.0,
        "sadness": 0.0,
        "surprise": 0.0,
        "anger": 0.0,
        "fear": 0.0,
        "disgust": 0.0,
        "neutral": 0.0
    }
    
    if smile_intensity > smile_threshold and mouth_curvature > 0.55 and mouth_width_ratio > mouth_threshold:
        emotions["happiness"] = 0.5 * smile_intensity + 0.3 * mouth_curvature + 0.2 * mouth_width_ratio
    if smile_intensity < 0.35 and eye_openness < 0.45 and eyebrow_height > brow_threshold:
        emotions["sadness"] = 0.5 * (1 - smile_intensity) + 0.3 * (1 - eye_openness) + 0.2 * eyebrow_height
    if eye_openness > eye_threshold and eyebrow_height > brow_threshold and mouth_width_ratio > 0.5:
        emotions["surprise"] = 0.5 * eye_openness + 0.3 * eyebrow_height + 0.2 * mouth_width_ratio
    if smile_intensity < 0.35 and eyebrow_height > brow_threshold and mouth_curvature < 0.45:
        emotions["anger"] = 0.4 * (1 - smile_intensity) + 0.4 * eyebrow_height + 0.2 * (1 - mouth_curvature)
    if eye_openness > eye_threshold and smile_intensity < 0.35 and eyebrow_height > brow_threshold:
        emotions["fear"] = 0.5 * eye_openness + 0.3 * (1 - smile_intensity) + 0.2 * eyebrow_height
    if smile_intensity < 0.35 and mouth_curvature < 0.35 and eyebrow_height > 0.55:
        emotions["disgust"] = 0.4 * (1 - mouth_curvature) + 0.3 * eyebrow_height + 0.3 * (1 - smile_intensity)
    emotions["neutral"] = 0.6 if max(emotions.values()) < 0.5 else 0.0
    
    emotion = max(emotions, key=emotions.get)
    confidence = emotions[emotion]
    if confidence < 0.4:
        emotion = "neutral"
        confidence = 0.6
    return emotion, confidence

def generate_frames():
    """Generate video frames with emotion overlay."""
    global current_emotion, emotion_confidence
    while True:
        success, frame = camera.read()
        if not success:
            continue
        
        frame = cv2.resize(frame, (640, 480))
        gray = preprocess_frame(frame)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(40, 40))
        
        emotion = "neutral"
        confidence = 0.0
        for (x, y, w, h) in faces:
            features = extract_facial_features(frame, x, y, w, h)
            emotion, confidence = detect_emotion(features)
            
            emotion_history.append((emotion, confidence))
            if len(emotion_history) >= 6:
                emotion_counts = {}
                total_weight = 0.0
                for i, (e, c) in enumerate(emotion_history):
                    weight = 1.0 + (i / len(emotion_history)) * 0.9
                    emotion_counts[e] = emotion_counts.get(e, 0) + weight * c
                    total_weight += weight
                emotion = max(emotion_counts, key=emotion_counts.get)
                confidence = emotion_counts[emotion] / total_weight
            
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