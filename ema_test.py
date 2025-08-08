import cv2
import numpy as np
from flask import Flask, Response, render_template
from collections import deque

app = Flask(__name__)

# Initialize webcam
camera = cv2.VideoCapture(0)
if not camera.isOpened():
    print("Error: Could not open camera")
    exit(1)

# Initialize Haar Cascade for face, eye, and smile detection
cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
if not os.path.exists(cascade_path):
    print(f"Error: Haar cascade file not found at {cascade_path}")
    exit(1)
face_cascade = cv2.CascadeClassifier(cascade_path)
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

# Emotion history for smoothing (increased to 7 frames)
emotion_history = deque(maxlen=7)

def preprocess_frame(frame):
    """Preprocess frame to improve detection on low-quality cameras."""
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Apply histogram equalization to enhance contrast
    gray = cv2.equalizeHist(gray)
    # Apply slight Gaussian blur to reduce noise
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    return gray

def extract_facial_features(frame, x, y, w, h):
    """Extract facial features with fallbacks for low-quality input."""
    gray = preprocess_frame(frame)
    face_roi = gray[y:y+h, x:x+w]
    
    # Detect eyes and smiles with relaxed parameters for low-quality cameras
    eyes = eye_cascade.detectMultiScale(face_roi, scaleFactor=1.2, minNeighbors=3)
    smiles = smile_cascade.detectMultiScale(face_roi, scaleFactor=1.6, minNeighbors=10)
    
    # Initialize features
    smile_intensity = min(1.0, len(smiles) * 0.4) if len(smiles) > 0 else 0.2
    eye_openness = min(1.0, len(eyes) * 0.6) if len(eyes) > 0 else 0.3
    eyebrow_height = 0.5
    mouth_curvature = 0.5
    face_aspect_ratio = w / h if h > 0 else 1.0
    
    # Approximate eyebrow height using face position
    if len(eyes) >= 2:
        eye1, eye2 = eyes[:2]
        ey1 = eye1[1]
        ey2 = eye2[1]
        eyebrow_height = min(1.0, max(0.0, (y - min(ey1, ey2)) / (h * 0.4)))
    else:
        # Fallback: use face aspect ratio to estimate eyebrow position
        eyebrow_height = min(1.0, max(0.0, 0.5 + (face_aspect_ratio - 1.0) * 0.5))
    
    # Approximate mouth curvature
    if len(smiles) > 0:
        sx, sy, sw, sh = smiles[0]
        mouth_curvature = min(1.0, max(0.0, sw / (sh * 1.5)))
        # Adjust based on mouth position (lower in face = flatter, e.g., disgust)
        mouth_position = sy / h
        if mouth_position > 0.7:
            mouth_curvature *= 0.8
    
    return [smile_intensity, eye_openness, eyebrow_height, mouth_curvature, face_aspect_ratio]

def detect_emotion(features):
    """Rule-based emotion detection with improved accuracy for low-quality input."""
    smile_intensity, eye_openness, eyebrow_height, mouth_curvature, face_aspect_ratio = features
    
    # Adaptive thresholds based on feature confidence
    smile_threshold = 0.6 if smile_intensity > 0.3 else 0.5
    eye_threshold = 0.7 if eye_openness > 0.4 else 0.5
    brow_threshold = 0.6 if eyebrow_height > 0.4 else 0.5
    
    # Refined rules for emotion classification
    if smile_intensity > smile_threshold and mouth_curvature > 0.5:
        return "happiness"
    elif smile_intensity < 0.3 and eye_openness < 0.4 and eyebrow_height > brow_threshold:
        return "sadness"
    elif eye_openness > eye_threshold and eyebrow_height > brow_threshold:
        return "surprise"
    elif smile_intensity < 0.3 and eyebrow_height > brow_threshold and mouth_curvature < 0.4:
        return "anger"
    elif eye_openness > eye_threshold and smile_intensity < 0.3 and eyebrow_height > brow_threshold:
        return "fear"
    elif smile_intensity < 0.3 and mouth_curvature < 0.3 and eyebrow_height > 0.5:
        return "disgust"
    else:
        return "neutral"

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            continue
        
        # Resize frame for faster processing
        frame = cv2.resize(frame, (640, 480))
        
        # Detect faces
        gray = preprocess_frame(frame)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=3)
        
        # Process each face
        emotion = "neutral"
        for (x, y, w, h) in faces:
            # Extract features and detect emotion
            features = extract_facial_features(frame, x, y, w, h)
            emotion = detect_emotion(features)
            
            # Smooth emotion predictions with weighted voting
            emotion_history.append(emotion)
            if len(emotion_history) >= 3:
                # Count frequency of emotions, weighting recent frames more
                emotion_counts = {}
                for i, e in enumerate(emotion_history):
                    weight = 1.0 + (i / len(emotion_history)) * 0.5  # Recent frames have slightly higher weight
                    emotion_counts[e] = emotion_counts.get(e, 0) + weight
                emotion = max(emotion_counts, key=emotion_counts.get)
            
            # Draw bounding box and emotion text
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f'{emotion.upper()}', (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        # Yield frame for streaming
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)