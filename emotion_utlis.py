import cv2, torch, numpy as np
from transformers import AutoFeatureExtractor, AutoModelForImageClassification
import logging

logger = logging.getLogger(__name__)

try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except ImportError:
    DEEPFACE_AVAILABLE = False
    logger.warning("DeepFace not available, using HF models only")

class EmotionDetector:
    def __init__(self, default_model='hf'):
        self.current_model = default_model
        self.emotion_labels = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear', 'contempt']
        self.hf_models_loaded = False
        self.load_hf_models()
    
    def load_hf_models(self):
        try:
            self.emo_model_name = "nateraw/ferplus"
            self.gen_model_name = "nateraw/gender-classification-retail-0009"
            
            self.emo_extractor = AutoFeatureExtractor.from_pretrained(self.emo_model_name)
            self.emo_model = AutoModelForImageClassification.from_pretrained(self.emo_model_name)
            self.gen_extractor = AutoFeatureExtractor.from_pretrained(self.gen_model_name)
            self.gen_model = AutoModelForImageClassification.from_pretrained(self.gen_model_name)
            
            self.hf_models_loaded = True
            logger.info("HF models loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load HF models: {e}")
            self.hf_models_loaded = False
    
    def preprocess_image(self, img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return cv2.resize(img_rgb, (64, 64))
    
    def predict_emotion_hf(self, frame):
        if not self.hf_models_loaded:
            return "neutral"
        
        try:
            img = self.preprocess_image(frame)
            inputs = self.emo_extractor(images=img, return_tensors="pt")
            with torch.no_grad():
                outputs = self.emo_model(**inputs)
            probs = outputs.logits.softmax(dim=1)[0].cpu().numpy()
            return self.emotion_labels[np.argmax(probs)]
        except Exception as e:
            logger.error(f"HF emotion prediction error: {e}")
            return "neutral"
    
    def predict_gender_hf(self, frame):
        if not self.hf_models_loaded:
            return "unknown"
        
        try:
            img = self.preprocess_image(frame)
            inputs = self.gen_extractor(images=img, return_tensors="pt")
            with torch.no_grad():
                outputs = self.gen_model(**inputs)
            probs = outputs.logits.softmax(dim=1)[0].cpu().numpy()
            return "male" if probs[0] > probs[1] else "female"
        except Exception as e:
            logger.error(f"HF gender prediction error: {e}")
            return "unknown"
    
    def predict_deepface(self, frame):
        if not DEEPFACE_AVAILABLE:
            return "neutral", "unknown"
        
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            analysis = DeepFace.analyze(rgb_frame, actions=['emotion', 'gender'], enforce_detection=False)
            
            if isinstance(analysis, list):
                analysis = analysis[0]
            
            emotion = analysis.get('dominant_emotion', 'neutral')
            gender_data = analysis.get('gender', {})
            
            if isinstance(gender_data, dict):
                gender = 'male' if gender_data.get('Man', 0) > gender_data.get('Woman', 0) else 'female'
            else:
                gender = str(gender_data).lower()
            
            return emotion, gender
        except Exception as e:
            logger.error(f"DeepFace prediction error: {e}")
            return "neutral", "unknown"
    
    def detect_faces(self, frame):
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            return faces
        except Exception as e:
            logger.error(f"Face detection error: {e}")
            return []
    
    def detect_emotion_gender(self, frame):
        if frame is None or frame.size == 0:
            return "neutral", "unknown"
        
        faces = self.detect_faces(frame)
        if len(faces) == 0:
            # No faces detected, analyze full frame
            face_region = frame
        else:
            # Use the largest detected face
            largest_face = max(faces, key=lambda f: f[2] * f[3])
            x, y, w, h = largest_face
            face_region = frame[y:y+h, x:x+w]
        
        if self.current_model == 'hf':
            emotion = self.predict_emotion_hf(face_region)
            gender = self.predict_gender_hf(face_region)
            return emotion, gender
        elif self.current_model == 'deepface':
            return self.predict_deepface(face_region)
        else:
            # Fallback to HF if available, otherwise deepface
            if self.hf_models_loaded:
                emotion = self.predict_emotion_hf(face_region)
                gender = self.predict_gender_hf(face_region)
                return emotion, gender
            else:
                return self.predict_deepface(face_region)
    
    def set_model(self, model_name):
        valid_models = ['hf', 'deepface', 'local']
        if model_name not in valid_models:
            raise ValueError(f"Invalid model: {model_name}. Valid options: {valid_models}")
        
        if model_name == 'deepface' and not DEEPFACE_AVAILABLE:
            raise ValueError("DeepFace not available. Install with: pip install deepface")
        
        if model_name == 'hf' and not self.hf_models_loaded:
            raise ValueError("HF models not loaded properly")
        
        self.current_model = model_name
        logger.info(f"Switched to model: {model_name}")
    
    def get_current_model(self):
        return self.current_model
    
    def get_available_models(self):
        models = []
        if self.hf_models_loaded:
            models.append('hf')
        if DEEPFACE_AVAILABLE:
            models.append('deepface')
        return models

# Emoji mapping for compatibility
EMOJI_MAP = {"neutral": "😐", "happiness": "😄", "sadness": "😢", "anger": "😠", "surprise": "😲", "fear": "😨", "disgust": "🤢", "contempt": "😒"}

# Legacy function for backward compatibility
def detect_emotion_gender(frame, use_hf=True):
    detector = EmotionDetector('hf' if use_hf else 'deepface')
    return detector.detect_emotion_gender(frame)

def detect_emotion_gender_hf(frame):
    detector = EmotionDetector('hf')
    return detector.detect_emotion_gender(frame)