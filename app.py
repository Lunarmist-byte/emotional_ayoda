from flask import Flask, render_template, Response, jsonify, request, session, redirect, url_for
import cv2, random, os, threading, pygame, time, logging
from datetime import datetime
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from emotion_utils import EmotionDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'supersecretkey-change-in-production')

class Config:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MEMES_DIR = os.path.join(BASE_DIR, "static", "memes")
    AUDIOS_DIR = os.path.join(BASE_DIR, "static", "audio")
    WALLPAPERS_DIR = os.path.join(BASE_DIR, "static", "wallpapers")
    
    SPOTIFY_CLIENT_ID = os.environ.get('SPOTIFY_CLIENT_ID', 'YOUR_CLIENT_ID')
    SPOTIFY_CLIENT_SECRET = os.environ.get('SPOTIFY_CLIENT_SECRET', 'YOUR_CLIENT_SECRET')
    SPOTIFY_REDIRECT_URI = os.environ.get('SPOTIFY_REDIRECT_URI', 'http://localhost:5000/callback')
    SPOTIFY_SCOPE = "user-read-playback-state,user-modify-playback-state"
    
    DETECTION_INTERVAL = 2.5
    FRAME_WIDTH = 640
    FRAME_HEIGHT = 480

config = Config()

EMOJI_MAP = {"neutral": "😐", "happiness": "😄", "sadness": "😢", "anger": "😠", "surprise": "😲", "fear": "😨", "disgust": "🤢", "contempt": "😒"}

SPOTIFY_PLAYLISTS = {
    "neutral": "spotify:playlist:37i9dQZF1DXcBWIGoYBM5M",
    "happiness": "spotify:playlist:37i9dQZF1DXdPec7aLTmlC",
    "sadness": "spotify:playlist:37i9dQZF1DX7qK8ma5wgG1",
    "anger": "spotify:playlist:37i9dQZF1DWZJM4X7N4z9J",
    "surprise": "spotify:playlist:37i9dQZF1DX4fpCWaHOned",
    "fear": "spotify:playlist:37i9dQZF1DX0Yxoavh5qJV",
    "disgust": "spotify:playlist:37i9dQZF1DX2d9bfVyTiXJ",
    "contempt": "spotify:playlist:37i9dQZF1DX0SM0LYsmbMT"
}

class EmotionApp:
    def __init__(self):
        self.camera = None
        self.detector = EmotionDetector()
        self.current_emotion = "neutral"
        self.current_gender = "unknown"
        self.current_meme_path = ""
        self.current_audio_path = ""
        self.detection_thread = None
        self.running = False
        self.last_detection_time = 0
        
        try:
            pygame.mixer.init()
            logger.info("Audio system initialized")
        except Exception as e:
            logger.error(f"Failed to initialize audio: {e}")
        
        self.initialize_camera()
        self.create_directories()
    
    def create_directories(self):
        dirs = [config.MEMES_DIR, config.AUDIOS_DIR, config.WALLPAPERS_DIR]
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)
            for emotion in EMOJI_MAP.keys():
                os.makedirs(os.path.join(dir_path, emotion), exist_ok=True)
    
    def initialize_camera(self):
        try:
            self.camera = cv2.VideoCapture(0)
            if self.camera.isOpened():
                self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
                self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
                logger.info("Camera initialized successfully")
            else:
                logger.error("Failed to open camera")
                self.camera = None
        except Exception as e:
            logger.error(f"Camera initialization error: {e}")
            self.camera = None
    
    def play_audio(self, path):
        if not os.path.exists(path):
            logger.warning(f"Audio file not found: {path}")
            return
        try:
            pygame.mixer.music.load(path)
            pygame.mixer.music.play()
            logger.info(f"Playing audio: {path}")
        except Exception as e:
            logger.error(f"Audio play error: {e}")
    
    def set_wallpaper(self, path):
        if not os.path.exists(path):
            return
        try:
            import platform
            if platform.system() == "Windows":
                import ctypes
                ctypes.windll.user32.SystemParametersInfoW(20, 0, path, 3)
            elif platform.system() == "Darwin":  # macOS
                os.system(f"osascript -e 'tell application \"Finder\" to set desktop picture to POSIX file \"{path}\"'")
            elif platform.system() == "Linux":
                os.system(f"gsettings set org.gnome.desktop.background picture-uri file://{path}")
        except Exception as e:
            logger.error(f"Wallpaper setting error: {e}")
    
    def get_random_media(self, emotion):
        meme_dir = os.path.join(config.MEMES_DIR, emotion)
        audio_dir = os.path.join(config.AUDIOS_DIR, emotion)
        
        meme_file = ""
        audio_file = ""
        
        if os.path.exists(meme_dir) and os.listdir(meme_dir):
            meme_file = f"/static/memes/{emotion}/{random.choice(os.listdir(meme_dir))}"
        
        if os.path.exists(audio_dir) and os.listdir(audio_dir):
            audio_file = os.path.join(audio_dir, random.choice(os.listdir(audio_dir)))
        
        return meme_file, audio_file
    
    def get_spotify_client(self):
        try:
            auth = SpotifyOAuth(
                client_id=config.SPOTIFY_CLIENT_ID,
                client_secret=config.SPOTIFY_CLIENT_SECRET,
                redirect_uri=config.SPOTIFY_REDIRECT_URI,
                scope=config.SPOTIFY_SCOPE,
                cache_path=".cache-" + session.get('uuid', 'default')
            )
            token_info = auth.get_cached_token()
            if not token_info:
                return None
            return spotipy.Spotify(auth=token_info['access_token'])
        except Exception as e:
            logger.error(f"Spotify client error: {e}")
            return None
    
    def play_spotify_playlist(self, emotion):
        sp = self.get_spotify_client()
        if not sp:
            return
        
        try:
            devices = sp.devices()
            if devices['devices']:
                playlist_uri = SPOTIFY_PLAYLISTS.get(emotion, SPOTIFY_PLAYLISTS["neutral"])
                sp.start_playback(device_id=devices['devices'][0]['id'], context_uri=playlist_uri)
                logger.info(f"Playing Spotify playlist for {emotion}")
        except Exception as e:
            logger.error(f"Spotify playback error: {e}")
    
    def emotion_detection_loop(self):
        self.running = True
        logger.info("Starting emotion detection loop")
        
        while self.running:
            if not self.camera or not self.camera.isOpened():
                time.sleep(1)
                continue
            
            ret, frame = self.camera.read()
            if not ret:
                time.sleep(0.1)
                continue
            
            current_time = time.time()
            if current_time - self.last_detection_time < config.DETECTION_INTERVAL:
                time.sleep(0.1)
                continue
            
            try:
                emotion, gender = self.detector.detect_emotion_gender(frame)
                
                if emotion != self.current_emotion:
                    logger.info(f"Emotion changed from {self.current_emotion} to {emotion}")
                    self.current_emotion = emotion
                    self.current_gender = gender
                    
                    self.current_meme_path, self.current_audio_path = self.get_random_media(emotion)
                    
                    if self.current_audio_path:
                        threading.Thread(target=self.play_audio, args=(self.current_audio_path,), daemon=True).start()
                    
                    wallpaper_path = os.path.join(config.WALLPAPERS_DIR, f"{emotion}.jpg")
                    if os.path.exists(wallpaper_path):
                        threading.Thread(target=self.set_wallpaper, args=(wallpaper_path,), daemon=True).start()
                    
                    threading.Thread(target=self.play_spotify_playlist, args=(emotion,), daemon=True).start()
                
                self.last_detection_time = current_time
                
            except Exception as e:
                logger.error(f"Detection loop error: {e}")
                time.sleep(1)
    
    def start_detection(self):
        if not self.detection_thread or not self.detection_thread.is_alive():
            self.detection_thread = threading.Thread(target=self.emotion_detection_loop, daemon=True)
            self.detection_thread.start()
    
    def stop_detection(self):
        self.running = False
        if self.detection_thread:
            self.detection_thread.join(timeout=5)
    
    def get_video_frame(self):
        if not self.camera or not self.camera.isOpened():
            return None
        
        ret, frame = self.camera.read()
        if not ret:
            return None
        
        if self.current_meme_path:
            try:
                meme_img_path = self.current_meme_path[1:]  # remove leading /
                if os.path.exists(meme_img_path):
                    overlay = cv2.imread(meme_img_path)
                    if overlay is not None:
                        overlay = cv2.resize(overlay, (150, 150))
                        frame[10:160, 10:160] = overlay
            except Exception as e:
                logger.error(f"Overlay error: {e}")
        
        return frame
    
    def cleanup(self):
        self.stop_detection()
        if self.camera:
            self.camera.release()
        cv2.destroyAllWindows()

# Global app instance
emotion_app = EmotionApp()

@app.route("/")
def index():
    return render_template("index.html", 
                         emotion=emotion_app.current_emotion, 
                         gender=emotion_app.current_gender, 
                         meme=emotion_app.current_meme_path, 
                         emoji=EMOJI_MAP.get(emotion_app.current_emotion, "😐"))

@app.route("/video_feed")
def video_feed():
    def generate_frames():
        while True:
            frame = emotion_app.get_video_frame()
            if frame is None:
                time.sleep(0.1)
                continue
            
            _, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/status")
def status():
    return jsonify({
        "emotion": emotion_app.current_emotion,
        "gender": emotion_app.current_gender,
        "emoji": EMOJI_MAP.get(emotion_app.current_emotion, "😐"),
        "timestamp": datetime.now().isoformat()
    })

@app.route("/set_model/<model>")
def set_model(model):
    if model in ['local', 'hf', 'deepface']:
        try:
            emotion_app.detector.set_model(model)
            return jsonify({"status": "success", "model": model})
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 400
    return jsonify({"status": "error", "message": "Invalid model"}), 400

@app.route("/login")
def login():
    try:
        sp_oauth = SpotifyOAuth(
            client_id=config.SPOTIFY_CLIENT_ID,
            client_secret=config.SPOTIFY_CLIENT_SECRET,
            redirect_uri=config.SPOTIFY_REDIRECT_URI,
            scope=config.SPOTIFY_SCOPE
        )
        auth_url = sp_oauth.get_authorize_url()
        return redirect(auth_url)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/callback")
def callback():
    try:
        sp_oauth = SpotifyOAuth(
            client_id=config.SPOTIFY_CLIENT_ID,
            client_secret=config.SPOTIFY_CLIENT_SECRET,
            redirect_uri=config.SPOTIFY_REDIRECT_URI,
            scope=config.SPOTIFY_SCOPE
        )
        session.clear()
        code = request.args.get("code")
        token_info = sp_oauth.get_access_token(code)
        session['token_info'] = token_info
        return redirect("/")
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/start_detection")
def start_detection():
    emotion_app.start_detection()
    return jsonify({"status": "Detection started"})

@app.route("/stop_detection")
def stop_detection():
    emotion_app.stop_detection()
    return jsonify({"status": "Detection stopped"})

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    try:
        emotion_app.start_detection()
        app.run(debug=True, host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        emotion_app.cleanup()