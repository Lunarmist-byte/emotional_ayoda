from flask import Flask,render_template,Response,jsonify,request,session,redirect,url_for
import cv2,random,os,threading,pygame
from emotion_utils import detect_emotion_gender
import spotipy
from spotipy.oauth2 import SpotifyOAuth

app=Flask(__name__)
app.secret_key = 'supersecretkey'  # Change this in production!

BASE=os.path.dirname(os.path.abspath(__file__))
MEMES=os.path.join(BASE,"static","memes")
AUDIOS=os.path.join(BASE,"static","audio")
WALLS=os.path.join(BASE,"static","wallpapers")

EMOJI_MAP = {
  "neutral":"😐",
  "happiness":"😄",
  "sadness":"😢",
  "anger":"😠",
  "surprise":"😲",
  "fear":"😨",
  "disgust":"🤢",
  "contempt":"😒"
}

cam=cv2.VideoCapture(0)
cur_emo="neutral"
cur_gen="unknown"
meme_path=""
audio_path=""
pygame.mixer.init()

# Spotify OAuth config (fill with your credentials)
SPOTIFY_CLIENT_ID = "YOUR_CLIENT_ID"
SPOTIFY_CLIENT_SECRET = "YOUR_CLIENT_SECRET"
SPOTIFY_REDIRECT_URI = "http://localhost:5000/callback"
SCOPE = "user-read-playback-state,user-modify-playback-state"

def play_audio(path):
    try:
        pygame.mixer.music.load(path)
        pygame.mixer.music.play()
    except: pass

def set_wallpaper(path):
    import platform
    if platform.system()=="Windows":
        import ctypes
        ctypes.windll.user32.SystemParametersInfoW(20,0,path,3)

def random_media(emo):
    mf=os.path.join(MEMES,emo)
    af=os.path.join(AUDIOS,emo)
    m=random.choice(os.listdir(mf)) if os.path.exists(mf) else ""
    a=random.choice(os.listdir(af)) if os.path.exists(af) else ""
    return (f"/static/memes/{emo}/{m}" if m else "", os.path.join(af,a) if a else "")

# Spotify auth
def get_spotify_auth():
    return SpotifyOAuth(client_id=SPOTIFY_CLIENT_ID,
                        client_secret=SPOTIFY_CLIENT_SECRET,
                        redirect_uri=SPOTIFY_REDIRECT_URI,
                        scope=SCOPE,
                        cache_path=".cache-"+session.get('uuid', 'default'))

def get_spotify_client():
    auth = get_spotify_auth()
    token_info = auth.get_cached_token()
    if not token_info:
        return None
    return spotipy.Spotify(auth=token_info['access_token'])

# Play playlist based on emotion
def spotify_play_playlist(sp,emotion):
    # simple mapping, update with your own playlist URIs
    playlists = {
        "neutral":"spotify:playlist:37i9dQZF1DXcBWIGoYBM5M",
        "happiness":"spotify:playlist:37i9dQZF1DXdPec7aLTmlC",
        "sadness":"spotify:playlist:37i9dQZF1DX7qK8ma5wgG1",
        "anger":"spotify:playlist:37i9dQZF1DWZJM4X7N4z9J",
        "surprise":"spotify:playlist:37i9dQZF1DX4fpCWaHOned",
        "fear":"spotify:playlist:37i9dQZF1DX0Yxoavh5qJV",
        "disgust":"spotify:playlist:37i9dQZF1DX2d9bfVyTiXJ",
        "contempt":"spotify:playlist:37i9dQZF1DX0SM0LYsmbMT"
    }
    uri=playlists.get(emotion,playlists["neutral"])
    devices=sp.devices()
    if devices['devices']:
        sp.start_playback(device_id=devices['devices'][0]['id'],context_uri=uri)

def mood_loop():
    global cur_emo,cur_gen,meme_path,audio_path
    while True:
        ret,frame=cam.read()
        if not ret: continue
        emo,gen=detect_emotion_gender(frame)
        if emo!=cur_emo:
            cur_emo=emo
            cur_gen=gen
            meme_path,audio_path=random_media(emo)
            if audio_path:
                threading.Thread(target=play_audio,args=(audio_path,),daemon=True).start()
            wall=os.path.join(WALLS,f"{emo}.jpg")
            if os.path.exists(wall):
                set_wallpaper(wall)

            sp=get_spotify_client()
            if sp:
                try:
                    spotify_play_playlist(sp,emo)
                except Exception as e:
                    print("Spotify play error:",e)

        cv2.waitKey(2500)

threading.Thread(target=mood_loop,daemon=True).start()

@app.route('/')
def index():
    return render_template("index.html",emotion=cur_emo,gender=cur_gen,meme=meme_path)

@app.route('/video_feed')
def video_feed():
    def gen():
        while True:
            s,frame=cam.read()
            if not s: break
            if meme_path:
                overlay=cv2.imread(meme_path[1:])
                if overlay is not None:
                    overlay=cv2.resize(overlay,(150,150))
                    frame[10:160,10:160]=overlay
            _,buf=cv2.imencode('.jpg',frame)
            yield(b'--frame\r\nContent-Type:image/jpeg\r\n\r\n'+buf.tobytes()+b'\r\n')
    return Response(gen(),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def status():
    return jsonify({"emotion":cur_emo,"gender":cur_gen})

@app.route('/login')
def login():
    sp_oauth=get_spotify_auth()
    auth_url=sp_oauth.get_authorize_url()
    return redirect(auth_url)

@app.route('/callback')
def callback():
    sp_oauth=get_spotify_auth()
    session.clear()
    code=request.args.get('code')
    token_info=sp_oauth.get_access_token(code)
    session['token_info']=token_info
    return redirect('/')

if __name__=="__main__":
    app.run(debug=True)
