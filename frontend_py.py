import kivy
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.switch import Switch
from kivy.uix.spinner import Spinner
from kivy.uix.popup import Popup
from kivy.uix.progressbar import ProgressBar
from kivy.uix.image import Image
from kivy.uix.card import MDCard
from kivy.clock import Clock
from kivy.graphics import Color, RoundedRectangle
from kivy.metrics import dp
from kivymd.app import MDApp
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.card import MDCard
from kivymd.uix.button import MDRaisedButton, MDIconButton
from kivymd.uix.label import MDLabel
from kivymd.uix.selectioncontrol import MDSwitch
from kivymd.uix.menu import MDDropdownMenu
from kivymd.uix.toolbar import MDTopAppBar
from kivymd.uix.gridlayout import MDGridLayout
from kivymd.uix.floatlayout import MDFloatLayout
from kivymd.uix.progressbar import MDProgressBar
from kivymd.theming import ThemableBehavior
import requests, threading, json, cv2
from io import BytesIO
from kivy.graphics.texture import Texture

class EmotionCard(MDCard):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.md_bg_color = [0.2, 0.2, 0.2, 0.1]
        self.radius = [15]
        self.elevation = 3
        self.padding = dp(20)
        self.spacing = dp(10)
        
        layout = MDBoxLayout(orientation='vertical', adaptive_height=True)
        
        self.emotion_label = MDLabel(
            text="😐 Neutral",
            font_style="H4",
            halign="center",
            size_hint_y=None,
            height=dp(80)
        )
        
        self.confidence_bar = MDProgressBar(
            value=75,
            size_hint_y=None,
            height=dp(10)
        )
        
        self.gender_label = MDLabel(
            text="Gender: Unknown",
            font_style="Subtitle1",
            halign="center",
            size_hint_y=None,
            height=dp(40)
        )
        
        layout.add_widget(self.emotion_label)
        layout.add_widget(self.confidence_bar)
        layout.add_widget(self.gender_label)
        self.add_widget(layout)

class ModelSwitcher(MDCard):
    def __init__(self, callback=None, **kwargs):
        super().__init__(**kwargs)
        self.callback = callback
        self.md_bg_color = [0.1, 0.1, 0.1, 0.1]
        self.radius = [15]
        self.elevation = 2
        self.padding = dp(20)
        
        layout = MDBoxLayout(orientation='vertical', spacing=dp(15))
        
        title = MDLabel(
            text="AI Model Selection",
            font_style="H6",
            size_hint_y=None,
            height=dp(40)
        )
        
        model_layout = MDBoxLayout(orientation='horizontal', spacing=dp(10))
        
        self.hf_btn = MDRaisedButton(
            text="🤗 HuggingFace",
            md_bg_color=[0.2, 0.6, 1, 1],
            size_hint=(0.5, None),
            height=dp(45)
        )
        self.hf_btn.bind(on_press=lambda x: self.switch_model('hf'))
        
        self.deepface_btn = MDRaisedButton(
            text="🧠 DeepFace",
            md_bg_color=[0.6, 0.6, 0.6, 1],
            size_hint=(0.5, None),
            height=dp(45)
        )
        self.deepface_btn.bind(on_press=lambda x: self.switch_model('deepface'))
        
        model_layout.add_widget(self.hf_btn)
        model_layout.add_widget(self.deepface_btn)
        
        layout.add_widget(title)
        layout.add_widget(model_layout)
        self.add_widget(layout)
        
    def switch_model(self, model):
        if model == 'hf':
            self.hf_btn.md_bg_color = [0.2, 0.6, 1, 1]
            self.deepface_btn.md_bg_color = [0.6, 0.6, 0.6, 1]
        else:
            self.hf_btn.md_bg_color = [0.6, 0.6, 0.6, 1]
            self.deepface_btn.md_bg_color = [0.2, 0.6, 1, 1]
        
        if self.callback:
            self.callback(model)

class SpotifyCard(MDCard):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.md_bg_color = [0.1, 0.8, 0.3, 0.1]
        self.radius = [15]
        self.elevation = 3
        self.padding = dp(20)
        self.connected = False
        
        layout = MDBoxLayout(orientation='vertical', spacing=dp(15))
        
        title = MDLabel(
            text="🎵 Spotify Integration",
            font_style="H6",
            size_hint_y=None,
            height=dp(40)
        )
        
        self.status_label = MDLabel(
            text="Status: Disconnected",
            font_style="Subtitle2",
            theme_text_color="Secondary",
            size_hint_y=None,
            height=dp(30)
        )
        
        button_layout = MDBoxLayout(orientation='horizontal', spacing=dp(10))
        
        self.connect_btn = MDRaisedButton(
            text="Connect",
            md_bg_color=[0.1, 0.8, 0.3, 1],
            size_hint=(0.5, None),
            height=dp(45)
        )
        self.connect_btn.bind(on_press=self.toggle_spotify)
        
        self.play_btn = MDRaisedButton(
            text="Play Mood",
            md_bg_color=[0.6, 0.6, 0.6, 1],
            size_hint=(0.5, None),
            height=dp(45),
            disabled=True
        )
        self.play_btn.bind(on_press=self.play_mood_playlist)
        
        button_layout.add_widget(self.connect_btn)
        button_layout.add_widget(self.play_btn)
        
        layout.add_widget(title)
        layout.add_widget(self.status_label)
        layout.add_widget(button_layout)
        self.add_widget(layout)
    
    def toggle_spotify(self, *args):
        if not self.connected:
            threading.Thread(target=self.connect_spotify, daemon=True).start()
        else:
            self.disconnect_spotify()
    
    def connect_spotify(self):
        try:
            response = requests.get("http://localhost:5000/login", timeout=5)
            if response.status_code == 200:
                self.connected = True
                Clock.schedule_once(self.update_connected_ui)
        except Exception as e:
            Clock.schedule_once(lambda dt: self.show_error(f"Connection failed: {str(e)}"))
    
    def update_connected_ui(self, *args):
        self.status_label.text = "Status: Connected ✅"
        self.connect_btn.text = "Disconnect"
        self.connect_btn.md_bg_color = [0.8, 0.2, 0.2, 1]
        self.play_btn.disabled = False
        self.play_btn.md_bg_color = [0.1, 0.8, 0.3, 1]
    
    def disconnect_spotify(self):
        self.connected = False
        self.status_label.text = "Status: Disconnected"
        self.connect_btn.text = "Connect"
        self.connect_btn.md_bg_color = [0.1, 0.8, 0.3, 1]
        self.play_btn.disabled = True
        self.play_btn.md_bg_color = [0.6, 0.6, 0.6, 1]
    
    def play_mood_playlist(self, *args):
        # This would trigger mood-based playlist
        pass
    
    def show_error(self, message):
        popup = Popup(
            title="Error",
            content=MDLabel(text=message),
            size_hint=(0.8, 0.4)
        )
        popup.open()

class VideoFeed(Image):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.allow_stretch = True
        self.keep_ratio = True
        Clock.schedule_interval(self.update_frame, 1/30)
    
    def update_frame(self, dt):
        try:
            # Simulate getting frame from Flask backend
            # In real implementation, you'd get this from your video feed
            response = requests.get("http://localhost:5000/video_feed", stream=True, timeout=1)
            if response.status_code == 200:
                # Process video frame here
                pass
        except:
            pass

class ControlPanel(MDCard):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.md_bg_color = [0.1, 0.1, 0.1, 0.05]
        self.radius = [15]
        self.elevation = 2
        self.padding = dp(20)
        
        layout = MDBoxLayout(orientation='vertical', spacing=dp(15))
        
        title = MDLabel(
            text="⚙️ Controls",
            font_style="H6",
            size_hint_y=None,
            height=dp(40)
        )
        
        self.detection_running = False
        
        button_layout = MDBoxLayout(orientation='horizontal', spacing=dp(10))
        
        self.start_btn = MDRaisedButton(
            text="▶️ Start Detection",
            md_bg_color=[0.2, 0.8, 0.2, 1],
            size_hint=(0.5, None),
            height=dp(45)
        )
        self.start_btn.bind(on_press=self.toggle_detection)
        
        self.settings_btn = MDIconButton(
            icon="cog",
            theme_icon_color="Primary",
            size_hint=(None, None),
            size=(dp(45), dp(45))
        )
        self.settings_btn.bind(on_press=self.show_settings)
        
        button_layout.add_widget(self.start_btn)
        button_layout.add_widget(self.settings_btn)
        
        layout.add_widget(title)
        layout.add_widget(button_layout)
        self.add_widget(layout)
    
    def toggle_detection(self, *args):
        if not self.detection_running:
            self.start_detection()
        else:
            self.stop_detection()
    
    def start_detection(self):
        try:
            requests.get("http://localhost:5000/start_detection", timeout=2)
            self.detection_running = True
            self.start_btn.text = "⏹️ Stop Detection"
            self.start_btn.md_bg_color = [0.8, 0.2, 0.2, 1]
        except:
            pass
    
    def stop_detection(self):
        try:
            requests.get("http://localhost:5000/stop_detection", timeout=2)
            self.detection_running = False
            self.start_btn.text = "▶️ Start Detection"
            self.start_btn.md_bg_color = [0.2, 0.8, 0.2, 1]
        except:
            pass
    
    def show_settings(self, *args):
        content = MDBoxLayout(orientation='vertical', spacing=dp(20), size_hint_y=None)
        content.bind(minimum_height=content.setter('height'))
        
        # Add settings options here
        settings_label = MDLabel(text="Settings will be here", halign="center")
        content.add_widget(settings_label)
        
        popup = Popup(
            title="Settings",
            content=content,
            size_hint=(0.8, 0.6)
        )
        popup.open()

class EmotionDetectionApp(MDApp):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.title = "Emotion Detection Dashboard"
        self.theme_cls.theme_style = "Light"
        self.theme_cls.primary_palette = "Blue"
        self.theme_cls.accent_palette = "Amber"
        
    def build(self):
        main_layout = MDBoxLayout(orientation='vertical')
        
        # Top App Bar
        toolbar = MDTopAppBar(
            title="🧠 Emotion Detection Dashboard",
            right_action_items=[
                ["theme-light-dark", lambda x: self.toggle_theme()],
                ["information", lambda x: self.show_info()]
            ]
        )
        
        # Main content
        content = MDBoxLayout(orientation='horizontal', spacing=dp(20), padding=dp(20))
        
        # Left side - Video and controls
        left_panel = MDBoxLayout(orientation='vertical', spacing=dp(20), size_hint_x=0.6)
        
        # Video feed
        video_card = MDCard(md_bg_color=[0.1, 0.1, 0.1, 0.1], radius=[15], elevation=3, padding=dp(20))
        video_layout = MDBoxLayout(orientation='vertical')
        
        video_title = MDLabel(text="📹 Live Feed", font_style="H6", size_hint_y=None, height=dp(40))
        self.video_feed = VideoFeed(size_hint=(1, 1))
        
        video_layout.add_widget(video_title)
        video_layout.add_widget(self.video_feed)
        video_card.add_widget(video_layout)
        
        # Control panel
        self.control_panel = ControlPanel(size_hint_y=None, height=dp(120))
        
        left_panel.add_widget(video_card)
        left_panel.add_widget(self.control_panel)
        
        # Right side - Status and controls
        right_panel = MDBoxLayout(orientation='vertical', spacing=dp(20), size_hint_x=0.4)
        
        # Emotion display
        self.emotion_card = EmotionCard(size_hint_y=None, height=dp(200))
        
        # Model switcher
        self.model_switcher = ModelSwitcher(
            callback=self.switch_model,
            size_hint_y=None,
            height=dp(140)
        )
        
        # Spotify integration
        self.spotify_card = SpotifyCard(size_hint_y=None, height=dp(160))
        
        right_panel.add_widget(self.emotion_card)
        right_panel.add_widget(self.model_switcher)
        right_panel.add_widget(self.spotify_card)
        
        content.add_widget(left_panel)
        content.add_widget(right_panel)
        
        main_layout.add_widget(toolbar)
        main_layout.add_widget(content)
        
        # Start status updates
        Clock.schedule_interval(self.update_status, 2.0)
        
        return main_layout
    
    def toggle_theme(self):
        if self.theme_cls.theme_style == "Light":
            self.theme_cls.theme_style = "Dark"
        else:
            self.theme_cls.theme_style = "Light"
    
    def switch_model(self, model):
        try:
            response = requests.get(f"http://localhost:5000/set_model/{model}", timeout=2)
            if response.status_code == 200:
                print(f"Switched to {model} model")
        except Exception as e:
            print(f"Failed to switch model: {e}")
    
    def update_status(self, dt):
        try:
            response = requests.get("http://localhost:5000/status", timeout=1)
            if response.status_code == 200:
                data = response.json()
                emotion = data.get('emotion', 'neutral')
                gender = data.get('gender', 'unknown')
                emoji = data.get('emoji', '😐')
                
                self.emotion_card.emotion_label.text = f"{emoji} {emotion.title()}"
                self.emotion_card.gender_label.text = f"Gender: {gender.title()}"
                
                # Update progress bar based on emotion confidence (simulated)
                confidence = hash(emotion) % 40 + 60  # Simulate 60-100% confidence
                self.emotion_card.confidence_bar.value = confidence
        except:
            pass
    
    def show_info(self):
        content = MDBoxLayout(orientation='vertical', spacing=dp(20))
        
        info_text = """
🧠 Emotion Detection Dashboard

Features:
• Real-time emotion detection
• Gender recognition
• Model switching (HuggingFace/DeepFace)
• Spotify integration
• Dark/Light theme
• Live video feed

Created with Kivy & KivyMD
        """
        
        info_label = MDLabel(
            text=info_text.strip(),
            halign="center",
            valign="middle"
        )
        
        content.add_widget(info_label)
        
        popup = Popup(
            title="About",
            content=content,
            size_hint=(0.8, 0.6)
        )
        popup.open()

if __name__ == "__main__":
    EmotionDetectionApp().run()