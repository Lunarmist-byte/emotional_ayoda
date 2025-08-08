import cv2
import torch
import numpy as np
from transformers import AutoFeatureExtractor, AutoModelForImageClassification

USE_HF=True

emotion_labels=['neutral','happiness','surprise','sadness','anger','disgust','fear','contempt']

if USE_HF:
    emo_model_name="nateraw/ferplus"
    gen_model_name="nateraw/gender-classification-retail-0009"
    emo_extractor=AutoFeatureExtractor.from_pretrained(emo_model_name)
    emo_model=AutoModelForImageClassification.from_pretrained(emo_model_name)
    gen_extractor=AutoFeatureExtractor.from_pretrained(gen_model_name)
    gen_model=AutoModelForImageClassification.from_pretrained(gen_model_name)

def prep(img):
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    return cv2.resize(img,(64,64))

def predict_emotion_hf(frame):
    img=prep(frame)
    inputs=emo_extractor(images=img,return_tensors="pt")
    with torch.no_grad():
        out=emo_model(**inputs)
    probs=out.logits.softmax(dim=1)[0].cpu().numpy()
    return emotion_labels[np.argmax(probs)]

def predict_gender_hf(frame):
    img=prep(frame)
    inputs=gen_extractor(images=img,return_tensors="pt")
    with torch.no_grad():
        out=gen_model(**inputs)
    probs=out.logits.softmax(dim=1)[0].cpu().numpy()
    return"male" if probs[0]>probs[1] else"female"

def detect_emotion_gender(frame):
    if not USE_HF:
        return"neutral","unknown"
    try:
        return predict_emotion_hf(frame),predict_gender_hf(frame)
    except:
        return"neutral","unknown"

EMOJI_MAP={
  "neutral":"😐",
  "happiness":"😄",
  "sadness":"😢",
  "anger":"😠",
  "surprise":"😲",
  "fear":"😨",
  "disgust":"🤢",
  "contempt":"😒"
}
