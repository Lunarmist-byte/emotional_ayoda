import cv2
from emotion_utlis import EmotionDetector

detector = EmotionDetector()

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    emo, gen = detector.detect_emotion_gender(frame)
    print(f"Emotion: {emo}, Gender: {gen}")
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
