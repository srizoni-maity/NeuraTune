import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import pywhatkit
import os

# Load trained model
model = load_model("model/best_model.h5")

# Emotion labels
labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Folder containing emoji images
emoji_path = "emojis"

# Load all emoji images into a dictionary
emoji_images = {}
for label in labels:
    path = os.path.join(emoji_path, f"{label}.png")
    emoji = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if emoji is not None:
        emoji_images[label] = emoji
    else:
        print(f"Warning: Could not load image for {label}")

# Overlay emoji image on frame
def overlay_emoji(frame, emoji, x, y):
    try:
        h, w = emoji.shape[:2]
        if y - h < 0: y = h  # avoid overflow
        emoji_rgb = emoji[:, :, :3]
        alpha = emoji[:, :, 3] / 255.0

        for c in range(3):
            frame[y - h:y, x:x + w, c] = (
                alpha * emoji_rgb[:, :, c] +
                (1 - alpha) * frame[y - h:y, x:x + w, c]
            )
    except:
        pass  # just skip if size doesn't match or out of bounds

# Ask user for mode
mode = input("Choose mode:\n- Type 'face' to detect emotion via webcam\n- Type 'text' to enter emotion manually\nYour choice: ").strip().lower()
detected_emotion = None

if mode == 'face':
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi = cv2.resize(roi_gray, (48, 48))
            roi = roi.astype("float") / 255.0
            roi = np.expand_dims(roi, axis=0)
            roi = np.expand_dims(roi, axis=-1)

            preds = model.predict(roi)[0]
            emotion_idx = np.argmax(preds)
            detected_emotion = labels[emotion_idx]

            # Draw face rectangle
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            # Put emotion label
            cv2.putText(frame, detected_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # Overlay emoji if available
            if detected_emotion in emoji_images:
                emoji = emoji_images[detected_emotion]
                overlay_emoji(frame, emoji, x, y)

        cv2.imshow("Real-Time Emotion Detection", frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

elif mode == 'text':
    detected_emotion = input("Enter your current emotion (e.g., Happy, Sad, Angry): ").capitalize()
    if detected_emotion not in labels:
        print("Invalid emotion entered. Please restart and enter a valid emotion.")
        exit()

else:
    print("Invalid mode selected. Please run again and choose 'face' or 'text'.")
    exit()

# Ask for language and singer
if detected_emotion:
    language = input("Enter your preferred language (e.g., English, Hindi, Bengali): ")
    singer = input("Enter your favorite singer’s name: ")
    query = f"{singer} {language} {detected_emotion} song"
    print(f"\nOpening YouTube for: {query}")
    pywhatkit.playonyt(query)
else:
    print("No emotion detected.")
