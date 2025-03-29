# -*- coding: utf-8 -*-
"""
Created on Sat Mar 29 12:05:59 2025

@author: vicky
"""
import streamlit as st
import cv2
from deepface import DeepFace
import numpy as np
from PIL import Image

# Set page config
st.set_page_config(page_title="Emotion Detection App", layout="wide")

# Title
st.title("😊 Real-time Emotion Detection with DeepFace 🎥")

# Select mode: Webcam or Upload
option = st.sidebar.radio("Choose Input Method:", ["📸 Upload Image", "🎥 Live Webcam"])

# Emoji for emotions
emotion_emoji = {
    "happy": "😊",
    "sad": "😔",
    "angry": "😡",
    "surprise": "😮",
    "fear": "😨",
    "neutral": "😐",
    "disgust": "🤢"
}

# Function to analyze emotions
def analyze_emotion(image):
    try:
        # Analyze emotions using DeepFace
        result = DeepFace.analyze(image, actions=['emotion'], enforce_detection=False)
        emotion = result[0]['dominant_emotion']
        emoji = emotion_emoji.get(emotion, "😐")
        return emotion, emoji
    except Exception as e:
        return "No face detected", "❌"

# 📸 Handle image upload
if option == "📸 Upload Image":
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Convert image to array for processing
        image = np.array(Image.open(uploaded_file))

        # Analyze emotion
        emotion, emoji = analyze_emotion(image)

        # Display uploaded image
        st.image(image, caption=f"Detected Emotion: {emotion} {emoji}", use_column_width=True)

# 🎥 Handle live webcam
elif option == "🎥 Live Webcam":
    # Start the webcam
    cap = cv2.VideoCapture(0)
    stframe = st.empty()

    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture frame. Please check your webcam.")
            break

        # Convert to RGB (for compatibility with DeepFace)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Analyze emotion
        emotion, emoji = analyze_emotion(frame_rgb)

        # Add text to frame
        cv2.putText(frame_rgb, f"Emotion: {emotion} {emoji}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the frame
        stframe.image(frame_rgb, channels="RGB", use_column_width=True)

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
