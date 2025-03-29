# -*- coding: utf-8 -*-
"""
Created on Sat Mar 29 20:45:38 2025

@author: vicky
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
from deepface import DeepFace

# Set Streamlit page layout
st.set_page_config(page_title="Race/Ethnicity Detection", layout="wide")

# Title and Description
st.title("🌎 Race/Ethnicity Detection")
st.write("Upload an image or use a webcam to detect the person's race/ethnicity.")

# Function to predict race
def analyze_race(image):
    try:
        # Analyze the image for race detection
        result = DeepFace.analyze(image, actions=['race'], enforce_detection=False)
        race_results = result[0]['race']
        detected_race = max(race_results, key=race_results.get)  # Get race with highest confidence
        return detected_race, race_results
    except Exception as e:
        return None, str(e)

# 📸 Upload Image Option
option = st.sidebar.radio("Choose Input Method:", ["📸 Upload Image", "🎥 Live Webcam"])

# Handle image upload
if option == "📸 Upload Image":
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Convert uploaded image to array
        image = np.array(Image.open(uploaded_file))

        # Predict race
        race, race_scores = analyze_race(image)

        if race is not None:
            st.image(image, caption=f"Predicted Race: {race}", use_column_width=True)
            st.write("🔍 **Confidence Scores:**")
            st.json(race_scores)
        else:
            st.error("⚠️ No face detected. Please try another image.")

# 🎥 Webcam Option
elif option == "🎥 Live Webcam":
    cap = cv2.VideoCapture(0)
    stframe = st.empty()

    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("⚠️ Failed to capture frame. Please check your webcam.")
            break

        # Convert frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Predict race
        race, race_scores = analyze_race(frame_rgb)

        # Display the result
        if race is not None:
            cv2.putText(frame_rgb, f"Race: {race}", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Show the webcam feed in Streamlit
        stframe.image(frame_rgb, channels="RGB", use_column_width=True)

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
