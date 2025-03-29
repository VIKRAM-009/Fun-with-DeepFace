# -*- coding: utf-8 -*-
"""
Created on Sat Mar 29 19:33:19 2025

@author: vicky
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
from deepface import DeepFace

# Set Streamlit page layout
st.set_page_config(page_title="Gender Prediction App", layout="wide")

# Title
st.title("👩‍🦰👨 Gender Prediction with DeepFace")

# Sidebar - Choose Input Method
option = st.sidebar.radio("Choose Input Method:", ["📸 Upload Image", "🎥 Live Webcam"])

# Function to analyze gender
def analyze_gender(image):
    try:
        # Analyze gender using DeepFace
        result = DeepFace.analyze(image, actions=['gender'], enforce_detection=False)
        gender = result[0]['dominant_gender']
        confidence = result[0]['gender'][gender]
        return gender, confidence
    except Exception as e:
        return "No face detected", 0

# 📸 Handle image upload
if option == "📸 Upload Image":
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Convert image to array
        image = np.array(Image.open(uploaded_file))

        # Analyze gender
        gender, confidence = analyze_gender(image)

        # Display the result
        st.image(image, caption=f"Predicted Gender: {gender} ({confidence:.2f}% confidence)", use_column_width=True)

# 🎥 Handle live webcam
elif option == "🎥 Live Webcam":
    # Start webcam
    cap = cv2.VideoCapture(0)
    stframe = st.empty()

    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture frame. Please check your webcam.")
            break

        # Convert to RGB for DeepFace compatibility
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Analyze gender
        gender, confidence = analyze_gender(frame_rgb)

        # Display gender on frame
        cv2.putText(frame_rgb, f"Gender: {gender} ({confidence:.2f}%)", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display webcam feed in Streamlit
        stframe.image(frame_rgb, channels="RGB", use_column_width=True)

        # Break loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
