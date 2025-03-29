# -*- coding: utf-8 -*-
"""
Created on Sat Mar 29 20:34:05 2025

@author: vicky
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
from deepface import DeepFace

# Set Streamlit page layout
st.set_page_config(page_title="Age Prediction App", layout="wide")

# Title and Description
st.title("🎂 Age Prediction Using DeepFace")
st.write("Upload an image or use a webcam to predict the person's age.")

# Function to predict age
def analyze_age(image):
    try:
        # Analyze the image for age prediction
        result = DeepFace.analyze(image, actions=['age'], enforce_detection=False)
        age = result[0]['age']
        return age
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

        # Predict age
        age = analyze_age(image)

        if age is not None:
            st.image(image, caption=f"Predicted Age: {age}", use_column_width=True)
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

        # Predict age
        age = analyze_age(frame_rgb)

        # Display the result
        if age is not None:
            cv2.putText(frame_rgb, f"Age: {age}", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Show the webcam feed in Streamlit
        stframe.image(frame_rgb, channels="RGB", use_column_width=True)

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
