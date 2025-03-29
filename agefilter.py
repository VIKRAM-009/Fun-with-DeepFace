# -*- coding: utf-8 -*-
"""
Created on Sat Mar 29 21:45:43 2025

@author: vicky
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
from deepface import DeepFace

# Set Streamlit page layout
st.set_page_config(page_title="Age Progression & Regression", layout="wide")

# Title and description
st.title("🕰️ Age Progression/Regression Analysis")
st.write("Upload an image or use a webcam to see how age progression/regression affects facial features.")

# Function to predict age
def predict_age(image):
    """Predicts age from the image using DeepFace."""
    try:
        result = DeepFace.analyze(image, actions=['age'], enforce_detection=False)
        age = result[0]['age']
        return age
    except Exception as e:
        st.error(f"Error: {e}")
        return None

# Function to apply age progression/regression effect
def apply_age_effect(image, age_diff):
    """Applies age progression/regression effect by adjusting brightness and sharpness."""
    # Convert to PIL format
    pil_img = Image.fromarray(image)

    # Simulate age progression
    if age_diff > 0:
        # Increase brightness and sharpness for progression
        enhanced_img = pil_img.point(lambda p: p * (1 + age_diff * 0.05)).convert('RGB')
    else:
        # Decrease brightness for regression
        enhanced_img = pil_img.point(lambda p: p * (1 - abs(age_diff) * 0.05)).convert('RGB')

    return np.array(enhanced_img)

# Sidebar options for input
option = st.sidebar.radio("Choose Input Method:", ["📸 Upload Image", "🎥 Live Webcam"])
age_option = st.sidebar.slider("Select Age Progression/Regression (years):", -30, 30, 0)

# 📸 Handle image upload
if option == "📸 Upload Image":
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Convert uploaded image to array
        image = np.array(Image.open(uploaded_file))

        # Predict current age
        predicted_age = predict_age(image)

        if predicted_age is not None:
            # Apply age effect based on selected age difference
            modified_image = apply_age_effect(image, age_option)

            # Display original and modified images
            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption=f"Original Age: {predicted_age}", use_column_width=True)
            with col2:
                st.image(modified_image, caption=f"Modified Age: {predicted_age + age_option}", use_column_width=True)
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

        # Convert frame to RGB for Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Predict age from the frame
        predicted_age = predict_age(frame_rgb)

        if predicted_age is not None:
            # Apply age effect based on selected age difference
            modified_frame = apply_age_effect(frame_rgb, age_option)

            # Display the webcam frame with modification
            cv2.putText(modified_frame, f"Modified Age: {predicted_age + age_option}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

            stframe.image(modified_frame, channels="RGB", use_column_width=True)

        # Exit the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
