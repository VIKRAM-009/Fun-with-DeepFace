# -*- coding: utf-8 -*-
"""
Created on Sat Mar 29 21:39:27 2025

@author: vicky
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Set Streamlit page layout
st.set_page_config(page_title="AI Face Beautification & Filters", layout="wide")

# Title and Description
st.title("🎨 AI-Powered Face Beautification & Filters")
st.write("Upload an image or use your webcam to apply AI-powered face filters!")

# Define available filters
filters = ["Original", "Grayscale", "Sketch", "Blur", "Beautify"]

# 📸 Upload Image Option
option = st.sidebar.radio("Choose Input Method:", ["📸 Upload Image", "🎥 Live Webcam"])
selected_filter = st.sidebar.selectbox("Choose a Filter", filters)

# Beautification function
def beautify_face(image):
    """Applies a smoothening effect to the face using bilateral filtering."""
    smoothed = cv2.bilateralFilter(image, 9, 75, 75)
    return smoothed

# Apply selected filter
def apply_filter(image, filter_name):
    """Applies the selected filter to the image."""
    if filter_name == "Grayscale":
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif filter_name == "Sketch":
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        inverted_img = cv2.bitwise_not(gray_img)
        blurred = cv2.GaussianBlur(inverted_img, (21, 21), 0)
        inverted_blur = cv2.bitwise_not(blurred)
        sketch = cv2.divide(gray_img, inverted_blur, scale=256.0)
        return sketch
    elif filter_name == "Blur":
        return cv2.GaussianBlur(image, (15, 15), 0)
    elif filter_name == "Beautify":
        return beautify_face(image)
    else:
        return image

# 📸 Handle Image Upload
if option == "📸 Upload Image":
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Convert uploaded image to numpy array
        image = np.array(Image.open(uploaded_file))

        # Apply the selected filter
        result_image = apply_filter(image, selected_filter)

        # Display the processed image
        st.image(result_image, caption=f"Filter: {selected_filter}", use_column_width=True)

# 🎥 Webcam Option
elif option == "🎥 Live Webcam":
    cap = cv2.VideoCapture(0)
    stframe = st.empty()

    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("⚠️ Failed to capture frame. Please check your webcam.")
            break

        # Flip and convert to RGB for Streamlit
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Apply the selected filter
        result_frame = apply_filter(frame_rgb, selected_filter)

        # Handle grayscale or sketch separately
        if selected_filter in ["Grayscale", "Sketch"]:
            stframe.image(result_frame, channels="GRAY", use_column_width=True)
        else:
            stframe.image(result_frame, channels="RGB", use_column_width=True)

        # Exit the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
