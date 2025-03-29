# -*- coding: utf-8 -*-
"""
Created on Sat Mar 29 19:45:23 2025

@author: vicky
"""

import streamlit as st
from deepface import DeepFace
import numpy as np
from PIL import Image

# Set Streamlit page layout
st.set_page_config(page_title="Face Verification App", layout="wide")

# Title and Description
st.title("🔍 Face Verification Using DeepFace")
st.write("Upload two images to verify if they belong to the same person.")

# Function to verify faces
def verify_faces(img1, img2):
    try:
        # Save uploaded images temporarily
        img1_path = r"C:\Users\vicky\Downloads\IMG_20241119_122631.jpg"
        img2_path = r"C:\Users\vicky\Downloads\IMG_20241007_170003.jpg"

        img1.save(img1_path)
        img2.save(img2_path)

        # Perform face verification
        result = DeepFace.verify(img1_path, img2_path, enforce_detection=False)

        # Get verification results
        verified = result["verified"]
        confidence = result["distance"]
        
        return verified, confidence
    except Exception as e:
        return None, str(e)

# 📸 Upload Images
col1, col2 = st.columns(2)

with col1:
    img1 = st.file_uploader("Upload First Image", type=["jpg", "jpeg", "png"])

with col2:
    img2 = st.file_uploader("Upload Second Image", type=["jpg", "jpeg", "png"])

# Check if both images are uploaded
if img1 and img2:
    # Convert images to PIL format
    img1_pil = Image.open(img1)
    img2_pil = Image.open(img2)

    # Display the uploaded images
    col1.image(img1_pil, caption="First Image", use_column_width=True)
    col2.image(img2_pil, caption="Second Image", use_column_width=True)

    # Verify the images when button is clicked
    if st.button("🔍 Verify Faces"):
        with st.spinner("Verifying... Please wait ⏳"):
            verified, confidence = verify_faces(img1_pil, img2_pil)

            if verified is not None:
                if verified:
                    st.success(f"✅ Verified! The images are of the same person. Confidence: {1 - confidence:.2f}")
                else:
                    st.error(f"❌ Not Verified! The images are of different persons. Confidence: {1 - confidence:.2f}")
            else:
                st.error(f"⚠️ Error: {confidence}")

else:
    st.info("Please upload both images to proceed.")
