import cv2
import numpy as np
import streamlit as st
import CIE
import water
import joblib

st.title("Image-to-Wavelength Conversion Model ")

# Load the trained model
model = joblib.load('wavelength_model.pkl')

# Upload an image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image_data = uploaded_file.read()
    nparr = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
    
    # Save the uploaded image as "analyze.jpg"
    path = "analyze.jpg"
    with open(path, "wb") as f:
        f.write(image_data)
    
    st.image(image, caption="Uploaded Image", width = 250)
    
    # Analyze the uploaded image using the water.py script
    r1 = water.analyze_image(path)
    st.write(r1)

    # Process the image using CIE and wave
    x = CIE.run(image)

    # Predict the wavelength
    x_array = np.array(x)  # Convert the list to a NumPy array
    predicted_wavelength = model.predict([x_array])  # Wrap x_array in a list
    st.write(f"Predicted Wavelength: {predicted_wavelength[0]} nm")
