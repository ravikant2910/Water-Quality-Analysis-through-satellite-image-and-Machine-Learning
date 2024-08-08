import cv2
import numpy as np
import streamlit as st
import CIE
import water
import joblib
st.set_page_config(layout="wide", page_title="Water Analysis App")
# Add custom CSS to remove space above the title
custom_css = """
<style>
.stApp {
    margin-top: 0;
    padding-top: 0;
    
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# Set the page title
#st.set_page_config(layout="wide", page_title="Water Analysis App")

# Create a centered title using HTML and CSS within Markdown
centered_title = """
<div style="text-align: center;">
    <h1>Water Quality Analysis using Satellite Images Model</h1>
</div>
"""


st.markdown(centered_title, unsafe_allow_html=True)

# Load the trained model
model = joblib.load('wavelength_model.pkl')

# Upload an image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Analyze the uploaded image using the water.py script
    image_data = uploaded_file.read()
    nparr = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
    path = "analyze.jpg"
    with open(path, "wb") as f:
        f.write(image_data)
    r1 = water.analyze_image(path)

    # Process the image using CIE and wave
    x = CIE.run(image)

    # Predict the wavelength
    x_array = np.array(x)
    predicted_wavelength = model.predict([x_array])

    # Define a layout with two columns
    col1, col2 = st.columns(2)

    # Left column for image, r1, and wavelength
    with col1:
        st.image(image, caption="Uploaded Image", width=400)
        st.write(r1)
        st.write("Predicted Wavelength:", predicted_wavelength[0], "nm")

    # Right column (for now, it's blank)
    with col2:
        pass
