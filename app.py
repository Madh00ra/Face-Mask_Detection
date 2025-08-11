import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import gdown

FILE_ID = "1tAx5Ld2Gmwew2eiHU7ZIW249Ti7AeKAI"
MODEL_PATH = "FaceMaskDetection.keras"

# Function to download the model if not present
def download_model():
    if not os.path.exists(MODEL_PATH):
        st.info("Downloading model from Google Drive... Please wait.")
        url = f"https://drive.google.com/uc?id={FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)
        st.success("Model downloaded successfully!")

# Load model
model = tf.keras.models.load_model("FaceMaskDetection.keras")

# Title
st.title("😷 Face Mask Detection")

# Choose input method
option = st.radio("Select input method:", ["Upload Image", "Use Camera"])

# Handle input
if option == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
elif option == "Use Camera":
    camera_image = st.camera_input("Take a photo")
    if camera_image is not None:
        image = Image.open(camera_image).convert("RGB")

# Predict if image is available
if 'image' in locals():
    st.image(image, caption="Selected Image", use_column_width=True)

    # Preprocess
    img = image.resize((150, 150))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)[0][0]
    label = "Mask" if prediction < 0.5 else "No Mask"
    confidence = 1 - prediction if prediction < 0.5 else prediction

    #Display
    st.markdown(f"### Prediction: `{label}`")
    st.markdown(f"Confidence: `{confidence:.2f}`")

