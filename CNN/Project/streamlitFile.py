import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import streamlit as st


# Load model

model = load_model(r"C:\Users\mrtat\Downloads\NTI\Project\my_model.h5")
class_names = ["Normal", "Pneumonia"]


# Image Preprocessing

def preprocess_image(image):
    image = image.convert("RGB")  # Ensure it's 3-channel
    img = image.resize((64, 64))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


# Streamlit UI

st.set_page_config(page_title="Pneumonia Detection", layout="centered")
st.title(" Pneumonia Detection from X-ray")

uploaded_files = st.file_uploader(
    "Upload one or more X-ray images (PNG, JPG, JPEG)",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True
)

if uploaded_files:
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file)
        st.image(image, caption=f"Uploaded: {uploaded_file.name}", use_container_width=True)

        input_array = preprocess_image(image)
        prediction = model.predict(input_array)
        predicted_class = class_names[np.argmax(prediction)]
        accuracy = predicted_class[prediction>.5]

        st.markdown(f"### Prediction: **{predicted_class}**")
        st.markdown(f"Accuracy: `{accuracy:.2f}%`")
        st.divider()
#streamlit run C:\Users\mrtat\Downloads\NTI\Project\streamlitFile.py  