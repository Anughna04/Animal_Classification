# app.py
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Class labels (same order as training)
class_labels = ['Bear', 'Bird', 'Cat', 'Cow', 'Deer', 'Dog', 'Dolphin', 'Elephant', 'Giraffe', 'Horse', 'Kangaroo', 'Lion', 'Panda', 'Tiger', 'Zebra']

# Load the MobileNetV2 model
@st.cache_resource
def load_mobilenet_model():
    return load_model("animal_classifier_model.h5")

model = load_mobilenet_model()

# Streamlit UI
st.title("🐾 Animal Image Classifier")
st.markdown("Upload an image of an animal and get the predicted class instantly.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocessing
    img = img.resize((224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)[0]
    predicted_index = np.argmax(prediction)
    predicted_label = class_labels[predicted_index]
    confidence = prediction[predicted_index] * 100

    # Result
    st.success(f"Predicted Animal: **{predicted_label}**")
    st.info(f"Confidence: **{confidence:.2f}%**")
