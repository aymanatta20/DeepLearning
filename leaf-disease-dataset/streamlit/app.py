import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image


MODEL_PATH = "LeafDisease.h5"
model = tf.keras.models.load_model(MODEL_PATH)

st.title(" Leaf Disease Classification")
st.write("Upload a leaf image and the model will predict the disease.")

uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    img_resized = image.resize((224, 224))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  

    
    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction, axis=1)[0]

    st.write(" Prediction:", class_idx)
