import streamlit as st
import numpy as np
import tensorflow as tf
import keras  # ✅ إضافة استيراد keras مباشرة
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.resnet_v2 import preprocess_input
import cv2
from PIL import Image
import os

# Set page config
st.set_page_config(
    page_title="Brain Tumor AI System",
    page_icon="🧠",
    layout="wide"
)

# --- Mode Selection ---
st.sidebar.title("🔬 Select Mode")
mode = st.sidebar.radio("Choose Task:", ["Classification", "Segmentation"], index=0)

# Title and description
st.title("🧠 Brain Tumor AI System")
st.markdown("""
<div style='color:#4F8BF9; font-size:20px;'>
    Upload a brain MRI image to detect a tumor (Classification) or segment tumor area (Segmentation).
</div>
""", unsafe_allow_html=True)
st.markdown("---")

if mode == "Classification":
    # Load the model
    @st.cache_resource
    def load_model():
        try:
            model_path ="brain_tumor_classifier.h5"
            if os.path.exists(model_path):
                # ✅ استخدام keras بدل tf.keras
                model = keras.models.load_model(model_path, compile=False)
                st.success("✅ Model loaded successfully!")
                return model
            else:
                st.error("❌ Model file not found!")
                return None
        except Exception as e:
            st.error(f"❌ Error loading model: {str(e)}")
            return None

    # باقي الكود زي ما هو...
    # -----------------------------
    def preprocess_image(image, target_size=(224, 224)):
        try:
            if isinstance(image, Image.Image):
                img_array = img_to_array(image)
            else:
                img_array = image
            img_resized = cv2.resize(img_array, target_size)
            if len(img_resized.shape) == 2:
                img_resized = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
            elif img_resized.shape[2] == 1:
                img_resized = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
            img_normalized = img_resized.astype('float32') / 255.0
            img_batch = np.expand_dims(img_normalized, axis=0)
            return img_batch
        except Exception as e:
            st.error(f"Error preprocessing image: {str(e)}")
            return None

    def predict_tumor(model, image):
        try:
            processed_img = preprocess_image(image)
            if processed_img is None:
                return None, None
            prediction = model.predict(processed_img, verbose=0)
            probability = prediction[0][0]
            predicted_class = "Tumor Detected" if probability > 0.5 else "No Tumor"
            return predicted_class, probability
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            return None, None

    model = load_model()
    # باقي عرض النتائج زي كودك الأصلي...
    # -----------------------------

else:
    st.subheader("🧠 Tumor Segmentation (Unet)")
    uploaded_file = st.file_uploader(
        "Choose a brain MRI image for segmentation...",
        type=['jpg', 'jpeg', 'png', 'bmp', 'tif', 'tiff'],
        key="segmentation_uploader"
    )

    @st.cache_resource
    def load_unet_model():
        try:
            model_path = "brain_tumor_segmentation_model.h5"
            if os.path.exists(model_path):
                # ✅ استخدام keras بدل tf.keras
                model = keras.models.load_model(model_path, compile=False)
                st.success("✅ Unet model loaded successfully!")
                return model
            else:
                st.error("❌ Unet model file not found!")
                return None
        except Exception as e:
            st.error(f"❌ Error loading Unet model: {str(e)}")
            return None

    # باقي كود الـ segmentation زي ما هو...
