import os
import streamlit as tf_streamlit
import tensorflow as tf
from PIL import Image
import numpy as np

tf_streamlit.title("🐱 Are You a Cat? Production System")
tf_streamlit.write("Upload a selfie to check if you are a Cat, a Dog, or a Human!")

MODEL_PATH = "baseline_cat_model.h5"

if os.path.exists(MODEL_PATH):
    @tf_streamlit.cache_resource
    def load_my_model():
        return tf.keras.models.load_model(MODEL_PATH)
    
    model = load_my_model()
    tf_streamlit.success("🤖 Core AI Model loaded successfully into production!")
else:
    tf_streamlit.error(f"❌ Model file '{MODEL_PATH}' not found. Please run train_baseline.py first.")

uploaded_file = tf_streamlit.file_uploader("Choose an image file...", type=["jpg", "jpeg", "png"])

CLASS_NAMES = ['cats', 'dogs', 'humans']

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    tf_streamlit.image(image, caption="Your Uploaded Photo", width=300)
    
    tf_streamlit.write("🧠 Analyzing image patterns...")
    
    # 🔥 FIX: Force conversion to standard 3-channel RGB (removes alpha/transparency layer)
    image_rgb = image.convert('RGB')
    
    # Process the safe RGB image copy instead
    img_resized = image_rgb.resize((150, 150))
    img_array = tf.keras.utils.img_to_array(img_resized)
    img_tensor = np.expand_dims(img_array, axis=0) 
    
    # Execute inference prediction safely
    predictions = model.predict(img_tensor)
    score = predictions[0] 
    
    predicted_class_index = np.argmax(score)
    final_prediction = CLASS_NAMES[predicted_class_index]
    confidence_percentage = float(score[predicted_class_index] * 100)
    
    tf_streamlit.subheader(f"🎯 Prediction: You are a **{final_prediction.upper()}**!")
    tf_streamlit.info(f"Confidence Level: **{confidence_percentage:.2f}%**")