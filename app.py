import os
import streamlit as tf_streamlit
import tensorflow as tf
from PIL import Image, ImageOps  # Integrated ImageOps here for handling rotation tags
import numpy as np

# Page configuration for a more professional layout
tf_streamlit.set_page_config(page_title="Are You a Cat?", layout="centered")

tf_streamlit.title("🐱 Are You a Cat? Production System")
tf_streamlit.write("Upload a selfie to check if you are a Cat, a Dog, or a Human!")

# Define the path to our saved model from Week 1
MODEL_PATH = "baseline_cat_model.h5"

# Load the model into memory safely
if os.path.exists(MODEL_PATH):
    # We use a caching decorator so Streamlit doesn't reload the heavy model file on every click
    @tf_streamlit.cache_resource
    def load_my_model():
        return tf.keras.models.load_model(MODEL_PATH)
    
    model = load_my_model()
    tf_streamlit.success("🤖 Core AI Model loaded successfully into production!")
else:
    tf_streamlit.error(f"❌ Model file '{MODEL_PATH}' not found. Please run train_baseline.py first.")

# Create the File Uploader interface element
uploaded_file = tf_streamlit.file_uploader("Choose an image file...", type=["jpg", "jpeg", "png"])

# Define the exact class ordering matching our training data structures
CLASS_NAMES = ['cats', 'dogs', 'humans']

if uploaded_file is not None:
    # Open the uploaded file using PIL
    raw_image = Image.open(uploaded_file)
    
    # 🔥 FIX: Auto-rotate the image to its true orientation based on phone camera EXIF data
    image = ImageOps.exif_transpose(raw_image)
    
    # Display layout split into two columns (Left: Image, Right: Analysis)
    col1, col2 = tf_streamlit.columns(2)
    
    with col1:
        # Streamlit now prints the correctly oriented image copy
        tf_streamlit.image(image, caption="Uploaded Photo", use_container_width=True)
    
    # Preprocess image format cleanly using our oriented copy
    image_rgb = image.convert('RGB')
    img_resized = image_rgb.resize((150, 150))
    img_array = tf.keras.utils.img_to_array(img_resized)
    img_tensor = np.expand_dims(img_array, axis=0) # Converts shape layout to (1, 150, 150, 3)
    
    # Execute inference prediction
    with col2:
        tf_streamlit.write("🧠 **AI Analysis Running...**")
        predictions = model.predict(img_tensor)
        score = predictions[0] # Array of 3 probabilities (e.g., [0.1, 0.1, 0.8])
        
        # Find the index with the highest probability score value
        predicted_class_index = np.argmax(score)
        final_prediction = CLASS_NAMES[predicted_class_index]
        confidence_percentage = float(score[predicted_class_index] * 100)
        
        # --- UX FEATURE: CONFIDENCE GRAPH ---
        tf_streamlit.write("📊 **Class Breakdown:**")
        for i, class_name in enumerate(CLASS_NAMES):
            class_prob = float(score[i])
            tf_streamlit.text(f"{class_name.capitalize()}: {class_prob*100:.1f}%")
            tf_streamlit.progress(class_prob) # Displays a visual loading-style bar
            
        tf_streamlit.divider()
        
        # --- THRESHOLD GUARDRAIL ---
        # If the highest prediction score is less than 60%, trigger an uncertainty alert
        CONFIDENCE_THRESHOLD = 60.0
        
        if confidence_percentage < CONFIDENCE_THRESHOLD:
            tf_streamlit.warning(
                f"⚠️ **Result is Uncertain.** The model's top guess is **{final_prediction.upper()}** "
                f"but with only **{confidence_percentage:.1f}%** confidence. "
                f"The image might be too close up, blurry, or poorly cropped!"
            )
        else:
            tf_streamlit.subheader(f"🎯 Prediction: **{final_prediction.upper()}**")
            tf_streamlit.info(f"Confidence Level: **{confidence_percentage:.1f}%**")