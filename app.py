import os
import streamlit as tf_streamlit
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import datetime

tf_streamlit.set_page_config(page_title="Are You a Cat? Production", layout="wide")

MODEL_PATH = "baseline_cat_model.h5"
DATA_DIR = "data"
LOG_DIR = os.path.join(DATA_DIR, "logged_feedback")
CLASS_NAMES = ['cats', 'dogs', 'humans']

# Sidebar Analytics Panel
with tf_streamlit.sidebar:
    tf_streamlit.header("📊 Production Analytics")
    tf_streamlit.write("Current state of your dataset repository:")
    for c_name in CLASS_NAMES:
        folder_path = os.path.join(DATA_DIR, c_name)
        if os.path.exists(folder_path):
            num_files = len([f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))])
            tf_streamlit.metric(label=f"Total {c_name.capitalize()} Images", value=num_files)
        else:
            tf_streamlit.metric(label=f"Total {c_name.capitalize()} Images", value=0)
            
    tf_streamlit.divider()
    if os.path.exists(LOG_DIR):
        total_logs = len([f for f in os.listdir(LOG_DIR) if os.path.isfile(os.path.join(LOG_DIR, f))])
        tf_streamlit.metric(label="⚙️ Logged User Feedbacks", value=total_logs)

# Main Interface Header
tf_streamlit.title("🐱 Continuous Learning AI Production System")

with tf_streamlit.expander("💡 How to get 100% accurate results?"):
    tf_streamlit.write("""
    * **📸 Crop it close:** Upload a close-up portrait or selfie.
    * **🧍 Center the subject:** Make sure the face/pet is the main focus.
    * **🌿 Clean backgrounds:** Avoid heavily patterned or crowded backgrounds.
    """)

@tf_streamlit.cache_resource
def load_my_model():
    if os.path.exists(MODEL_PATH):
        return tf.keras.models.load_model(MODEL_PATH)
    return None

model = load_my_model()

if model is not None:
    tf_streamlit.success("🤖 Core AI Model loaded successfully into production!")
else:
    tf_streamlit.error(f"❌ Model file '{MODEL_PATH}' not found. Run training below to generate it.")

uploaded_file = tf_streamlit.file_uploader("Choose an image file...", type=["jpg", "jpeg", "png"])

def log_user_feedback(image_to_save, prediction_label, user_status):
    os.makedirs(LOG_DIR, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{user_status}_pred_{prediction_label}_{timestamp}.jpg"
    full_save_path = os.path.join(LOG_DIR, filename)
    image_to_save.save(full_save_path)
    return full_save_path

if uploaded_file is not None:
    raw_image = Image.open(uploaded_file)
    image = ImageOps.exif_transpose(raw_image)
    
    col1, col2 = tf_streamlit.columns(2)
    
    with col1:
        tf_streamlit.image(image, caption="Uploaded Photo", use_container_width=True)
    
    image_rgb = image.convert('RGB')
    img_resized = image_rgb.resize((150, 150))
    img_array = tf.keras.utils.img_to_array(img_resized)
    img_tensor = np.expand_dims(img_array, axis=0) 
    
    with col2:
        if model is not None:
            tf_streamlit.write("🧠 **AI Analysis Running...**")
            predictions = model.predict(img_tensor)
            score = predictions[0] 
            
            predicted_class_index = np.argmax(score)
            final_prediction = CLASS_NAMES[predicted_class_index]
            confidence_percentage = float(score[predicted_class_index] * 100)
            
            tf_streamlit.write("📊 **Class Breakdown:**")
            for i, class_name in enumerate(CLASS_NAMES):
                class_prob = float(score[i])
                tf_streamlit.text(f"{class_name.capitalize()}: {class_prob*100:.1f}%")
                tf_streamlit.progress(class_prob) 
                
            tf_streamlit.divider()
            
            CONFIDENCE_THRESHOLD = 75.0
            if confidence_percentage < CONFIDENCE_THRESHOLD:
                tf_streamlit.warning(f"⚠️ **Uncertain:** Top guess is {final_prediction.upper()} ({confidence_percentage:.1f}%).")
            else:
                tf_streamlit.subheader(f"🎯 Prediction: **{final_prediction.upper()}**")
                tf_streamlit.info(f"Confidence Level: **{confidence_percentage:.1f}%**")
        else:
            tf_streamlit.warning("Please train the model using the dashboard below first.")

        # Day 11 & 12 Functional Feedback Layout
        tf_streamlit.write("---")
        tf_streamlit.write("📝 **Was this prediction accurate?**")
        
        btn_col1, btn_col2 = tf_streamlit.columns(2)
        
        with btn_col1:
            if tf_streamlit.button("✅ Correct", use_container_width=True):
                saved_path = log_user_feedback(image_rgb, final_prediction, "CORRECT")
                tf_streamlit.success("Logged! Thank you for confirming.")
                tf_streamlit.rerun()
                
        with btn_col2:
            if tf_streamlit.button("❌ Incorrect", use_container_width=True):
                saved_path = log_user_feedback(image_rgb, final_prediction, "INCORRECT")
                tf_streamlit.error("Logged! Discrepancy sent to auditing.")
                tf_streamlit.rerun()

        # --- ACTIVE LEARNING DATA COLLECTOR ---
        tf_streamlit.write("---")
        tf_streamlit.write("🛠️ **Active Learning Data Collector**")
        
        correct_label = tf_streamlit.selectbox(
            "What is the correct classification for this image?",
            options=CLASS_NAMES
        )
        
        if tf_streamlit.button("📥 Submit Image to Training Dataset"):
            target_folder = os.path.join(DATA_DIR, correct_label)
            os.makedirs(target_folder, exist_ok=True)
            
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"user_feedback_{timestamp}.jpg"
            filepath = os.path.join(target_folder, filename)
            
            image_rgb.save(filepath)
            tf_streamlit.success(f"✅ Saved directly to `{correct_label}` dataset! Refreshing metrics...")
            tf_streamlit.rerun()

# --- 🔥 DAY 13 OPTIMIZED TRAINING DASHBOARD ---
tf_streamlit.write("---")
with tf_streamlit.expander("⚙️ Admin MLOps Training Dashboard", expanded=False):
    tf_streamlit.write("Click below to retrain the MobileNetV2 network using all accumulated user data simultaneously.")
    
    if tf_streamlit.button("⚡ Start Background Retraining Loop"):
        with tf_streamlit.spinner("Training in progress... Do not close this tab."):
            IMAGE_SIZE = (150, 150)
            BATCH_SIZE = 32
            
            # Loading latest dataset including newly injected user photos
            train_ds = tf.keras.utils.image_dataset_from_directory(
                DATA_DIR, validation_split=0.2, subset="training", seed=123,
                image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, label_mode="categorical",
                class_names=CLASS_NAMES
            )

            val_ds = tf.keras.utils.image_dataset_from_directory(
                DATA_DIR, validation_split=0.2, subset="validation", seed=123,
                image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, label_mode="categorical",
                class_names=CLASS_NAMES
            )
            
            # --- 🔥 DAY 13 OPTIMIZATION: ON-THE-FLY DATA AUGMENTATION ---
            data_augmentation = tf.keras.Sequential([
                tf.keras.layers.RandomFlip("horizontal"),
                tf.keras.layers.RandomRotation(0.15),
                tf.keras.layers.RandomZoom(0.1)
            ])
            
            base_model = tf.keras.applications.MobileNetV2(
                input_shape=(150, 150, 3), include_top=False, weights='imagenet'
            )
            base_model.trainable = False
            
            retrained_model = tf.keras.models.Sequential([
                tf.keras.layers.Input(shape=(150, 150, 3)),
                data_augmentation,  # Injected augmentation here to process new data dynamically
                tf.keras.layers.Rescaling(1./127.5, offset=-1),
                base_model,
                tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.Dropout(0.3),  # Increased slightly to prevent memorizing new edge cases
                tf.keras.layers.Dense(3, activation='softmax')
            ])
            
            retrained_model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
                loss='categorical_crossentropy', 
                metrics=['accuracy']
            )
            
            # 🔥 Extended to 7 epochs so the network thoroughly ingests the new user items
            retrained_model.fit(train_ds, validation_data=val_ds, epochs=7)
            
            retrained_model.save(MODEL_PATH)
            tf_streamlit.cache_resource.clear()
            
        tf_streamlit.success("🎉 Day 13 Retraining Complete! Model successfully updated live.")
        tf_streamlit.rerun()