import os
import glob
import datetime
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

import streamlit as st





import mlflow
import mlflow.keras

st.set_page_config(page_title="Are You a Cat?", page_icon="🐱", layout="wide")

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)
DEFAULT_MODEL_PATH = os.path.join(MODEL_DIR, "baseline_cat_model.h5")
DATA_DIR = "data"
CLASS_NAMES = ['cats', 'dogs', 'humans']

# Configure MLflow SQLite backend
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Cat_Dog_Human_Classification_Tracker")

def get_available_models():
    model_files = glob.glob(os.path.join(MODEL_DIR, "*.h5"))
    if not model_files:
        return []
    return sorted(model_files, key=os.path.getmtime, reverse=True)

if "selected_model_path" not in st.session_state:
    st.session_state.selected_model_path = DEFAULT_MODEL_PATH

@st.cache_resource
def load_trained_model(model_path):
    if os.path.exists(model_path):
        try:
            return tf.keras.models.load_model(model_path)
        except Exception:
            return None
    available = get_available_models()
    if available:
        try:
            st.session_state.selected_model_path = available[0]
            return tf.keras.models.load_model(available[0])
        except Exception:
            return None
    return None

model = load_trained_model(st.session_state.selected_model_path)

if "prediction_results" not in st.session_state:
    st.session_state.prediction_results = None
if "current_image_name" not in st.session_state:
    st.session_state.current_image_name = None

# UI Header
st.title("🐱 Are You a Cat? – Computer Vision System")
st.write("Classify images using MobileNetV2 with active learning feedback and experiment tracking.")

current_model = model if model is not None else load_trained_model(st.session_state.selected_model_path)

if current_model is not None:
    st.caption(f"⚙️ **Active Model Checkpoint:** `{os.path.basename(st.session_state.selected_model_path)}`")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    
    if current_model is not None:
        if st.session_state.current_image_name != uploaded_file.name or st.session_state.prediction_results is None:
            st.session_state.current_image_name = uploaded_file.name
            
            img_resized = image.resize((150, 150))
            img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
            img_array = np.expand_dims(img_array, axis=0)
            
            predictions = current_model.predict(img_array)
            scores = predictions[0]
            predicted_index = np.argmax(scores)
            
            st.session_state.prediction_results = {
                "scores": scores,
                "predicted_class": CLASS_NAMES[predicted_index],
                "confidence": scores[predicted_index] * 100
            }

    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.image(image, caption='Uploaded Image', use_container_width=True)
    
    with col2:
        st.subheader("Classification Results")
        
        if st.session_state.prediction_results is not None:
            res = st.session_state.prediction_results
            scores = res["scores"]
            predicted_class = res["predicted_class"]
            confidence = res["confidence"]
            
            if confidence >= 75.0:
                st.success(f"**Prediction:** {predicted_class.upper()} ({confidence:.2f}% Confidence)")
            else:
                st.warning(f"**Low Confidence Prediction:** {predicted_class.upper()} ({confidence:.2f}% Confidence)")
            
            st.write("### Class Probabilities:")
            for name, score in zip(CLASS_NAMES, scores):
                st.progress(float(score), text=f"{name.capitalize()}: {score*100:.2f}%")
            
            st.markdown("---")
            st.subheader("📝 Active Learning Feedback Collector")
            correct_label = st.selectbox("Is this prediction wrong? Select actual category:", CLASS_NAMES)
            
            if st.button("💾 Save to Dataset for Retraining"):
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = os.path.join(DATA_DIR, correct_label, f"feedback_{correct_label}_{timestamp}.jpg")
                image.save(save_path)
                st.success(f"✅ Saved image to `data/{correct_label}/` successfully!")
        else:
            st.error("No model checkpoint loaded. Run training script in terminal first.")

else:
    st.session_state.prediction_results = None
    st.session_state.current_image_name = None

st.markdown("---")

# Admin Panel
with st.expander("⚙️ Admin Management & MLflow Tracking"):
    st.subheader("🔄 Model Version Rollback")
    available_models = get_available_models()
    
    if available_models:
        selected_version = st.selectbox(
            "Select Checkpoint to Activate:",
            available_models,
            index=0 if st.session_state.selected_model_path not in available_models else available_models.index(st.session_state.selected_model_path)
        )
        if st.button("🔄 Activate Selected Version"):
            st.session_state.selected_model_path = selected_version
            st.session_state.prediction_results = None
            st.cache_resource.clear()
            st.success(f"Switched active model to `{os.path.basename(selected_version)}`!")
            st.rerun()

    st.markdown("---")
    st.subheader("⚡ Trigger MLflow-Tracked Retraining")
    
    learning_rate = st.selectbox("Select Learning Rate", [0.001, 0.01], index=0)
    epochs = st.slider("Select Epochs", min_value=3, max_value=15, value=7)
    
    if st.button("⚡ Start Retraining Loop"):
        with st.spinner("Training model & logging metrics to MLflow..."):
            mlflow.tensorflow.autolog()
            
            with mlflow.start_run(run_name=f"Retrain_LR_{learning_rate}"):
                train_ds = tf.keras.utils.image_dataset_from_directory(
                    DATA_DIR, labels='inferred', label_mode='int',
                    class_names=CLASS_NAMES, validation_split=0.2, subset="training",
                    seed=123, image_size=(150, 150), batch_size=32
                )
                val_ds = tf.keras.utils.image_dataset_from_directory(
                    DATA_DIR, labels='inferred', label_mode='int',
                    class_names=CLASS_NAMES, validation_split=0.2, subset="validation",
                    seed=123, image_size=(150, 150), batch_size=32
                )
                
                base_model = tf.keras.applications.MobileNetV2(
                    input_shape=(150, 150, 3), include_top=False, weights='imagenet'
                )
                base_model.trainable = False
                
                retrained_model = tf.keras.models.Sequential([
                    tf.keras.layers.Input(shape=(150, 150, 3)),
                    tf.keras.layers.Rescaling(1./127.5, offset=-1),
                    base_model,
                    tf.keras.layers.GlobalAveragePooling2D(),
                    tf.keras.layers.Dropout(0.3),
                    tf.keras.layers.Dense(3, activation='softmax')
                ])
                
                retrained_model.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy']
                )
                
                history = retrained_model.fit(train_ds, validation_data=val_ds, epochs=epochs)
                final_val_acc = history.history['val_accuracy'][-1]
                
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                new_model_filename = f"model_v_{timestamp}.h5"
                new_model_path = os.path.join(MODEL_DIR, new_model_filename)
                
                retrained_model.save(new_model_path)
                retrained_model.save(DEFAULT_MODEL_PATH)
                
                st.session_state.selected_model_path = new_model_path
                st.session_state.prediction_results = None
                st.cache_resource.clear()
                
                st.success(f"🎉 Retraining complete! Saved `{new_model_filename}` (Val Accuracy: {final_val_acc*100:.2f}%).")
                st.rerun()