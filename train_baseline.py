import os
import tensorflow as tf

# Suppress unnecessary log clutter in your terminal
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

DATA_DIR = os.path.join("data")
BATCH_SIZE = 32
IMAGE_SIZE = (150, 150)

print("🔄 Loading datasets from disk...")

train_dataset = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical"
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical"
)

class_names = train_dataset.class_names
print(f"\n✅ Successfully identified classes: {class_names}")

AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.cache().prefetch(buffer_size=AUTOTUNE)

print("🚀 Day 4 Setup Complete!")

print("\n🧠 Building the Convolutional Neural Network (CNN)...")

# FIXED Keras 3 Warning Layout by separating the Input definition layer
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(150, 150, 3)), 
    tf.keras.layers.Rescaling(1./255),
    
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

EPOCHS = 5
print(f"\n🚀 Starting training for {EPOCHS} epochs...")

history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS
)

# ==========================================
# 💾 DAY 6 ADDITION: SAVE THE MODEL ARTIFACT
# ==========================================
print("\n💾 Saving trained model to disk...")
MODEL_NAME = "baseline_cat_model.h5"
model.save(MODEL_NAME)

if os.path.exists(MODEL_NAME):
    print(f"🎉 Success! Your model file was created: '{MODEL_NAME}'")
    print("Day 6 complete. You are ready to start building the frontend UI next week!")
else:
    print("⚠️ Warning: Model file was not detected on disk.")