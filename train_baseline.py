import os
import tensorflow as tf

# Suppress log clutter
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

AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.cache().prefetch(buffer_size=AUTOTUNE)

# --- 🔥 UPGRADE: DATA AUGMENTATION LAYER ---
# This forces the model to look at shapes instead of just colors/backgrounds
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.5),
    tf.keras.layers.RandomZoom(0.1),
])

print("\n🧠 Building the Upgraded CNN Model...")

model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(150, 150, 3)),
    
    # Apply data augmentation to training data
    data_augmentation,
    
    tf.keras.layers.Rescaling(1./255),
    
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    
    # Dropout layer drops 30% of neurons randomly to prevent overfitting/memorization
    tf.keras.layers.Dropout(0.3), 
    
    tf.keras.layers.Dense(3, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# --- 📈 UPGRADE: MORE EPOCHS ---
# 5 epochs wasn't enough time for the model to understand the features. Let's do 15!
EPOCHS = 15
print(f"\n🚀 Training upgraded model for {EPOCHS} epochs...")

model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS
)

print("\n💾 Saving updated model...")
model.save("baseline_cat_model.h5")
print("🎉 Done! Re-run your Streamlit app now and test the image again.")