import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

DATA_DIR = os.path.join("data")
BATCH_SIZE = 32
IMAGE_SIZE = (150, 150) # MobileNetV2 handles 150x150 processing efficiently

print("🔄 Loading datasets from disk...")
train_dataset = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR, validation_split=0.2, subset="training", seed=123,
    image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, label_mode="categorical"
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR, validation_split=0.2, subset="validation", seed=123,
    image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, label_mode="categorical"
)

AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.cache().prefetch(buffer_size=AUTOTUNE)

# Data Augmentation to keep the model flexible
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
])

print("\n🚀 Downloading pre-trained MobileNetV2 weights from Google...")
# We load the model WITHOUT the top classification layer because we are building our own
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(150, 150, 3),
    include_top=False,
    weights='imagenet'
)

# CRITICAL STEP: Freeze the pre-trained weights so we don't destroy Google's training
base_model.trainable = False

print("🧠 Assembling the Transfer Learning Architecture...")
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(150, 150, 3)),
    data_augmentation,
    
    # MobileNetV2 expects pixel values scaled between -1 and 1 instead of 0 and 255
    tf.keras.layers.Rescaling(1./127.5, offset=-1),
    
    base_model, # The frozen core containing 1.4 million images worth of features
    
    tf.keras.layers.GlobalAveragePooling2D(), # Flattens the 2D feature maps into a 1D vector
    tf.keras.layers.Dropout(0.2),             # Helps prevent overfitting on your dataset
    tf.keras.layers.Dense(3, activation='softmax') # Final 3 classes output
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Because MobileNetV2 is already highly educated, it trains incredibly fast!
EPOCHS = 10
print(f"\n⚡ Fine-tuning head layers for {EPOCHS} epochs...")
history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS
)

print("\n💾 Saving your upgraded production model...")
model.save("baseline_cat_model.h5")
print("🎉 Upgrade Complete!")