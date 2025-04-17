from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

# data path
dataset_path = "dataset"

# No validation_split here
image_gen = ImageDataGenerator(rescale=1. / 255)

train_data = image_gen.flow_from_directory(
    dataset_path,
    target_size=(224, 224),
    class_mode='binary',
    batch_size=2
)

# Define simple CNN model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train only on training data
model.fit(train_data, epochs=5)

# Save the trained model
model.save("accident_detector.h5")
print("✅ Model trained and saved!")
