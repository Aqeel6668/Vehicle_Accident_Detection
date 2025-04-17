import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Load model
model = tf.keras.models.load_model("accident_detector.h5")

# Load and preprocess test image
img_path = "test_images/test1.jpg"  # change to your test image path
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Predict
prediction = model.predict(img_array)
if prediction[0][0] > 0.5:
    print("🚗 No Accident")
else:
    print("💥 Accident Detected")
