from flask import Flask, request, jsonify, render_template
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# Load your trained model
model = load_model('accident_detector.h5')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            print("❌ No file in request.files")
            return jsonify({'result': 'No file part'}), 400

        file = request.files['file']

        if file.filename == '':
            print("❌ No selected file")
            return jsonify({'result': 'No file selected'}), 400

        print(f"📂 File received: {file.filename}")

        # Save file to temporary folder
        os.makedirs('test_images', exist_ok=True)
        file_path = os.path.join('test_images', file.filename)
        file.save(file_path)
        print(f"✅ File saved to: {file_path}")

        # Preprocess image
        img = image.load_img(file_path, target_size=(224, 224))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        prediction = model.predict(img_array)
        print(f"📊 Prediction: {prediction}")

        result = "No Accident Detected" if prediction[0][0] > 0.5 else "Accident Detected"
        return jsonify({'result': result})

    except Exception as e:
        print(f"❌ Exception: {str(e)}")
        return jsonify({'result': 'An error occurred. Please try again.'}), 500

if __name__ == '__main__':
    app.run(debug=True)
