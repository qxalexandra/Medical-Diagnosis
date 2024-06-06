import cv2
from flask import Flask, request, jsonify, render_template
import numpy as np
import io
from PIL import Image
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Încarcă modelul Keras folosind calea relativă sau absolută
model_path = 'C:/Users/AlexandraStan/Desktop/facultate/An4/Sem 2/Licenta/models/ocular/4__MobileNetV2_model__9551.keras'  # Calea relativă
# model_path = '/absolute/path/to/models/model.h5'  # Calea absolută (dacă este necesar)
model = load_model(model_path)

def preprocess_image(image):
    image = np.array(image)
    image = cv2.resize(image, (64, 64))
    image = image / 255.0  # Scalarea pixelilor între 0 și 1
    return np.expand_dims(image, axis=0)  # Adaugă dimensiunea batch

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/MedicalDiagnosis/')
def medical_diagnosis():
    return 'Welcome to Medical Diagnosis'

@app.route('/predict', methods=['POST'])
def predict():
    print("Received POST request for prediction.")
    if 'file' not in request.files:
        print("No file part in request.")
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        print("No selected file.")
        return jsonify({'error': 'No selected file'})

    try:
        image = Image.open(io.BytesIO(file.read()))
        print("Image loaded successfully.")
        processed_image = preprocess_image(image)  # Nu este necesar să specificăm dimensiunea target_size aici
        print("Image preprocessed.")

        # Efectuează predicția
        prediction = model.predict(processed_image)
        print("Prediction made.")

        # Identifică clasa cu cea mai mare probabilitate
        clasa_prezisa = np.argmax(prediction)

        # Lista cu numele claselor asociate cu fiecare indice
        nume_clase = ["normal", "cataract", "glaucoma", "myopia"]

        # Afisează clasa prezisă
        boala_prezisa = nume_clase[clasa_prezisa]
        print("Boala prezisă este:", boala_prezisa)

        # Returnează rezultatul
        return jsonify({'prediction': boala_prezisa})
    except Exception as e:
        print("Error occurred during prediction:", e)
        return jsonify({'error': str(e)})

# Rulează serverul Flask
app.run(debug=True, host='127.0.0.1', port=8000)
