import cv2
from flask import Flask, request, jsonify, render_template
import numpy as np
import io
from PIL import Image
from tensorflow.keras.models import load_model

app = Flask(__name__)

ocular_model_path = 'C:/Users/AlexandraStan/Desktop/facultate/An4/Sem 2/Licenta/models/ocular/4__MobileNetV2_model__9551.keras'  # Calea relativă
cervical_model_path = 'C:/Users/AlexandraStan/Desktop/facultate/An4/Sem 2/Licenta/models/cervical/MobileNetV2_model__9539.keras'

ocular_model = load_model(ocular_model_path)
cervical_model = load_model(cervical_model_path)

class_names_ocular = ["normal", "cataract", "glaucoma", "myopia"]
class_names_cervical = ["Dyskeratotic", "Koilocytotic", "Metaplastic", "Parabasal", "Superficial-Intermediate"]

def preprocess_image_ocular(image):
    image = np.array(image)
    image = cv2.resize(image, (64, 64))
    image = image / 255.0
    return np.expand_dims(image, axis=0)

def preprocess_image_cervical(image):
    image = np.array(image)
    image = cv2.resize(image, (96, 96))
    image = image / 255.0
    return np.expand_dims(image, axis=0)

@app.route('/SelectCondition/get_diagnosed.html')
def get_diagnosed():
    category = request.args.get('category')
    if category:
        return render_template('get_diagnosed.html', category=category)
    else:
        return render_template('error.html', message='Category not provided')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/SelectCondition/')
def medical_diagnosis():
    return render_template('select_condition.html')

@app.route('/predict/ocular', methods=['POST'])
def predict_ocular():
    print("Received POST request for ocular prediction.")
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
        processed_image = preprocess_image_ocular(image)


        prediction = ocular_model.predict(processed_image)
        print("Prediction made.")


        predicted_class_index = np.argmax(prediction)
        predicted_class = class_names_ocular[predicted_class_index]


        return jsonify({'prediction': predicted_class})

    except Exception as e:
        print("Error occurred during prediction:", e)
        return jsonify({'error': str(e)})

@app.route('/predict/cervical', methods=['POST'])
def predict_cervical():
    print("Received POST request for cervical prediction.")
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
        processed_image = preprocess_image_cervical(image)


        prediction = cervical_model.predict(processed_image)
        print("Prediction made.")


        predicted_class_index = np.argmax(prediction)
        predicted_class = class_names_cervical[predicted_class_index]


        return jsonify({'prediction': predicted_class})

    except Exception as e:
        print("Error occurred during prediction:", e)
        return jsonify({'error': str(e)})


app.run(debug=True, host='127.0.0.1', port=8000)
