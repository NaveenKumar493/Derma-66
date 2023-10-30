from flask import render_template, jsonify, Flask, redirect, url_for, request
import random
import os
import numpy as np
from keras.applications.mobilenet import MobileNet
from keras.preprocessing import image
from keras.applications.mobilenet import preprocess_input, decode_predictions
from keras.models import model_from_json
import keras
from keras import backend as K
from PIL import Image
import io

app = Flask(__name__)

# Define the SKIN_CLASSES dictionary here at the global scope
SKIN_CLASSES = {
  0: 'Actinic Keratoses (Solar Keratoses) or intraepithelial Carcinoma (Bowens disease)',
  1: 'Basal Cell Carcinoma',
  2: 'Benign Keratosis',
  3: 'Dermatofibroma',
  4: 'Melanoma',
  5: 'Melanocytic Nevi',
  6: 'Vascular skin lesion'
}


# Define your SKIN_CLASSES dictionary here

@app.route('/')
def index():
    return render_template('index.html', title='Home')


@app.route('/uploaded', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']
        if f.filename == '':
            return render_template('uploaded.html', title='Upload an Image')

        # Read the image from the file storage and convert it to a PIL Image
        image_bytes = f.read()
        img = Image.open(io.BytesIO(image_bytes))
        img = img.resize((224, 224))  # Resize the image to your target size

        # Convert the PIL image to a NumPy array
        img_array = np.array(img)

        # Preprocess the image as needed
        img_array = img_array / 255.0  # Normalize pixel values

        # Reshape the image to match the model's input shape
        img_array = img_array.reshape((1, 224, 224, 3))

        # Load the model and make a prediction
        j_file = open('modelnew.json', 'r')
        loaded_json_model = j_file.read()
        j_file.close()
        model = model_from_json(loaded_json_model)
        model.load_weights('modelnew.h5')
        prediction = model.predict(img_array)
        pred = np.argmax(prediction)
        disease = SKIN_CLASSES[pred]
        accuracy = prediction[0][pred]
        K.clear_session()

        return render_template('uploaded.html', title='Success', predictions=disease, acc=accuracy * 100,
                               img_file=f.filename)

    return render_template('uploaded.html', title='Upload an Image')


if __name__ == "__main__":
    app.run(debug=True)
