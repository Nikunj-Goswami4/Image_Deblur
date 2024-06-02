from flask import Flask, render_template, request
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import pickle

with open('model.pkl', 'rb') as f:
    autoencoder = pickle.load(f)

app = Flask(__name__)

def predict_image(image_path, model):
    image = load_img(image_path, target_size=(256, 256))
    image = img_to_array(image).astype('float32') / 255
    image = np.expand_dims(image, axis=0)


    result = model.predict(image)
    result = np.squeeze(result)  # Remove the batch dimension
    return result


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':

        file = request.files['imagefile']

        if file.filename == '':
            return render_template('index.html', prediction_text='No selected file')

        file_path = 'uploads/' + file.filename
        file.save(file_path)

        predicted_image = predict_image(file_path, autoencoder)

        # os.remove(file_path)

        predicted_image_path = 'predictions/' + file.filename
        tf.keras.preprocessing.image.save_img(predicted_image_path, predicted_image)

        return render_template('index.html', predicted_image=predicted_image_path)
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run(port=3000, debug=True)
