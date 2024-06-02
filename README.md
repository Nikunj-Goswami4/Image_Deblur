# Image Deblurring Model

## Author & Description
**Author:** Nikunj Goswami  
**Aim:** A model that corrects blurred images and converts them to clear images.

![image_deblur_image](https://github.com/Nikunj-Goswami4/Image_Deblur/assets/92319257/0854eaaa-1bb7-41b8-add8-fd9656208090)

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Training the Model](#training-the-model)
- [Prediction](#prediction)
- [Flask Application](#flask-application)
- [Results](#results)
- [Contributing](#contributing)
- [Demonstration](#demonstration)

## Introduction
This repository contains code to train an image deblurring model using a Convolutional Neural Network (CNN). The model takes blurred images as input and outputs clear images. The model architecture is based on an autoencoder.

## Dataset
The dataset used for this project is from Kaggle and consists of two folders:
- `defocused_blurred`: Contains 350 blurred images.
- `sharp`: Contains 350 sharp and clear images.

[Dataset Link](https://www.kaggle.com/datasets/kwentar/blur-dataset)

## Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/Nikunj-Goswami4/Image_Deblur.git
    cd Image_Deblur
    ```
2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage
### Training the Model
1. Place the dataset folders (`defocused_blurred` and `sharp`) in the `data/` directory.
2. Run the Jupyter notebook `image_deblur.ipynb` to train and save the model.
3. The saved model is then used in `app.py`.

### Flask Application
1. Start the Flask server:
    ```bash
    python app.py
    ```
2. Open your web browser and go to `http://127.0.0.1:3000`.

### Making Predictions
1. Upload a blurred image using the web interface.
2. The predicted clear image will be displayed.

## Model Architecture
The model is an autoencoder consisting of an encoder and a decoder:
- **Encoder:** Composed of several convolutional layers to encode the input image to a latent space.
- **Decoder:** Composed of several transpose convolutional layers to decode the latent space back to the clear image.

## Training the Model
The model is trained using the Mean Squared Error (MSE) loss function and the Adam optimizer. The learning rate is reduced on a plateau.

## Prediction
Use the trained model to predict clear images from blurred images. Preprocess the image and pass it through the autoencoder model to get the deblurred result.

## Flask Application
The Flask application allows users to upload blurred images and receive deblurred images. It uses the trained model to predict the clear images and displays them on the web interface.

## Results
- The training and validation loss and accuracy are plotted to evaluate the model performance.
- Sample input, original, and predicted images are visualized for qualitative analysis.

## Contributing
Contributions are welcome! Please submit a pull request or open an issue to discuss what you would like to change.

## Demonstration
https://github.com/Nikunj-Goswami4/Image_Deblur/assets/92319257/1d98c01e-c09c-43f6-8783-63a17bd8ac88


