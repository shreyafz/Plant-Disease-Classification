# Plant Disease Classification using Computer Vision

## Description:

This project implements a Convolutional Neural Network (CNN) model to classify citrus leaf diseases. The model can differentiate between four different diseases: Black spot, Canker, Greening, and Healthy. The project explores both training a CNN model from scratch and utilizing transfer learning with pre-trained models like EfficientNetB0 and VGG16.

## Getting Started:
### Prerequisites:

- Python 3.x
- TensorFlow
- TensorFlow Datasets
- matplotlib

## Installation:

Install required libraries: pip install -r requirements.txt
## Running the Script:

#### Train the model from scratch: 
- python train.py -- model_type scratch
#### Train a model using transfer learning (e.g., EfficientNetB0):
- python train.py --model_type efficientnetb0
## Structure:
- train.py: Script for training the CNN model.
data_utils.py: Utility functions for data preprocessing and augmentation.
- models.py: Functions to build CNN models from scratch and using transfer learning.
- requirements.txt: Text file specifying required Python libraries.
## Project link:
https://colab.research.google.com/drive/1NGU36RObYM8puViloJ2Yg4vqgDsmiq20?usp=drive_link
## Further Exploration:

Experiment with different hyperparameters (learning rate, epochs, network architecture).
Explore additional data augmentation techniques.
Try using different pre-trained models (e.g., VGG16, ResNet50).
#### Feel free to contribute to this project by creating pull requests!
