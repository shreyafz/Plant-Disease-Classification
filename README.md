# Plant Disease Classification using Computer Vision

## Description:

This repository implements Convolutional Neural Networks (CNNs) for a computer vision task: classifying citrus leaf diseases. It differentiates between Black spot, Canker, Greening, and Healthy leaves. The project explores training a CNN from scratch and utilizing transfer learning with pre-trained models like EfficientNetB0 and VGG16.
- This project demonstrates the application of computer vision techniques, specifically deep learning with CNNs, to an agricultural problem.
- By analyzing image data of citrus leaves, the model can learn to identify visual patterns associated with different diseases.

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
