# Faster R-CNN Waste Detection

This project implements an object detection system for waste classification using Faster R-CNN.

## Classes
- glass
- metal
- paper
- plastic
- background

## Model
The model is based on Faster R-CNN implemented with PyTorch.

## Training
The model was trained using 5-fold cross validation.

## Best Fold
Fold 5 achieved the best performance.

Accuracy: 0.9259  
mAP@0.50:0.95: 0.786

## How to use

Clone the repository:

git clone https://github.com/Senkara/fasterrcnn-waste-detection.git

Install dependencies:

pip install torch torchvision

Run the project:

python main.py
