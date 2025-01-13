# Animal Classification with Continuous Attribute Learning (CAL)

This project implements a deep learning model for animal image classification using a novel Continuous Attribute Learning approach. The model combines traditional image classification with attribute prediction to improve overall classification accuracy and model interpretability. It includes both training and analysis components for comprehensive model evaluation and visualization.

## Project Overview

The model architecture consists of:
- A ResNet50 backbone for feature extraction
- An attribute prediction head that learns continuous animal attributes
- A class prediction head that uses predicted attributes for final classification

Key features:
- Label smoothing for better generalization
- Dynamic loss weighting between attribute and classification tasks
- One-cycle learning rate scheduling
- Advanced data augmentation pipeline
- Gradient clipping for stable training

## Requirements

- Python 3.7+
- PyTorch 1.7+
- torchvision
- pandas
- numpy
- Pillow
- scikit-learn
- tqdm
- matplotlib
- seaborn

## Model Architecture

### CALModel
The model uses a two-stage architecture:
1. ResNet50 backbone extracts features from images
2. These features are used to:
   - Predict continuous attributes (85 dimensions)
   - Use predicted attributes to classify images into 50 classes

### Key Components

- **Attribute Predictor**: Multi-layer network that predicts continuous attributes
- **Class Predictor**: Takes predicted attributes and maps them to final classes
- **Label Smoothing**: Implements smooth labels for better generalization
- **Dynamic Loss Weighting**: Balances attribute and classification learning

## Training Configuration

The model uses the following default configuration:
- Batch size: 32
- Learning rate: 0.001
- Number of epochs: 50
- Image size: 224x224
- Label smoothing: 0.1
- Weight decay: 0.01

## Data Augmentation

Training transforms include:
- Random horizontal flips
- Random rotation (±10 degrees)
- Color jittering (brightness, contrast, saturation)
- Normalization using ImageNet statistics

## Usage

### Training
1. Prepare your dataset following the structure above
2. Update the CONFIG dictionary with your desired parameters
3. Run the training

### Analysis
The project includes comprehensive analysis tools to evaluate model performance:

The analysis script generates several visualizations and metrics:
- Confusion matrix for class predictions
- Class-wise F1 scores visualization
- Attribute prediction accuracy analysis
- Performance summaries for both training and validation sets

Analysis outputs are saved in an 'analysis' directory and include:
- Confusion matrix heatmaps
- Class accuracy bar plots
- Attribute prediction accuracy visualizations
- Detailed performance reports including:
  - Overall accuracy metrics
  - Top/bottom performing classes
  - Best/worst predicted attributes

## Training Process

The training pipeline includes:
1. Loading and preprocessing data
2. Initializing model with pretrained ResNet50 backbone
3. Training with dynamic loss weighting
4. Validation after each epoch
5. Model checkpointing (saves best model based on validation accuracy)

## Model Evaluation

### Training Metrics
During training, the model tracks:
- Training loss (attribute and classification)
- Validation loss
- Classification accuracy
- Attribute prediction accuracy

### Analysis Metrics
Post-training analysis provides:
1. Classification Performance
   - Confusion matrix for all classes
   - Per-class F1 scores
   - Detailed analysis of best and worst performing classes

2. Attribute Prediction Analysis
   - Accuracy for each predicted attribute
   - Analysis of most and least accurately predicted attributes
   - Attribute prediction distribution visualization

3. Visual Analysis
   - Heatmap visualizations of confusion matrices
   - Bar plots of class-wise performance
   - Attribute prediction accuracy distributions

## Acknowledgments

This implementation uses PyTorch and the ResNet50 architecture pretrained on ImageNet. The continuous attribute learning approach is inspired by recent advances in attribute-based learning for image classification.
