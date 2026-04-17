# Violence Detection Model

This project implements a deep learning model for detecting violence in video frames using Convolutional Neural Networks (CNNs).

## Dataset

The model uses the `violence_frames` dataset which contains:
- `violence` directory: ~9,999 images of violent content
- `nonviolence` directory: ~9,997 images of non-violent content

The images are named in the format:
- Violence: `V_XXX_fY.jpg` (e.g., `V_1000_f0.jpg`)
- Non-violence: `NV_XXX_fY.jpg` (e.g., `NV_1000_f0.jpg`)

## Model Architecture

The model uses transfer learning with pre-trained CNN architectures:
- **ResNet50** (default): Good balance of performance and speed
- **VGG16**: Reliable and well-understood architecture
- **EfficientNetB0**: Efficient with good performance

Alternatively, a custom CNN architecture is available for training from scratch.

## File Structure

```
violence_detection/
├── violence_frames/              # Dataset directory
│   ├── violence/                # Violent frame images
│   └── nonviolence/             # Non-violent frame images
├── src/                         # Source code
│   ├── preprocess.py            # Data preprocessing utilities
│   ├── model.py                 # Model architecture definitions
│   ├── trainer.py               # Training utilities and workflow
│   └── evaluator.py             # Model evaluation utilities
├── main_violence_detection.py   # Main execution script
├── violence_detection_model_plan.md  # Implementation plan
├── requirements.txt             # Dependencies
└── README.md                   # This file
```

## Usage

### Installation

```bash
pip install -r requirements.txt
```

### Training

Run the main script to train the violence detection model:

```bash
python main_violence_detection.py
```

This will:
1. Load and preprocess the dataset
2. Build the model architecture (ResNet50 by default)
3. Train the model with initial frozen base
4. Fine-tune the model
5. Evaluate the model on the test set
6. Save the trained model as `violence_detection_model_resnet.h5`
7. Generate evaluation plots

### Model Types

You can specify different model architectures by modifying the `model_type` parameter in `main_violence_detection.py`:
- `'resnet'` for ResNet50 (default)
- `'vgg'` for VGG16
- `'efficientnet'` for EfficientNetB0
- `'custom'` for a custom CNN from scratch

## Model Details

### Transfer Learning Approach
1. Use a pre-trained base model (ResNet50, VGG16, or EfficientNetB0)
2. Freeze the base model initially and train only the classifier head
3. Fine-tune by unfreezing part of the base model with a lower learning rate

### Architecture
- Input: 224x224x3 RGB images
- Base model: Pre-trained CNN without top classification layer
- Global Average Pooling
- Dense layers with batch normalization and dropout for regularization
- Output: Single sigmoid neuron for binary classification

### Training Process
- Loss function: Binary cross-entropy
- Optimizer: Adam with learning rate scheduling
- Data augmentation: Rotation, shifting, flipping, zooming
- Callbacks: Early stopping, learning rate reduction, model checkpointing

## Evaluation Metrics

The model is evaluated using:
- Accuracy
- Precision
- Recall
- F1-score
- ROC AUC score
- Confusion matrix
- Classification report

## Important Notes

- **BERT models**: Are NOT appropriate for this task as they are designed for Natural Language Processing, not image classification
- **PKL serialization**: Models can be saved in various formats including pickle if needed, though Keras native format (.h5) is recommended
- The model is designed for binary classification of video frames as violent or non-violent
- For production use, consider additional validation and bias testing

## Results

After training, the model will generate:
- Training history plots
- Evaluation metrics
- Confusion matrix visualization
- ROC curve
- Prediction distribution plots
- Saved model file

The trained model can be integrated into video analysis pipelines for real-time violence detection.