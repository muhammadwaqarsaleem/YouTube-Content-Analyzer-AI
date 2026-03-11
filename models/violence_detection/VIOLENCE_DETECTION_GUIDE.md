# Violence Detection Model - Complete Implementation Guide

## 🎯 Executive Summary

This project implements a **state-of-the-art violence detection system** using deep learning CNNs with transfer learning. The model classifies video frames as violent or non-violent with high accuracy.

### Key Features
✅ **2-Phase Training**: Feature Extraction → Fine-Tuning  
✅ **Multiple Architectures**: ResNet50, EfficientNet, VGG16, Vision Transformer  
✅ **Enhanced Data Augmentation**: Rotation, flip, brightness, color variation  
✅ **Comprehensive Evaluation**: Accuracy, Precision, Recall, F1, AUC, MCC  
✅ **Production Ready**: Multiple model formats (.h5, SavedModel, .pkl)  
✅ **Deployment Scripts**: Single image, batch, video processing  

---

## 📊 Dataset Analysis

### Violence Frames Dataset
- **Location**: `violence_frames/`
- **Classes**: 
  - `violence/`: ~9,999 images
  - `nonviolence/`: ~9,997 images
- **Total**: ~19,996 images
- **Format**: Video frame extractions (V_XXX_fY.jpg, NV_XXX_fY.jpg)
- **Task**: Binary image classification
- **Balance**: Approximately 50/50 (balanced)

---

## 🤖 Model Architecture Decision

### ❌ BERT-Type Models (NOT Recommended)
**Why Not:**
- BERT is designed for **NLP/text** tasks, NOT images
- Processes tokenized text, not pixel data
- Your task is **visual**, not textual

**When to Use BERT:**
- Analyzing video titles/descriptions for violent language
- Classifying text comments about videos
- Multi-modal would need: CLIP, ViLBERT, LXMERT

### ✅ CNN Transfer Learning (RECOMMENDED)

**Your Current Approach - PERFECT!**

#### Architecture Options:

1. **ResNet50** ⭐⭐⭐⭐⭐ (Best Balance)
   - Pre-trained on ImageNet
   - Residual connections prevent vanishing gradients
   - Fast inference (~30ms/image on GPU)
   - Expected accuracy: 90-95%

2. **EfficientNetB0** ⭐⭐⭐⭐ (Most Efficient)
   - Smaller, faster than ResNet
   - Better accuracy per parameter
   - Mobile-friendly

3. **VGG16** ⭐⭐⭐ (Simplest)
   - Simple architecture
   - Well-understood behavior
   - Larger model size (~500MB)

4. **Vision Transformer (ViT)** ⭐⭐⭐ (Experimental)
   - Transformer-based approach
   - Requires more data (100k+ images optimal)
   - Computationally expensive

### ❌ Traditional ML (SVM/Random Forest) - Not Recommended
- Lower accuracy (70-80%)
- Requires manual feature extraction
- Only use if no GPU available

---

## 🏗️ Installation

### Prerequisites
- Python 3.8+
- GPU recommended (CUDA 11.2+)
- 8GB+ RAM
- 10GB free disk space

### Install Dependencies

```bash
pip install -r requirements.txt
```

**requirements.txt:**
```
tensorflow>=2.8.0
opencv-python>=4.5.0
numpy>=1.19.0
scikit-learn>=1.0.0
matplotlib>=3.3.0
seaborn>=0.11.0
pandas>=1.3.0
transformers>=4.20.0
Pillow>=8.0.0
```

---

## 🚀 Quick Start

### Training the Model

**Basic Training (Recommended):**
```bash
python main_violence_detection.py
```

This will:
1. Load all ~20k images from `violence_frames/`
2. Apply data augmentation
3. Train with 2-phase approach (30 epochs total)
4. Evaluate on test set
5. Save models in multiple formats

**Training Configuration:**
- Epochs: 30 (Phase 1: 15, Phase 2: 15)
- Batch Size: 32
- Image Size: 224x224
- Optimizer: Adam (LR: 0.001 → 0.00005)
- Data Augmentation: Enabled

**Expected Training Time:**
- Phase 1 (frozen): 1-2 hours (GPU)
- Phase 2 (fine-tune): 3-5 hours (GPU)
- Total: 4-7 hours

### Model Selection

Edit `main_violence_detection.py` to change architecture:

```python
model_type = 'resnet'        # Recommended (default)
model_type = 'efficientnet'  # More efficient
model_type = 'vgg'           # Simpler
model_type = 'vit'           # Experimental
model_type = 'custom'        # From scratch
```

---

## 📁 File Structure

```
Project69/
├── violence_frames/              # Dataset
│   ├── violence/                # Violent frames (~10k images)
│   └── nonviolence/             # Non-violent frames (~10k images)
├── src/                         # Source code
│   ├── preprocess.py            # Data preprocessing
│   ├── model.py                 # Model architectures
│   ├── trainer.py               # Training utilities
│   └── evaluator.py             # Evaluation metrics
├── main_violence_detection.py   # Main training script
├── predict_violence.py          # Prediction/deployment
├── convert_model_format.py      # Format conversion
├── requirements.txt             # Dependencies
└── VIOLENCE_DETECTION_GUIDE.md  # This file
```

---

## 🎓 Training Process

### Phase 1: Feature Extraction (Frozen Base)
```python
base_model.trainable = False  # Freeze pre-trained weights
optimizer = Adam(learning_rate=0.001)
epochs = 10-15
```
- Train only classifier head
- Quick convergence
- Preserves pre-trained features

### Phase 2: Fine-Tuning
```python
base_model.trainable = True
for layer in base_model.layers[:80]:
    layer.trainable = False  # Keep early layers frozen
optimizer = Adam(learning_rate=0.00005)
epochs = 10-15
```
- Unfreeze top 80 layers
- Lower learning rate
- Improves accuracy by 2-5%

---

## 📊 Expected Performance

### ResNet50 (Recommended):
- **Training Accuracy**: 95-98%
- **Validation Accuracy**: 92-95%
- **Test Accuracy**: 90-94%
- **Precision**: 0.90-0.95
- **Recall**: 0.88-0.93
- **F1-Score**: 0.89-0.94
- **ROC AUC**: 0.94-0.97
- **MCC**: 0.80-0.88

### Performance by Architecture:

| Architecture | Accuracy | Speed | Model Size | Best For |
|-------------|----------|-------|------------|----------|
| ResNet50 | 90-94% | Fast | ~100MB | General use |
| EfficientNet | 89-93% | Faster | ~50MB | Mobile/Edge |
| VGG16 | 87-91% | Slow | ~500MB | Simplicity |
| Custom CNN | 75-85% | Fastest | ~10MB | Educational |
| ViT | 92-95%* | Slowest | ~350MB | Research (*needs 100k+ images) |

---

## 💾 Model Formats

### Output Files After Training:

1. **Keras H5 Format** (Recommended for most uses)
   ```
   violence_detection_model_resnet.h5
   ```
   - Single file (~100MB)
   - Includes architecture + weights
   - Easy to load and use

2. **TensorFlow SavedModel** (Production deployment)
   ```
   violence_detection_model_resnet_savedmodel/
   ```
   - Directory format
   - TF Serving, TFLite, TF.js compatible
   - Best for production

3. **Pickle Format** (Python-specific)
   ```
   violence_detection_model_resnet.pkl
   ```
   - Python serialization
   - Version-dependent
   - Use with caution

### Converting Between Formats:

```bash
# Convert to all formats
python convert_model_format.py

# Or programmatically:
from convert_model_format import ModelFormatConverter

# H5 to SavedModel
ModelFormatConverter.h5_to_savedmodel(
    'model.h5', 
    'savedmodel_output'
)

# H5 to Pickle
ModelFormatConverter.h5_to_pickle(
    'model.h5', 
    'model.pkl'
)
```

---

## 🔮 Deployment & Inference

### Method 1: Single Image Prediction

```python
from predict_violence import ViolenceDetectionPredictor

# Initialize predictor
predictor = ViolenceDetectionPredictor('violence_detection_model_resnet.h5')

# Predict single image
result = predictor.predict_single('test_image.jpg')

print(f"Label: {result['label']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Violence Probability: {result['probability_violence']*100:.2f}%")
```

**Output:**
```
Label: Violence
Confidence: 94.23%
Violence Probability: 94.23%
```

### Method 2: Batch Prediction

```python
# Predict entire directory
results = predictor.predict_from_directory(
    'path/to/images/',
    threshold=0.5
)

print(f"Total: {results['total_images']}")
print(f"Violent: {results['violent_images']}")
print(f"Non-Violent: {results['non_violent_images']}")
```

### Method 3: Video Processing

```python
# Analyze video
results = predictor.predict_video(
    'input_video.mp4',
    output_path='output_annotated.mp4',
    threshold=0.5,
    frame_interval=5  # Process every 5th frame
)

print(f"Overall: {results['overall_assessment']}")
print(f"Violent Frames: {results['violent_frames_count']}")
print(f"Violence %: {results['violence_percentage']:.2f}%")
```

### Method 4: Real-Time Detection

```python
import cv2
import numpy as np

cap = cv2.VideoCapture(0)  # Webcam
predictor = ViolenceDetectionPredictor('model.h5')

while True:
    ret, frame = cap.read()
    
    # Preprocess
    img_resized = cv2.resize(frame, (224, 224))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_normalized = img_rgb.astype('float32') / 255.0
    img_batch = np.expand_dims(img_normalized, axis=0)
    
    # Predict
    prediction = predictor.model.predict(img_batch, verbose=0)[0][0]
    
    # Annotate
    if prediction > 0.5:
        label = f"VIOLENCE: {prediction:.2f}"
        color = (0, 0, 255)  # Red
    else:
        label = f"SAFE: {1-prediction:.2f}"
        color = (0, 255, 0)  # Green
    
    cv2.putText(frame, label, (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
    cv2.imshow('Violence Detection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

---

## 📈 Advanced Configuration

### Adjusting Classification Threshold

Default threshold is 0.5, but you can optimize for different metrics:

```python
# Higher threshold = fewer false positives (more precise)
result = predictor.predict_single('image.jpg', threshold=0.7)

# Lower threshold = fewer false negatives (higher recall)
result = predictor.predict_single('image.jpg', threshold=0.3)
```

**Threshold Selection Guide:**
- **0.5** (default): Balanced precision/recall
- **0.3-0.4**: Maximize recall (catch all violence, accept some false positives)
- **0.6-0.7**: Maximize precision (minimize false alarms)

### Class Weights for Imbalanced Data

If your dataset becomes imbalanced:

```python
# Automatically calculated in main_violence_detection.py
class_weights = preprocessor.get_class_weights(y_train)

# Manual override
class_weights = {0: 1.0, 1: 2.0}  # Weight violence class 2x more
```

### Custom Data Augmentation

Modify `src/preprocess.py`:

```python
self.datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,      # Critical for violence
    zoom_range=0.15,
    shear_range=0.1,
    brightness_range=[0.8, 1.2],  # Lighting variation
    channel_shift_range=10       # Color variation
)
```

---

## 🔍 Model Evaluation

### Comprehensive Metrics

The evaluator provides:

1. **Primary Metrics:**
   - Accuracy: Overall correctness
   - Precision: True positives / (true + false positives)
   - Recall: True positives / (true + false negatives)
   - F1-Score: Harmonic mean of precision/recall
   - ROC AUC: Area under ROC curve

2. **Advanced Metrics:**
   - Average Precision (AP)
   - Matthews Correlation Coefficient (MCC)
   - Sensitivity (TPR)
   - Specificity (TNR)

3. **Visualizations:**
   - Confusion Matrix
   - ROC Curve
   - Precision-Recall Curve
   - Prediction Distribution

### Running Evaluation

```python
from src.evaluator import ViolenceDetectionEvaluator

evaluator = ViolenceDetectionEvaluator(model)
results = evaluator.generate_evaluation_report(X_test, y_test)
```

**Sample Output:**
```
======================================================================
VIOLENCE DETECTION MODEL - COMPREHENSIVE EVALUATION REPORT
======================================================================

📊 PRIMARY METRICS:
----------------------------------------------------------------------
  Test Accuracy:      0.9234 (92.34%)
  Test Precision:     0.9312
  Test Recall:        0.9156
  Test AUC:           0.9456
  F1-Score:           0.9233
  ROC AUC:            0.9512
  Average Precision:  0.9423
  MCC:                0.8467

📈 PER-CLASS METRICS:
----------------------------------------------------------------------
  Sensitivity (TPR):  0.9156
  Specificity (TNR):  0.9312

🔍 CONFUSION MATRIX:
----------------------------------------------------------------------
  True Negatives:  1387
  False Positives: 98
  False Negatives: 126
  True Positives:  1389
```

---

## 🎯 Error Analysis

### Understanding Misclassifications

After evaluation, analyze errors:

```python
import numpy as np

# Get predictions
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int).flatten()

# Find false positives (non-violent predicted as violent)
false_positives = np.where((y_pred == 1) & (y_test == 0))[0]

# Find false negatives (violent predicted as non-violent)
false_negatives = np.where((y_pred == 0) & (y_test == 1))[0]

print(f"False Positives: {len(false_positives)}")
print(f"False Negatives: {len(false_negatives)}")
```

### Common Failure Modes

1. **False Positives** (safe → violent):
   - Fast motion blur
   - Certain sports (wrestling, boxing)
   - Crowd scenes
   - Dark/low-light conditions

2. **False Negatives** (violent → safe):
   - Stylized/cartoon violence
   - Blood without action
   - Psychological violence (no physical contact)
   - Occluded violence (partially visible)

### Improving Performance

1. **Add more training data** for problematic cases
2. **Adjust threshold** based on use case
3. **Use ensemble methods** (multiple models)
4. **Apply temporal smoothing** for videos (use surrounding frames)
5. **Fine-tune longer** or with different learning rates

---

## 🚀 Production Deployment

### Web API (Flask)

```python
from flask import Flask, request, jsonify
from predict_violence import ViolenceDetectionPredictor
import cv2
import numpy as np

app = Flask(__name__)
predictor = ViolenceDetectionPredictor('model.h5')

@app.route('/predict', methods=['POST'])
def predict():
    # Get image from request
    file = request.files['image']
    
    # Read image
    npimg = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    
    # Predict
    result = predictor.predict_single(img)
    
    return jsonify({
        'is_violent': result['is_violent'],
        'confidence': result['confidence'],
        'label': result['label']
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

**Usage:**
```bash
curl -X POST -F "image=@test.jpg" http://localhost:5000/predict
```

### Docker Deployment

```dockerfile
FROM tensorflow/tensorflow:2.8.0-gpu

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 5000

CMD ["python", "app.py"]
```

### Mobile Deployment (TFLite)

```python
# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# Save
with open('violence_detection.tflite', 'wb') as f:
    f.write(tflite_model)

# Use in Android/iOS app
```

---

## 🧪 Experimentation

### Trying Different Architectures

```bash
# Edit main_violence_detection.py
model_type = 'efficientnet'  # Change architecture

# Run training
python main_violence_detection.py
```

### Comparing Models

```python
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

models = {
    'ResNet50': 'violence_detection_model_resnet.h5',
    'EfficientNet': 'violence_detection_model_efficientnet.h5',
    'VGG16': 'violence_detection_model_vgg.h5'
}

plt.figure(figsize=(10, 8))

for name, path in models.items():
    model = tf.keras.models.load_model(path)
    y_pred = model.predict(X_test).flatten()
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)
    
    plt.plot(fpr, tpr, label=f'{name} (AUC={roc_auc:.4f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend()
plt.show()
```

---

## 📝 Troubleshooting

### Common Issues

**1. Out of Memory (OOM) Error**
```python
# Reduce batch size
batch_size = 16  # Instead of 32

# Or limit dataset
preprocessor.prepare_datasets(data_dir, limit=5000)
```

**2. Slow Training**
- Use GPU (CUDA-enabled TensorFlow)
- Reduce image size temporarily: `(128, 128)` instead of `(224, 224)`
- Use mixed precision training

**3. Overfitting**
```python
# Increase dropout
layers.Dropout(0.6)  # Instead of 0.5

# Add more augmentation
rotation_range=20
zoom_range=0.2

# Use early stopping
EarlyStopping(patience=5)  # Instead of 10
```

**4. Underfitting**
```python
# Train longer
epochs = 50  # Instead of 30

# Unfreeze more layers in fine-tuning
fine_tune_layers=100  # Instead of 80

# Higher learning rate
Adam(learning_rate=0.0001)  # Instead of 0.00005
```

---

## 📚 References & Further Reading

### Papers
1. **ResNet**: Deep Residual Learning for Image Recognition (He et al., 2015)
2. **EfficientNet**: Rethinking Model Scaling for Convolutional Neural Networks (Tan & Le, 2019)
3. **Vision Transformers**: An Image is Worth 16x16 Words (Dosovitskiy et al., 2020)

### Resources
- [TensorFlow Keras Documentation](https://www.tensorflow.org/api_docs/python/tf/keras)
- [Transfer Learning Guide](https://www.tensorflow.org/tutorials/images/transfer_learning)
- [OpenCV Documentation](https://docs.opencv.org/)

---

## 🎓 Conclusion

### Key Takeaways

✅ **CNN Transfer Learning** is the right choice (not BERT)  
✅ **ResNet50** offers best balance of accuracy and speed  
✅ **2-Phase Training** (frozen → fine-tuned) improves accuracy  
✅ **Enhanced augmentation** prevents overfitting  
✅ **Multiple model formats** enable flexible deployment  
✅ **Comprehensive evaluation** ensures reliability  

### Next Steps

1. **Train the model** using `main_violence_detection.py`
2. **Evaluate performance** on your specific use case
3. **Deploy** using the prediction scripts
4. **Monitor** and retrain with new data as needed

### Future Enhancements

- Temporal analysis (LSTM on frame sequences)
- Multi-modal (combine with audio/text analysis)
- Real-time streaming implementation
- Active learning pipeline for continuous improvement

---

## 📞 Support

For issues or questions:
1. Check this guide first
2. Review error messages carefully
3. Inspect training logs and plots
4. Verify data integrity

**Good luck with your violence detection project! 🚀**
