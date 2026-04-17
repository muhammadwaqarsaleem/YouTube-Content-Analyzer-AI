"""
Main script for violence detection model training and evaluation
Implements 2-phase training: Feature Extraction + Fine-Tuning
"""

from src.preprocess import ViolenceDataPreprocessor
from src.model import ViolenceDetectionModel, CustomCNNModel
from src.trainer import ViolenceDetectionTrainer, train_violence_detection_model
from src.evaluator import ViolenceDetectionEvaluator


def main():
    """
    Main function to run the violence detection model training and evaluation
    """
    print("\n" + "="*70)
    print("VIOLENCE DETECTION MODEL - TRAINING PIPELINE")
    print("="*70 + "\n")
    
    # Define dataset path
    data_dir = "violence_frames"
    
    # Model selection
    print("📋 MODEL SELECTION:")
    print("-" * 70)
    print("Available architectures:")
    print("  1. 'resnet'        - ResNet50 (Recommended - Best balance)")
    print("  2. 'efficientnet'  - EfficientNetB0 (Most efficient)")
    print("  3. 'vgg'           - VGG16 (Simplest)")
    print("  4. 'vit'           - Vision Transformer (Experimental)")
    print("  5. 'custom'        - Custom CNN from scratch")
    print()
    
    # Specify model type: 'resnet', 'vgg', 'efficientnet', 'vit', or 'custom'
    model_type = 'resnet'  # Recommended for good balance of performance and speed
    
    print(f"✅ Selected Model Architecture: {model_type.upper()}")
    print()
    
    # Training configuration
    print("⚙️  TRAINING CONFIGURATION:")
    print("-" * 70)
    epochs = 30  # Total epochs (split between Phase 1 and Phase 2)
    batch_size = 32
    print(f"  Epochs: {epochs} (Phase 1: ~15, Phase 2: ~15)")
    print(f"  Batch Size: {batch_size}")
    print(f"  Image Size: 224x224")
    print(f"  Data Augmentation: Enabled (rotation, flip, brightness, etc.)")
    print(f"  2-Phase Training: Enabled (Frozen Base → Fine-Tuning)")
    print()
    
    # Initialize preprocessor with optional limit for testing
    preprocessor = ViolenceDataPreprocessor(img_size=(224, 224))
    
    # Load datasets (set limit=None to use all images)
    print("📂 LOADING DATASET:")
    print("-" * 70)
    print(f"Dataset directory: {data_dir}")
    print("Note: To limit dataset for testing, use: prepare_datasets(data_dir, limit=1000)")
    print("⚠️  For systems with <16GB RAM, use limit=5000 or less")
    print()
    
    # Use limit=5000 for systems with limited RAM
    train_data, val_data, test_data = preprocessor.prepare_datasets(data_dir, limit=5000)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = train_data, val_data, test_data
    
    # Calculate class weights if needed
    class_weights = preprocessor.get_class_weights(y_train)
    print(f"Class Weights: {class_weights}")
    print()
    
    # Initialize trainer
    trainer = ViolenceDetectionTrainer(model_type=model_type)
    
    # Train the model with 2-phase approach
    print("🚀 STARTING TRAINING...")
    print("="*70 + "\n")
    
    model = trainer.train_full_model(
        train_data, val_data, 
        epochs=epochs, 
        batch_size=batch_size,
        fine_tune=True,  # Enable Phase 2 fine-tuning
        class_weights=class_weights
    )
    
    print("\n" + "="*70)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*70 + "\n")
    
    # Evaluate on test set
    print("📊 EVALUATING ON TEST SET...")
    print("="*70 + "\n")
    
    evaluator = ViolenceDetectionEvaluator(model)
    evaluation_results = evaluator.generate_evaluation_report(
        X_test, y_test, 
        threshold=0.5,
        save_plots=True
    )
    
    # Save the model in multiple formats
    print("\n💾 SAVING MODEL...")
    print("-" * 70)
    
    # Save as .h5 (Keras format)
    h5_path = f'violence_detection_model_{model_type}.h5'
    model.save(h5_path)
    print(f"  ✓ Saved Keras model: {h5_path}")
    
    # Save as SavedModel format (for production)
    savedmodel_path = f'violence_detection_model_{model_type}_savedmodel'
    model.save(savedmodel_path)
    print(f"  ✓ Saved TensorFlow SavedModel: {savedmodel_path}")
    
    print("\n📝 TRAINING SUMMARY:")
    print("="*70)
    print(f"  Model Architecture: {model_type.upper()}")
    print(f"  Training Samples: {len(X_train)}")
    print(f"  Validation Samples: {len(X_val)}")
    print(f"  Test Samples: {len(X_test)}")
    print(f"  Final Test Accuracy: {evaluation_results['test_accuracy']*100:.2f}%")
    print(f"  Final Test Precision: {evaluation_results['test_precision']:.4f}")
    print(f"  Final Test Recall: {evaluation_results['test_recall']:.4f}")
    print(f"  Final F1-Score: {evaluation_results['f1_score']:.4f}")
    print(f"  ROC AUC: {evaluation_results['roc_auc']:.4f}")
    print("="*70)
    print("\n✅ Model training and evaluation completed successfully!")
    print("   Model files ready for deployment.\n")
    

if __name__ == "__main__":
    main()