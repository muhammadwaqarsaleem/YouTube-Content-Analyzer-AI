import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from src.model import ViolenceDetectionModel, CustomCNNModel, get_training_callbacks
from src.preprocess import ViolenceDataPreprocessor


class ViolenceDetectionTrainer:
    """
    Trainer class for violence detection model
    """
    
    def __init__(self, model_type='resnet', input_shape=(224, 224, 3)):
        self.model_type = model_type
        self.input_shape = input_shape
        self.model = None
        self.history = None
        
    def build_model(self):
        """
        Build the violence detection model
        """
        if self.model_type == 'custom':
            self.model = CustomCNNModel(input_shape=self.input_shape, num_classes=1)
        else:
            self.model = ViolenceDetectionModel(
                input_shape=self.input_shape, 
                num_classes=1, 
                model_type=self.model_type
            )
        return self.model.get_compiled_model()
    
    def train_initial_model(self, train_data, val_data, epochs=20, batch_size=32, class_weights=None):
        """
        Train the model with frozen base (initial training) - Phase 1
        Args:
            train_data: Tuple of (X_train, y_train)
            val_data: Tuple of (X_val, y_val)
            epochs: Number of training epochs
            batch_size: Batch size
            class_weights: Optional class weights for imbalanced data
        """
        (X_train, y_train) = train_data
        (X_val, y_val) = val_data
        
        model = self.build_model()
        
        print("="*60)
        print("PHASE 1: Training with Frozen Base (Feature Extraction)")
        print("="*60)
        print(f"Epochs: {epochs}, Batch Size: {batch_size}")
        if class_weights:
            print(f"Using class weights: {class_weights}")
        
        self.history = model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_val, y_val),
            callbacks=get_training_callbacks(),
            class_weight=class_weights,
            verbose=1
        )
        
        # Evaluate after Phase 1
        val_results = model.evaluate(X_val, y_val, verbose=0)
        val_loss, val_acc, val_prec, val_rec, val_auc = val_results
        print(f"\nPhase 1 Validation Accuracy: {val_acc:.4f}")
        print(f"Phase 1 Validation Precision: {val_prec:.4f}")
        print(f"Phase 1 Validation Recall: {val_rec:.4f}")
        print(f"Phase 1 Validation AUC: {val_auc:.4f}")
        
        return model
    
    def fine_tune_model(self, model, train_data, val_data, fine_tune_epochs=10, batch_size=32, fine_tune_layers=50):
        """
        Fine-tune the model by unfreezing top layers of the base model - Phase 2
        Args:
            model: Trained model from Phase 1
            train_data: Tuple of (X_train, y_train)
            val_data: Tuple of (X_val, y_val)
            fine_tune_epochs: Number of fine-tuning epochs
            batch_size: Batch size
            fine_tune_layers: Number of layers to unfreeze from the end
        """
        (X_train, y_train) = train_data
        (X_val, y_val) = val_data
        
        # Unfreeze the top layers of the base model for fine-tuning
        if hasattr(model, 'unfreeze_base_model'):
            model.unfreeze_base_model(fine_tune_at=fine_tune_layers)
            print(f"Unfroze top {fine_tune_layers} layers for fine-tuning")
        
        # Use a much lower learning rate for fine-tuning
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.00005),  # Reduced LR
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall', tf.keras.metrics.AUC(name='auc')]
        )
        
        print("\n" + "="*60)
        print("PHASE 2: Fine-Tuning Top Layers")
        print("="*60)
        print(f"Epochs: {fine_tune_epochs}, Batch Size: {batch_size}")
        print(f"Learning Rate: 0.00005 (reduced for fine-tuning)")
        
        # Create specific callbacks for fine-tuning
        fine_tune_callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=8,
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-7,
                verbose=1
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath='best_fine_tuned_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            )
        ]
        
        fine_tune_history = model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=fine_tune_epochs,
            validation_data=(X_val, y_val),
            callbacks=fine_tune_callbacks,
            verbose=1
        )
        
        # Evaluate after Phase 2
        val_results = model.evaluate(X_val, y_val, verbose=0)
        val_loss, val_acc, val_prec, val_rec, val_auc = val_results
        print(f"\nPhase 2 Validation Accuracy: {val_acc:.4f}")
        print(f"Phase 2 Validation Precision: {val_prec:.4f}")
        print(f"Phase 2 Validation Recall: {val_rec:.4f}")
        print(f"Phase 2 Validation AUC: {val_auc:.4f}")
        
        return model, fine_tune_history
    
    def train_full_model(self, train_data, val_data, epochs=30, batch_size=32, fine_tune=True, class_weights=None):
        """
        Complete training process with Phase 1 (frozen) and Phase 2 (fine-tuning)
        Args:
            train_data: Tuple of (X_train, y_train)
            val_data: Tuple of (X_val, y_val)
            epochs: Total epochs (split between phases)
            batch_size: Batch size
            fine_tune: Whether to do Phase 2 fine-tuning
            class_weights: Optional class weights
        """
        # Phase 1: Initial training with frozen base
        phase1_epochs = max(10, epochs // 2)
        model = self.train_initial_model(
            train_data, val_data, 
            epochs=phase1_epochs, 
            batch_size=batch_size,
            class_weights=class_weights
        )
        
        if fine_tune:
            # Phase 2: Fine-tuning
            phase2_epochs = epochs - phase1_epochs
            model, fine_tune_history = self.fine_tune_model(
                model, train_data, val_data, 
                fine_tune_epochs=phase2_epochs, 
                batch_size=batch_size,
                fine_tune_layers=80  # Unfreeze top 80 layers
            )
            
            # Combine histories if needed
            if self.history and fine_tune_history:
                for key in fine_tune_history.history:
                    if key in self.history.history:
                        self.history.history[key] += fine_tune_history.history[key]
                    else:
                        self.history.history[key] = fine_tune_history.history[key]
        
        return model
    
    def plot_training_history(self):
        """
        Plot training history
        """
        if self.history is None:
            print("No training history available.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot accuracy
        axes[0, 0].plot(self.history.history['accuracy'], label='Training Accuracy')
        axes[0, 0].plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        
        # Plot loss
        axes[0, 1].plot(self.history.history['loss'], label='Training Loss')
        axes[0, 1].plot(self.history.history['val_loss'], label='Validation Loss')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        
        # Plot precision
        if 'precision' in self.history.history:
            axes[1, 0].plot(self.history.history['precision'], label='Training Precision')
            axes[1, 0].plot(self.history.history['val_precision'], label='Validation Precision')
            axes[1, 0].set_title('Model Precision')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Precision')
            axes[1, 0].legend()
        
        # Plot recall
        if 'recall' in self.history.history:
            axes[1, 1].plot(self.history.history['recall'], label='Training Recall')
            axes[1, 1].plot(self.history.history['val_recall'], label='Validation Recall')
            axes[1, 1].set_title('Model Recall')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Recall')
            axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig('training_history.png')
        plt.show()


def train_violence_detection_model(data_dir, model_type='resnet', epochs=30, batch_size=32):
    """
    Complete workflow to train the violence detection model
    """
    # Initialize preprocessor
    preprocessor = ViolenceDataPreprocessor()
    
    # Load and prepare data
    train_data, val_data, test_data = preprocessor.prepare_datasets(data_dir)
    
    # Initialize trainer
    trainer = ViolenceDetectionTrainer(model_type=model_type)
    
    # Train the model
    model = trainer.train_full_model(train_data, val_data, epochs=epochs, batch_size=batch_size)
    
    # Evaluate on test set
    X_test, y_test = test_data
    test_loss, test_acc, test_precision, test_recall = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test Recall: {test_recall:.4f}")
    
    # Plot training history
    trainer.plot_training_history()
    
    # Save the model
    model.save(f'violence_detection_model_{model_type}.h5')
    print(f"Model saved as violence_detection_model_{model_type}.h5")
    
    return model, trainer