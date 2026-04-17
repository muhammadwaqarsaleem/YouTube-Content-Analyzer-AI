"""
Fine-tune Violence Detection Model
Using labeled data from violence_frames/ directory
"""

import os
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from sklearn.model_selection import train_test_split
import cv2
from pathlib import Path
import time

class ViolenceModelFinetuner:
    """Fine-tune violence detection model with labeled data"""
    
    def __init__(self, base_model_path, data_dir='violence_frames'):
        self.data_dir = Path(data_dir)
        self.violence_dir = self.data_dir / 'violence'
        self.nonviolence_dir = self.data_dir / 'nonviolence'
        self.base_model_path = base_model_path
        
        # Image settings
        self.img_size = (224, 224)
        
        # Training settings
        self.batch_size = 32
        self.epochs = 15
        
        print("="*80)
        print("VIOLENCE MODEL FINE-TUNING")
        print("="*80)
        print(f"\n📂 Data Directory: {self.data_dir}")
        print(f"  Violence samples: {self._count_images(self.violence_dir)}")
        print(f"  Non-violence samples: {self._count_images(self.nonviolence_dir)}")
        print(f"\n🎯 Base Model: {self.base_model_path}")
        print(f"⚙️  Settings: {self.epochs} epochs, batch size {self.batch_size}")
        print("="*80)
    
    def _count_images(self, directory):
        """Count images in directory"""
        if not directory.exists():
            return 0
        return len(list(directory.glob('*.jpg')) + list(directory.glob('*.png')))
    
    def load_and_preprocess_data(self):
        """Load and preprocess all images"""
        print("\n[1/4] Loading and preprocessing data...")
        
        X = []
        y = []
        
        # Load violence frames (label = 1)
        print("  Loading violence frames...")
        violence_images = list(self.violence_dir.glob('*.jpg')) + list(self.violence_dir.glob('*.png'))
        
        for i, img_path in enumerate(violence_images):
            try:
                img = cv2.imread(str(img_path))
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, self.img_size)
                    img = img.astype('float32') / 255.0  # Normalize
                    X.append(img)
                    y.append(1)  # Violence label
                    
                    if (i + 1) % 1000 == 0:
                        print(f"    Loaded {i+1}/{len(violence_images)} violence images")
            except Exception as e:
                print(f"    ⚠️  Error loading {img_path}: {e}")
        
        # Load non-violence frames (label = 0)
        print("  Loading non-violence frames...")
        nonviolence_images = list(self.nonviolence_dir.glob('*.jpg')) + list(self.nonviolence_dir.glob('*.png'))
        
        for i, img_path in enumerate(nonviolence_images):
            try:
                img = cv2.imread(str(img_path))
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, self.img_size)
                    img = img.astype('float32') / 255.0  # Normalize
                    X.append(img)
                    y.append(0)  # Non-violence label
                    
                    if (i + 1) % 1000 == 0:
                        print(f"    Loaded {i+1}/{len(nonviolence_images)} non-violence images")
            except Exception as e:
                print(f"    ⚠️  Error loading {img_path}: {e}")
        
        # Convert to numpy arrays
        X = np.array(X)
        y = np.array(y)
        
        print(f"\n  ✓ Total samples loaded: {len(X)}")
        print(f"    - Violence: {np.sum(y == 1)}")
        print(f"    - Non-violence: {np.sum(y == 0)}")
        print(f"  ✓ Data shape: {X.shape}")
        
        return X, y
    
    def build_fine_tuned_model(self):
        """Build fine-tuned model based on existing architecture"""
        print("\n[2/4] Building fine-tuned model...")
        
        # Load base model without top layers
        base_model = keras.models.load_model(self.base_model_path)
        
        # Freeze early layers, fine-tune later layers
        for layer in base_model.layers[:-50]:
            layer.trainable = False
        
        # Add new classification head
        x = base_model.output
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.5)(x)  # Increased dropout for regularization
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        predictions = layers.Dense(1, activation='sigmoid')(x)
        
        fine_tuned_model = models.Model(inputs=base_model.input, outputs=predictions)
        
        # Compile with lower learning rate for fine-tuning
        fine_tuned_model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0001),
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
        )
        
        print(f"  ✓ Model architecture created")
        print(f"  ✓ Trainable layers: {sum(1 for l in fine_tuned_model.layers if l.trainable)}/{len(fine_tuned_model.layers)}")
        print(f"  ✓ Total parameters: {fine_tuned_model.count_params():,}")
        
        return fine_tuned_model
    
    def train(self, X, y, model):
        """Train the fine-tuned model"""
        print("\n[3/4] Training model...")
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"  Training set: {len(X_train)} samples")
        print(f"  Validation set: {len(X_val)} samples")
        
        # Callbacks
        checkpoint_dir = Path('models/violence_detection/fine_tuned')
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        best_callback = callbacks.ModelCheckpoint(
            str(checkpoint_dir / 'best_model.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        )
        
        early_stop = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        )
        
        # Train
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=[best_callback, early_stop, reduce_lr],
            verbose=1
        )
        
        print("\n  ✓ Training complete!")
        
        # Evaluate on validation set
        val_loss, val_acc, val_prec, val_rec = model.evaluate(X_val, y_val, verbose=0)
        
        print(f"\n  📊 Validation Results:")
        print(f"    Accuracy: {val_acc*100:.2f}%")
        print(f"    Precision: {val_prec*100:.2f}%")
        print(f"    Recall: {val_rec*100:.2f}%")
        print(f"    Loss: {val_loss:.4f}")
        
        return history, model
    
    def evaluate_and_save(self, model, X, y):
        """Evaluate model and save final version"""
        print("\n[4/4] Evaluating and saving model...")
        
        # Test on full dataset
        loss, acc, prec, rec = model.evaluate(X, y, verbose=0)
        
        print(f"\n📈 Final Evaluation (Full Dataset):")
        print(f"  Accuracy: {acc*100:.2f}%")
        print(f"  Precision: {prec*100:.2f}%")
        print(f"  Recall: {rec*100:.2f}%")
        
        # Save fine-tuned model
        output_path = 'models/violence_detection/fine_tuned/final_model.h5'
        model.save(output_path)
        print(f"\n💾 Model saved to: {output_path}")
        
        # Save training metadata
        metadata = {
            'training_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'dataset': {
                'total_samples': len(X),
                'violence_samples': int(np.sum(y == 1)),
                'nonviolence_samples': int(np.sum(y == 0))
            },
            'performance': {
                'accuracy': f"{acc*100:.2f}%",
                'precision': f"{prec*100:.2f}%",
                'recall': f"{rec*100:.2f}%"
            },
            'settings': {
                'epochs': self.epochs,
                'batch_size': self.batch_size,
                'img_size': self.img_size
            }
        }
        
        import json
        metadata_path = 'models/violence_detection/fine_tuned/training_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"📝 Metadata saved to: {metadata_path}")
        
        print("\n" + "="*80)
        print("✅ FINE-TUNING COMPLETE!")
        print("="*80)
        print(f"\nExpected Improvements:")
        print(f"  ✅ Reduced false positives on nature documentaries")
        print(f"  ✅ Better discrimination between real violence and natural phenomena")
        print(f"  ✅ Maintained high accuracy on actual violent content")
        print("="*80)


def main():
    """Main fine-tuning function"""
    
    # Initialize finetuner
    finetuner = ViolenceModelFinetuner(
        base_model_path='models/violence_detection/violence_detection_model_resnet.h5',
        data_dir='violence_frames'
    )
    
    # Load data
    X, y = finetuner.load_and_preprocess_data()
    
    # Build model
    model = finetuner.build_fine_tuned_model()
    
    # Train
    history, model = finetuner.train(X, y, model)
    
    # Evaluate and save
    finetuner.evaluate_and_save(model, X, y)
    
    print("\n🎉 Fine-tuning completed successfully!")
    print("\nNext steps:")
    print("1. Update services/violence_service.py to use the new model")
    print("2. Run false positive test again to verify improvements")
    print("3. Deploy to production if results are satisfactory")


if __name__ == "__main__":
    main()
