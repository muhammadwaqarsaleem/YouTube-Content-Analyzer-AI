import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50, VGG16, EfficientNetB0
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint


class ViolenceDetectionModel:
    """
    Violence Detection CNN Model using transfer learning
    Supports: ResNet50, VGG16, EfficientNetB0, and Vision Transformer (ViT)
    """
    
    def __init__(self, input_shape=(224, 224, 3), num_classes=1, model_type='resnet'):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model_type = model_type
        self.model = None
        self._build_model()
    
    def _build_model(self):
        """
        Build the CNN model using transfer learning
        """
        if self.model_type == 'resnet':
            base_model = ResNet50(
                weights='imagenet',
                include_top=False,
                input_shape=self.input_shape
            )
        elif self.model_type == 'vgg':
            base_model = VGG16(
                weights='imagenet',
                include_top=False,
                input_shape=self.input_shape
            )
        elif self.model_type == 'efficientnet':
            base_model = EfficientNetB0(
                weights='imagenet',
                include_top=False,
                input_shape=self.input_shape
            )
        elif self.model_type == 'vit':
            # Vision Transformer option for experimentation
            try:
                from transformers import TFAutoModelForImageClassification
                base_model = TFAutoModelForImageClassification.from_pretrained(
                    "google/vit-base-patch16-224",
                    ignore_mismatched_sizes=True
                )
                # ViT doesn't use typical pooling, so we'll use its pooled output
                self._build_vit_model(base_model)
                return
            except ImportError:
                print("Warning: transformers library not installed. Falling back to ResNet50.")
                base_model = ResNet50(
                    weights='imagenet',
                    include_top=False,
                    input_shape=self.input_shape
                )
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        # Freeze the base model initially
        base_model.trainable = False
        
        # Add custom classifier on top
        self.model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(self.num_classes, activation='sigmoid' if self.num_classes == 1 else 'softmax')
        ])
        
        # Compile the model
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy' if self.num_classes == 1 else 'categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall', tf.keras.metrics.AUC(name='auc')]
        )
    
    def _build_vit_model(self, vit_base):
        """
        Special builder for Vision Transformer
        """
        # ViT already has its own head, but we can add custom layers
        inputs = layers.Input(shape=self.input_shape)
        
        # Get ViT features
        vit_outputs = vit_base(inputs)
        pooled_output = vit_outputs.pooler_output
        
        # Add custom classification head
        x = layers.Dense(128, activation='relu')(pooled_output)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(self.num_classes, activation='sigmoid' if self.num_classes == 1 else 'softmax')(x)
        
        self.model = models.Model(inputs=inputs, outputs=outputs)
        
        # Compile
        self.model.compile(
            optimizer=Adam(learning_rate=0.0001),  # Lower LR for ViT
            loss='binary_crossentropy' if self.num_classes == 1 else 'categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall', tf.keras.metrics.AUC(name='auc')]
        )
        
    def unfreeze_base_model(self, fine_tune_at=None):
        """
        Unfreeze the base model for fine-tuning
        If fine_tune_at is specified, only layers after that index will be trainable
        """
        base_model = self.model.layers[0]
        base_model.trainable = True
        
        if fine_tune_at:
            # Freeze all layers before fine_tune_at
            for layer in base_model.layers[:fine_tune_at]:
                layer.trainable = False
    
    def get_model_summary(self):
        """
        Print model summary
        """
        return self.model.summary()
    
    def get_compiled_model(self):
        """
        Return the compiled model
        """
        return self.model


class CustomCNNModel:
    """
    Custom CNN model from scratch (alternative to transfer learning)
    """
    
    def __init__(self, input_shape=(224, 224, 3), num_classes=1):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        self._build_model()
    
    def _build_model(self):
        """
        Build a custom CNN model
        """
        self.model = models.Sequential([
            # First convolutional block
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            # Second convolutional block
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            # Third convolutional block
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            # Fourth convolutional block
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            # Fifth convolutional block
            layers.Conv2D(512, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            # Flatten and dense layers
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(self.num_classes, activation='sigmoid' if self.num_classes == 1 else 'softmax')
        ])
        
        # Compile the model
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy' if self.num_classes == 1 else 'categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
    
    def get_model_summary(self):
        """
        Print model summary
        """
        return self.model.summary()
    
    def get_compiled_model(self):
        """
        Return the compiled model
        """
        return self.model


def get_training_callbacks(model_save_path='best_violence_model.h5'):
    """
    Get common callbacks for training
    """
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=model_save_path,
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        )
    ]
    return callbacks