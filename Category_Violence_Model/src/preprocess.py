import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

class ViolenceDataPreprocessor:
    """
    Data preprocessing class for violence detection dataset
    """
    
    def __init__(self, img_size=(224, 224), validation_split=0.2, test_split=0.15):
        self.img_size = img_size
        self.validation_split = validation_split
        self.test_split = test_split
        # Enhanced data augmentation for better generalization
        self.datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=15,  # Reduced from 20 for more realistic augmentations
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,  # Critical for violence detection
            zoom_range=0.15,
            shear_range=0.1,
            brightness_range=[0.8, 1.2],  # Added brightness variation
            fill_mode='nearest',
            channel_shift_range=10  # Added color variation
        )
        
    def load_images_from_directory(self, directory_path, limit=None):
        """
        Load and preprocess images from a given directory
        Args:
            directory_path: Path to the violence_frames directory
            limit: Maximum number of images to load per class (None for all)
        """
        images = []
        labels = []
        
        # Process violence images
        violence_dir = os.path.join(directory_path, 'violence')
        print(f"Loading violence images from {violence_dir}")
        violence_files = [f for f in os.listdir(violence_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if limit:
            violence_files = violence_files[:limit]
        
        for i, filename in enumerate(violence_files):
            img_path = os.path.join(violence_dir, filename)
            try:
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.resize(img, self.img_size)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    images.append(img)
                    labels.append(1)  # 1 for violence
                    
                    if (i + 1) % 1000 == 0:
                        print(f"  Loaded {i + 1}/{len(violence_files)} violence images")
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
        
        # Process non-violence images
        nonviolence_dir = os.path.join(directory_path, 'nonviolence')
        print(f"Loading non-violence images from {nonviolence_dir}")
        nonviolence_files = [f for f in os.listdir(nonviolence_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if limit:
            nonviolence_files = nonviolence_files[:limit]
        
        for i, filename in enumerate(nonviolence_files):
            img_path = os.path.join(nonviolence_dir, filename)
            try:
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.resize(img, self.img_size)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    images.append(img)
                    labels.append(0)  # 0 for non-violence
                    
                    if (i + 1) % 1000 == 0:
                        print(f"  Loaded {i + 1}/{len(nonviolence_files)} non-violence images")
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
        
        print(f"Total loaded: {len(images)} images")
        return np.array(images), np.array(labels)
    
    def prepare_datasets(self, data_dir, limit=None):
        """
        Prepare train, validation, and test datasets
        Args:
            data_dir: Path to violence_frames directory
            limit: Max images per class (None for all)
        """
        print("Loading images...")
        images, labels = self.load_images_from_directory(data_dir, limit=limit)
        
        print(f"Loaded {len(images)} images with shape {images.shape}")
        print(f"Label distribution: Violence={np.sum(labels)}, Non-violence={len(labels) - np.sum(labels)}")
        
        # Split the data into train+validation and test
        X_temp, X_test, y_temp, y_test = train_test_split(
            images, labels, 
            test_size=self.test_split, 
            random_state=42, 
            stratify=labels
        )
        
        # Calculate adjusted validation split for remaining data
        adjusted_val_split = self.validation_split / (1 - self.test_split)
        
        # Split the remaining data into train and validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=adjusted_val_split,
            random_state=42,
            stratify=y_temp
        )
        
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Validation set: {X_val.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        
        # Normalize the images
        X_train = X_train.astype('float32') / 255.0
        X_val = X_val.astype('float32') / 255.0
        X_test = X_test.astype('float32') / 255.0
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
    
    def create_data_generator(self, x, y, batch_size=32, shuffle=True):
        """
        Create augmented data generator
        Args:
            x: Input images
            y: Labels
            batch_size: Batch size for training
            shuffle: Whether to shuffle data
        """
        generator = self.datagen.flow(x, y, batch_size=batch_size, shuffle=shuffle)
        return generator
    
    def get_class_weights(self, y_train):
        """
        Calculate class weights for imbalanced datasets
        """
        from sklearn.utils.class_weight import compute_class_weight
        
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        
        return dict(enumerate(class_weights))