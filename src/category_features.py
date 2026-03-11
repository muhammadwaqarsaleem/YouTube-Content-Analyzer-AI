"""
Category Feature Extraction Module
Extracts and combines multi-modal features for category prediction model
"""

import numpy as np
import cv2
import pickle
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler


class CategoryFeatureExtractor:
    """
    Extract features for category prediction model
    Combines image features, text features, and numerical features
    """
    
    def __init__(self, feature_dir='features', img_size=(224, 224)):
        """
        Initialize the feature extractor
        
        Args:
            feature_dir: Directory containing saved feature encoders
            img_size: Target image dimensions for thumbnail processing
        """
        self.feature_dir = feature_dir
        self.img_size = img_size
        
        # Load pre-trained encoders
        self.tfidf_vectorizer = None
        self.num_scaler = None
        self.cat_vectorizer = None
        self.target_encoder = None
        
        self._load_encoders()
    
    def _load_encoders(self):
        """Load saved feature encoders"""
        try:
            # Load TF-IDF vectorizer
            tfidf_path = Path(self.feature_dir) / 'tfidf_vectorizer.pkl'
            if tfidf_path.exists():
                with open(tfidf_path, 'rb') as f:
                    self.tfidf_vectorizer = pickle.load(f)
                print(f"✓ Loaded TF-IDF vectorizer from {tfidf_path}")
            
            # Load numerical scaler
            num_scaler_path = Path(self.feature_dir) / 'num_scaler.pkl'
            if num_scaler_path.exists():
                with open(num_scaler_path, 'rb') as f:
                    self.num_scaler = pickle.load(f)
                print(f"✓ Loaded numerical scaler from {num_scaler_path}")
            
            # Load categorical vectorizer
            cat_vectorizer_path = Path(self.feature_dir) / 'cat_vectorizer.pkl'
            if cat_vectorizer_path.exists():
                with open(cat_vectorizer_path, 'rb') as f:
                    self.cat_vectorizer = pickle.load(f)
                print(f"✓ Loaded categorical vectorizer from {cat_vectorizer_path}")
            
            # Load target encoder
            target_encoder_path = Path(self.feature_dir) / 'target_encoder.pkl'
            if target_encoder_path.exists():
                with open(target_encoder_path, 'rb') as f:
                    self.target_encoder = pickle.load(f)
                print(f"✓ Loaded target encoder from {target_encoder_path}")
                
        except Exception as e:
            print(f"Warning: Could not load some encoders: {e}")
            print("Will need to retrain or fit encoders on new data")
    
    def load_thumbnail(self, thumbnail_path):
        """
        Load and preprocess thumbnail image
        
        Args:
            thumbnail_path: Path to thumbnail image
            
        Returns:
            Preprocessed image array (reduced dimensions)
        """
        img = cv2.imread(str(thumbnail_path))
        
        if img is None:
            raise FileNotFoundError(f"Could not load thumbnail: {thumbnail_path}")
        
        # Resize to standard size
        img_resized = cv2.resize(img, self.img_size)
        
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        
        # Reduce dimensions for feature extraction
        # CRITICAL FIX: Original model was trained with minimal/no image features
        # Total expected: 34,358 = text(5000) + numerical(10) + categorical(29,347) + image(1)
        # Using single grayscale pixel as image feature to match training
        img_resized = cv2.resize(img, self.img_size)  # Resize to standard size first
        img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        
        # Extract only 1 feature: mean brightness of center region
        h, w = img_gray.shape
        center_region = img_gray[h//4:3*h//4, w//4:3*w//4]  # Center 50% of image
        img_mean_brightness = np.mean(center_region) / 255.0  # Normalize to 0-1
        
        # Return as 1D array with single feature
        img_normalized = np.array([img_mean_brightness]).astype('float32')
        
        return img_normalized
    
    def extract_text_features(self, metadata):
        """
        Extract TF-IDF features from text fields
        
        Args:
            metadata: Dictionary with video metadata
            
        Returns:
            TF-IDF feature vector
        """
        # Combine text fields
        text_parts = []
        
        title = metadata.get('title', '')
        if title:
            text_parts.append(title)
        
        description = metadata.get('description', '')
        if description:
            text_parts.append(description[:500])  # Limit description length
        
        tags = metadata.get('tags', [])
        if tags:
            text_parts.append(' '.join(tags))
        
        channel = metadata.get('channel', '')
        if channel:
            text_parts.append(channel)
        
        combined_text = ' | '.join(text_parts) if text_parts else ''
        
        # Use existing vectorizer if loaded, otherwise create new one
        if self.tfidf_vectorizer is not None:
            try:
                text_features = self.tfidf_vectorizer.transform([combined_text])
                print(f"✓ Extracted text features: {text_features.shape}")
                return text_features
            except Exception as e:
                print(f"Warning: TF-IDF transformation failed: {e}")
                # Create minimal fallback features
                return np.zeros((1, 1000))
        else:
            # Fallback: create simple word count features
            words = combined_text.lower().split()
            word_counts = {}
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1
            
            # Return top 1000 most common words as features
            sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:1000]
            features = np.array([count for _, count in sorted_words]).reshape(1, -1)
            
            # Pad if necessary
            if features.shape[1] < 1000:
                padding = np.zeros((1, 1000 - features.shape[1]))
                features = np.hstack([features, padding])
            
            print(f"✓ Created fallback text features: {features.shape}")
            return features
    
    def extract_numerical_features(self, metadata):
        """
        Extract and scale numerical features
        
        Args:
            metadata: Dictionary with video metadata
            
        Returns:
            Scaled numerical feature vector
        """
        # Extract ONLY the core numerical fields that match training data
        # Model expects 10 features, so we use these 7 + 3 derived = 10 total
        features = {
            'duration': float(metadata.get('duration') or 0),
            'view_count': float(metadata.get('view_count') or 0),
            'like_count': float(metadata.get('like_count') or 0),
            'comment_count': float(metadata.get('comment_count') or 0),
            'fps': float(metadata.get('fps') or 0),
            'width': float(metadata.get('width') or 0),
            'height': float(metadata.get('height') or 0)
        }
        
        # Create log-transformed engagement features (3 additional = 10 total)
        if features['view_count'] > 0:
            features['log_views'] = np.log1p(features['view_count'])
        else:
            features['log_views'] = 0.0
        
        if features['like_count'] > 0:
            features['log_likes'] = np.log1p(features['like_count'])
        else:
            features['log_likes'] = 0.0
        
        if features['comment_count'] > 0:
            features['log_comments'] = np.log1p(features['comment_count'])
        else:
            features['log_comments'] = 0.0
        
        # Convert to array in specific order
        feature_names = [
            'duration', 'view_count', 'like_count', 'comment_count',
            'fps', 'width', 'height',
            'log_views', 'log_likes', 'log_comments'
        ]
        feature_values = [features[name] for name in feature_names]
        features_array = np.array(feature_values).reshape(1, -1)
        
        # Scale using existing scaler if available
        if self.num_scaler is not None:
            try:
                scaled_features = self.num_scaler.transform(features_array)
                print(f"✓ Extracted and scaled numerical features: {scaled_features.shape}")
                return scaled_features
            except Exception as e:
                print(f"Warning: Scaling failed: {e}")
                # Return unscaled features
                return features_array
        else:
            # Simple normalization fallback
            mean = np.mean(features_array, axis=1, keepdims=True)
            std = np.std(features_array, axis=1, keepdims=True) + 1e-8
            normalized = (features_array - mean) / std
            print(f"✓ Created normalized numerical features: {normalized.shape}")
            return normalized
    
    def extract_categorical_features(self, metadata):
        """
        Encode categorical variables with robust type handling
        
        Args:
            metadata: Dictionary with video metadata
            
        Returns:
            Encoded categorical features
        """
        # Extract categorical fields - handle any data type robustly
        categories = []
        
        # Extractor (platform) - Handle any type
        extractor = metadata.get('extractor', 'youtube')
        if extractor is None:
            extractor = 'youtube'
        elif isinstance(extractor, list):
            extractor = ' '.join(map(str, extractor)) if extractor else 'youtube'
        elif not isinstance(extractor, str):
            extractor = str(extractor)
        categories.append(extractor.lower().strip())
        
        # Resolution category - Always string, but handle numeric inputs
        width = metadata.get('width', 0)
        height = metadata.get('height', 0)
        try:
            height = int(height) if height else 0
        except (ValueError, TypeError):
            height = 0
            
        if height >= 1080:
            resolution = 'HD_1080+'
        elif height >= 720:
            resolution = 'HD_720'
        elif height >= 480:
            resolution = 'SD_480'
        else:
            resolution = 'LOW'
        categories.append(resolution.lower())
        
        # Duration category - Handle any type
        duration = metadata.get('duration', 0)
        try:
            duration = float(duration) if duration else 0
        except (ValueError, TypeError):
            duration = 0
            
        if duration > 600:  # > 10 minutes
            duration_cat = 'LONG'
        elif duration > 180:  # > 3 minutes
            duration_cat = 'MEDIUM'
        else:
            duration_cat = 'SHORT'
        categories.append(duration_cat.lower())
        
        # Debug logging
        print(f"\nDEBUG: Categorical inputs:")
        print(f"  Categories list: {categories}")
        print(f"  Category types: {[type(c) for c in categories]}")
        print(f"  Vectorizer type: {type(self.cat_vectorizer)}")
        
        # Encode using vectorizer - TF-IDF expects a single string, not list of strings!
        if self.cat_vectorizer is not None:
            try:
                # Join categories into single string for TF-IDF vectorizer
                categories_string = ' '.join(categories)
                cat_features = self.cat_vectorizer.transform([categories_string])
                print(f"✓ Encoded categorical features: {cat_features.shape}")
                return cat_features
            except Exception as e:
                print(f"Warning: Categorical encoding failed: {e}")
                import traceback
                traceback.print_exc()
                # Return zeros with reasonable shape as workaround
                return np.zeros((1, 30)).astype(np.float32)
        else:
            return np.zeros((1, 30)).astype(np.float32)
    
    def combine_features(self, image_features=None, text_features=None, 
                        numerical_features=None, categorical_features=None):
        """
        Combine all feature types into final feature vector
        
        Args:
            image_features: Image feature array (optional)
            text_features: Text feature array (sparse or dense)
            numerical_features: Numerical feature array
            categorical_features: Categorical feature array
            
        Returns:
            Combined feature vector
        """
        all_features = []
        
        # Flatten image features if provided
        if image_features is not None:
            img_flat = image_features.flatten()
            all_features.append(img_flat)
            print(f"  Image features: {img_flat.shape}")
        
        # Add text features
        if text_features is not None:
            if hasattr(text_features, 'toarray'):
                text_flat = text_features.toarray().flatten()
            else:
                text_flat = text_features.flatten()
            all_features.append(text_flat)
            print(f"  Text features: {text_flat.shape}")
        
        # Add numerical features
        if numerical_features is not None:
            num_flat = numerical_features.flatten()
            all_features.append(num_flat)
            print(f"  Numerical features: {num_flat.shape}")
        
        # Add categorical features
        if categorical_features is not None:
            if hasattr(categorical_features, 'toarray'):
                cat_flat = categorical_features.toarray().flatten()
            else:
                cat_flat = categorical_features.flatten()
            all_features.append(cat_flat)
            print(f"  Categorical features: {cat_flat.shape}")
        
        if len(all_features) == 0:
            raise ValueError("No features provided to combine")
        
        # Concatenate all features
        combined = np.hstack(all_features)
        
        print(f"✓ Combined feature vector: {combined.shape}")
        
        return combined
    
    def extract_all_features(self, thumbnail_path, metadata):
        """
        Extract all features from thumbnail and metadata
        
        Args:
            thumbnail_path: Path to thumbnail image
            metadata: Dictionary with video metadata
            
        Returns:
            Combined feature vector ready for model prediction
        """
        print("Extracting multi-modal features...")
        
        # Extract image features (flattened pixel values)
        img = self.load_thumbnail(thumbnail_path)
        image_features = img.flatten()
        
        # Extract text features
        text_features = self.extract_text_features(metadata)
        
        # Extract numerical features
        numerical_features = self.extract_numerical_features(metadata)
        
        # Extract categorical features
        categorical_features = self.extract_categorical_features(metadata)
        
        # Combine all features
        final_features = self.combine_features(
            image_features=image_features,
            text_features=text_features,
            numerical_features=numerical_features,
            categorical_features=categorical_features
        )
        
        return final_features


def main():
    """
    Test the feature extractor
    """
    extractor = CategoryFeatureExtractor()
    
    # Example usage
    test_thumbnail = input("Enter thumbnail path (or press Enter to skip): ")
    
    if test_thumbnail and Path(test_thumbnail).exists():
        test_metadata = {
            'title': 'Test Video Title',
            'description': 'This is a test description',
            'tags': ['test', 'example'],
            'channel': 'Test Channel',
            'duration': 300,
            'view_count': 10000,
            'like_count': 500,
            'comment_count': 50
        }
        
        features = extractor.extract_all_features(test_thumbnail, test_metadata)
        print(f"\nFinal feature vector shape: {features.shape}")
        print(f"Feature vector dtype: {features.dtype}")
    else:
        print("No test thumbnail provided")


if __name__ == "__main__":
    main()
