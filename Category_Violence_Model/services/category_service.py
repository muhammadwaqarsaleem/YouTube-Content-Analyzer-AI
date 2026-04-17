"""
Category Prediction Service
Wraps the category prediction model for easy integration
"""

import numpy as np
import pickle
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from src.category_features import CategoryFeatureExtractor
from services.category_detectors import get_all_detectors


class CategoryPredictionService:
    """
    Service class for YouTube video category prediction
    """
    
    def __init__(self, 
                 model_path='models/logistic_regression_model.pkl',
                 feature_dir='features'):
        """
        Initialize the category prediction service
        
        Args:
            model_path: Path to trained logistic regression model
            feature_dir: Directory containing feature encoders
        """
        print(f"Loading category prediction model from {model_path}...")
        
        self.model_path = model_path
        self.feature_dir = feature_dir
        self.model = None
        self.feature_extractor = None
        self.detectors = get_all_detectors()  # Load all category detectors
        
        self._load_model()
        self._init_feature_extractor()
        
        print("✓ Category prediction service initialized")
    
    def _load_model(self):
        """Load the trained logistic regression model"""
        try:
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            print(f"✓ Model loaded successfully")
            
            # Get model info if available
            if hasattr(self.model, 'classes_'):
                print(f"  Number of classes: {len(self.model.classes_)}")
                
        except FileNotFoundError:
            print(f"Warning: Model not found at {self.model_path}")
            print("Will use fallback prediction method")
            self.model = None
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None
    
    def _init_feature_extractor(self):
        """Initialize the feature extractor"""
        try:
            self.feature_extractor = CategoryFeatureExtractor(
                feature_dir=self.feature_dir
            )
            print("✓ Feature extractor initialized")
        except Exception as e:
            print(f"Warning: Could not initialize feature extractor: {e}")
            self.feature_extractor = None
    
    def _decode_categories(self, encoded_categories):
        """
        Decode numeric category labels to human-readable names
        
        Args:
            encoded_categories: List or array of encoded category indices
            
        Returns:
            List of decoded category names
        """
        # Mapping from numeric indices to category names
        # Based on YouTube API categories
        category_map = {
            0: 'Autos & Vehicles',
            1: 'Comedy',
            2: 'Education',
            3: 'Entertainment',
            4: 'Film & Animation',
            5: 'Gaming',
            6: 'Howto & Style',
            7: 'Music',
            8: 'News & Politics',
            9: 'Nonprofits & Activism',
            10: 'People & Blogs',
            11: 'Pets & Animals',
            12: 'Science & Technology',
            13: 'Shows',
            14: 'Sports',
            15: 'Travel & Events'
        }
        
        decoded = []
        for cat in encoded_categories:
            # Handle both numpy arrays and regular lists
            cat_idx = int(cat) if hasattr(cat, '__int__') else int(cat)
            decoded.append(category_map.get(cat_idx, f'Category {cat_idx}'))
        
        return decoded
    
    def predict_category(self, thumbnail_path, metadata):
        """
        Predict video category from thumbnail and metadata
        
        Args:
            thumbnail_path: Path to thumbnail image (can be None)
            metadata: Dictionary with video metadata
            
        Returns:
            Dictionary with category predictions
        """
        # Handle missing thumbnail - use fallback prediction
        if thumbnail_path is None or not Path(thumbnail_path).exists():
            print("No thumbnail provided or file doesn't exist - using fallback prediction")
            return self._fallback_prediction(metadata)
        
        if self.model is None or self.feature_extractor is None:
            return self._fallback_prediction(metadata)
        
        print("Predicting category...")
        
        # Extract features
        features = self.feature_extractor.extract_all_features(
            thumbnail_path, 
            metadata
        )
        
        # Make prediction
        try:
            prediction = self.model.predict([features])[0]
            probabilities = self.model.predict_proba([features])[0]
            
            # Get class labels
            if hasattr(self.model, 'classes_'):
                classes = self.model.classes_
            else:
                # Fallback for simple models
                classes = list(range(len(probabilities)))
            
            # Create category-probability mapping with decoded names
            category_names = self._decode_categories(classes)
            category_probs = []
            for i, prob in enumerate(probabilities):
                if i < len(classes):
                    category_name = category_names[i] if i < len(category_names) else str(classes[i])
                    # Ensure probability is not None and is a valid number
                    prob_value = float(prob) if prob is not None else 0.0
                    category_probs.append({
                        'category': category_name,
                        'probability': prob_value
                    })
            
            # Sort by probability
            category_probs.sort(key=lambda x: x['probability'], reverse=True)
            
            # DEBUG: Show original model prediction BEFORE overrides
            print(f"   Top 3 categories from ML model:")
            for i, cat in enumerate(category_probs[:3], 1):
                print(f"   {i}. {cat['category']} ({cat['probability']*100:.2f}%)")
            
            # RUN ALL CATEGORY OVERRIDE DETECTORS
            # Check each category in priority order until one matches
            result = {'all_categories': category_probs}  # Default: use model prediction
            detection_applied = False
            override_details = None  # Track why override was applied
            
            for detector in self.detectors:
                detection_result = detector.detect(metadata, category_probs)
                
                if detection_result.is_detected:
                    print(f"\n⚠️  {detector.category_name.upper()} DETECTED (score: {detection_result.score}/10)")
                    print(f"   Indicators: {', '.join(detection_result.indicators)}")
                    print(f"   Forcing category to '{detection_result.category}' at {detection_result.confidence*100:.0f}% confidence")
                    
                    result = {
                        'primary_category': detection_result.category,
                        'primary_probability': detection_result.confidence,
                        'all_categories': detection_result.all_categories,
                        'is_multi_label': False,
                        'multi_label_categories': [detection_result.category]
                    }
                    detection_applied = True
                    
                    # Store override details for frontend display
                    override_details = {
                        'triggered': True,
                        'detector': detector.category_name,
                        'score': detection_result.score,
                        'max_score': 10,
                        'indicators': detection_result.indicators,
                        'original_model_prediction': category_probs[0]['category'],
                        'original_model_confidence': category_probs[0]['probability'],
                        'override_reason': f"Detected {detection_result.score}/10 confidence signals for {detector.category_name}"
                    }
                    break  # Use first matching detector (highest priority)
            
            # Final fallback - use model prediction if no detector triggered
            if not detection_applied:
                print(f"\nℹ️  NO OVERRIDE TRIGGERED - USING MODEL PREDICTION")
                print(f"   Model's top choice: {category_probs[0]['category']} ({category_probs[0]['probability']*100:.2f}%)")
                
                # Store that no override was triggered
                override_details = {
                    'triggered': False,
                    'reason': 'No override detector found strong enough signals',
                    'model_prediction_used': category_probs[0]['category'],
                    'model_confidence': category_probs[0]['probability']
                }
            
            # Determine if multi-label (multiple categories above threshold)
            threshold = 0.1
            high_prob_categories = [cp for cp in result.get('all_categories', category_probs) if cp['probability'] > threshold]
            is_multi_label = len(high_prob_categories) > 1
            
            final_result = {
                'primary_category': result.get('primary_category', category_probs[0]['category']),
                'primary_probability': result.get('primary_probability', category_probs[0]['probability']),
                'all_categories': result.get('all_categories', category_probs)[:10],
                'is_multi_label': is_multi_label,
                'multi_label_categories': [cp['category'] for cp in high_prob_categories[:3]],
                'override_details': override_details  # Include override explanation
            }
            
            print(f"\n✓ Prediction complete:")
            print(f"  Primary category: {final_result['primary_category']}")
            print(f"  Confidence: {final_result['primary_probability']*100:.2f}%")
            if is_multi_label:
                print(f"  Multi-label detected: {', '.join(final_result['multi_label_categories'])}")
            
            return final_result
            
        except Exception as e:
            print(f"Error making prediction: {e}")
            return self._fallback_prediction(metadata)
    
    def _fallback_prediction(self, metadata):
        """
        Fallback prediction when model is not available
        
        Uses simple heuristics based on metadata
        """
        print("Using fallback prediction method...")
        
        # Simple keyword-based categorization
        title = metadata.get('title', '').lower()
        description = metadata.get('description', '').lower()
        tags = ' '.join(metadata.get('tags', [])).lower()
        
        combined_text = f"{title} {description} {tags}"
        
        # Define category keywords (using generic names for internal matching)
        category_keywords = {
            'Entertainment': ['entertainment', 'fun', 'show', 'music', 'video', 'movie'],
            'Comedy': ['comedy', 'funny', 'humor', 'joke', 'laugh', 'lol'],
            'Tech': ['tech', 'technology', 'computer', 'software', 'programming', 'code'],
            'Science': ['science', 'research', 'discovery', 'experiment', 'physics', 'biology'],
            'News': ['news', 'breaking', 'report', 'current', 'affairs', 'politics'],
            'Food': ['food', 'cooking', 'recipe', 'kitchen', 'chef', 'restaurant'],
            'Blog': ['blog', 'vlog', 'daily', 'life', 'personal', 'story'],
            'Automobile': ['car', 'auto', 'vehicle', 'drive', 'automotive', 'bike'],
            'Informative': ['informative', 'educational', 'tutorial', 'learn', 'guide'],
            'VideoGames': ['game', 'gaming', 'videogame', 'esports', 'playthrough'],
        }
        
        # Map generic category names to YouTube API standard categories
        CATEGORY_MAPPING = {
            'Informative': 'Education',
            'VideoGames': 'Gaming',
            'Tech': 'Science & Technology',
            'Blog': 'People & Blogs',
            'Automobile': 'Autos & Vehicles',
            'News': 'News & Politics',
            'Science': 'Science & Technology',
            # Keep these as-is (already match YouTube API)
            'Entertainment': 'Entertainment',
            'Comedy': 'Comedy',
            'Food': 'Entertainment',  # Food falls under Entertainment in YouTube API
        }
        
        # Count matches
        category_scores = {}
        for category, keywords in category_keywords.items():
            score = sum(1 for keyword in keywords if keyword in combined_text)
            category_scores[category] = score
        
        # Sort by score
        sorted_categories = sorted(category_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Get top categories
        top_score = sorted_categories[0][1] if sorted_categories else 0
        top_categories = [(cat, score) for cat, score in sorted_categories if score > 0]
        
        if not top_categories:
            top_categories = [('Entertainment', 1)]  # Default
        
        # Normalize scores to probabilities and apply category mapping
        total_score = sum(score for _, score in top_categories)
        if total_score == 0:
            total_score = 1
        
        category_probs = []
        for category, score in top_categories[:10]:
            prob = score / total_score
            # Map to YouTube API category name
            mapped_category = CATEGORY_MAPPING.get(category, category)
            category_probs.append({
                'category': mapped_category,
                'probability': prob
            })
        
        result = {
            'primary_category': category_probs[0]['category'],
            'primary_probability': category_probs[0]['probability'],
            'all_categories': category_probs,
            'is_multi_label': len(category_probs) > 1,
            'multi_label_categories': [cp['category'] for cp in category_probs[:3]]
        }
        
        print(f"Fallback prediction: {result['primary_category']}")
        
        return result



def main():
    """
    Test the category prediction service
    """
    service = CategoryPredictionService()
    
    # Example test
    test_thumbnail = input("Enter thumbnail path (or press Enter to skip): ")
    
    if test_thumbnail and Path(test_thumbnail).exists():
        test_metadata = {
            'title': 'Amazing Tech Review 2024',
            'description': 'In-depth review of latest technology',
            'tags': ['tech', 'review', 'gadgets'],
            'channel': 'Tech Channel',
            'duration': 600,
            'view_count': 50000,
            'like_count': 2000,
            'comment_count': 150
        }
        
        result = service.predict_category(test_thumbnail, test_metadata)
        
        print(f"\nPrediction Results:")
        print(f"  Primary: {result['primary_category']} ({result['primary_probability']*100:.2f}%)")
        print(f"  Top categories:")
        for cat in result['all_categories'][:5]:
            print(f"    - {cat['category']}: {cat['probability']*100:.2f}%")
    else:
        print("No test thumbnail provided")


if __name__ == "__main__":
    main()
