"""
Category Prediction Service
Wraps the category prediction model for easy integration
"""

import numpy as np
import pickle
import re
from pathlib import Path
import sys
from datetime import datetime
from scipy.sparse import hstack

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from services.category_detectors import get_all_detectors


class CategoryPredictionService:
    """
    Service class for YouTube video category prediction
    """
    
    def __init__(self, 
                 model_path='models/yt_category_model.pkl',
                 feature_dir='features'):
        """
        Initialize the category prediction service
        
        Args:
            model_path: Path to trained LightGBM model dictionary (model, tfidf, le)
            feature_dir: Directory containing feature encoders (deprecated for new model)
        """
        print(f"Loading new category prediction model from {model_path}...")
        
        self.model_path = model_path
        self.model_dict = None
        self.model = None
        self.tfidf = None
        self.le = None
        self.detectors = get_all_detectors()  # Load all category detectors
        
        self._load_model()
        
        print("[OK] Category prediction service initialized (New LightGBM Pipeline)")
    
    def _load_model(self):
        """Load the trained LightGBM model dictionary"""
        try:
            with open(self.model_path, 'rb') as f:
                self.model_dict = pickle.load(f)
            
            self.model = self.model_dict.get('model')
            self.tfidf = self.model_dict.get('tfidf')
            self.le = self.model_dict.get('le')
            
            if self.model and self.tfidf and self.le:
                print(f"[OK] New LightGBM model, TF-IDF, and LabelEncoder loaded successfully")
            else:
                missing = [k for k in ['model', 'tfidf', 'le'] if not self.model_dict.get(k)]
                print(f"Error: Model dictionary missing keys: {missing}")
                
        except FileNotFoundError:
            print(f"Warning: Model not found at {self.model_path}")
            self.model = None
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None

    def _clean_text(self, text):
        r"""
        Clean text according to integration specs:
        1. Convert to lowercase.
        2. Remove all non-alphanumeric characters [^a-z0-9\s].
        3. Collapse multiple spaces.
        4. Trim whitespace.
        """
        if not text:
            return ""
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
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
        Predict video category from metadata using LightGBM pipeline
        
        Args:
            thumbnail_path: Path to thumbnail image (not used in this new model version)
            metadata: Dictionary with video metadata
            
        Returns:
            Dictionary with category predictions
        """
        if self.model is None or self.tfidf is None or self.le is None:
            return self._fallback_prediction(metadata)
        
        print("Predicting category (LightGBM Pipeline)...")
        
        try:
            # 1. Text Preprocessing
            title = self._clean_text(metadata.get('title', ''))
            tags_list = metadata.get('tags_list', [])
            if not tags_list and metadata.get('tags'):
                tags_list = metadata.get('tags', '').split(',')
            tags = self._clean_text(' '.join(tags_list))
            
            desc_raw = metadata.get('description', '')
            desc_words = self._clean_text(desc_raw).split()
            desc_snippet = ' '.join(desc_words[:100])
            
            # Construct Weighted Input String
            weighted_text = (title + " ") * 3 + (tags + " ") * 2 + desc_snippet
            weighted_text = weighted_text.strip()
            
            # 2. Numerical Features
            vc = float(metadata.get('view_count', 0))
            lc = float(metadata.get('like_count', 0))
            cc = float(metadata.get('comment_count', 0))
            
            views = np.log1p(vc)
            likes = np.log1p(lc)
            comments = np.log1p(cc)
            
            like_rate = likes / (views + 1)
            comment_rate = comments / (views + 1)
            
            # Timestamp logic
            published_at = metadata.get('published_at')
            if published_at:
                dt = datetime.fromtimestamp(published_at)
                publish_hour = dt.hour
                publish_weekday = dt.weekday()
            else:
                publish_hour = 0
                publish_weekday = 0
                
            title_len = len(metadata.get('title', ''))
            tag_count = len(tags_list)
            
            num_features = np.array([[
                views, likes, comments, 
                like_rate, comment_rate, 
                publish_hour, publish_weekday, 
                title_len, tag_count
            ]], dtype=np.float32)
            
            # 3. Data Pipeline
            text_sparse = self.tfidf.transform([weighted_text])
            X = hstack([text_sparse, num_features])
            
            # 4. Inference
            probabilities = self.model.predict_proba(X)[0]
            classes = self.le.classes_
            
            # Create category-probability mapping
            category_probs = []
            for i, prob in enumerate(probabilities):
                category_probs.append({
                    'category': str(classes[i]),
                    'probability': float(prob)
                })
            
            # Sort by probability
            category_probs.sort(key=lambda x: x['probability'], reverse=True)
            
            # DEBUG
            print(f"   Top 3 from LightGBM model:")
            for i, cat in enumerate(category_probs[:3], 1):
                print(f"   {i}. {cat['category']} ({cat['probability']*100:.2f}%)")
            
            # RUN ALL CATEGORY OVERRIDE DETECTORS
            result = {'all_categories': category_probs}
            detection_applied = False
            override_details = None
            
            for detector in self.detectors:
                detection_result = detector.detect(metadata, category_probs)
                if detection_result.is_detected:
                    print(f"\n[!] {detector.category_name.upper()} DETECTED (score: {detection_result.score}/10)")
                    result = {
                        'primary_category': detection_result.category,
                        'primary_probability': detection_result.confidence,
                        'all_categories': detection_result.all_categories,
                        'is_multi_label': False,
                        'multi_label_categories': [detection_result.category]
                    }
                    detection_applied = True
                    override_details = {
                        'triggered': True,
                        'detector': detector.category_name,
                        'indicators': detection_result.indicators,
                        'original_model_prediction': category_probs[0]['category'],
                        'original_model_confidence': category_probs[0]['probability'],
                        'override_reason': f"Detected {detection_result.score}/10 signals for {detector.category_name}"
                    }
                    break
            
            if not detection_applied:
                override_details = {'triggered': False}
            
            threshold = 0.2
            high_prob_categories = [cp for cp in result.get('all_categories', category_probs) if cp['probability'] > threshold]
            
            final_result = {
                'primary_category': result.get('primary_category', category_probs[0]['category']),
                'primary_probability': result.get('primary_probability', category_probs[0]['probability']),
                'all_categories': result.get('all_categories', category_probs)[:10],
                'is_multi_label': len(high_prob_categories) > 1,
                'multi_label_categories': [cp['category'] for cp in high_prob_categories[:3]],
                'override_details': override_details
            }
            
            return final_result
            
        except Exception as e:
            print(f"Error making prediction: {e}")
            import traceback
            traceback.print_exc()
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
