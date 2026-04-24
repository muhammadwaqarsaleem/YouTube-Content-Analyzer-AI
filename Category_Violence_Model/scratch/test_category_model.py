import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(os.getcwd()) / "Category_Violence_Model"))

from services.category_service import CategoryPredictionService

def test():
    try:
        service = CategoryPredictionService(model_path="Category_Violence_Model/models/yt_category_model.pkl")
        metadata = {
            'title': 'Test Tech Review 2024',
            'description': 'This is a long description about the latest smartphone technology and its features.',
            'view_count': 100000,
            'like_count': 5000,
            'comment_count': 300,
            'tags_list': ['tech', 'smartphone', 'review'],
            'published_at': 1713960000
        }
        result = service.predict_category(None, metadata)
        print("\nSUCCESS!")
        print(f"Primary Category: {result['primary_category']}")
        print(f"Confidence: {result['primary_probability']:.2f}")
    except Exception as e:
        print(f"\nFAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test()
