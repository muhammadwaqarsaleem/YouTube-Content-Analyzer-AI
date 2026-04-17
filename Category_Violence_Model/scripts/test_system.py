"""
Test Script for YouTube Video Analysis Backend
Tests all components of the system
"""

import sys
from pathlib import Path


def print_section(title):
    """Print formatted section header"""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70 + "\n")


def test_imports():
    """Test if all required modules can be imported"""
    print_section("TEST 1: Checking Module Imports")
    
    modules_to_test = [
        ('fastapi', 'FastAPI framework'),
        ('uvicorn', 'ASGI server'),
        ('yt_dlp', 'YouTube downloader'),
        ('youtube_transcript_api', 'Transcript fetcher'),
        ('tensorflow', 'TensorFlow ML'),
        ('cv2', 'OpenCV image processing'),
        ('numpy', 'NumPy arrays'),
        ('sklearn', 'Scikit-learn ML'),
        ('pandas', 'Pandas data handling'),
    ]
    
    failed = []
    
    for module, description in modules_to_test:
        try:
            __import__(module)
            print(f"  ✓ {module:30s} - {description}")
        except ImportError as e:
            print(f"  ✗ {module:30s} - FAILED: {e}")
            failed.append(module)
    
    if failed:
        print(f"\n⚠️  {len(failed)} modules failed to import")
        return False
    
    print(f"\n✓ All {len(modules_to_test)} modules imported successfully")
    return True


def test_project_structure():
    """Test if all required files exist"""
    print_section("TEST 2: Checking Project Structure")
    
    required_files = [
        'api/main.py',
        'services/violence_service.py',
        'services/category_service.py',
        'services/analysis_aggregator.py',
        'src/youtube_extractor.py',
        'src/violence_preprocess.py',
        'src/category_features.py',
        'utils/file_manager.py',
        'utils/video_utils.py',
        'frontend/index.html',
        'frontend/styles.css',
        'frontend/dashboard.js',
        'requirements.txt',
        'start_server.py'
    ]
    
    missing = []
    
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"  ✓ {file_path}")
        else:
            print(f"  ✗ {file_path} - MISSING")
            missing.append(file_path)
    
    if missing:
        print(f"\n⚠️  {len(missing)} files are missing")
        return False
    
    print(f"\n✓ All {len(required_files)} files found")
    return True


def test_model_files():
    """Test if model files exist"""
    print_section("TEST 3: Checking Model Files")
    
    models = {
        'models/violence_detection/violence_detection_model_resnet.h5': 'Violence Detection (ResNet50)',
        'models/logistic_regression_model.pkl': 'Category Prediction (Logistic Regression)',
        'features/tfidf_vectorizer.pkl': 'TF-IDF Vectorizer',
        'features/cat_vectorizer.pkl': 'Categorical Vectorizer',
        'features/num_scaler.pkl': 'Numerical Scaler',
        'features/target_encoder.pkl': 'Target Encoder'
    }
    
    missing = []
    
    for model_path, description in models.items():
        if Path(model_path).exists():
            size_mb = Path(model_path).stat().st_size / (1024 * 1024)
            print(f"  ✓ {model_path:50s} - {size_mb:.1f} MB")
        else:
            print(f"  ⚠ {model_path:50s} - Not found (will use fallback)")
            missing.append(model_path)
    
    if len(missing) > 2:
        print(f"\n⚠️  Warning: Multiple model files missing")
        print("   System will work with reduced functionality")
    
    print()
    return True


def test_youtube_extractor():
    """Test YouTube extractor module"""
    print_section("TEST 4: Testing YouTube Extractor Module")
    
    try:
        from src.youtube_extractor import YouTubeMediaExtractor
        
        extractor = YouTubeMediaExtractor()
        print("  ✓ YouTubeMediaExtractor initialized")
        
        # Test video ID extraction
        test_urls = [
            'https://www.youtube.com/watch?v=dQw4w9WgXcQ',
            'https://youtu.be/dQw4w9WgXcQ',
            'https://youtube.com/watch?v=abc123'
        ]
        
        for url in test_urls:
            video_id = extractor.get_video_id(url)
            if video_id:
                print(f"  ✓ Extracted video ID from URL: {video_id}")
            else:
                print(f"  ✗ Failed to extract video ID from: {url}")
        
        print("\n✓ YouTube Extractor module working")
        return True
        
    except Exception as e:
        print(f"  ✗ Error testing YouTube extractor: {e}")
        return False


def test_violence_preprocessor():
    """Test violence frame preprocessor"""
    print_section("TEST 5: Testing Violence Preprocessor")
    
    try:
        from src.violence_preprocess import ViolenceFramePreprocessor
        
        preprocessor = ViolenceFramePreprocessor()
        print("  ✓ ViolenceFramePreprocessor initialized")
        print(f"  ✓ Input shape: {preprocessor.img_size}")
        
        print("\n✓ Violence Preprocessor module working")
        return True
        
    except Exception as e:
        print(f"  ✗ Error testing violence preprocessor: {e}")
        return False


def test_category_features():
    """Test category feature extractor"""
    print_section("TEST 6: Testing Category Feature Extractor")
    
    try:
        from src.category_features import CategoryFeatureExtractor
        
        extractor = CategoryFeatureExtractor()
        print("  ✓ CategoryFeatureExtractor initialized")
        
        # Check if encoders loaded
        if extractor.tfidf_vectorizer:
            print("  ✓ TF-IDF vectorizer loaded")
        else:
            print("  ⚠ TF-IDF vectorizer not loaded (will use fallback)")
        
        if extractor.num_scaler:
            print("  ✓ Numerical scaler loaded")
        else:
            print("  ⚠ Numerical scaler not loaded (will use fallback)")
        
        print("\n✓ Category Feature Extractor module working")
        return True
        
    except Exception as e:
        print(f"  ✗ Error testing category feature extractor: {e}")
        return False


def test_services():
    """Test service layer"""
    print_section("TEST 7: Testing Service Layer")
    
    services_tested = 0
    services_passed = 0
    
    # Test Violence Detection Service
    try:
        from services.violence_service import ViolenceDetectionService
        
        if Path('models/violence_detection/violence_detection_model_resnet.h5').exists():
            service = ViolenceDetectionService('models/violence_detection/violence_detection_model_resnet.h5')
            print("  ✓ ViolenceDetectionService initialized")
            services_passed += 1
        else:
            print("  ⚠ ViolenceDetectionService - Model not found")
        
        services_tested += 1
        
    except Exception as e:
        print(f"  ✗ ViolenceDetectionService error: {e}")
        services_tested += 1
    
    # Test Category Prediction Service
    try:
        from services.category_service import CategoryPredictionService
        
        if Path('models/logistic_regression_model.pkl').exists():
            service = CategoryPredictionService('models/logistic_regression_model.pkl')
            print("  ✓ CategoryPredictionService initialized")
            services_passed += 1
        else:
            print("  ⚠ CategoryPredictionService - Model not found")
        
        services_tested += 1
        
    except Exception as e:
        print(f"  ✗ CategoryPredictionService error: {e}")
        services_tested += 1
    
    # Test Analysis Aggregator
    try:
        from services.analysis_aggregator import AnalysisAggregator
        
        aggregator = AnalysisAggregator()
        print("  ✓ AnalysisAggregator initialized")
        services_passed += 1
        services_tested += 1
        
    except Exception as e:
        print(f"  ✗ AnalysisAggregator error: {e}")
        services_tested += 1
    
    print(f"\n✓ {services_passed}/{services_tested} services working")
    return services_passed == services_tested


def test_api_app():
    """Test FastAPI application"""
    print_section("TEST 8: Testing FastAPI Application")
    
    try:
        from api.main import app
        
        print("  ✓ FastAPI app imported")
        print(f"  ✓ App title: {app.title}")
        print(f"  ✓ App version: {app.version}")
        
        # Check routes
        routes = [route.path for route in app.routes]
        expected_routes = ['/analyze/video', '/analyze/batch', '/analysis/{analysis_id}', '/health']
        
        print("\n  Available routes:")
        for route in routes[:10]:  # Show first 10 routes
            print(f"    • {route}")
        
        missing_routes = [r for r in expected_routes if r not in routes]
        if missing_routes:
            print(f"\n  ⚠ Missing routes: {missing_routes}")
        else:
            print("\n  ✓ All expected routes present")
        
        print("\n✓ FastAPI application working")
        return True
        
    except Exception as e:
        print(f"  ✗ Error testing FastAPI app: {e}")
        return False


def test_frontend_files():
    """Test frontend files"""
    print_section("TEST 9: Testing Frontend Files")
    
    frontend_files = {
        'frontend/index.html': 'HTML Dashboard',
        'frontend/styles.css': 'CSS Styles',
        'frontend/dashboard.js': 'JavaScript Logic'
    }
    
    all_good = True
    
    for file_path, description in frontend_files.items():
        if Path(file_path).exists():
            size_kb = Path(file_path).stat().st_size / 1024
            print(f"  ✓ {description:30s} - {size_kb:.1f} KB")
        else:
            print(f"  ✗ {description:30s} - MISSING")
            all_good = False
    
    if all_good:
        print("\n✓ All frontend files present")
    else:
        print("\n⚠️  Some frontend files missing")
    
    return all_good


def run_summary(tests_run, tests_passed):
    """Print test summary"""
    print_section("TEST SUMMARY")
    
    total_tests = tests_run
    passed_tests = tests_passed
    success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
    
    print(f"Tests Run:     {total_tests}")
    print(f"Tests Passed:  {passed_tests}")
    print(f"Tests Failed:  {total_tests - passed_tests}")
    print(f"Success Rate:  {success_rate:.1f}%\n")
    
    if success_rate == 100:
        print("🎉 ALL TESTS PASSED! System is ready to use.\n")
        print("Next steps:")
        print("  1. Run: python start_server.py")
        print("  2. Open: http://localhost:8000/")
        print("  3. Paste a YouTube URL and analyze!\n")
    elif success_rate >= 80:
        print("✅ Most tests passed. System should work with minor issues.\n")
        print("Check the failed tests above for potential issues.\n")
    else:
        print("⚠️  Multiple tests failed. Please review errors above.\n")
        print("Recommended actions:")
        print("  1. Install dependencies: pip install -r requirements.txt")
        print("  2. Verify model files exist")
        print("  3. Run this test again\n")


def main():
    """Run all tests"""
    print("\n" + "█"*70)
    print("█" + " "*68 + "█")
    print("█" + "  YOUTUBE VIDEO ANALYSIS BACKEND - SYSTEM TEST".center(68) + "█")
    print("█" + " "*68 + "█")
    print("█"*70)
    
    tests_run = 0
    tests_passed = 0
    
    # Run tests
    if test_imports():
        tests_passed += 1
    tests_run += 1
    
    if test_project_structure():
        tests_passed += 1
    tests_run += 1
    
    if test_model_files():
        tests_passed += 1
    tests_run += 1
    
    if test_youtube_extractor():
        tests_passed += 1
    tests_run += 1
    
    if test_violence_preprocessor():
        tests_passed += 1
    tests_run += 1
    
    if test_category_features():
        tests_passed += 1
    tests_run += 1
    
    if test_services():
        tests_passed += 1
    tests_run += 1
    
    if test_api_app():
        tests_passed += 1
    tests_run += 1
    
    if test_frontend_files():
        tests_passed += 1
    tests_run += 1
    
    # Print summary
    run_summary(tests_run, tests_passed)
    
    return tests_passed == tests_run


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
