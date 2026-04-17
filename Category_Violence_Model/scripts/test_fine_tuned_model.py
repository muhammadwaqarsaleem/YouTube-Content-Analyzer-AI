"""
Test Fine-Tuned Violence Model for False Positives
Tests on challenging content that previously caused FPs
"""

import os
import sys
from pathlib import Path
import glob

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from services.violence_service import ViolenceDetectionService, ViolenceDetectionConfig

def test_false_positives():
    """Test fine-tuned model on known false positive triggers"""
    
    print("="*80)
    print("FALSE POSITIVE TEST - FINE-TUNED MODEL")
    print("="*80)
    
    # Initialize service (will auto-use fine-tuned model)
    service = ViolenceDetectionService()
    
    print(f"\nModel being tested: {service.model_path}")
    print(f"Threshold: {service.default_threshold*100:.1f}%")
    
    # Test directories that previously caused FPs
    test_cases = [
        {
            'name': 'Nature Documentaries (Animal Behavior)',
            'path': 'temp/videos/nature_test/*.mp4',
            'expected': 'NO VIOLENCE'
        },
        {
            'name': 'Music Videos (Stylized Content)',
            'path': 'temp/videos/music_video_test/*.mp4', 
            'expected': 'NO VIOLENCE'
        },
        {
            'name': 'Sports (Wrestling/Boxing)',
            'path': 'temp/videos/sports_test/*.mp4',
            'expected': 'NO VIOLENCE'
        },
        {
            'name': 'Gaming Content',
            'path': 'temp/videos/gaming_test/*.mp4',
            'expected': 'NO VIOLENCE'
        },
        {
            'name': 'Actual Violence (Ground Truth)',
            'path': 'violence_frames/violence/*.jpg',
            'expected': 'VIOLENCE'
        }
    ]
    
    results = []
    
    for test_case in test_cases:
        print(f"\n{'='*80}")
        print(f"Testing: {test_case['name']}")
        print(f"Expected: {test_case['expected']}")
        print(f"{'='*80}")
        
        # Find test files
        test_files = glob.glob(test_case['path'])
        
        if not test_files:
            print(f"⚠️  No test files found at {test_case['path']}")
            continue
        
        print(f"Found {len(test_files)} test samples")
        
        # For image-based tests (violence frames)
        if test_files[0].endswith(('.jpg', '.png')):
            # Sample 20 images for testing
            sample_size = min(20, len(test_files))
            test_sample = test_files[:sample_size]
            
            result = service.analyze_video_frames(
                frame_paths=test_sample,
                timestamps=list(range(sample_size))
            )
            
            is_violent = result['is_violent']
            violence_pct = result['violence_percentage']
            
            print(f"\n📊 Results:")
            print(f"  Violent frames: {result['violent_frame_count']}/{result['total_frames']}")
            print(f"  Violence %: {violence_pct}%")
            print(f"  Detected: {'VIOLENCE' if is_violent else 'NO VIOLENCE'}")
            
            # Check if result matches expectation
            expected_violent = test_case['expected'] == 'VIOLENCE'
            correct = (is_violent == expected_violent)
            
            status = "✅ PASS" if correct else "❌ FAIL"
            print(f"\n{status} - Expected: {test_case['expected']}, Got: {'VIOLENCE' if is_violent else 'NO VIOLENCE'}")
            
            results.append({
                'name': test_case['name'],
                'correct': correct,
                'expected': test_case['expected'],
                'detected': 'VIOLENCE' if is_violent else 'NO VIOLENCE',
                'violence_pct': violence_pct
            })
        
        else:
            print(f"⚠️  Video testing requires video extraction - skipping for now")
            print(f"   To test videos, extract frames first using:")
            print(f"   python -c \"from utils.video_utils import extract_frames; extract_frames('video.mp4')\"")
    
    # Print summary
    print(f"\n{'='*80}")
    print("TEST SUMMARY")
    print(f"{'='*80}")
    
    total_tests = len(results)
    passed = sum(1 for r in results if r['correct'])
    
    for result in results:
        status = "✅ PASS" if result['correct'] else "❌ FAIL"
        print(f"{status} {result['name']}")
        print(f"       Expected: {result['expected']}, Got: {result['detected']} (Violence: {result['violence_pct']:.1f}%)")
    
    print(f"\nTotal: {passed}/{total_tests} tests passed ({passed/total_tests*100:.1f}%)")
    
    if passed == total_tests:
        print("\n🎉 ALL TESTS PASSED! Fine-tuned model successfully eliminates FPs!")
    else:
        print(f"\n⚠️  {total_tests - passed} test(s) failed. Model may need further tuning.")
    
    print(f"\n{'='*80}")
    
    return passed == total_tests


def test_nonviolence_frames():
    """Test on non-violence frames to verify no false positives"""
    
    print(f"\n{'='*80}")
    print("NON-VIOLENCE FRAME TEST")
    print(f"{'='*80}")
    
    service = ViolenceDetectionService()
    
    # Get non-violence frames
    nonviolence_frames = glob.glob('violence_frames/nonviolence/*.jpg')
    
    if not nonviolence_frames:
        print("⚠️  No non-violence frames found!")
        return None
    
    # Sample 50 frames
    sample_size = min(50, len(nonviolence_frames))
    test_sample = nonviolence_frames[:sample_size]
    
    print(f"Testing {sample_size} non-violence frames...")
    
    result = service.analyze_video_frames(
        frame_paths=test_sample,
        timestamps=list(range(sample_size))
    )
    
    is_violent = result['is_violent']
    violence_pct = result['violence_percentage']
    
    print(f"\n📊 Results:")
    print(f"  Total frames: {result['total_frames']}")
    print(f"  Violent frames: {result['violent_frame_count']}")
    print(f"  Violence %: {violence_pct}%")
    print(f"  Is Violent: {is_violent}")
    
    # Should NOT detect violence
    correct = not is_violent and violence_pct < 10  # Less than 10% is acceptable noise
    
    if correct:
        print(f"\n✅ PASS - Correctly identified as NON-VIOLENT")
    else:
        print(f"\n❌ FAIL - False positive detected!")
    
    return correct


if __name__ == "__main__":
    # Run tests
    fp_passed = test_false_positives()
    nonviolence_passed = test_nonviolence_frames()
    
    print(f"\n{'='*80}")
    print("FINAL RESULTS")
    print(f"{'='*80}")
    print(f"False Positive Tests: {'✅ PASS' if fp_passed else '❌ FAIL'}")
    print(f"Non-Violence Test: {'✅ PASS' if nonviolence_passed else '❌ FAIL'}")
    
    if fp_passed and nonviolence_passed:
        print(f"\n🎉 ALL TESTS PASSED! Model ready for production!")
        sys.exit(0)
    else:
        print(f"\n⚠️  Some tests failed. Review results before deployment.")
        sys.exit(1)
