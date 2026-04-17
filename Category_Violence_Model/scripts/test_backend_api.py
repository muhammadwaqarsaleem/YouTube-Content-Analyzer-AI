"""
Test Backend API Violence Detection
Compare backend results with our custom analysis
"""

import requests
import time
import json


def test_video_analysis(video_url, video_name):
    """
    Test a single video through the backend API
    
    Args:
        video_url: YouTube video URL
        video_name: Name for display
    """
    print(f"\n{'='*80}")
    print(f"Testing: {video_name}")
    print(f"URL: {video_url}")
    print(f"{'='*80}")
    
    try:
        # Step 1: Start analysis
        print("\n📡 Starting analysis...")
        response = requests.post(
            'http://localhost:8000/analyze/video',
            json={
                'video_url': video_url,
                'extract_all_frames': True,
                'max_frames': 100  # Limit to 100 frames for comparison
            },
            timeout=10
        )
        
        if response.status_code != 200:
            print(f"❌ Error starting analysis: {response.text}")
            return None
        
        data = response.json()
        analysis_id = data['analysis_id']
        print(f"✓ Analysis started with ID: {analysis_id}")
        print(f"⏱️  Estimated time: {data.get('data', {}).get('estimated_time_seconds', 'Unknown')} seconds")
        
        # Step 2: Poll for results
        print("\n⏳ Waiting for results...")
        max_attempts = 60  # Wait up to 2 minutes
        attempt = 0
        
        while attempt < max_attempts:
            time.sleep(3)  # Wait 3 seconds between checks
            
            try:
                result_response = requests.get(
                    f'http://localhost:8000/analysis/{analysis_id}',
                    timeout=10
                )
                
                if result_response.status_code == 200:
                    result_data = result_response.json()
                    status = result_data.get('status')
                    
                    if status == 'complete':
                        print(f"\n✅ Analysis complete!")
                        return result_data
                    
                    elif status == 'processing':
                        step = result_data.get('step', 'processing')
                        elapsed = result_data.get('elapsed_seconds', 0)
                        print(f"  ⚙️  Processing... Step: {step}, Elapsed: {elapsed}s")
                    
                    elif status == 'error':
                        print(f"❌ Error: {result_data.get('error', 'Unknown error')}")
                        return None
                
                attempt += 1
                
            except requests.exceptions.Timeout:
                print(f"  ⏱️  Request timeout, retrying...")
                continue
            except Exception as e:
                print(f"  ❌ Error polling: {e}")
                attempt += 1
        
        print(f"\n❌ Timeout: Analysis did not complete in time")
        return None
        
    except requests.exceptions.ConnectionError:
        print("\n❌ Cannot connect to backend API!")
        print("   Make sure the server is running:")
        print("   python start_server.py")
        return None
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return None


def display_results(result_data):
    """Display results in a readable format"""
    if not result_data or 'data' not in result_data:
        print("\n❌ No results to display")
        return
    
    data = result_data['data']
    
    print(f"\n{'='*80}")
    print("BACKEND API RESULTS")
    print(f"{'='*80}")
    
    # Video Info
    if 'videoInfo' in data:
        info = data['videoInfo']
        print(f"\n📹 Video Information:")
        print(f"  Title: {info.get('title', 'N/A')}")
        print(f"  Channel: {info.get('channel', 'N/A')}")
        print(f"  Duration: {info.get('duration', 0)} seconds")
    
    # Violence Metrics
    if 'violenceMetrics' in data:
        violence = data['violenceMetrics']
        print(f"\n🔍 Violence Detection:")
        print(f"  Is Violent: {'⚠️ YES' if violence.get('is_violent') else '✅ NO'}")
        print(f"  Violence Percentage: {violence.get('percentage', 0):.2f}%")
        print(f"  Violent Frames: {violence.get('violent_count', 0)}/{violence.get('total_frames', 0)}")
        
        # Add severity rating like our custom analysis
        percentage = violence.get('percentage', 0)
        if percentage > 30:
            severity = "⚠️ HIGH VIOLENCE"
        elif percentage > 10:
            severity = "⚠️ MODERATE VIOLENCE"
        elif percentage > 0:
            severity = "⚠️ LOW VIOLENCE"
        else:
            severity = "✅ NO VIOLENCE"
        print(f"  Severity: {severity}")
    
    # Category Metrics
    if 'categoryMetrics' in data:
        category = data['categoryMetrics']
        print(f"\n🏷️  Category Prediction:")
        print(f"  Primary: {category.get('primary', 'N/A')} ({category.get('confidence', 0)*100:.1f}%)")
        print(f"  All Categories: {', '.join(category.get('all_categories', []))}")
    
    # Overall Rating
    if 'overallRating' in data:
        rating = data['overallRating']
        print(f"\n🏆 Overall Rating: {rating}")
    
    # Processing Stats
    if 'processingStats' in data:
        stats = data['processingStats']
        print(f"\n⚙️  Processing Stats:")
        print(f"  Time: {stats.get('processing_time', 0):.2f} seconds")
        print(f"  Frames Analyzed: {stats.get('frames_analyzed', 0)}")
    
    print(f"\n{'='*80}")


def main():
    """Main test function"""
    print("="*80)
    print("BACKEND API VIOLENCE DETECTION TEST")
    print("="*80)
    
    # Test videos from our previous analysis
    test_videos = [
        {
            'url': 'https://www.youtube.com/watch?v=4jHY-keVG4s',
            'name': 'Roman Reigns vs. Goldberg (Expected: HIGH VIOLENCE ~40%)'
        },
        {
            'url': 'https://www.youtube.com/watch?v=NHSL9zck318',
            'name': 'Reigns vs. Owens vs. Rollins (Expected: LOW VIOLENCE ~5%)'
        },
        {
            'url': 'https://www.youtube.com/watch?v=vBtKF5M4FwA',
            'name': 'Undertaker vs. Brock Lesnar (Expected: MODERATE ~17%)'
        }
    ]
    
    results = []
    
    for video in test_videos:
        result = test_video_analysis(video['url'], video['name'])
        
        if result:
            display_results(result)
            results.append({
                'video': video['name'],
                'result': result
            })
            
            # Save individual result
            video_id = video['url'].split('v=')[1]
            with open(f'temp/results/backend_test_{video_id}.json', 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"💾 Results saved to: temp/results/backend_test_{video_id}.json")
        else:
            print(f"\n❌ Failed to analyze: {video['name']}")
    
    # Summary
    if results:
        print(f"\n{'='*80}")
        print("SUMMARY OF ALL TESTS")
        print(f"{'='*80}")
        
        for r in results:
            violence = r['result']['data'].get('violenceMetrics', {})
            percentage = violence.get('percentage', 0)
            is_violent = violence.get('is_violent', False)
            
            print(f"\n{r['video']}")
            print(f"  Violence: {percentage:.2f}%")
            print(f"  Violent: {'YES ⚠️' if is_violent else 'NO ✅'}")
        
        print(f"\n{'='*80}")
        print("✅ Backend testing complete!")
        print(f"{'='*80}")


if __name__ == "__main__":
    main()
