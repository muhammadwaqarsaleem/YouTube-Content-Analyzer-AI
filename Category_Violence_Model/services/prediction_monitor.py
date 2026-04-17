"""
Monitoring and Edge Case Logging Module
Tracks prediction quality and flags uncertain cases for review
"""

import json
from datetime import datetime
from pathlib import Path
import os


class PredictionMonitor:
    """
    Monitors predictions for quality assurance and edge case detection
    Logs uncertain predictions for human review and model improvement
    """
    
    def __init__(self, log_dir='monitoring_logs'):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Thresholds for flagging
        self.uncertainty_thresholds = {
            'category_confidence': 0.70,  # Flag if primary category confidence < 70%
            'violence_margin': 0.15,  # Flag if violence confidence is within 15% of threshold
            'context_mismatch': True,  # Flag if category and violence seem mismatched
            'sparse_violence': 0.10  # Flag if violence ratio < 10% (edge case territory)
        }
        
        # Counters for session stats
        self.stats = {
            'total_analyzed': 0,
            'flagged_for_review': 0,
            'combat_sports_detected': 0,
            'edge_cases_found': 0
        }
    
    def analyze_prediction(self, analysis_result):
        """
        Analyze a completed prediction for quality and edge cases
        
        Args:
            analysis_result: Complete analysis result from aggregator
            
        Returns:
            Dict with monitoring results and flags
        """
        self.stats['total_analyzed'] += 1
        
        flags = []
        edge_case_type = None
        
        # Extract key metrics
        category = analysis_result.get('category_prediction', {})
        violence = analysis_result.get('violence_analysis', {})
        metadata = analysis_result.get('metadata', {})
        
        # Check 1: Low category confidence
        category_conf = category.get('primary_probability', 0)
        if category_conf < self.uncertainty_thresholds['category_confidence']:
            flags.append({
                'type': 'LOW_CATEGORY_CONFIDENCE',
                'severity': 'MEDIUM',
                'details': f"Category confidence {category_conf*100:.1f}% below threshold"
            })
        
        # Check 2: Violence detection near threshold
        violence_pct = violence.get('violence_percentage', 0)
        is_violent = violence.get('is_violent', False)
        
        # Flag if violence percentage is close to decision boundary
        if 5 <= violence_pct <= 15:  # Around our 6% threshold
            flags.append({
                'type': 'VIOLENCE_EDGE_CASE',
                'severity': 'HIGH' if not is_violent else 'LOW',
                'details': f"Violence at {violence_pct:.1f}% - near decision threshold"
            })
            edge_case_type = 'SPARSE_VIOLENCE'
            self.stats['edge_cases_found'] += 1
        
        # Check 3: Context mismatch (high violence but entertainment category)
        primary_cat = category.get('primary_category', '')
        if violence_pct > 50 and primary_cat in ['Entertainment', 'Film & Animation']:
            flags.append({
                'type': 'CONTEXT_MISMATCH',
                'severity': 'HIGH',
                'details': f"High violence ({violence_pct:.0f}%) but categorized as {primary_cat}"
            })
            edge_case_type = 'CONTEXT_ANOMALY'
            self.stats['edge_cases_found'] += 1
        
        # Check 4: Combat sports without Sports category (should be caught by override)
        title = metadata.get('title', '').lower()
        channel = metadata.get('channel', '').lower()
        
        combat_keywords = ['ufc', 'wwe', 'boxing', 'mma', 'wrestling']
        has_combat_keywords = any(kw in f"{title} {channel}" for kw in combat_keywords)
        
        if has_combat_keywords and primary_cat != 'Sports':
            flags.append({
                'type': 'COMBAT_SPORTS_OVERRIDE_FAILED',
                'severity': 'CRITICAL',
                'details': f"Combat sports detected but category is {primary_cat}, not Sports"
            })
        elif has_combat_keywords:
            self.stats['combat_sports_detected'] += 1
        
        # Check 5: Very sparse violence pattern (< 6% but still flagged as violent)
        total_frames = violence.get('total_frames', 0)
        violent_frames = violence.get('violent_frame_count', 0)
        
        if total_frames > 0:
            violence_ratio = violent_frames / total_frames
            
            if violence_ratio < 0.06 and is_violent:
                flags.append({
                    'type': 'VERY_SPARSE_VIOLENCE',
                    'severity': 'MEDIUM',
                    'details': f"Only {violent_frames}/{total_frames} frames ({violence_ratio*100:.1f}%) flagged as violent"
                })
                edge_case_type = 'MINIMAL_VIOLENCE'
        
        # Determine if should be logged for review
        should_review = len(flags) > 0 and any(f['severity'] in ['HIGH', 'CRITICAL'] for f in flags)
        
        if should_review:
            self.stats['flagged_for_review'] += 1
            self._log_for_review(analysis_result, flags, edge_case_type)
        
        return {
            'flags': flags,
            'should_review': should_review,
            'edge_case_type': edge_case_type,
            'stats': self.stats
        }
    
    def _log_for_review(self, analysis_result, flags, edge_case_type):
        """Log prediction for human review"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        video_id = analysis_result.get('video_id', 'unknown')
        
        log_entry = {
            'timestamp': timestamp,
            'video_id': video_id,
            'edge_case_type': edge_case_type,
            'flags': flags,
            'prediction_summary': {
                'category': analysis_result.get('category_prediction', {}),
                'violence': analysis_result.get('violence_analysis', {}),
                'metadata': {
                    'title': analysis_result.get('metadata', {}).get('title'),
                    'channel': analysis_result.get('metadata', {}).get('channel'),
                    'duration': analysis_result.get('metadata', {}).get('duration')
                }
            },
            'full_analysis': analysis_result
        }
        
        # Write to log file
        log_file = self.log_dir / f"review_{timestamp}_{video_id}.json"
        
        with open(log_file, 'w') as f:
            json.dump(log_entry, f, indent=2, default=str)
        
        print(f"\n🔍 EDGE CASE LOGGED: {log_file}")
        print(f"   Type: {edge_case_type or 'MULTIPLE_FLAGS'}")
        print(f"   Flags: {len(flags)} issues detected")
    
    def get_session_stats(self):
        """Get statistics for current session"""
        return {
            **self.stats,
            'flag_rate': self.stats['flagged_for_review'] / max(1, self.stats['total_analyzed']) * 100,
            'edge_case_rate': self.stats['edge_cases_found'] / max(1, self.stats['total_analyzed']) * 100
        }
    
    def print_quality_report(self):
        """Print quality assurance report"""
        stats = self.get_session_stats()
        
        print("\n" + "="*80)
        print("PREDICTION QUALITY REPORT")
        print("="*80)
        
        print(f"\nTotal Videos Analyzed: {stats['total_analyzed']}")
        print(f"Flagged for Review: {stats['flagged_for_review']} ({stats['flag_rate']:.1f}%)")
        print(f"Edge Cases Detected: {stats['edge_cases_found']} ({stats['edge_case_rate']:.1f}%)")
        print(f"Combat Sports Videos: {stats['combat_sports_detected']}")
        
        print("\n" + "="*80)
        print("QUALITY METRICS")
        print("="*80)
        
        if stats['flag_rate'] < 5:
            print("✅ Flag rate is LOW (<5%) - System performing well")
        elif stats['flag_rate'] < 15:
            print("⚠️  Flag rate is MODERATE (5-15%) - Monitor closely")
        else:
            print("❌ Flag rate is HIGH (>15%) - Investigation needed")
        
        if stats['edge_case_rate'] < 10:
            print("✅ Edge case rate is acceptable (<10%)")
        else:
            print("⚠️  High edge case rate (>10%) - Consider model retraining")
        
        print("\n" + "="*80)


def main():
    """Test the monitoring system"""
    print("="*80)
    print("PREDICTION MONITORING SYSTEM TEST")
    print("="*80)
    
    monitor = PredictionMonitor()
    
    # Test case 1: Edge case - sparse violence
    print("\n\nTEST 1: Sparse Violence Edge Case")
    print("-"*80)
    
    test_analysis_1 = {
        'video_id': 'test_video_001',
        'category_prediction': {
            'primary_category': 'Entertainment',
            'primary_probability': 0.65,  # Below threshold
            'all_categories': []
        },
        'violence_analysis': {
            'is_violent': False,
            'violence_percentage': 6.0,  # Exactly at threshold
            'violent_frame_count': 3,
            'total_frames': 50,
            'severity': 'NONE'
        },
        'metadata': {
            'title': 'Random Video',
            'channel': 'Some Channel',
            'duration': 300
        }
    }
    
    result = monitor.analyze_prediction(test_analysis_1)
    
    print(f"\nFlags: {len(result['flags'])}")
    for flag in result['flags']:
        print(f"  - [{flag['severity']}] {flag['type']}: {flag['details']}")
    
    print(f"\nShould Review: {result['should_review']}")
    
    # Test case 2: Combat sports detection
    print("\n\nTEST 2: Combat Sports Detection")
    print("-"*80)
    
    test_analysis_2 = {
        'video_id': 'test_video_002',
        'category_prediction': {
            'primary_category': 'Sports',
            'primary_probability': 0.95,
            'all_categories': []
        },
        'violence_analysis': {
            'is_violent': True,
            'violence_percentage': 87.0,
            'violent_frame_count': 4,
            'total_frames': 50,
            'severity': 'HIGH'
        },
        'metadata': {
            'title': 'Khabib vs McGregor | FULL FIGHT | UFC',
            'channel': 'UFC',
            'duration': 1200
        }
    }
    
    result = monitor.analyze_prediction(test_analysis_2)
    
    print(f"\nFlags: {len(result['flags'])}")
    for flag in result['flags']:
        print(f"  - [{flag['severity']}] {flag['type']}: {flag['details']}")
    
    print(f"\nShould Review: {result['should_review']}")
    
    # Print final stats
    monitor.print_quality_report()
    
    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
