"""
Analysis Aggregator Service
Combines results from violence detection and category prediction services
"""

import json
from datetime import datetime
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))


class AnalysisAggregator:
    """
    Aggregate and format analysis results from multiple services
    """
    
    def __init__(self, output_dir='temp/results'):
        """
        Initialize the aggregator
        
        Args:
            output_dir: Directory to save analysis results
        """
        self.output_dir = output_dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        print("✓ Analysis aggregator initialized")
    
    def aggregate_results(self, video_id, violence_result, category_result, 
                         metadata, processing_time=None):
        """
        Combine all analysis results into a single report
        
        Args:
            video_id: YouTube video ID
            violence_result: Violence detection analysis result
            category_result: Category prediction result
            metadata: Video metadata
            processing_time: Total processing time in seconds
            
        Returns:
            Comprehensive analysis report dictionary
        """
        timestamp = datetime.now().isoformat()
        
        # Create summary
        summary = self._create_summary(violence_result, category_result)
        
        # Build complete report
        report = {
            'analysis_id': f"analysis_{video_id}_{int(datetime.now().timestamp())}",
            'timestamp': timestamp,
            'video_id': video_id,
            
            'summary': summary,
            
            'violence_analysis': violence_result,
            'category_prediction': category_result,
            
            'metadata': {
                'title': metadata.get('title', ''),
                'channel': metadata.get('channel', ''),
                'duration': metadata.get('duration', 0),
                'view_count': metadata.get('view_count', 0),
                'upload_date': metadata.get('upload_date', ''),
                'thumbnail_url': metadata.get('thumbnail_url', ''),
                'webpage_url': metadata.get('webpage_url', '')
            },
            
            'processing_info': {
                'processing_time_seconds': processing_time,
                'analysis_timestamp': timestamp,
                'model_versions': {
                    'violence_detection': 'resnet_v1.0',
                    'category_prediction': 'logistic_regression_v1.0'
                }
            }
        }
        
        return report
    
    # ─── SEVERITY LEVELS (ordered for comparison) ───
    SEVERITY_ORDER = ['NONE', 'LOW', 'MODERATE', 'HIGH', 'EXTREME']
    
    # ─── FICTIONAL VIOLENCE CATEGORIES (dampen severity) ───
    FICTIONAL_CATEGORIES = {'Gaming', 'Film & Animation', 'Entertainment', 'Comedy'}
    
    # ─── SERIOUS VIOLENCE CATEGORIES (boost severity) ───
    SERIOUS_CATEGORIES = {'News & Politics', 'Nonprofits & Activism', 'Education'}
    
    def _severity_index(self, severity):
        """Get numeric index for severity comparison"""
        try:
            return self.SEVERITY_ORDER.index(severity)
        except ValueError:
            return 0  # Default to NONE
    
    def _shift_severity(self, severity, delta):
        """Shift severity up (+) or down (-) by delta levels, clamped to valid range"""
        idx = self._severity_index(severity)
        new_idx = max(0, min(len(self.SEVERITY_ORDER) - 1, idx + delta))
        return self.SEVERITY_ORDER[new_idx]
    
    def _create_summary(self, violence_result, category_result):
        """
        Create a high-level summary with CROSS-SIGNAL VALIDATION.
        The aggregator is the SINGLE source of truth for severity, rating, and recommendation.
        
        Violence + Category signals are cross-validated:
        - Violent + News/Politics → severity boosted (real-world violence is more concerning)
        - Violent + Gaming/Film  → severity dampened (fictional violence is less alarming)
        """
        # ─── 1. RAW VIOLENCE SIGNALS ───
        raw_severity = self._get_raw_severity(violence_result)
        is_violent = violence_result.get('is_violent', False)
        violence_pct = violence_result.get('violence_percentage', 0.0)
        max_confidence = violence_result.get('max_confidence', 0.0)
        content_type = violence_result.get('detection_config', {}).get('content_type', 'general')
        
        # ─── 2. CATEGORY SIGNALS ───
        primary_category = category_result.get('primary_category', 'Unknown')
        raw_prob = category_result.get('primary_probability', 0.0)
        # Normalize: some detectors return 0.95, others 95.0
        category_confidence = raw_prob if raw_prob <= 1.0 else raw_prob / 100.0
        
        # ─── 3. CROSS-SIGNAL VALIDATION ───
        context_modifier = 0  # -1 = dampen, 0 = neutral, +1 = boost
        context_reason = None
        
        if is_violent:
            if primary_category in self.SERIOUS_CATEGORIES:
                context_modifier = +1
                context_reason = f"Violence in {primary_category} is more concerning (real-world context)"
                print(f"  🔺 CROSS-SIGNAL: Boosting severity (+1) — {context_reason}")
            elif primary_category in self.FICTIONAL_CATEGORIES:
                context_modifier = -1
                context_reason = f"Violence in {primary_category} is likely fictional/staged"
                print(f"  🔻 CROSS-SIGNAL: Dampening severity (-1) — {context_reason}")
        
        # Apply cross-signal modifier to severity
        final_severity = self._shift_severity(raw_severity, context_modifier)
        
        if context_modifier != 0:
            print(f"  📊 Severity: {raw_severity} → {final_severity} (modifier: {context_modifier:+d})")
        
        # CRITICAL FIX: If not violent, severity MUST be NONE regardless of raw score
        if not is_violent:
            final_severity = 'NONE'
            print(f"  ✓ Not violent (is_violent=False) → Setting severity to NONE")
        
        # ─── 4. BUILD SUMMARIES ───
        violence_summary = {
            'is_violent': is_violent,
            'violence_percentage': violence_pct,
            'violent_frame_count': violence_result.get('violent_frame_count', 0),
            'severity': final_severity,
            'raw_severity': raw_severity,
            'max_confidence': max_confidence,
            'content_type': content_type
        }
        
        category_summary = {
            'primary_category': primary_category,
            'confidence': category_confidence,  # Always 0-1 range
            'is_multi_label': category_result.get('is_multi_label', False),
            'top_categories': [
                cat['category'] for cat in category_result.get('all_categories', [])[:3]
            ]
        }
        
        # ─── 5. CONFIDENCE BREAKDOWN (transparency) ───
        confidence_breakdown = self._build_confidence_breakdown(
            violence_result, category_result, context_modifier, context_reason
        )
        
        # ─── 6. OVERALL RATING + RECOMMENDATION ───
        overall_rating = self._calculate_overall_rating(violence_summary, category_summary)
        recommendation = self._generate_recommendation(violence_summary, category_summary)
        
        return {
            'violence': violence_summary,
            'category': category_summary,
            'overall_rating': overall_rating,
            'recommendation': recommendation,
            'confidence_breakdown': confidence_breakdown,
            'cross_signal': {
                'modifier': context_modifier,
                'reason': context_reason
            }
        }
    
    def _get_raw_severity(self, violence_result):
        """
        Get the raw severity from violence service.
        Falls back to percentage-based calculation for backward compat.
        """
        if 'severity' in violence_result:
            return violence_result['severity']
        
        percentage = violence_result.get('violence_percentage', 0.0)
        if percentage == 0:
            return 'NONE'
        elif percentage < 10:
            return 'LOW'
        elif percentage < 30:
            return 'MODERATE'
        else:
            return 'HIGH'
    
    def _build_confidence_breakdown(self, violence_result, category_result,
                                     context_modifier, context_reason):
        """
        Build a transparency breakdown showing WHY the system made its decisions.
        This helps users trust or question the prediction.
        """
        breakdown = {
            'violence_signals': {
                'visual_confidence': violence_result.get('max_confidence', 0.0),
                'frame_ratio': violence_result.get('violence_percentage', 0.0) / 100.0,
                'content_type': violence_result.get('detection_config', {}).get('content_type', 'general'),
                'threshold_used': violence_result.get('detection_config', {}).get('threshold_used', 0.85),
                'frames_analyzed': violence_result.get('total_frames', 0),
                'violent_frames': violence_result.get('violent_frame_count', 0)
            },
            'category_signals': {
                'primary': category_result.get('primary_category', 'Unknown'),
                'model_confidence': category_result.get('primary_probability', 0.0),
                'override_applied': bool(category_result.get('override_details', {}).get('triggered', False)),
                'override_detector': category_result.get('override_details', {}).get('detector', None)
            },
            'cross_validation': {
                'applied': context_modifier != 0,
                'modifier': context_modifier,
                'reason': context_reason
            }
        }
        return breakdown
    
    def _calculate_overall_rating(self, violence_summary, category_summary):
        """
        Calculate overall content rating using BOTH violence severity AND category context.
        
        Rating matrix:
        - NONE severity → SAFE
        - LOW severity  → SAFE (with note)
        - MODERATE + serious category → CAUTION
        - MODERATE + fictional category → SAFE
        - HIGH/EXTREME → RESTRICTED
        """
        severity = violence_summary['severity']
        category = category_summary['primary_category']
        is_serious = category in self.SERIOUS_CATEGORIES
        
        if severity in ['NONE']:
            return 'SAFE'
        elif severity == 'LOW':
            return 'CAUTION' if is_serious else 'SAFE'
        elif severity == 'MODERATE':
            return 'CAUTION'
        elif severity in ['HIGH', 'EXTREME']:
            return 'RESTRICTED'
        else:
            return 'SAFE'
    
    def _generate_recommendation(self, violence_summary, category_summary):
        """
        Generate a context-aware recommendation using BOTH signals.
        """
        severity = violence_summary['severity']
        category = category_summary['primary_category']
        is_violent = violence_summary['is_violent']
        is_serious = category in self.SERIOUS_CATEGORIES
        is_fictional = category in self.FICTIONAL_CATEGORIES
        
        if not is_violent or severity == 'NONE':
            return "Content appears safe for general audiences."
        
        # Fictional violence (Gaming, Film, Entertainment)
        if is_fictional:
            if severity in ['HIGH', 'EXTREME']:
                return f"Intense fictional violence ({category}). Viewer discretion advised (16+)."
            elif severity == 'MODERATE':
                return f"Moderate fictional violence ({category}). Parental guidance suggested (13+)."
            else:
                return f"Mild action content ({category}). Generally suitable for audiences."
        
        # Serious/real-world violence (News, Education)
        if is_serious:
            if severity in ['HIGH', 'EXTREME']:
                return f"Graphic real-world violence ({category}). Restrict to mature audiences (18+)."
            elif severity == 'MODERATE':
                return f"Real-world violence ({category}). Parental guidance strongly suggested (13+)."
            else:
                return f"Mild violence in {category} content. Consider parental guidance."
        
        # General content
        if severity in ['HIGH', 'EXTREME']:
            return "Significant violent content detected. Recommended for mature audiences only (16+)."
        elif severity == 'MODERATE':
            return "Moderate violence present. Parental guidance suggested (13+)."
        else:
            return "Mild violence detected. Consider parental guidance."
    
    def save_report(self, report, filename=None):
        """
        Save analysis report to JSON file
        
        Args:
            report: Analysis report dictionary
            filename: Optional custom filename
            
        Returns:
            Path to saved report
        """
        if filename is None:
            filename = f"{report['analysis_id']}.json"
        
        output_path = Path(self.output_dir) / filename
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Report saved to: {output_path}")
        
        return output_path
    
    def load_report(self, analysis_id):
        """
        Load a previously saved analysis report
        
        Args:
            analysis_id: Analysis ID or filename
            
        Returns:
            Report dictionary or None if not found
        """
        # Try as filename first
        report_path = Path(self.output_dir) / f"{analysis_id}.json"
        
        if not report_path.exists():
            # Try as full filename
            report_path = Path(self.output_dir) / analysis_id
            if not report_path.suffix:
                report_path = report_path.with_suffix('.json')
        
        if report_path.exists():
            with open(report_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        # Silently return None if report doesn't exist (expected during processing)
        return None
    
    def format_for_display(self, report):
        """
        Format report for web dashboard display
        
        Args:
            report: Analysis report dictionary
            
        Returns:
            Formatted display data
        """
        # Normalize category probability to 0-1 range (some overrides return 95.0 instead of 0.95)
        raw_prob = report['category_prediction'].get('primary_probability', 0.0)
        normalized_confidence = raw_prob if raw_prob <= 1.0 else raw_prob / 100.0
        
        # Use the AGGREGATOR's final severity/rating (not violence_service's raw output)
        summary = report.get('summary', {})
        violence_summary = summary.get('violence', {})
        
        display_data = {
            'videoInfo': {
                'title': report['metadata']['title'],
                'channel': report['metadata']['channel'],
                'duration': self._format_duration(report['metadata']['duration']),
                'views': self._format_number(report['metadata']['view_count']),
                'videoId': report['video_id']
            },
            
            'violenceMetrics': {
                'isViolent': report['violence_analysis']['is_violent'],
                'percentage': report['violence_analysis']['violence_percentage'],
                'severity': violence_summary.get('severity', report['violence_analysis'].get('severity', 'NONE')),
                'max_confidence': report['violence_analysis'].get('max_confidence', 0),
                'recommendation': summary.get('recommendation', ''),
                'content_type': violence_summary.get('content_type', 'general'),
            },
            
            'categoryMetrics': {
                'primary': report['category_prediction']['primary_category'],
                'confidence': normalized_confidence,  # Always 0-1, frontend handles display
                'isMultiLabel': report['category_prediction']['is_multi_label'],
                'categories': report['category_prediction']['all_categories'],
                'overrideDetails': report['category_prediction'].get('override_details', None)
            },
            
            'rating': {
                'overall': summary.get('overall_rating', 'SAFE'),
                'recommendation': summary.get('recommendation', 'Content appears safe for general audiences.')
            },
            
            'confidenceBreakdown': summary.get('confidence_breakdown', {}),
            
            'crossSignal': summary.get('cross_signal', {}),
            
            'processingInfo': {
                'time': report['processing_info']['processing_time_seconds'],
                'timestamp': report['timestamp']
            }
        }
        
        return display_data
    
    def _format_duration(self, seconds):
        """Format duration in seconds to MM:SS"""
        if not seconds:
            return "0:00"
        
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}:{secs:02d}"
    
    def _format_number(self, num):
        """Format large numbers with K/M suffixes"""
        if not num:
            return "0"
        
        if num >= 1_000_000:
            return f"{num / 1_000_000:.1f}M"
        elif num >= 1_000:
            return f"{num / 1_000:.1f}K"
        else:
            return str(num)


def main():
    """
    Test the aggregator
    """
    aggregator = AnalysisAggregator()
    
    # Sample test data
    violence_result = {
        'is_violent': True,
        'violence_percentage': 15.5,
        'violent_frame_count': 155,
        'total_frames': 1000,
        'timeline': []
    }
    
    category_result = {
        'primary_category': 'Entertainment',
        'primary_probability': 0.85,
        'is_multi_label': True,
        'all_categories': [
            {'category': 'Entertainment', 'probability': 0.85},
            {'category': 'Comedy', 'probability': 0.45}
        ]
    }
    
    metadata = {
        'title': 'Test Video',
        'channel': 'Test Channel',
        'duration': 300,
        'view_count': 50000,
        'upload_date': '2024-01-01'
    }
    
    report = aggregator.aggregate_results(
        video_id='test123',
        violence_result=violence_result,
        category_result=category_result,
        metadata=metadata,
        processing_time=45.2
    )
    
    print("\nGenerated Report:")
    print(json.dumps(report, indent=2))
    
    # Test formatting
    display_data = aggregator.format_for_display(report)
    print("\nFormatted for Display:")
    print(json.dumps(display_data, indent=2))


if __name__ == "__main__":
    main()
