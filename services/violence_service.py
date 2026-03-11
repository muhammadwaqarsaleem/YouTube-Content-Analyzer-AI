"""
Violence Detection Service
Wraps the violence detection model for easy integration
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import warnings
warnings.filterwarnings('ignore', message='.*Compiled the loaded model.*')

import logging
logging.getLogger('absl').setLevel(logging.ERROR)

import numpy as np
import tensorflow as tf
from pathlib import Path
import sys
import glob
# Add parent directory to path to import existing modules
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent / 'models' / 'violence_detection'))
from predict_violence import ViolenceDetectionPredictor


class ViolenceDetectionService:
    """
    Service class for violence detection on video frames
    """
    
    def __init__(self, model_path='models/violence_detection/violence_detection_model_resnet.h5'):
        """
        Initialize the violence detection service
        
        Args:
            model_path: Path to trained violence detection model
        """
        print(f"Loading violence detection model from {model_path}...")
        
        self.model_path = model_path
        self.predictor = ViolenceDetectionPredictor(model_path, model_format='h5')
        
        # Content-type aware thresholds (dynamic, set per video)
        self.default_threshold = 0.70  # Lowered from 0.85 to catch brief/distant violence
        self.sports_entertainment_threshold = 0.70  # Wrestling/fighting sports
        self.news_education_threshold = 0.75  # News/raw footage
        
        # High-confidence override: If model is extremely certain, ignore the frame ratio requirement
        self.high_confidence_override = 0.95  # Increased from 0.88 to avoid flagging cartoons
        self.override_min_frames = 3  # Require at least 3 frames for override
        self.default_min_frames = 2  # Minimum 2 frames required (avoids single-frame glitches)
        self.sports_entertainment_min_frames = 3  # Keep 3 for combat sports (staged violence)
        
        print("✓ Violence detection service initialized")
    

    def analyze_video_frames(self, frame_paths, timestamps=None, batch_size=32, category_prediction=None, transcript=None, metadata=None):
        """
        Analyze multiple frames from a video with context-aware detection
        
        Args:
            frame_paths: List of frame paths
            timestamps: Optional list of timestamps (in seconds)
            batch_size: Batch size for processing
            category_prediction: Optional category prediction result from category_service
            transcript: Optional video transcript text for context analysis
            metadata: Optional video metadata dictionary (title, channel, description)
            
        Returns:
            Dictionary with comprehensive analysis results
        """
        print(f"Analyzing {len(frame_paths)} video frames...")
        
        if len(frame_paths) == 0:
            return {
                'is_violent': False,
                'violence_percentage': 0.0,
                'violent_frame_count': 0,
                'total_frames': 0,
                'timeline': [],
                'violent_frame_timestamps': []
            }
        
        # Get timestamps if not provided
        if timestamps is None:
            timestamps = list(range(len(frame_paths)))
            
        from typing import List, Any, cast
        timestamps = cast(List[Any], timestamps)  # Type explicitly for linters
        
        # Detect content type and set dynamic thresholds
        content_type = self._detect_content_type(category_prediction, transcript, metadata)
        threshold, min_frames = self._get_dynamic_threshold(content_type)
        
        print(f"Content type detected: {content_type}")
        print(f"Using threshold: {threshold*100:.1f}%, min frames: {min_frames}")
        
        # Process in batches to manage memory
        all_results = []
        
        for i in range(0, len(frame_paths), batch_size):
            batch_paths = frame_paths[i:i+batch_size]
            batch_timestamps = timestamps[i:i+batch_size]  # type: ignore
            
            try:
                # Use a LOWER threshold for prediction to catch more potential violence
                # We'll filter with our custom threshold later
                prediction_threshold = threshold - 0.10  # 10% lower for initial detection
                
                batch_predictions = self.predictor.predict_batch(
                    batch_paths, 
                    threshold=prediction_threshold,  # Lower to catch more frames
                    batch_size=batch_size
                )
                
                # Add timestamps to results
                for pred, ts in zip(batch_predictions, batch_timestamps):
                    pred['timestamp'] = ts
                    all_results.append(pred)
                
                if (i + len(batch_paths)) % 1000 == 0:
                    print(f"  Processed {i + len(batch_paths)}/{len(frame_paths)} frames")
                    
            except Exception as e:
                print(f"Error processing batch {i}: {e}")
                continue
        
        if len(all_results) == 0:
            print("Warning: No frames were successfully analyzed")
            return {
                'is_violent': False,
                'violence_percentage': 0.0,
                'violent_frame_count': 0,
                'total_frames': 0,
                'timeline': [],
                'violent_frame_timestamps': []
            }
        
        # Calculate statistics with STANDARDIZED threshold logic
        # STEP 1: Apply content-aware threshold consistently
        frame_ratio_threshold = self._get_frame_ratio_threshold(content_type)
        
        print(f"\n🔍 VIOLENCE THRESHOLD CONFIGURATION:")
        print(f"  Content type: {content_type}")
        print(f"  Confidence threshold: {threshold*100:.1f}%")
        print(f"  Minimum frames: {min_frames}")
        print(f"  Frame ratio threshold: {frame_ratio_threshold*100:.1f}%")
        
        # STEP 2: Filter frames using ONLY our custom threshold (ignore model's is_violent flag)
        # Use raw probability_violence for consistent filtering
        violent_frames = [
            r for r in all_results 
            if r.get('probability_violence', 0) >= threshold  # Use raw probability
        ]
        
        non_violent_frames = [r for r in all_results if r not in violent_frames]
        
        print(f"\n📊 FRAME ANALYSIS:")
        print(f"  Total frames analyzed: {len(all_results)}")
        print(f"  Violent frames (≥{threshold*100:.1f}%): {len(violent_frames)}")
        print(f"  Non-violent frames: {len(non_violent_frames)}")
        
        # STEP 3: Calculate violence percentage
        violence_percentage = len(violent_frames) / len(all_results) if all_results else 0
        
        print(f"  Violence percentage: {violence_percentage*100:.1f}%")
        print(f"  Required minimum: {frame_ratio_threshold*100:.1f}%")
        
        # STEP 4: Apply temporal clustering check & HIGH-CONFIDENCE OVERRIDE
        is_violent = False
        
        # Check for High-Confidence Override (extremely obvious violence, even if brief)
        # For longer videos, 2 frames is too few to trigger a global override. Demand 3 frames or 5% of total frames.
        high_conf_frames = [r for r in violent_frames if r.get('probability_violence', 0) >= self.high_confidence_override]
        required_override_frames = max(self.override_min_frames, int(len(all_results) * 0.05))
        
        if len(high_conf_frames) >= required_override_frames:
            print(f"\n🚨 HIGH-CONFIDENCE OVERRIDE TRIGGERED!")
            print(f"  {len(high_conf_frames)} frames exceeded {self.high_confidence_override*100:.1f}%. Bypassing ratio requirement.")
            is_violent = True
            
        # Normal check (meets minimum count and ratio)
        elif len(violent_frames) >= min_frames and violence_percentage >= frame_ratio_threshold:
            # Check clustering
            from typing import List, Any, cast
            # Extract timestamps as a list of floats
            raw_timestamps: List[Any] = [r.get('timestamp', 0) for r in violent_frames]
            violent_timestamps: List[float] = sorted([float(ts) for ts in raw_timestamps])
            
            if len(violent_timestamps) >= 2:
                time_diffs = [violent_timestamps[i+1] - violent_timestamps[i] 
                              for i in range(len(violent_timestamps)-1)]
                avg_gap = sum(time_diffs) / len(time_diffs)
                
                max_allowed_gap = 5.0 if content_type == 'sports_entertainment' else 3.0
                
                is_clustered = avg_gap < max_allowed_gap
                
                print(f"\n⏰ TEMPORAL CLUSTERING ANALYSIS:")
                # Avoid slicing to satisfy restrictive type checkers (Pyre2)
                sample_size = min(10, len(violent_timestamps))
                timestamps_sample = [violent_timestamps[i] for i in range(sample_size)]
                print(f"  Violent timestamps: {timestamps_sample}{'...' if len(violent_timestamps) > 10 else ''}")
                print(f"  Average gap: {avg_gap:.2f}s")
                print(f"  Max allowed gap: {max_allowed_gap:.1f}s")
                print(f"  Clustering result: {'PASS' if is_clustered else 'FAIL'}")
                
                is_violent = is_clustered
            else:
                # Single cluster point - check if meets minimum
                is_violent = len(violent_frames) >= min_frames
                print(f"\n⏰ SINGLE CLUSTER: Only {len(violent_timestamps)} violent timestamp, using frame count")
        
        # STEP 5: Calculate severity based on actual violent frames
        if violent_frames:
            max_confidence = max([r.get('probability_violence', 0) for r in violent_frames])
            avg_confidence = sum([r.get('probability_violence', 0) for r in violent_frames]) / len(violent_frames)
            
            # Calculate context weight based on content type and transcript
            context_weight = self._calculate_context_weight(content_type, transcript, len(violent_frames))
            
            # CONTENT-TYPE-AWARE SCORING WEIGHTS
            # Different content types need different signal balancing
            weights = self._get_scoring_weights(content_type)
            
            # Multi-signal scoring: max_conf, avg_conf, density (frame ratio), context
            violence_score = (
                max_confidence * weights['max_confidence'] +
                avg_confidence * weights['avg_confidence'] +
                violence_percentage * weights['density'] +
                context_weight * weights['context']
            )
            
            # Use violence_score for severity with content-type aware calibration
            severity = self._calculate_severity(violence_score, content_type, len(violent_frames), is_violent)
        else:
            max_confidence = 0.0
            severity = "NONE"
            violence_score = 0.0
        
        print(f"\n✅ FINAL VIOLENCE ASSESSMENT:")
        print(f"  Is violent: {is_violent}")
        print(f"  Severity: {severity}")
        
        # Create timeline (kept for backend logs, will be hidden from frontend)
        timeline = []
        violent_timestamps_final = []
        
        for result in all_results:
            entry = {
                'frame_index': len(timeline),
                'timestamp': result.get('timestamp', 0),
                'is_violent': result['is_violent'],
                'confidence': result['confidence'],
                'probability_violence': result['probability_violence'],
                'label': result['label']
            }
            timeline.append(entry)
            
            # Use consistent threshold for timeline marking
            if result.get('probability_violence', 0) >= threshold:
                violent_timestamps_final.append(result.get('timestamp', 0))
        
        # Generate recommendation based on content type and severity
        recommendation = self._generate_recommendation(is_violent, severity, content_type)
        
        # Additional context for frontend
        override_details = None
        if is_violent and violence_score < 0.65:
            # This was flagged by dynamic thresholds taking precedence
            override_details = {
                "reason": f"Elevated by threshold context ({content_type})",
                "original_score": violence_score
            }
        
        analysis_result = {
            'is_violent': is_violent,
            'violence_percentage': round(violence_percentage * 100, 2),  # Use actual percentage
            'violent_frame_count': len(violent_frames),  # Count of frames above threshold
            'non_violent_frame_count': len(non_violent_frames),
            'total_frames': len(all_results),
            'max_confidence': max_confidence,  # Highest confidence detection
            'severity': severity,
            'timeline': timeline,  # For backend logs, not frontend
            'violent_frame_timestamps': violent_timestamps_final,
            'recommendation': recommendation,
            'detection_config': {
                'content_type': content_type,
                'threshold_used': threshold,
                'min_frames_used': min_frames,
                'frame_ratio_threshold': frame_ratio_threshold
            }
        }
        
        print(f"\n✓ Analysis complete:")
        print(f"  Total frames: {analysis_result['total_frames']}")
        print(f"  High-confidence violent frames: {analysis_result['violent_frame_count']} ({analysis_result['violence_percentage']:.2f}% max confidence)")
        print(f"  Non-violent frames: {analysis_result['non_violent_frame_count']}")
        print(f"  Severity: {analysis_result['severity']}")
        print(f"  Overall assessment: {'VIOLENT' if is_violent else 'NON-VIOLENT'}")
        print(f"  Recommendation: {recommendation}")
        
        return analysis_result
    
    def _generate_recommendation(self, is_violent, severity, content_type='general'):
        """
        Generate content recommendation based on violence severity and content type
        
        Args:
            is_violent: Whether video contains violence
            severity: Weighted violence severity string ('EXTREME', 'HIGH', 'MODERATE', 'LOW', 'NONE')
            content_type: Type of content (sports_entertainment, news, general)
            
        Returns:
            Recommendation string
        """
        if not is_violent or severity == 'NONE':
            return "Content appears safe for general audiences."
            
        # Adjust recommendations based on content type
        if content_type == 'sports_entertainment':
            if severity in ['EXTREME', 'HIGH']:
                return "HIGH sports violence detected. Entertainment content (WWE/wrestling). Viewer discretion advised (16+)."
            elif severity == 'MODERATE':
                return "MODERATE sports violence present. Entertainment fighting content (13+)."
            else:
                return "Mild sports action detected. Generally suitable for audiences."
        
        # General/news content recommendations
        if severity == 'EXTREME':
            return "EXTREME violence detected. Restrict to mature audiences only (18+)."
        elif severity == 'HIGH':
            return "HIGH violence content. Recommended for mature audiences (16+)."
        elif severity == 'MODERATE':
            return "MODERATE violence present. Parental guidance suggested (13+)."
        else:
            return "Mild violence detected. Consider parental guidance."
    
    def _detect_content_type(self, category_prediction, transcript, metadata=None):
        """
        Detect content type using category prediction, transcript, and metadata analysis
        
        Args:
            category_prediction: Result from category_service.predict_category()
            transcript: Video transcript text
            metadata: Video metadata dictionary (title, channel, description)
            
        Returns:
            Content type string: 'sports_entertainment', 'news_education', or 'general'
        """
        print(f"\n🔍 CONTENT TYPE DETECTION:")
        
        # Check category prediction first
        primary_category = None
        category_confidence = 0.0
        
        if category_prediction:
            primary_category = category_prediction.get('primary_category', '')
            category_confidence = category_prediction.get('primary_probability', 0.0)
        
        print(f"  Primary category: {primary_category}")
        print(f"  Category confidence: {category_confidence:.2f}")
        print(f"  Transcript length: {len(transcript) if transcript else 0} chars")
        
        # Sports/Entertainment keywords in transcript
        sports_ent_keywords = ['wwe', 'wrestling', 'match', 'fight night', 'highlights', 'raw', 'smackdown',
                               'boxing', 'mma', 'ufc', 'kickboxing', 'sumo', 'judо', 'karate']
        
        # News/Education keywords
        news_edu_keywords = ['news', 'breaking', 'report', 'politics', 'government', 'election',
                            'educational', 'tutorial', 'learn', 'science', 'research', 'study']
        
        # Analyze transcript if available
        transcript_sports_score = 0
        transcript_news_score = 0
        
        if transcript:
            transcript_lower = transcript.lower()
            transcript_sports_score = sum(1 for kw in sports_ent_keywords if kw in transcript_lower)
            transcript_news_score = sum(1 for kw in news_edu_keywords if kw in transcript_lower)
            
            print(f"  Transcript sports keywords: {transcript_sports_score}")
            print(f"  Transcript news keywords: {transcript_news_score}")
        
        # Check 1: High confidence Entertainment/Sports + transcript keywords
        if primary_category in ['Entertainment', 'Sports'] and category_confidence > 0.6:
            if transcript_sports_score >= 2:
                print(f"  ✓ Detected sports/entertainment (high confidence category + keywords)")
                return 'sports_entertainment'
        
        # Check 2: Strong transcript signals regardless of category confidence
        # This catches WWE videos where ML model is uncertain but metadata is clear
        if transcript_sports_score >= 3:
            print(f"  ✓ Detected sports/entertainment (strong transcript signals: {transcript_sports_score} keywords)")
            return 'sports_entertainment'
        
        # Check 3: Channel-based detection (WWE channel = WWE content)
        if metadata:
            channel = metadata.get('channel', '').lower()
            title = metadata.get('title', '').lower()
            combat_channels = ['wwe', 'ufc', 'bellator', 'one championship', 'pfl']
            
            if any(kw in channel for kw in combat_channels):
                print(f"  ✓ Detected sports/entertainment (combat sports channel: {channel})")
                return 'sports_entertainment'
            
            # Also check title for strong combat sports indicators
            title_combat_keywords = ['wwe', 'ufc', 'wrestling', 'raw', 'smackdown', 'fight night']
            if any(kw in title for kw in title_combat_keywords):
                print(f"  ✓ Detected sports/entertainment (combat keywords in title)")
                return 'sports_entertainment'
        
        # News/Education detection
        if primary_category in ['News & Politics', 'Education'] and category_confidence > 0.6:
            if transcript_news_score >= 2:
                print(f"  ✓ Detected news/education content (category={primary_category}, news_keywords={transcript_news_score})")
                return 'news_education'

        # Crime footage detection (bodycam, CCTV, law enforcement) - use title as signal
        # Since CCTV/surveillance footage has low visual quality, the model returns lower scores
        if metadata:
            title = metadata.get('title', '').lower()
            channel = metadata.get('channel', '').lower()
            crime_title_keywords = [
                'brutal attack', 'caught on video', 'caught on camera', 'caught on tape',
                'shooting', 'stabbing', 'murder', 'homicide', 'assault caught',
                'attack caught', 'bodycam', 'dashcam', 'cctv', 'surveillance footage',
                'bystander video', 'street attack', 'robbery on camera'
            ]
            law_enforcement_channels = [
                'police department', 'police dept', 'sheriff', 'lapd', 'nypd',
                'highway patrol', 'state police', 'metropolitan police', 'county sheriff',
                'police service', 'department of justice', 'fbi'
            ]
            is_crime_title = any(kw in title for kw in crime_title_keywords)
            is_law_enforcement_channel = any(kw in channel for kw in law_enforcement_channels)

            if is_crime_title or is_law_enforcement_channel:
                reason = 'law_enforcement_channel' if is_law_enforcement_channel else 'crime_keywords_in_title'
                print(f"  ✓ Detected crime footage ({reason}) → lowering visual threshold")
                return 'crime_footage'

        # Default to general content
        print(f"  ✓ Using general content detection (category={primary_category}, confidence={category_confidence:.2f})")
        return 'general'
    
    def _get_dynamic_threshold(self, content_type):
        """
        Get appropriate threshold and minimum frames based on content type
        
        Args:
            content_type: 'sports_entertainment', 'news_education', or 'general'
            
        Returns:
            Tuple of (threshold, min_frames)
        """
        if content_type == 'sports_entertainment':
            return self.sports_entertainment_threshold, self.sports_entertainment_min_frames
        elif content_type == 'news_education':
            return self.news_education_threshold, 2  # Stricter, fewer frames needed
        elif content_type == 'crime_footage':
            # CCTV/bodycam footage has lower visual quality → lower confidence required
            return 0.60, 2  # 60% threshold, only 2 frames needed
        else:
            return self.default_threshold, self.default_min_frames
    
    def _get_frame_ratio_threshold(self, content_type):
        """
        Get frame ratio threshold based on content type
        
        Args:
            content_type: 'sports_entertainment', 'news_education', or 'general'
            
        Returns:
            Frame ratio threshold (0.0 to 1.0)
        """
        if content_type == 'sports_entertainment':
            return 0.06  # 6% for combat sports (sparse but intense violence)
        elif content_type == 'news_education':
            return 0.15  # 15% for serious content
        elif content_type == 'crime_footage':
            return 0.10  # 10% — CCTV clips are short, fewer frames needed
        else:
            return 0.15  # 15% for general content (lowered from 30% to catch brief real-world violence)
    
    # ─── CONTENT-TYPE-AWARE SCORING WEIGHTS ───
    # Each content type prioritizes different signals
    SCORING_WEIGHTS = {
        'crime_footage': {
            # Crime footage: max confidence matters most (even a single clear frame is significant)
            # Low density is expected (brief violent moments in CCTV clips)
            'max_confidence': 0.40,
            'avg_confidence': 0.20,
            'density': 0.15,
            'context': 0.25
        },
        'news_education': {
            # News: balanced between visual evidence and context
            # Context is important because news discusses violence without showing it
            'max_confidence': 0.30,
            'avg_confidence': 0.25,
            'density': 0.20,
            'context': 0.25
        },
        'sports_entertainment': {
            # Sports: density matters more (sustained action sequences)
            # Context weight reduced because sports keywords inflate false positives
            'max_confidence': 0.25,
            'avg_confidence': 0.30,
            'density': 0.30,
            'context': 0.15
        },
        'general': {
            # General: balanced, slightly favoring visual confidence
            'max_confidence': 0.35,
            'avg_confidence': 0.25,
            'density': 0.20,
            'context': 0.20
        }
    }
    
    def _get_scoring_weights(self, content_type):
        """Get the scoring weight matrix for the given content type."""
        return self.SCORING_WEIGHTS.get(content_type, self.SCORING_WEIGHTS['general'])
    
    def _calculate_context_weight(self, content_type, transcript, violent_frame_count):
        """
        Calculate context weight based on content signals.
        This measures how much the contextual evidence supports the visual detection.
        
        Returns: 0.0 (no context support) to 1.0 (strong context support)
        """
        base_weight = 0.3  # Low neutral baseline — require evidence to increase
        
        # Crime footage: metadata already confirmed violence is real
        if content_type == 'crime_footage':
            base_weight = 0.8  # High baseline — bodycam/CCTV confirms violence
        
        # News: violence discussed but may or may not be shown
        elif content_type == 'news_education':
            if violent_frame_count > 0:
                base_weight = 0.7  # Frames detected + news context = confirm
            else:
                base_weight = 0.2  # News discussing violence ≠ showing violence
        
        # Sports: staged violence gets moderate context
        elif content_type == 'sports_entertainment':
            base_weight = 0.5  # Known violent context, but fictional/staged
        
        # Transcript keyword analysis
        if transcript:
            transcript_lower = transcript.lower()
            
            # Generic violence keywords
            violence_keywords = ['fight', 'attack', 'violence', 'aggressive', 'combat',
                                 'strike', 'kill', 'stab', 'shoot', 'blood', 'injury']
            kw_count = sum(1 for kw in violence_keywords if kw in transcript_lower)
            
            # Scale keyword boost: 1 keyword = +0.05, 3+ = +0.15, 5+ = +0.25
            if kw_count >= 5:
                base_weight += 0.25
            elif kw_count >= 3:
                base_weight += 0.15
            elif kw_count >= 1:
                base_weight += 0.05
            
            # Combat sports specific keywords (only boost if sports type)
            if content_type == 'sports_entertainment':
                combat_keywords = ['wwe', 'wrestling', 'ufc', 'mma', 'boxing', 'knockout']
                combat_count = sum(1 for kw in combat_keywords if kw in transcript_lower)
                if combat_count >= 2:
                    base_weight += 0.1  # Moderate boost — confirms sports violence
        
        return max(0.0, min(1.0, base_weight))
    
    def _calculate_severity(self, violence_score, content_type, violent_frame_count, is_violent=False):
        """
        Calculate severity level with content-type aware thresholds.
        
        Uses a tiered threshold system:
        - Crime footage: lower thresholds (real-world violence is more concerning)
        - Sports/entertainment: higher thresholds (staged violence is less alarming)
        - General: balanced thresholds
        """
        # Content-type specific severity thresholds
        # Format: [(min_score, severity_level), ...] in descending order
        THRESHOLDS = {
            'crime_footage': [
                (0.75, 'EXTREME'),
                (0.60, 'HIGH'),
                (0.45, 'MODERATE'),
                (0.30, 'LOW'),
            ],
            'news_education': [
                (0.85, 'EXTREME'),
                (0.70, 'HIGH'),
                (0.55, 'MODERATE'),
                (0.40, 'LOW'),
            ],
            'sports_entertainment': [
                (0.92, 'EXTREME'),  # Very hard to reach — staged violence rarely extremes
                (0.80, 'HIGH'),
                (0.65, 'MODERATE'),
                (0.50, 'LOW'),
            ],
            'general': [
                (0.88, 'EXTREME'),
                (0.75, 'HIGH'),
                (0.60, 'MODERATE'),
                (0.45, 'LOW'),
            ]
        }
        
        thresholds = THRESHOLDS.get(content_type, THRESHOLDS['general'])
        
        # If system detected violence through frame analysis but score is below thresholds,
        # guarantee at least LOW severity (prevents false negatives)
        if is_violent:
            for min_score, level in thresholds:
                if violence_score >= min_score:
                    return level
            return 'LOW'  # Guaranteed minimum when frames were flagged
        
        # Non-flagged: must meet threshold to get severity
        for min_score, level in thresholds:
            if violence_score >= min_score:
                return level
        
        return 'NONE'
    


def main():
    """
    Test the violence detection service
    """
    
    # Check if model exists
    model_path = 'models/violence_detection/violence_detection_model_resnet.h5'
    
    if not Path(model_path).exists():
        print(f"Model not found at {model_path}")
        return
    
    service = ViolenceDetectionService(model_path)
    
    # Test on sample frames if available
    test_frames = glob.glob('violence_frames/violence/*.jpg')[:10]  # type: ignore
    
    if test_frames:
        print(f"\nTesting on {len(test_frames)} sample frames...")
        result = service.analyze_video_frames(test_frames)
        print(f"\nTest Results:")
        print(f"  Violent frames detected: {result['violent_frame_count']}")
        print(f"  Violence percentage: {result['violence_percentage']:.2f}%")
    else:
        print("\nNo test frames found in violence_frames/violence/")


if __name__ == "__main__":
    main()
