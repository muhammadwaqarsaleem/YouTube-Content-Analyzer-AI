"""
Violence Detection Service - Refactored Version
Wraps the violence detection model for clean, production-ready integration
"""

import os
import glob
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

# Suppress TensorFlow/Keras log noise before any TF imports
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import warnings
warnings.filterwarnings('ignore', message='.*Compiled the loaded model.*')

import logging
logging.getLogger('absl').setLevel(logging.ERROR)

import numpy as np
import tensorflow as tf

# Add parent directory to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent / 'models' / 'violence_detection'))

from predict_violence import ViolenceDetectionPredictor


# ============================================================================
# CONFIGURATION & CONSTANTS
# ============================================================================

class ViolenceDetectionConfig:
    """Configuration for violence detection service"""
    
    # Model paths
    DEFAULT_MODEL_PATH = 'models/violence_detection/fine_tuned/final_model.h5'  # Updated to fine-tuned model
    FINETUNED_MODEL_PATH = 'models/violence_detection/fine_tuned/final_model.h5'
    BASE_MODEL_PATH = 'models/violence_detection/violence_detection_model_resnet.h5'
    
    # Content-type aware thresholds - EMERGENCY FIX for nature video FPs
    DEFAULT_THRESHOLD = 0.95  # Increased from 0.85 - only very high confidence
    SPORTS_ENTERTAINMENT_THRESHOLD = 0.95
    NEWS_EDUCATION_THRESHOLD = 0.95
    CRIME_FOOTAGE_THRESHOLD = 0.90  # Slightly lower for real crime footage
    GAMING_ANIMATION_THRESHOLD = 0.95  # All content treated equally
    
    # Minimum frame requirements - INCREASED EMERGENCY FIX
    DEFAULT_MIN_FRAMES = 8  # Increased from 5 - need much more evidence
    SPORTS_ENTERTAINMENT_MIN_FRAMES = 8
    OVERRIDE_MIN_FRAMES = 8  # Increased from 5
    
    # High-confidence override settings - EXTREMELY CONSERVATIVE
    HIGH_CONFIDENCE_OVERRIDE = 0.99  # Increased from 0.98 - only 99%+
    HIGH_CONFIDENCE_MIN_FRAMES = 20  # Increased from 15 - need MANY extreme frames
    OVERRIDE_FRAME_PERCENTAGE = 0.25  # Increased from 0.20 - 25% of frames must be extreme
    
    # Processing settings
    DEFAULT_BATCH_SIZE = 32
    FRAMES_PER_BATCH_NOTIFICATION = 1000


# ============================================================================
# SERVICE CLASS
# ============================================================================

class ViolenceDetectionService:
    """
    Service class for violence detection on video frames.
    
    Provides context-aware violence detection with dynamic thresholds based on:
    - Content type (sports, news, general, crime footage)
    - Transcript analysis
    - Metadata signals
    
    Attributes:
        model_path: Path to the trained violence detection model
        predictor: Violence detection predictor instance
        default_threshold: Default confidence threshold for violence detection
        high_confidence_override: Threshold for high-confidence override
    """
    
    def __init__(self, model_path: str = None):
        """
        Initialize the violence detection service.
        
        Args:
            model_path: Path to trained violence detection model
                       Defaults to fine-tuned model if available, otherwise base model
        """
        # Auto-detect best available model
        if model_path is None:
            finetuned_path = 'models/violence_detection/fine_tuned/final_model.h5'
            base_path = 'models/violence_detection/violence_detection_model_resnet.h5'
            
            if Path(finetuned_path).exists():
                model_path = finetuned_path
                print("✓ Using FINE-TUNED model (reduced false positives)")
            else:
                model_path = base_path
                print("⚠️  Fine-tuned model not found, using base model")
        
        print(f"Loading violence detection model from {model_path}...")
        
        self.model_path = model_path
        self.predictor = ViolenceDetectionPredictor(model_path, model_format='h5')
        
        # Initialize thresholds
        self.default_threshold = ViolenceDetectionConfig.DEFAULT_THRESHOLD
        self.high_confidence_override = ViolenceDetectionConfig.HIGH_CONFIDENCE_OVERRIDE
        
        print("✓ Violence detection service initialized")
    
    def analyze_video_frames(
        self,
        frame_paths: List[str],
        timestamps: Optional[List[float]] = None,
        batch_size: int = ViolenceDetectionConfig.DEFAULT_BATCH_SIZE,
        category_prediction: Optional[Dict[str, Any]] = None,
        transcript: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Analyze multiple frames from a video with context-aware detection.
        
        Args:
            frame_paths: List of paths to video frames
            timestamps: Optional list of timestamps (in seconds) for each frame
            batch_size: Number of frames to process in each batch
            category_prediction: Optional category prediction result for context
            transcript: Optional video transcript text for context analysis
            metadata: Optional video metadata (title, channel, description)
            
        Returns:
            Dictionary containing comprehensive analysis results:
            - is_violent: Boolean indicating overall violence detection
            - violence_percentage: Percentage of violent frames
            - violent_frame_count: Number of violent frames detected
            - severity: Severity level (NONE, LOW, MODERATE, HIGH, EXTREME)
            - timeline: Detailed frame-by-frame analysis
            - recommendation: Content recommendation string
        """
        print(f"Analyzing {len(frame_paths)} video frames...")
        
        # Handle empty input
        if not frame_paths:
            return self._create_empty_result()
        
        # Initialize timestamps if not provided
        if timestamps is None:
            timestamps = list(range(len(frame_paths)))
        
        # Detect content type and configure thresholds
        content_type = self._detect_content_type(
            category_prediction, transcript, metadata
        )
        threshold, min_frames = self._get_dynamic_threshold(content_type)
        
        print(f"Content type detected: {content_type}")
        print(f"Using threshold: {threshold*100:.1f}%, min frames: {min_frames}")
        
        # Process frames in batches
        all_results = self._process_frames_in_batches(
            frame_paths, timestamps, batch_size, threshold
        )
        
        # Handle processing failures
        if not all_results:
            print("Warning: No frames were successfully analyzed")
            return self._create_empty_result()
        
        # Analyze results
        return self._analyze_results(all_results, content_type, min_frames)
    
    def _create_empty_result(self) -> Dict[str, Any]:
        """Create empty result dictionary for edge cases"""
        return {
            'is_violent': False,
            'violence_percentage': 0.0,
            'violent_frame_count': 0,
            'total_frames': 0,
            'timeline': [],
            'violent_frame_timestamps': []
        }
    
    def _process_frames_in_batches(
        self,
        frame_paths: List[str],
        timestamps: List[float],
        batch_size: int,
        prediction_threshold: float
    ) -> List[Dict[str, Any]]:
        """
        Process frames in batches to manage memory usage.
        
        Args:
            frame_paths: List of frame paths to process
            timestamps: Corresponding timestamps for each frame
            batch_size: Number of frames per batch
            prediction_threshold: Initial prediction threshold (lower than final)
            
        Returns:
            List of prediction results with timestamps attached
        """
        all_results = []
        total_frames = len(frame_paths)
        
        for i in range(0, total_frames, batch_size):
            batch_paths = frame_paths[i:i + batch_size]
            batch_timestamps = timestamps[i:i + batch_size]
            
            try:
                # Use lower threshold for initial detection to catch more potential violence
                batch_predictions = self.predictor.predict_batch(
                    batch_paths,
                    threshold=prediction_threshold - 0.10,  # 10% lower for sensitivity
                    batch_size=batch_size
                )
                
                # Attach timestamps to predictions
                for pred, ts in zip(batch_predictions, batch_timestamps):
                    pred['timestamp'] = ts
                    all_results.append(pred)
                
                # Progress notification
                if (i + len(batch_paths)) % ViolenceDetectionConfig.FRAMES_PER_BATCH_NOTIFICATION == 0:
                    print(f"  Processed {i + len(batch_paths)}/{total_frames} frames")
                    
            except Exception as e:
                print(f"Error processing batch {i}: {e}")
                continue
        
        return all_results
    
    def _analyze_results(
        self,
        all_results: List[Dict[str, Any]],
        content_type: str,
        min_frames: int
    ) -> Dict[str, Any]:
        """
        Analyze prediction results to determine overall violence assessment.
        
        Args:
            all_results: List of all frame predictions
            content_type: Detected content type
            min_frames: Minimum number of violent frames required
            
        Returns:
            Comprehensive analysis result dictionary
        """
        # Get frame ratio threshold for this content type
        frame_ratio_threshold = self._get_frame_ratio_threshold(content_type)
        
        # Log configuration
        print(f"\n🔍 VIOLENCE THRESHOLD CONFIGURATION:")
        print(f"  Content type: {content_type}")
        print(f"  Confidence threshold: {self._get_threshold_for_content_type(content_type)*100:.1f}%")
        print(f"  Minimum frames: {min_frames}")
        print(f"  Frame ratio threshold: {frame_ratio_threshold*100:.1f}%")
        
        # Filter violent frames using custom threshold
        violent_frames = self._filter_violent_frames(all_results, content_type)
        non_violent_frames = [r for r in all_results if r not in violent_frames]
        
        # Calculate statistics
        violence_percentage = len(violent_frames) / len(all_results) if all_results else 0
        
        print(f"\n📊 FRAME ANALYSIS:")
        print(f"  Total frames analyzed: {len(all_results)}")
        print(f"  Violent frames: {len(violent_frames)} ({violence_percentage*100:.1f}%)")
        print(f"  Non-violent frames: {len(non_violent_frames)}")
        
        # Determine if violent (with high-confidence override)
        is_violent = self._assess_violence(
            violent_frames, all_results, content_type, min_frames, frame_ratio_threshold
        )
        
        # Calculate severity and scoring
        max_confidence, severity, violence_score = self._calculate_severity_and_scoring(
            violent_frames, content_type, violence_percentage, is_violent
        )
        
        # Build timeline
        timeline, violent_timestamps_final = self._build_timeline(
            all_results, content_type
        )
        
        # Generate recommendation
        recommendation = self._generate_recommendation(is_violent, severity, content_type)
        
        # Build final result
        return {
            'is_violent': is_violent,
            'violence_percentage': round(violence_percentage * 100, 2),
            'violent_frame_count': len(violent_frames),
            'non_violent_frame_count': len(non_violent_frames),
            'total_frames': len(all_results),
            'max_confidence': max_confidence,
            'severity': severity,
            'timeline': timeline,
            'violent_frame_timestamps': violent_timestamps_final,
            'recommendation': recommendation,
            'detection_config': {
                'content_type': content_type,
                'threshold_used': self._get_threshold_for_content_type(content_type),
                'min_frames_used': min_frames,
                'frame_ratio_threshold': frame_ratio_threshold
            }
        }
    
    def _filter_violent_frames(
        self,
        all_results: List[Dict[str, Any]],
        content_type: str
    ) -> List[Dict[str, Any]]:
        """Filter frames that exceed the violence threshold"""
        threshold = self._get_threshold_for_content_type(content_type)
        return [
            r for r in all_results
            if r.get('probability_violence', 0) >= threshold
        ]
    
    def _assess_violence(
        self,
        violent_frames: List[Dict[str, Any]],
        all_results: List[Dict[str, Any]],
        content_type: str,
        min_frames: int,
        frame_ratio_threshold: float
    ) -> bool:
        """
        Assess whether video contains violence with enhanced false positive prevention.
        
        Checks:
        1. NEW: Nature documentary false positive pattern detection
        2. High-confidence override (extremely obvious violence)
        3. Normal check (meets minimum count and ratio with temporal clustering)
        """
        total_frames = len(all_results)
        violent_count = len(violent_frames)
        
        # ─────────────────────────────────────────────────────────────
        # CRITICAL FIX #3: NATURE DOCUMENTARY FALSE POSITIVE DETECTION
        # ─────────────────────────────────────────────────────────────
        if violent_count > 0 and violent_count >= 15:
            # Analyze score distribution to detect nature FP pattern
            violence_scores = [r.get('probability_violence', 0) for r in all_results]
            
            # DEBUG: Print ALL scores to see what's happening
            print(f"\n🔍 DEBUG: Analyzing violence score distribution...")
            print(f"   Total frames: {len(all_results)}")
            print(f"   Violent frames (≥85%): {violent_count}")
            
            # Show score breakdown
            score_ranges = [
                ('<70%', sum(1 for s in violence_scores if s < 0.70)),
                ('70-84%', sum(1 for s in violence_scores if 0.70 <= s < 0.85)),
                ('85-91%', sum(1 for s in violence_scores if 0.85 <= s < 0.92)),
                ('92-95%', sum(1 for s in violence_scores if 0.92 <= s < 0.96)),
                ('96%+', sum(1 for s in violence_scores if s >= 0.96)),
            ]
            for range_name, count in score_ranges:
                print(f"   {range_name}: {count} frames")
            
            # Count frames in different confidence ranges
            moderate_high = sum(1 for s in violence_scores if 0.85 <= s < 0.92)  # 85-92%
            high = sum(1 for s in violence_scores if 0.92 <= s < 0.96)  # 92-96%
            extreme = sum(1 for s in violence_scores if s >= 0.96)  # 96%+
            
            print(f"\n   Nature FP Check:")
            print(f"   - Moderate-High (85-92%): {moderate_high} (need ≥10)")
            print(f"   - Extreme (96%+): {extreme} (need <5)")
            
            # Nature FP pattern: Many moderate-high scores, few extreme
            # Real violence: Many EXTREME scores (96%+) even if total % is lower
            # NEW: Check for "real violence" pattern - lots of 96%+ frames
            if extreme >= 50:
                print(f"\n✅ REAL VIOLENCE PATTERN DETECTED!")
                print(f"   {extreme} frames at 96%+ indicates genuine violent content")
                print(f"   (Wrestling, fights, action sequences show this pattern)")
                # Don't return - continue to extreme violence check below
            elif moderate_high >= 10 and extreme < 5:
                print(f"\n⚠️  NATURE FALSE POSITIVE PATTERN DETECTED!")
                print(f"   Score distribution suggests non-violent content:")
                print(f"   - Moderate-High (85-92%): {moderate_high} frames")
                print(f"   - High (92-96%): {high} frames")
                print(f"   - Extreme (96%+): {extreme} frames")
                print(f"   Nature/water/animal content typically shows this pattern")
                return False  # Override - likely false positive from nature scenes
            else:
                print(f"   ✓ Pattern does NOT match nature FP (continuing to normal checks)")
        
        # Check for high-confidence override
        high_conf_frames = [
            r for r in violent_frames
            if r.get('probability_violence', 0) >= self.high_confidence_override
        ]
        
        required_override_frames = max(
            ViolenceDetectionConfig.OVERRIDE_MIN_FRAMES,  # Now 10 frames
            int(len(all_results) * ViolenceDetectionConfig.OVERRIDE_FRAME_PERCENTAGE)
        )
        
        # CRITICAL FIX: Detect REAL violence by checking for many extreme (96%+) frames
        # Wrestling/fights have diluted % but MANY ultra-high confidence frames
        extreme_frames_count = sum(1 for r in all_results if r.get('probability_violence', 0) >= 0.96)
        
        if extreme_frames_count >= 50:
            print(f"\n🚨 EXTREME VIOLENCE DETECTED!")
            print(f"   {extreme_frames_count} frames at 96%+ confidence")
            print(f"   This indicates genuine violent content (wrestling, fights, action)")
            return True
        
        # Only trigger override if MANY frames are extremely high confidence
        # This prevents false positives from nature docs, music videos, etc.
        if len(high_conf_frames) >= required_override_frames and len(high_conf_frames) >= 15:
            print(f"\n🚨 HIGH-CONFIDENCE OVERRIDE TRIGGERED!")
            print(f"  {len(high_conf_frames)} frames exceeded {self.high_confidence_override*100:.1f}%")
            return True
        
        # Normal check: meets minimum count and ratio
        if len(violent_frames) >= min_frames and \
           len(violent_frames) / len(all_results) >= frame_ratio_threshold:
            return self._check_temporal_clustering(violent_frames, content_type)
        
        return False
    
    def _check_temporal_clustering(
        self,
        violent_frames: List[Dict[str, Any]],
        content_type: str
    ) -> bool:
        """
        Check if violent frames are temporally clustered (not random)
        
        Enhanced with sliding window verification to reduce false positives
        """
        violent_timestamps = sorted([
            float(r.get('timestamp', 0)) for r in violent_frames
        ])
        
        if len(violent_timestamps) < 2:
            return len(violent_frames) >= ViolenceDetectionConfig.DEFAULT_MIN_FRAMES
        
        # Calculate average gap between violent timestamps
        time_diffs = [
            violent_timestamps[i+1] - violent_timestamps[i]
            for i in range(len(violent_timestamps) - 1)
        ]
        avg_gap = sum(time_diffs) / len(time_diffs)
        
        # Different content types have SAME clustering requirements now
        max_allowed_gap = 5.0  # Reduced from 7.0 - tighter clustering for real violence
        
        # NEW: Sliding window verification
        # Check if violent frames cluster in short time windows (more robust than average)
        is_clustered_in_windows = self._verify_with_sliding_window(
            violent_timestamps, window_size=5.0, min_violent_frames=5  # Increased to reduce nature FPs
        )
        
        # Both conditions must be met for stronger verification
        is_clustered = avg_gap < max_allowed_gap and is_clustered_in_windows
        
        print(f"\n⏰ TEMPORAL CLUSTERING ANALYSIS:")
        sample_timestamps = violent_timestamps[:10]
        print(f"  Violent timestamps: {sample_timestamps}{'...' if len(violent_timestamps) > 10 else ''}")
        print(f"  Average gap: {avg_gap:.2f}s (max allowed: {max_allowed_gap:.1f}s)")
        print(f"  Sliding window check: {'PASS' if is_clustered_in_windows else 'FAIL'}")
        print(f"  Clustering result: {'PASS' if is_clustered else 'FAIL'}")
        
        return is_clustered
    
    def _verify_with_sliding_window(
        self,
        violent_timestamps: List[float],
        window_size: float = 5.0,
        min_violent_frames: int = 3
    ) -> bool:
        """
        Verify violence using sliding time windows
        
        This reduces false positives by ensuring violent frames are truly clustered,
        not just randomly distributed across the video.
        
        Args:
            violent_timestamps: Sorted list of violent frame timestamps
            window_size: Size of sliding window in seconds
            min_violent_frames: Minimum violent frames required in a window
            
        Returns:
            True if at least one window has enough violent frames
        """
        if len(violent_timestamps) < min_violent_frames:
            return False
        
        # Slide window through timestamps
        for i in range(len(violent_timestamps)):
            window_start = violent_timestamps[i]
            window_end = window_start + window_size
            
            # Count violent frames in this window
            frames_in_window = sum(
                1 for ts in violent_timestamps[i:]
                if ts <= window_end
            )
            
            # Found a valid cluster
            if frames_in_window >= min_violent_frames:
                return True
        
        return False
    
    def _calculate_severity_and_scoring(
        self,
        violent_frames: List[Dict[str, Any]],
        content_type: str,
        violence_percentage: float,
        is_violent: bool
    ) -> Tuple[float, str, float]:
        """Calculate severity level using multi-signal scoring"""
        if not violent_frames:
            return 0.0, "NONE", 0.0
        
        # Extract confidence scores
        max_confidence = max(r.get('probability_violence', 0) for r in violent_frames)
        avg_confidence = sum(r.get('probability_violence', 0) for r in violent_frames) / len(violent_frames)
        
        # Get context weight and scoring weights
        context_weight = self._calculate_context_weight(content_type, None, len(violent_frames))
        weights = self._get_scoring_weights(content_type)
        
        # Multi-signal scoring
        violence_score = (
            max_confidence * weights['max_confidence'] +
            avg_confidence * weights['avg_confidence'] +
            violence_percentage * weights['density'] +
            context_weight * weights['context']
        )
        
        # Calculate severity level
        severity = self._calculate_severity_level(
            violence_score, content_type, len(violent_frames), is_violent
        )
        
        print(f"\n✅ FINAL VIOLENCE ASSESSMENT:")
        print(f"  Is violent: {is_violent}")
        print(f"  Severity: {severity}")
        
        return max_confidence, severity, violence_score
    
    def _get_threshold_for_content_type(self, content_type: str) -> float:
        """Get appropriate threshold for content type"""
        thresholds = {
            'sports_entertainment': ViolenceDetectionConfig.SPORTS_ENTERTAINMENT_THRESHOLD,
            'news_education': ViolenceDetectionConfig.NEWS_EDUCATION_THRESHOLD,
            'crime_footage': ViolenceDetectionConfig.CRIME_FOOTAGE_THRESHOLD,
            'gaming_animation': ViolenceDetectionConfig.GAMING_ANIMATION_THRESHOLD,  # Added gaming support
        }
        return thresholds.get(content_type, ViolenceDetectionConfig.DEFAULT_THRESHOLD)
    
    def _get_dynamic_threshold(self, content_type: str) -> Tuple[float, int]:
        """Get threshold and minimum frames for content type"""
        config_map = {
            'sports_entertainment': (
                ViolenceDetectionConfig.SPORTS_ENTERTAINMENT_THRESHOLD,
                ViolenceDetectionConfig.SPORTS_ENTERTAINMENT_MIN_FRAMES
            ),
            'news_education': (
                ViolenceDetectionConfig.NEWS_EDUCATION_THRESHOLD,
                2
            ),
            'crime_footage': (
                ViolenceDetectionConfig.CRIME_FOOTAGE_THRESHOLD,
                2
            ),
            'gaming_animation': (  # Added gaming support
                ViolenceDetectionConfig.GAMING_ANIMATION_THRESHOLD,
                3  # Standard min frames for gaming
            ),
        }
        
        threshold, min_frames = config_map.get(
            content_type,
            (ViolenceDetectionConfig.DEFAULT_THRESHOLD, ViolenceDetectionConfig.DEFAULT_MIN_FRAMES)
        )
        
        return threshold, min_frames
    
    def _get_frame_ratio_threshold(self, content_type: str) -> float:
        """Get frame ratio threshold for content type"""
        # Frame ratio threshold - EMERGENCY FIX for nature video false positives
        return 0.35  # 35% required (increased from 25%) - nature videos often have scattered "violent-looking" frames but not this high
    
    # ─── CONTENT-TYPE-AWARE SCORING WEIGHTS ───
    SCORING_WEIGHTS = {
        'crime_footage': {
            'max_confidence': 0.40,
            'avg_confidence': 0.20,
            'density': 0.15,
            'context': 0.25
        },
        'news_education': {
            'max_confidence': 0.30,
            'avg_confidence': 0.25,
            'density': 0.20,
            'context': 0.25
        },
        'sports_entertainment': {
            'max_confidence': 0.25,
            'avg_confidence': 0.30,
            'density': 0.30,
            'context': 0.15
        },
        'general': {
            'max_confidence': 0.35,
            'avg_confidence': 0.25,
            'density': 0.20,
            'context': 0.20
        }
    }
    
    def _get_scoring_weights(self, content_type: str) -> Dict[str, float]:
        """Get scoring weight matrix for content type"""
        return self.SCORING_WEIGHTS.get(content_type, self.SCORING_WEIGHTS['general'])
    
    def _calculate_context_weight(
        self,
        content_type: str,
        transcript: Optional[str],
        violent_frame_count: int
    ) -> float:
        """
        Calculate context weight - NOW ALWAYS NEUTRAL
        No bias based on content type or transcript
        """
        # Fixed neutral weight - NO BIAS
        return 0.5
    
    def _calculate_severity_level(
        self,
        violence_score: float,
        content_type: str,
        violent_frame_count: int,
        is_violent: bool
    ) -> str:
        """Calculate severity level with content-type aware thresholds"""
        
        # Content-type specific severity thresholds - ALL EQUAL NOW
        THRESHOLDS = {
            'crime_footage': [
                (0.75, 'EXTREME'),
                (0.60, 'HIGH'),
                (0.45, 'MODERATE'),
                (0.30, 'LOW'),
            ],
            'news_education': [
                (0.75, 'EXTREME'),
                (0.60, 'HIGH'),
                (0.45, 'MODERATE'),
                (0.30, 'LOW'),
            ],
            'sports_entertainment': [  # NO SPECIAL TREATMENT - same as others
                (0.75, 'EXTREME'),
                (0.60, 'HIGH'),
                (0.45, 'MODERATE'),
                (0.30, 'LOW'),
            ],
            'general': [
                (0.75, 'EXTREME'),
                (0.60, 'HIGH'),
                (0.45, 'MODERATE'),
                (0.30, 'LOW'),
            ]
        }
        
        thresholds = THRESHOLDS.get(content_type, THRESHOLDS['general'])
        
        # If violence was detected, guarantee at least LOW severity
        if is_violent:
            for min_score, level in thresholds:
                if violence_score >= min_score:
                    return level
            return 'LOW'
        
        # Otherwise, must meet threshold
        for min_score, level in thresholds:
            if violence_score >= min_score:
                return level
        
        return 'NONE'
    
    def _build_timeline(
        self,
        all_results: List[Dict[str, Any]],
        content_type: str
    ) -> Tuple[List[Dict[str, Any]], List[float]]:
        """Build detailed timeline of frame analysis"""
        timeline = []
        violent_timestamps = []
        threshold = self._get_threshold_for_content_type(content_type)
        
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
            
            # Mark violent frames using consistent threshold
            if result.get('probability_violence', 0) >= threshold:
                violent_timestamps.append(result.get('timestamp', 0))
        
        return timeline, violent_timestamps
    
    def _generate_recommendation(
        self,
        is_violent: bool,
        severity: str,
        content_type: str
    ) -> str:
        """Generate content recommendation based on violence assessment"""
        
        if not is_violent or severity == 'NONE':
            return "No significant violence detected."
        
        # Direct violence warnings - no special treatment for any content type
        if severity == 'EXTREME':
            return "⚠️ EXTREME VIOLENCE DETECTED. Graphic content. Mature audiences only (18+)."
        
        elif severity == 'HIGH':
            return "⚠️ HIGH VIOLENCE DETECTED. Intense violent content. Viewer discretion advised (16+)."
        
        elif severity == 'MODERATE':
            return "⚠️ MODERATE VIOLENCE DETECTED. Violent content present. Parental guidance suggested (13+)."
        
        else:
            return "⚠️ MILD VIOLENCE DETECTED. Some violent content."
    
    # ─── CONTENT TYPE DETECTION ───
    
    def _detect_content_type(
        self,
        category_prediction: Optional[Dict[str, Any]],
        transcript: Optional[str],
        metadata: Optional[Dict[str, Any]]
    ) -> str:
        """
        Detect content type using multiple signals.
        
        Analyzes:
        - Category prediction from ML model
        - Transcript keywords
        - Metadata (title, channel)
        
        Returns:
            Content type string: 'sports_entertainment', 'news_education', 'crime_footage', or 'general'
        """
        print(f"\n🔍 CONTENT TYPE DETECTION:")
        print(f"  ℹ️  All content treated equally - no special treatment")
        # Always return 'general' - NO CONTEXT BIAS
        # Violence is violence regardless of content type
        return 'general'


# ============================================================================
# MAIN (TEST FUNCTION)
# ============================================================================

def main():
    """Test the violence detection service"""
    
    model_path = ViolenceDetectionConfig.DEFAULT_MODEL_PATH
    
    if not Path(model_path).exists():
        print(f"Model not found at {model_path}")
        return
    
    service = ViolenceDetectionService(model_path)
    
    # Test on sample frames if available
    test_frames = glob.glob('violence_frames/violence/*.jpg')[:10]
    
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
