"""
Unified Category Override Detection Framework
Provides standardized interface for all category-specific detectors
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class DetectionResult:
    """Standardized result from category detection"""
    is_detected: bool
    category: str
    confidence: float  # 0.0 to 1.0
    score: int  # 0-10 scale
    indicators: List[str]
    all_categories: List[Dict[str, float]]  # Updated category probabilities


class CategoryOverrideDetector(ABC):
    """
    Base class for all category-specific detectors
    
    All detectors must implement:
    - _get_keywords(): Return category-specific keywords
    - _get_channels(): Return known channels for this category
    - _check_duration(): Return duration-based score
    - _check_title_pattern(): Return title pattern score
    """
    
    def __init__(self, category_name: str, priority: int):
        self.category_name = category_name
        self.priority = priority  # Lower number = higher priority
        self.keywords = self._get_keywords()
        self.channels = self._get_channels()
        
    @abstractmethod
    def _get_keywords(self) -> List[str]:
        """Return list of keywords for this category"""
        pass
    
    @abstractmethod
    def _get_channels(self) -> List[str]:
        """Return list of channel names/organizations for this category"""
        pass
    
    def _check_keywords(self, text: str) -> Tuple[int, List[str]]:
        """
        Check text for category keywords
        Returns: (score 0-3, list of matched indicators)
        """
        text_lower = text.lower()
        matches = sum(1 for kw in self.keywords if kw in text_lower)
        
        indicators = []
        if matches >= 5:
            score = 3
            indicators.append(f'keywords ({matches})')
        elif matches >= 3:
            score = 2
            indicators.append(f'keywords ({matches})')
        elif matches >= 1:
            score = 1
            indicators.append(f'keywords ({matches})')
        else:
            score = 0
            
        return score, indicators
    
    def _check_channel(self, channel: str) -> Tuple[int, List[str]]:
        """
        Check if channel is known for this category
        Returns: (score 0-3, list of matched indicators)
        """
        channel_lower = channel.lower()
        
        for ch in self.channels:
            if ch in channel_lower:
                return 3, ['verified_channel']
        
        return 0, []
    
    def _check_duration(self, duration: int, min_optimal: int, max_optimal: int) -> Tuple[int, List[str]]:
        """
        Check if duration matches typical range for this category
        Returns: (score 0-2, list of matched indicators)
        """
        if not duration:
            return 0, []
        
        if min_optimal <= duration <= max_optimal:
            return 2, ['duration_match']
        elif min_optimal * 0.5 <= duration <= max_optimal * 1.5:
            return 1, ['duration_partial']
        else:
            return 0, []
    
    def _check_title_pattern(self, title: str, patterns: List[str]) -> Tuple[int, List[str]]:
        """
        Check if title matches common patterns for this category
        Returns: (score 0-2, list of matched indicators)
        """
        title_lower = title.lower()
        
        for pattern in patterns:
            if pattern in title_lower:
                return 2, ['title_pattern']
        
        # Check for partial matches
        words = title_lower.split()
        for pattern in patterns:
            if any(pattern in word for word in words):
                return 1, ['title_partial']
        
        return 0, []
    
    def detect(self, metadata: Dict, category_probs: List[Dict]) -> DetectionResult:
        """
        Main detection method - implements standardized 10-point scoring
        
        Args:
            metadata: Video metadata (title, channel, description, duration, tags)
            category_probs: Current category predictions from ML model
            
        Returns:
            DetectionResult with detection decision and updated probabilities
        """
        title = metadata.get('title', '')
        channel = metadata.get('channel', '')
        description = metadata.get('description', '')
        duration = metadata.get('duration', 0)
        tags = ' '.join(metadata.get('tags', []))
        
        # Combine text sources
        combined_text = f"{title} {description} {tags}"
        
        # Scoring components (total: 10 points)
        total_score = 0
        all_indicators = []
        
        # 1. Keyword analysis (0-3 points)
        keyword_score, keyword_indicators = self._check_keywords(combined_text)
        total_score += keyword_score
        all_indicators.extend(keyword_indicators)
        
        # 2. Channel authority (0-3 points)
        channel_score, channel_indicators = self._check_channel(channel)
        total_score += channel_score
        all_indicators.extend(channel_indicators)
        
        # 3. Duration pattern (0-2 points) - implemented by subclasses
        duration_score, duration_indicators = self._get_duration_score(duration)
        total_score += duration_score
        all_indicators.extend(duration_indicators)
        
        # 4. Title/format structure (0-2 points) - implemented by subclasses
        title_score, title_indicators = self._get_title_score(title)
        total_score += title_score
        all_indicators.extend(title_indicators)
        
        # Decision logic based on score
        if total_score >= 8:
            confidence = 0.98
            is_detected = True
        elif total_score >= 6:
            confidence = 0.95
            is_detected = True
        elif total_score >= 4:
            confidence = 0.80
            is_detected = True
        else:
            confidence = 0.0
            is_detected = False
        
        # Update category probabilities if detected
        if is_detected:
            updated_categories = self._boost_category(category_probs, confidence)
        else:
            updated_categories = category_probs
        
        return DetectionResult(
            is_detected=is_detected,
            category=self.category_name,
            confidence=confidence,
            score=total_score,
            indicators=all_indicators,
            all_categories=updated_categories
        )
    
    @abstractmethod
    def _get_duration_score(self, duration: int) -> Tuple[int, List[str]]:
        """Return duration-based score for this category"""
        pass
    
    @abstractmethod
    def _get_title_score(self, title: str) -> Tuple[int, List[str]]:
        """Return title pattern score for this category"""
        pass
    
    def _boost_category(self, category_probs: List[Dict], confidence: float) -> List[Dict]:
        """Boost this category's probability to specified confidence level"""
        updated = []
        
        for cat in category_probs:
            if cat['category'] == self.category_name:
                cat['probability'] = confidence
            else:
                # Reduce other categories proportionally
                cat['probability'] *= (1.0 - confidence)
            updated.append(cat)
        
        # Sort by probability descending
        updated.sort(key=lambda x: x['probability'], reverse=True)
        
        return updated
