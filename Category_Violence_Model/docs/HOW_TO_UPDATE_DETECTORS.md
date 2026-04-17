# How to Update Category Detectors with New Keyword Databases

## Overview

This guide shows you exactly how to update your category detectors to use the new external keyword database files instead of hardcoded keywords.

## Current State

Your `category_detectors.py` file has hardcoded keywords like this:

```python
class GamingDetector(CategoryOverrideDetector):
    def _get_keywords(self) -> List[str]:
        return [
            'gameplay', 'walkthrough', 'let\'s play', 'gaming', 'esports',
            'streamer', 'twitch', 'youtube gaming', 'live stream',
            'minecraft', 'fortnite', 'gta v', 'call of duty', 'league of legends',
            # ... limited to ~15 keywords
        ]
```

## Target State

After update, it will load 617+ keywords from an external file:

```python
class GamingDetector(KeywordLoaderMixin, CategoryOverrideDetector):
    def __init__(self):
        super().__init__()
        self.keyword_file = 'gaming_keywords.txt'
    
    def _get_keywords(self) -> List[str]:
        keywords = self.load_keywords_from_file(self.keyword_file)
        
        # Fallback if file loading fails
        if not keywords:
            keywords = [
                'gameplay', 'walkthrough', 'let\'s play', 'gaming',
                # ... original hardcoded list as backup
            ]
        
        return keywords
```

## Step-by-Step Implementation

### Step 1: Add KeywordLoaderMixin Import

Open `services/category_detectors.py` and update imports:

```python
"""
Category-Specific Override Detectors
Implements detection logic for all 16 YouTube categories
"""

from .category_override_detector import CategoryOverrideDetector
from .keyword_loader_mixin import KeywordLoaderMixin  # ADD THIS LINE
from typing import List, Tuple, Dict
```

### Step 2: Update FilmAnimationDetector

**Find this code** (lines 12-25):
```python
class FilmAnimationDetector(CategoryOverrideDetector):
    """Detects Film & Animation content"""
    
    def __init__(self):
        super().__init__('Film & Animation', priority=4)
    
    def _get_keywords(self) -> List[str]:
        return [
            'trailer', 'official trailer', 'clip', 'scene', 'movie', 'film',
            'animation', 'animated', 'cartoon', 'teaser', 'preview',
            'full movie', 'streaming now', 'in theaters', 'box office',
            'marvel', 'dc comics', 'pixar', 'disney', 'warner bros',
            'universal pictures', 'paramount', 'sony pictures'
        ]
```

**Replace with**:
```python
class FilmAnimationDetector(KeywordLoaderMixin, CategoryOverrideDetector):
    """Detects Film & Animation content"""
    
    def __init__(self):
        super().__init__('Film & Animation', priority=4)
        self.keyword_file = 'film_animation_keywords.txt'
    
    def _get_keywords(self) -> List[str]:
        """Load keywords from external file with fallback"""
        keywords = self.load_keywords_from_file(self.keyword_file)
        
        # Fallback if file loading fails
        if not keywords:
            keywords = [
                'trailer', 'official trailer', 'clip', 'scene', 'movie', 'film',
                'animation', 'animated', 'cartoon', 'teaser', 'preview',
                'full movie', 'streaming now', 'in theaters', 'box office',
                'marvel', 'dc comics', 'pixar', 'disney', 'warner bros',
                'universal pictures', 'paramount', 'sony pictures'
            ]
        
        return keywords
```

### Step 3: Update GamingDetector

**Find this code** (lines 50-63):
```python
class GamingDetector(CategoryOverrideDetector):
    """Detects Gaming content"""
    
    def __init__(self):
        super().__init__('Gaming', priority=5)
    
    def _get_keywords(self) -> List[str]:
        return [
            'gameplay', 'walkthrough', 'let\'s play', 'gaming', 'esports',
            'streamer', 'twitch', 'youtube gaming', 'live stream',
            'minecraft', 'fortnite', 'gta v', 'call of duty', 'league of legends',
            'valorant', 'apex legends', 'roblox', 'among us', 'overwatch',
            'boss fight', 'ending', 'secret', 'easter egg', 'review gameplay'
        ]
```

**Replace with**:
```python
class GamingDetector(KeywordLoaderMixin, CategoryOverrideDetector):
    """Detects Gaming content"""
    
    def __init__(self):
        super().__init__('Gaming', priority=5)
        self.keyword_file = 'gaming_keywords.txt'
    
    def _get_keywords(self) -> List[str]:
        """Load keywords from external file with fallback"""
        keywords = self.load_keywords_from_file(self.keyword_file)
        
        # Fallback if file loading fails
        if not keywords:
            keywords = [
                'gameplay', 'walkthrough', 'let\'s play', 'gaming', 'esports',
                'streamer', 'twitch', 'youtube gaming', 'live stream',
                'minecraft', 'fortnite', 'gta v', 'call of duty', 'league of legends',
                'valorant', 'apex legends', 'roblox', 'among us', 'overwatch',
                'boss fight', 'ending', 'secret', 'easter egg', 'review gameplay'
            ]
        
        return keywords
```

### Step 4: Update HowtoStyleDetector

**Find** (lines 88-101) and **replace**:
```python
class HowtoStyleDetector(KeywordLoaderMixin, CategoryOverrideDetector):
    """Detects Howto & Style content"""
    
    def __init__(self):
        super().__init__('Howto & Style', priority=10)
        self.keyword_file = 'howto_style_keywords.txt'
    
    def _get_keywords(self) -> List[str]:
        keywords = self.load_keywords_from_file(self.keyword_file)
        
        if not keywords:
            keywords = [
                'tutorial', 'how to', 'diy', 'guide', 'makeup', 'cooking',
                'recipe', 'hack', 'tips', 'tricks', 'fashion', 'style',
                'beauty', 'skincare', 'haircare', 'workout', 'fitness',
                'home decor', 'craft', 'knitting', 'sewing', 'painting',
                'life hack', '5-minute crafts', 'buzzfeed'
            ]
        
        return keywords
```

### Step 5: Update EducationDetector

**Find** (lines 126-139) and **replace**:
```python
class EducationDetector(KeywordLoaderMixin, CategoryOverrideDetector):
    """Detects Education content"""
    
    def __init__(self):
        super().__init__('Education', priority=6)
        self.keyword_file = 'education_keywords.txt'
    
    def _get_keywords(self) -> List[str]:
        keywords = self.load_keywords_from_file(self.keyword_file)
        
        if not keywords:
            keywords = [
                'lecture', 'course', 'learn', 'education', 'documentary',
                'explained', 'history', 'science', 'mathematics', 'physics',
                'chemistry', 'biology', 'literature', 'philosophy',
                'ted talk', 'crash course', 'khan academy', 'university',
                'college', 'school', 'academic', 'research', 'study'
            ]
        
        return keywords
```

### Step 6: Update ComedyDetector

**Find** (lines 166-178) and **replace**:
```python
class ComedyDetector(KeywordLoaderMixin, CategoryOverrideDetector):
    """Detects Comedy content"""
    
    def __init__(self):
        super().__init__('Comedy', priority=7)
        self.keyword_file = 'comedy_keywords.txt'
    
    def _get_keywords(self) -> List[str]:
        keywords = self.load_keywords_from_file(self.keyword_file)
        
        if not keywords:
            keywords = [
                'comedy', 'funny', 'humor', 'joke', 'parody', 'skit',
                'stand-up', 'prank', 'compilation', 'lol', 'hilarious',
                'laugh', 'comedic', 'satire', 'mockumentary', 'improv',
                'snl', 'saturday night live', 'conan', 'fallon'
            ]
        
        return keywords
```

### Step 7: Update MusicDetector

**Find** (lines 459-502) and **replace**:
```python
class MusicDetector(KeywordLoaderMixin, CategoryOverrideDetector):
    """Detects Music videos - official releases and artist content"""
    
    def __init__(self):
        super().__init__('Music', priority=1)
        self.keyword_file = 'music_keywords.txt'
    
    def _get_keywords(self) -> List[str]:
        keywords = self.load_keywords_from_file(self.keyword_file)
        
        if not keywords:
            keywords = [
                # Music video formats
                'official video', 'official music video', 'lyric video', 'audio',
                'feat.', 'ft.', 'featuring', 'prod. by', 'directed by',
                'album', 'single', 'ep', 'lp', 'remix', 'cover',
                'music video', 'mv', 'teaser', 'trailer', 'visualizer',
                
                # Top global artists (abbreviated)
                'taylor swift', 'ariana grande', 'justin bieber', 'ed sheeran',
                'beyoncé', 'rihanna', 'drake', 'the weeknd', 'dua lipa',
                
                # Music genres
                'pop', 'rock', 'hip hop', 'rap', 'r&b', 'country', 'jazz',
                'electronic', 'edm', 'house', 'techno', 'dubstep', 'trap',
            ]
        
        return keywords
```

### Step 8: Update NewsDetector

**Find** (lines 538-585) and **replace**:
```python
class NewsDetector(KeywordLoaderMixin, CategoryOverrideDetector):
    """Detects News & Politics content - breaking news and political coverage"""
    
    def __init__(self):
        super().__init__('News & Politics', priority=2)
        self.keyword_file = 'news_politics_keywords.txt'
    
    def _get_keywords(self) -> List[str]:
        keywords = self.load_keywords_from_file(self.keyword_file)
        
        if not keywords:
            keywords = [
                # Breaking news & urgency
                'breaking', 'news', 'report', 'update', 'live',
                'just in', 'developing', 'exclusive', 'investigation',
                
                # Conflict & military terminology
                'war', 'attack', 'drone', 'strike', 'military',
                'missile', 'bombing', 'artillery', 'combat', 'frontline',
                
                # Government & politics
                'president', 'government', 'minister', 'parliament',
                'congress', 'senate', 'prime minister', 'secretary',
            ]
        
        return keywords
```

## Testing Your Changes

### Test 1: Verify File Loading

Run this test to ensure keywords are loading:

```python
from services.category_detectors import GamingDetector

detector = GamingDetector()
keywords = detector._get_keywords()

print(f"✓ Loaded {len(keywords)} keywords")
assert len(keywords) > 100, "Should load many keywords from file"
print("✓ Test passed!")
```

### Test 2: Verify Detection Still Works

```python
from services.category_service import CategoryService

service = CategoryService()

result = service.predict_category(
    thumbnail_path='test.jpg',
    metadata={
        'title': 'Minecraft Survival Gameplay Episode 1',
        'description': 'Let\'s play Minecraft survival mode',
        'tags': ['gaming', 'minecraft', 'gameplay']
    }
)

print(f"Primary Category: {result['primary_category']}")
assert result['primary_category'] == 'Gaming', "Should detect gaming content"
print("✓ Detection test passed!")
```

### Test 3: Run Integration Script

```bash
python scripts\integrate_keyword_database.py
```

This will verify all keyword files are loading correctly.

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'keyword_loader_mixin'"

**Solution**: Make sure the mixin file exists:
```bash
ls services/keyword_loader_mixin.py
```

If it doesn't exist, create it by running:
```bash
python scripts\integrate_keyword_database.py
```

### Issue: Keywords Not Loading (Getting Fallback List)

**Check file path**:
```python
import os
from pathlib import Path

# Check if file exists
filepath = Path('wordlists/gaming_keywords.txt')
print(f"File exists: {filepath.exists()}")
print(f"Absolute path: {filepath.absolute()}")
```

### Issue: Performance Slowdown

If keyword loading is slow, add caching:

```python
class GamingDetector(KeywordLoaderMixin, CategoryOverrideDetector):
    def __init__(self):
        super().__init__('Gaming', priority=5)
        self.keyword_file = 'gaming_keywords.txt'
        self._cached_keywords = None
    
    def _get_keywords(self) -> List[str]:
        if self._cached_keywords is not None:
            return self._cached_keywords
        
        self._cached_keywords = self.load_keywords_from_file(self.keyword_file)
        
        if not self._cached_keywords:
            self._cached_keywords = [/* fallback list */]
        
        return self._cached_keywords
```

## Verification Checklist

After making all updates, verify:

- [ ] All 7 detectors inherit from `KeywordLoaderMixin`
- [ ] Each detector has `self.keyword_file` set in `__init__`
- [ ] `_get_keywords()` calls `load_keywords_from_file()`
- [ ] Fallback keywords present in case file loading fails
- [ ] Tests pass with expected keyword counts
- [ ] Detection accuracy maintained or improved

## Expected Results

After successful integration:

| Detector | Before | After | Improvement |
|----------|--------|-------|-------------|
| Film & Animation | ~15 keywords | 353 keywords | +2,253% |
| Gaming | ~20 keywords | 617 keywords | +2,985% |
| Howto & Style | ~20 keywords | 512 keywords | +2,460% |
| Education | ~15 keywords | 420 keywords | +2,700% |
| Comedy | ~15 keywords | 409 keywords | +2,627% |
| Music | ~50 keywords | 189 keywords | +278% |
| News & Politics | ~50 keywords | 172 keywords | +244% |

## Next Steps

1. ✅ Complete all 7 detector updates
2. ✅ Run comprehensive tests
3. ✅ Monitor detection accuracy
4. ⏳ Create keyword files for remaining 8 categories
5. ⏳ Implement performance optimizations
6. ⏳ Add analytics tracking

## Rollback Plan

If issues occur, you can easily rollback:

```bash
# Revert category_detectors.py to previous version
git checkout HEAD -- services/category_detectors.py
```

Or manually restore the original hardcoded keyword lists.

---

**Need Help?** 
- See `KEYWORD_QUICK_REFERENCE.md` for quick tips
- Check `KEYWORD_DATABASE_EXPANSION.md` for full documentation
- Run test script: `python scripts/integrate_keyword_database.py`
