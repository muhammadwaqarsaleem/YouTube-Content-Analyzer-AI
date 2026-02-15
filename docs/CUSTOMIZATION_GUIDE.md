# 🔧 Customization Guide: Tuning the Labeling Pipeline

This guide shows you how to customize the weak supervision pipeline for your specific research needs.

## 📊 Adjusting Classification Thresholds

### Understanding Current Thresholds

**Location:** `batch_labeling_pipeline.py`, lines ~450-460 in `label_transcript()` method

**Current Logic:**
```python
if zero_tolerance_count >= 1:
    label = 'Mature'
elif mature_density >= 1.5:      # 1.5 keywords per 1,000 words
    label = 'Mature'
elif teen_density >= 2.5:        # 2.5 keywords per 1,000 words
    label = 'Teen'
else:
    label = 'General'
```

### When to Adjust Thresholds

**Make thresholds MORE STRICT (higher numbers):**
- You're getting too many Mature/Teen labels
- You want higher confidence in classifications
- Your content is generally more edgy (gaming channel, comedy)

**Make thresholds LESS STRICT (lower numbers):**
- You're missing obvious Mature/Teen content
- You want higher recall (catch more positives)
- Your content is generally more conservative (educational, news)

### Example Adjustments

#### 1. Conservative (Stricter) Thresholds

Use when your dataset skews toward mature content:

```python
# In label_transcript() method
if zero_tolerance_count >= 1:
    label = 'Mature'
elif mature_density >= 2.5:      # Increased from 1.5
    label = 'Mature'
elif teen_density >= 4.0:        # Increased from 2.5
    label = 'Teen'
else:
    label = 'General'
```

**Effect:**
- Fewer Mature labels (maybe 5-8% instead of 10-12%)
- More Teen content classified as General
- Higher precision, lower recall

#### 2. Aggressive (Looser) Thresholds

Use when your dataset is predominantly family-friendly:

```python
# In label_transcript() method
if zero_tolerance_count >= 1:
    label = 'Mature'
elif mature_density >= 1.0:      # Decreased from 1.5
    label = 'Mature'
elif teen_density >= 1.5:        # Decreased from 2.5
    label = 'Teen'
else:
    label = 'General'
```

**Effect:**
- More Mature labels (maybe 15-20%)
- More General content classified as Teen
- Higher recall, lower precision

#### 3. Domain-Specific: Gaming Content

Gaming videos have high "teen" keyword density naturally:

```python
# Adjust Teen threshold up, keep Mature the same
if zero_tolerance_count >= 1:
    label = 'Mature'
elif mature_density >= 1.5:
    label = 'Mature'
elif teen_density >= 3.5:        # Increased from 2.5
    label = 'Teen'
else:
    label = 'General'
```

#### 4. Multi-Tier Mature Classification

Add sub-categories within Mature:

```python
# Create Mature-Mild vs Mature-Extreme
if zero_tolerance_count >= 1:
    label = 'Mature-Extreme'
elif mature_density >= 3.0:
    label = 'Mature-Extreme'
elif mature_density >= 1.5:
    label = 'Mature-Mild'
elif teen_density >= 2.5:
    label = 'Teen'
else:
    label = 'General'
```

### Threshold Calibration Process

**Step 1: Run with default thresholds**
```bash
python batch_labeling_pipeline.py
```

**Step 2: Analyze distribution**
```bash
python validate_labels.py
```

**Step 3: Manual review**
```python
import pandas as pd

df = pd.read_csv('training_ready_dataset.csv')

# Check samples near boundaries
near_mature_boundary = df[
    (df['Mature_Density'] >= 1.3) & 
    (df['Mature_Density'] <= 1.7)
].sample(10)

for idx, row in near_mature_boundary.iterrows():
    print(f"Label: {row['Age_Label']}")
    print(f"Density: {row['Mature_Density']:.3f}")
    print(f"Title: {row['title']}")
    print("-" * 80)
```

**Step 4: Adjust based on findings**
- If boundary samples look mis-classified, adjust threshold
- Aim for ~80-85% accuracy on manual review

**Step 5: Re-run and validate**

---

## 🔤 Customizing Keyword Dictionaries

### Adding Domain-Specific Keywords

**Location:** `batch_labeling_pipeline.py`, lines 24-141

#### Example: Medical/Health Content

Add medical terminology to General keywords:

```python
GENERAL_KEYWORDS = {
    # ... existing keywords ...
    
    # Medical/Health (add these)
    'medical', 'medicine', 'doctor', 'physician', 'nurse', 'hospital',
    'clinic', 'treatment', 'diagnosis', 'therapy', 'patient', 'healthcare',
    'disease', 'illness', 'recovery', 'healing', 'prevention', 'vaccine',
    'anatomy', 'physiology', 'surgery', 'medication', 'prescription',
    'wellness program', 'public health', 'epidemiology', 'nutrition science'
}
```

#### Example: Technology Reviews

Add tech terminology to General keywords:

```python
GENERAL_KEYWORDS = {
    # ... existing keywords ...
    
    # Technology Reviews
    'smartphone', 'laptop', 'tablet', 'computer', 'processor', 'CPU', 'GPU',
    'RAM', 'storage', 'battery', 'display', 'camera', 'performance',
    'benchmark', 'specs', 'features', 'review', 'unboxing', 'comparison',
    'software', 'hardware', 'upgrade', 'tech specs', 'build quality'
}
```

#### Example: Sports Content

Differentiate sports action from violence:

```python
# Remove sports terms from Teen/Mature if present
# Add to General instead
GENERAL_KEYWORDS = {
    # ... existing keywords ...
    
    # Sports (Positive/Neutral Context)
    'sports', 'athlete', 'championship', 'tournament', 'competition',
    'team', 'player', 'coach', 'training', 'practice', 'score', 'goal',
    'touchdown', 'homerun', 'basket', 'football', 'basketball', 'baseball',
    'soccer', 'tennis', 'golf', 'olympics', 'sportsmanship', 'victory'
}
```

### Removing Problematic Keywords

If certain keywords cause false positives:

```python
# Example: Remove "kill" from Mature if it appears in gaming context
MATURE_KEYWORDS = {
    # ... other keywords ...
    # 'kill',  # Commented out - too common in gaming
    'murder', 'killing spree',  # Keep specific violent phrases
    # ... rest of keywords ...
}

# Add gaming-specific "kill" to Teen instead
TEEN_KEYWORDS = {
    # ... other keywords ...
    'kill streak', 'killing it', 'killer', 'kills',  # Gaming context
}
```

### Language-Specific Dictionaries

For multilingual datasets:

```python
# Create separate dictionaries per language
GENERAL_KEYWORDS_ES = {  # Spanish
    'educación', 'aprender', 'enseñar', 'familia', 'niños', ...
}

GENERAL_KEYWORDS_FR = {  # French
    'éducation', 'apprendre', 'enseigner', 'famille', 'enfants', ...
}

# Then in __init__, detect language and use appropriate dict
# (Requires language detection integration)
```

### Weighted Keywords (Advanced)

Give different keywords different weights:

```python
# Instead of just counting, apply weights
MATURE_KEYWORDS_WEIGHTED = {
    'fuck': 2.0,      # Strong profanity = 2x weight
    'damn': 0.5,      # Mild profanity = 0.5x weight
    'murder': 3.0,    # Extreme violence = 3x weight
    'fight': 0.3,     # Common word = 0.3x weight
}

# Then in _calculate_keyword_matches(), track weights
# (Requires code modification)
```

---

## 🎯 Advanced Customizations

### 1. Context-Aware Classification

Add logic to check keyword context:

```python
def _is_educational_context(self, text: str, keyword_pos: int) -> bool:
    """Check if keyword appears in educational context."""
    # Get surrounding text (±100 chars)
    start = max(0, keyword_pos - 100)
    end = min(len(text), keyword_pos + 100)
    context = text[start:end].lower()
    
    # Check for educational indicators
    edu_indicators = ['learn', 'teach', 'explain', 'understand', 'tutorial']
    return any(indicator in context for indicator in edu_indicators)

# Then in label_transcript():
# Filter out mature keywords in educational context
```

### 2. Multi-Signal Ensemble

Combine keyword density with other signals:

```python
def label_transcript_ensemble(self, transcript: str, metadata: dict) -> str:
    """Use multiple signals for classification."""
    
    # Signal 1: Keyword density (existing)
    keyword_label = self.label_transcript(transcript)
    
    # Signal 2: Channel reputation (if available)
    channel_rating = metadata.get('channel_rating', 'Unknown')
    
    # Signal 3: Video metadata
    title = metadata.get('title', '').lower()
    description = metadata.get('description', '').lower()
    
    # Combine signals
    if channel_rating == 'Verified-FamilyFriendly':
        # Trust channel, downgrade from Mature to Teen if borderline
        if keyword_label == 'Mature' and mature_density < 2.0:
            return 'Teen'
    
    # Check title/description for explicit warnings
    if 'explicit' in title or 'nsfw' in title:
        return 'Mature'  # Override keyword-based label
    
    return keyword_label
```

### 3. Time-Based Analysis

Weight keywords by position in transcript:

```python
def _calculate_weighted_density(self, matches: List[Tuple[str, int]], 
                                total_words: int) -> float:
    """
    Weight keywords more heavily if they appear early.
    Rationale: Thumbnails/intros set content expectations.
    """
    weighted_count = 0
    
    for keyword, char_pos in matches:
        # Calculate position as fraction of transcript
        word_pos = char_pos / 6  # Approx chars per word
        position_fraction = word_pos / total_words
        
        # Weight: 1.5x for first 10%, 1.0x thereafter
        if position_fraction < 0.10:
            weight = 1.5
        else:
            weight = 1.0
        
        weighted_count += weight
    
    # Calculate density with weighted count
    return (weighted_count / total_words) * 1000
```

### 4. Confidence Scores

Add confidence to predictions:

```python
def label_transcript_with_confidence(self, transcript: str) -> Tuple[str, float]:
    """Return label and confidence score (0-1)."""
    
    label, severity, mature_dens, teen_dens, flags = self.label_transcript(transcript)
    
    # Calculate confidence based on distance from threshold
    if label == 'Mature':
        if zero_tolerance_count > 0:
            confidence = 1.0  # Certain
        else:
            # Distance above threshold
            distance = mature_dens - 1.5
            confidence = min(0.99, 0.7 + (distance * 0.1))
    
    elif label == 'Teen':
        distance = teen_dens - 2.5
        confidence = min(0.99, 0.6 + (distance * 0.1))
    
    else:  # General
        # Lower density = higher confidence
        max_density = max(mature_dens, teen_dens)
        confidence = max(0.5, 1.0 - (max_density * 0.2))
    
    return label, confidence

# Then filter by confidence
# df = df[df['Confidence'] > 0.7]  # Keep only high-confidence labels
```

---

## 🧪 Testing Your Customizations

### Create a Test Set

```python
# test_customizations.py
import pandas as pd
from batch_labeling_pipeline import WeakSupervisionLabeler

# Create test cases with known labels
test_data = pd.DataFrame({
    'transcript': [
        # Should be General
        "This tutorial explains how to code in Python. Learn the basics of programming.",
        
        # Should be Teen
        "Epic gaming montage! This Fortnite gameplay is insane. Clutch wins all day bro!",
        
        # Should be Mature
        "This shit is fucking crazy. Gore everywhere, blood and violence non-stop."
    ],
    'expected_label': ['General', 'Teen', 'Mature']
})

# Save to CSV
test_data.to_csv('test_samples.csv', index=False)

# Run labeling
labeler = WeakSupervisionLabeler('test_samples.csv', 'test_output.csv')
labeler.load_and_clean_data()
labeler.process_batch_labeling()

# Check accuracy
results = pd.read_csv('test_output.csv')
accuracy = (results['Age_Label'] == results['expected_label']).mean()
print(f"Test Accuracy: {accuracy * 100:.1f}%")
```

### A/B Test Different Thresholds

```python
# threshold_comparison.py
import pandas as pd

thresholds = [
    {'name': 'Conservative', 'mature': 2.5, 'teen': 4.0},
    {'name': 'Default', 'mature': 1.5, 'teen': 2.5},
    {'name': 'Aggressive', 'mature': 1.0, 'teen': 1.5}
]

for config in thresholds:
    print(f"\nTesting {config['name']} thresholds...")
    
    # Modify thresholds in code (requires parameterization)
    # Run labeling
    # Compare distributions
```

---

## 📝 Best Practices Checklist

Before finalizing your customizations:

- [ ] Document all threshold changes in comments
- [ ] Keep original values commented out for reference
- [ ] Test on a small sample first (~100 rows)
- [ ] Manually review 20-30 samples per category
- [ ] Check label distribution is reasonable (60-70% General)
- [ ] Validate severity scores align with labels
- [ ] Run validation script after each major change
- [ ] Track accuracy on manually labeled subset
- [ ] Version control your changes (Git commit messages)
- [ ] Update README with your custom thresholds

---

## 🔄 Reverting Changes

If customizations don't work:

```bash
# If using Git
git checkout batch_labeling_pipeline.py

# Or manually restore from backup
cp batch_labeling_pipeline.py.backup batch_labeling_pipeline.py
```

**Always keep backups:**
```bash
cp batch_labeling_pipeline.py batch_labeling_pipeline.py.backup
```

---

## 📞 Need Help?

If you're unsure about customizations:

1. Start with small threshold adjustments (±0.5)
2. Test on 10% sample before full dataset
3. Consult domain experts for keyword lists
4. Review related research papers for benchmark thresholds
5. Document your decision-making process for publications

**Remember:** Weak supervision is inherently noisy. Aim for 80-85% accuracy, not perfection!
