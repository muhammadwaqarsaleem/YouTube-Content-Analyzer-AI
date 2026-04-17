# Violence Detection - Quick Reference Guide

## 🎯 Core Logic (New)

### One High-Confidence Frame = Violent Video
```
IF any frame has confidence >= 70% for violence:
    → Video is flagged as VIOLENT
    → Violence % = MAXIMUM confidence among violent frames
    → Severity based on highest confidence level
```

### Confidence Thresholds
| Level | Confidence | Meaning |
|-------|-----------|---------|
| **EXTREME** | ≥95% | Extremely clear violence |
| **HIGH** | ≥85% | Very definite violence |
| **MODERATE** | ≥75% | Clear violence |
| **LOW** | ≥70% | Mild but detectable |
| **NONE** | <70% | Filtered as noise |

---

## 📊 Examples

### Example 1: Single Violent Frame
**Input**: 50 frames analyzed
- Frame 1-49: Clean (no violence)
- Frame 50: **92% confidence violence**

**Output**:
```
✅ is_violent: TRUE
✅ violence_percentage: 92%
✅ severity: HIGH
✅ Frontend shows: "92% VIOLENCE DETECTED"
```

---

### Example 2: Multiple Low-Confidence Frames
**Input**: 50 frames analyzed
- All frames: 50-65% confidence (below 70% threshold)

**Output**:
```
✅ is_violent: FALSE
✅ violence_percentage: 0%
✅ severity: NONE
✅ Frontend shows: "NO VIOLENCE DETECTED"
```

---

### Example 3: Mixed Confidence
**Input**: 50 frames analyzed
- 10 frames: 55% confidence (filtered out)
- 3 frames: 85-88% confidence (counted)
- 37 frames: Clean

**Output**:
```
✅ is_violent: TRUE (3 high-conf frames)
✅ violence_percentage: 88% (max confidence)
✅ severity: HIGH
✅ Frontend shows: "88% VIOLENCE DETECTED"
```

---

## 🔧 Configuration

### Adjust Sensitivity Threshold
File: `services/violence_service.py` (line 33)

```python
self.threshold = 0.7  # Change this value (0.5-0.95)
```

**Recommended Values**:
- **0.50-0.60**: Very sensitive (more false positives)
- **0.65-0.75**: Balanced (current setting)
- **0.80-0.95**: Very strict (may miss some violence)

---

## 🖥️ Frontend Display

### What Users See
```
🛡️ Violence Detection

[XX% VIOLENCE DETECTED] or [NO VIOLENCE DETECTED]
[SEVERITY BADGE]

Detection Confidence: XX.X%
Assessment: [Content recommendation message]
```

### What Was Removed
❌ Violent frames count  
❌ Total frames count  
❌ Frame-by-frame timeline  
❌ Percentage calculations  

**Why**: Too detailed, burdens users with information they don't need.

---

## 🧪 Testing

### Run Test Script
```bash
python scripts/test_violence_optimization.py
```

**Expected Output**:
```
✅ TEST 1: High-Confidence Single Frame - PASSED
✅ TEST 2: Multiple Low-Confidence Frames - PASSED
✅ TEST 3: Mixed Confidence Levels - PASSED
```

---

## 📈 Performance Impact

### Before Optimization
- Analyzed all frames equally
- Counted low-confidence detections
- Showed detailed timeline
- Violence % = frame percentage

### After Optimization
- Filters low-confidence noise (<70%)
- Focuses on high-certainty detections
- Shows one clear prediction
- Violence % = max confidence

**Result**: Cleaner results, fewer false positives, better UX.

---

## 🚨 Troubleshooting

### Issue: Too Many False Positives
**Solution**: Increase threshold in `violence_service.py`
```python
self.threshold = 0.8  # More strict
```

### Issue: Missing Violence
**Solution**: Decrease threshold (not recommended below 0.6)
```python
self.threshold = 0.6  # More sensitive
```

### Issue: Frontend Shows Old Format
**Solution**: Clear browser cache and reload
```
Ctrl + Shift + R (Windows)
Cmd + Shift + R (Mac)
```

---

## 📝 Key Files

| File | Purpose |
|------|---------|
| `services/violence_service.py` | Core detection logic |
| `frontend/index.html` | UI structure |
| `frontend/dashboard.js` | Display logic |
| `services/analysis_aggregator.py` | Data formatting |
| `scripts/test_violence_optimization.py` | Verification tests |

---

## 🎯 Success Metrics

✅ **Accuracy**: Only real violence detected (≥70% confidence)  
✅ **Clarity**: One big prediction, no confusion  
✅ **Actionability**: Clear recommendations for users  
✅ **Performance**: Fast analysis, simplified output  
✅ **UX**: Users understand results instantly  

---

## 🌐 Server Access

- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

---

**Last Updated**: March 4, 2026  
**Version**: v2.0 Optimized
