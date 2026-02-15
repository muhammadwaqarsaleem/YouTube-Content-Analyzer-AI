# 🚀 Quick Reference Card - Batch Labeling Pipeline

## Installation & Setup

```bash
# 1. Install dependencies (same as EDA)
pip install pandas numpy tqdm

# 2. Place files in same folder
youtube_data.csv
batch_labeling_pipeline.py

# 3. Run
python batch_labeling_pipeline.py
```

## File Structure

```
your-project/
├── youtube_data.csv                    # Input (your raw data)
├── batch_labeling_pipeline.py          # Main script
├── training_ready_dataset.csv          # Output (labeled data)
├── validate_labels.py                  # QA script (optional)
└── label_quality_report.txt            # Validation report
```

## Command Cheatsheet

```bash
# Basic run
python batch_labeling_pipeline.py

# Validate output
python validate_labels.py

# Check Python version
python --version

# List installed packages
pip list | grep -E "(pandas|numpy|tqdm)"
```

## Key Metrics to Monitor

### Label Distribution (Expected)
- **General:** 50-70%
- **Teen:** 20-35%
- **Mature:** 5-15%

### Red Flags
- ⚠️ General < 40% → Thresholds too aggressive
- ⚠️ Mature > 20% → Thresholds too loose
- ⚠️ All one category → Keywords/thresholds broken

## Classification Thresholds

**Current Defaults:**
```python
Zero Tolerance: ≥ 1 occurrence → Mature
Mature Density: ≥ 1.5 per 1K words → Mature  
Teen Density:   ≥ 2.5 per 1K words → Teen
Else:           → General
```

**Common Adjustments:**
```python
# More strict (fewer Mature/Teen)
Mature: 1.5 → 2.5
Teen:   2.5 → 4.0

# Less strict (more Mature/Teen)
Mature: 1.5 → 1.0
Teen:   2.5 → 1.5
```

## Output Columns Reference

| Column | Type | Description |
|--------|------|-------------|
| `Age_Label` | string | G / T / M classification |
| `Severity_Score` | int | Total keyword matches |
| `Mature_Density` | float | Mature keywords per 1K words |
| `Teen_Density` | float | Teen keywords per 1K words |
| `Transcript_Word_Count` | int | Total words |
| `Content_Flags` | string | Timestamped categories |

## Content Flags Format

```
Violence_Reference: 3 (approx min 2, min 14, min 21) | 
Strong_Language: 5 (approx min 1, min 8, min 15)
```

**Flag Types:**
- Violence_Reference
- Strong_Language  
- Substance_Reference
- Sexual_Content
- Gaming_Content
- Mild_Language

## Common Issues & Fixes

### "File not found"
```python
# Check path in script (line ~775)
INPUT_CSV_PATH = "youtube_data.csv"
```

### "Transcript column not found"
```python
# Check your column name
df.columns  # In Python
# Update script if different
```

### Script runs slow
- **Expected:** 40-50 transcripts/sec
- **If slower:** Check antivirus, close other apps
- **6K transcripts:** ~2-3 minutes

### Memory error
```python
# Process in chunks (modify script)
# Or use machine with more RAM
```

## Quick Data Checks

### Python (Quick Checks)

```python
import pandas as pd

# Load labeled data
df = pd.read_csv('training_ready_dataset.csv')

# Check distribution
print(df['Age_Label'].value_counts(normalize=True))

# Check high severity
print(df[df['Severity_Score'] > 100].shape)

# Sample Mature labels
print(df[df['Age_Label'] == 'Mature'].sample(5)[
    ['title', 'Severity_Score', 'Content_Flags']
])

# Check density ranges
print(df.groupby('Age_Label')[
    ['Mature_Density', 'Teen_Density']
].mean())
```

## Customization Shortcuts

### Change thresholds (line ~455)
```python
elif mature_density >= 1.5:  # Change this number
    label = 'Mature'
elif teen_density >= 2.5:    # Change this number
    label = 'Teen'
```

### Add keywords (lines 24-141)
```python
GENERAL_KEYWORDS = {
    # ... existing ...
    'your_new_keyword', 'another_keyword',
}
```

### Change output filename (line ~776)
```python
OUTPUT_CSV_PATH = "my_custom_name.csv"
```

## Validation Checklist

After labeling, check:

- [ ] Output file exists and is ~same size as input
- [ ] All original columns present
- [ ] 6 new columns added
- [ ] Label distribution reasonable
- [ ] No all-NaN columns
- [ ] Severity scores align with labels
- [ ] Content flags formatted correctly
- [ ] Manually review 10-20 samples

## Next Steps After Labeling

1. **Validate:** `python validate_labels.py`
2. **Spot-check:** Review random samples per category
3. **Train/test split:** Stratified by Age_Label
4. **Model training:** Use Age_Label as target
5. **Evaluate:** Check performance on test set

## Performance Benchmarks

| Dataset Size | Expected Time | RAM Usage |
|--------------|---------------|-----------|
| 1,000 rows   | ~25 seconds   | ~500 MB   |
| 6,000 rows   | ~2.5 minutes  | ~1 GB     |
| 16,000 rows  | ~7 minutes    | ~2 GB     |
| 50,000 rows  | ~20 minutes   | ~3 GB     |

## Directory Safety

**DO commit to GitHub:**
- ✅ `batch_labeling_pipeline.py`
- ✅ `validate_labels.py`
- ✅ `README.md` files
- ✅ `requirements.txt`

**DON'T commit:**
- ❌ `youtube_data.csv` (raw data)
- ❌ `training_ready_dataset.csv` (labeled data)
- ❌ `*.txt` reports
- ❌ Large files (>100MB)

## Emergency Commands

```bash
# Stop running script
Ctrl+C

# Check if script is running
# Windows:
tasklist | findstr python
# Mac/Linux:
ps aux | grep python

# Kill frozen process
# Windows:
taskkill /F /IM python.exe
# Mac/Linux:
killall python
```

## Getting Help

1. Check LABELING_README.md
2. Check CUSTOMIZATION_GUIDE.md
3. Run validation script
4. Review error messages carefully
5. Check GitHub issues
6. Contact team lead

## Key Files by Purpose

| Need to... | Use this file |
|------------|---------------|
| Run labeling | `batch_labeling_pipeline.py` |
| Validate results | `validate_labels.py` |
| Learn usage | `LABELING_README.md` |
| Customize | `CUSTOMIZATION_GUIDE.md` |
| Quick help | `QUICK_REFERENCE.md` (this file) |

## Research Documentation

**For your paper, report:**
- Total samples labeled: [number]
- Label distribution: G: [%], T: [%], M: [%]
- Thresholds used: Mature ≥ [X], Teen ≥ [Y]
- Keywords per category: G: [N], T: [N], M: [N]
- Processing time: [X] minutes
- Validation accuracy: [%] (if spot-checked)

---

**Quick tip:** Bookmark this file for fast reference during labeling runs! 📌
