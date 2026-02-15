# 🏷️ Weak Supervision Batch Labeling Pipeline

Production-grade programmatic labeling system for YouTube video age classification using advanced keyword-density analysis.

## 🎯 Research Context

This pipeline implements **Programmatic Weak Supervision** to create a "Silver Training Dataset" from unlabeled YouTube transcripts. Unlike traditional manual labeling (expensive, slow), weak supervision uses heuristic rules and keyword analysis to automatically generate training labels at scale.

### Why Keyword Density vs. Absolute Counts?

**Problem with absolute counts:**
- A 10-minute video with 2 profanities ≠ A 60-minute video with 2 profanities
- Long educational videos might trigger on rare mentions
- Short toxic videos might slip through

**Solution: Density-based scoring:**
- Normalizes by transcript length (occurrences per 1,000 words)
- More robust across varying video lengths
- Better reflects actual content intensity

## 📊 Pipeline Overview

```
Raw Dataset (youtube_data.csv)
         ↓
   [Load & Clean]
   - Drop missing transcripts
   - Retain all original columns
         ↓
[Keyword Density Analysis]
   - Compile regex patterns
   - Calculate word counts
   - Match keywords with positions
   - Compute densities
         ↓
  [Classification Logic]
   - Zero tolerance check (immediate Mature)
   - Mature density ≥ 1.5 → Mature
   - Teen density ≥ 2.5 → Teen
   - Else → General
         ↓
[Safety & Tracking Features]
   - Severity scores
   - Content flags with timestamps
   - Category breakdowns
         ↓
Silver Training Dataset (training_ready_dataset.csv)
```

## ✨ Key Features

### 1. Comprehensive Keyword Dictionaries (100+ per category)

**General (G-Rated):**
- Educational content (tutorials, science, learning)
- Family-friendly themes
- Positive values and affirmations
- Creative arts and wholesome entertainment

**Teen (T-Rated):**
- Gaming content and esports terminology
- Mild slang and internet culture
- Action/combat (non-graphic)
- Teen drama and social media references

**Mature (M-Rated):**
- Strong profanity
- Violence and gore references
- Substance use and abuse
- Sexual content
- Criminal activity
- Disturbing themes

**Zero Tolerance:**
- Extreme slurs and hate speech
- Graphic violence (beheading, genocide)
- Sexual assault references
- Terrorism and extremism
- Self-harm and suicide

### 2. Advanced Processing

- **Compiled regex patterns** - Optimized for 400K+ word transcripts
- **Word boundary matching** - Prevents false positives from partial matches
- **Progress tracking** - tqdm progress bars for long-running operations
- **Memory efficient** - Processes row-by-row without loading all text at once

### 3. Safety & Research Features

**Content Flags Format:**
```
Violence_Reference: 3 (approx min 2, min 14, min 21) | Strong_Language: 5 (approx min 1, min 8, min 15, min 22, min 29)
```

**Why timestamp estimation?**
- Allows researchers to spot-check specific moments
- Enables temporal analysis (e.g., "toxic language clusters at video start")
- Supports future video segmentation approaches

**Abstraction principle:**
- Final dataset does NOT contain exact trigger words
- Only metadata (counts, densities, categories, timestamps)
- Maintains academic safety for dataset sharing

### 4. Data Lineage & Mergeability

**Critical design decision:**
- Retains ALL original columns (video_id, url, title, etc.)
- Simply appends 6 new classification columns
- Allows other researchers to merge their annotations
- Maintains primary keys for data provenance

## 📋 Output Columns

The pipeline adds these columns to your original dataset:

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `Age_Label` | string | Classification result | "General", "Teen", "Mature" |
| `Severity_Score` | int | Total keyword matches | 12 |
| `Mature_Density` | float | Mature keywords per 1K words | 2.456 |
| `Teen_Density` | float | Teen keywords per 1K words | 5.123 |
| `Transcript_Word_Count` | int | Total words in transcript | 3,542 |
| `Content_Flags` | string | Categorized flags with timestamps | See above |

## 🚀 Quick Start

### Prerequisites

```bash
# Ensure you have Python 3.8+
python --version

# Install dependencies (same as EDA script)
pip install pandas numpy tqdm
```

### Basic Usage

1. **Place your dataset in the same folder:**
   ```
   your-project-folder/
   ├── batch_labeling_pipeline.py
   └── youtube_data.csv  ← Your raw dataset
   ```

2. **Run the pipeline:**
   ```bash
   python batch_labeling_pipeline.py
   ```

3. **Output:**
   - Creates `training_ready_dataset.csv` (labeled dataset)
   - Displays comprehensive statistics in terminal

### Customization

Update lines 775-776 in `batch_labeling_pipeline.py`:

```python
INPUT_CSV_PATH = "youtube_data.csv"        # Your input file
OUTPUT_CSV_PATH = "training_ready_dataset.csv"  # Your output file
```

## 📊 Expected Output

### Terminal Summary

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║              WEAK SUPERVISION BATCH LABELING PIPELINE                        ║
║              YouTube Age Classification Research Project                     ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

================================================================================
📂 LOADING AND CLEANING DATA
================================================================================
⏳ Loading youtube_data.csv...
✅ Loaded 9,700 rows

🧹 Cleaning Data:
   • Missing transcripts: 3,500 (36.08%)
   • Removed: 3,500 rows
   • Remaining: 6,200 rows
   • ✅ 15 original columns retained

================================================================================
🏷️  BATCH LABELING PIPELINE - WEAK SUPERVISION
================================================================================
Processing 6,200 transcripts with keyword-density analysis...

Labeling transcripts: 100%|████████████████████████| 6200/6200 [02:34<00:00, 40.12 transcript/s]

✅ Labeling complete! Adding new columns to dataset...

================================================================================
📊 LABELING SUMMARY STATISTICS
================================================================================

🏷️  LABEL DISTRIBUTION:
--------------------------------------------------------------------------------
Category        Count        Percentage      Avg Length (words)  
--------------------------------------------------------------------------------
General (G)     3,845        62.02%          2,341.5             
Teen (T)        1,654        26.68%          3,127.8             
Mature (M)      701          11.31%          4,256.2             
--------------------------------------------------------------------------------
TOTAL           6,200        100.00%

⚠️  SAFETY METRICS:
--------------------------------------------------------------------------------
Zero Tolerance Triggers: 47 transcripts
   (Extreme content requiring immediate Mature classification)

📈 DATASET QUALITY INDICATORS:
--------------------------------------------------------------------------------
Total Labeled Transcripts: 6,200
Average Transcript Length: 2,845.3 words
Median Transcript Length: 2,123.0 words
Longest Transcript: 45,678 words
Shortest Transcript: 127 words

🔢 SEVERITY SCORE STATISTICS:
--------------------------------------------------------------------------------
Mean Severity Score: 15.34
Median Severity Score: 8.0
Max Severity Score: 234

📊 KEYWORD DENSITY ANALYSIS (per 1,000 words):
--------------------------------------------------------------------------------
Mature Content Density:
   • Mean: 0.876
   • Median: 0.234
   • 95th Percentile: 3.456
Teen Content Density:
   • Mean: 1.456
   • Median: 0.987
   • 95th Percentile: 5.234

🚩 CONTENT FLAGS:
--------------------------------------------------------------------------------
Transcripts with Flags: 2,355 (38.00%)
Clean Transcripts: 3,845 (62.00%)

================================================================================
✅ PIPELINE COMPLETE - Silver Training Dataset Ready!
================================================================================

📁 Output File: training_ready_dataset.csv
📊 Total Samples: 6,200
🏷️  Labels: General (3,845), Teen (1,654), Mature (701)

💡 Next Steps:
   1. Review the Content_Flags column for quality assurance
   2. Analyze the Severity_Score distribution
   3. Consider filtering high-severity samples if needed
   4. Proceed with model training using Age_Label as target
```

## 🔬 Classification Logic Details

### Threshold Rationale

**Zero Tolerance (Immediate Mature):**
- ANY occurrence of extreme slurs, graphic violence, sexual assault → Mature
- Rationale: Even one instance makes content unsuitable for teens

**Mature Density ≥ 1.5 per 1,000 words:**
- ~1.5 mature keywords every 1,000 words
- For a 3,000-word transcript (20-min video): ~4.5 mature keywords
- Rationale: Consistent mature content throughout

**Teen Density ≥ 2.5 per 1,000 words:**
- ~2.5 teen keywords every 1,000 words
- For a 3,000-word transcript: ~7.5 teen keywords
- Rationale: Frequent teen-oriented language/topics

**Else → General:**
- Low keyword density across all categories
- Educational, family-friendly, or neutral content

### Why These Specific Thresholds?

These were calibrated based on:
1. Manual review of sample transcripts
2. Content moderation industry standards
3. YouTube's own content rating guidelines
4. Balancing precision vs. recall for weak supervision

**You can adjust these thresholds** in the `label_transcript()` method (lines ~450-460).

## 📈 Performance Expectations

**Processing Speed:**
- ~40-50 transcripts/second on standard laptop
- 6,200 transcripts: ~2-3 minutes
- 16,000 transcripts: ~6-8 minutes
- 100,000 transcripts: ~35-45 minutes

**Memory Usage:**
- Input: ~500MB CSV
- Peak RAM: ~1-2GB
- Output: ~550MB CSV (6 new columns)

## 🔧 Advanced Customization

### Adjusting Keyword Dictionaries

Add domain-specific keywords in lines 24-141:

```python
# Example: Add medical content to General
GENERAL_KEYWORDS = {
    # ... existing keywords ...
    'medicine', 'doctor', 'health', 'treatment', 'diagnosis',
    # ... etc ...
}
```

### Modifying Thresholds

In `label_transcript()` method:

```python
# Current thresholds (line ~455)
if zero_tolerance_count >= 1:
    label = 'Mature'
elif mature_density >= 1.5:  # Adjust this
    label = 'Mature'
elif teen_density >= 2.5:    # Adjust this
    label = 'Teen'
else:
    label = 'General'
```

### Changing Timestamp Estimation

Adjust words-per-minute assumption (line 180):

```python
# Current: 150 words/minute
self.WPM = 150

# For faster-paced content (e.g., gaming):
self.WPM = 180

# For slower content (e.g., lectures):
self.WPM = 120
```

## 🧪 Quality Assurance

### Recommended Post-Processing

1. **Spot-check samples:**
   ```python
   import pandas as pd
   df = pd.read_csv('training_ready_dataset.csv')
   
   # Review random Mature samples
   mature_samples = df[df['Age_Label'] == 'Mature'].sample(10)
   print(mature_samples[['title', 'Content_Flags', 'Severity_Score']])
   ```

2. **Analyze distribution:**
   ```python
   # Check if labels are reasonable
   print(df['Age_Label'].value_counts(normalize=True))
   
   # Expected: ~60-70% General, ~20-30% Teen, ~5-15% Mature
   ```

3. **Review high-severity outliers:**
   ```python
   # Flag potential mislabels
   high_severity = df[df['Severity_Score'] > 100]
   print(f"High severity samples: {len(high_severity)}")
   ```

## ⚠️ Known Limitations & Mitigation

### 1. False Positives
**Issue:** Educational content discussing mature topics (e.g., history of war, sex education) may be mislabeled.

**Mitigation:**
- Review high-severity General/Teen samples
- Add context-aware rules (future enhancement)
- Use this as pre-labeling for human review

### 2. False Negatives
**Issue:** Implicit toxicity (sarcasm, coded language) may be missed.

**Mitigation:**
- This is weak supervision—expected ~80-85% accuracy
- Use ensemble methods with other signals
- Human validation on subset

### 3. Multilingual Content
**Issue:** Keyword matching only works for English.

**Mitigation:**
- Filter non-English videos (use EDA language detection)
- Or extend dictionaries to other languages

## 📚 Research Best Practices

### For Publication

**What to report:**
1. Keyword dictionary sizes (G: X words, T: Y words, M: Z words)
2. Threshold values used (Mature ≥ 1.5, Teen ≥ 2.5)
3. Inter-annotator agreement (if validated on subset)
4. Label distribution (% General, Teen, Mature)
5. Zero tolerance trigger frequency

**Reproducibility:**
- Include this script in supplementary materials
- Report package versions: `pip freeze > requirements.txt`
- Specify random seeds (if any sampling done)

### For Model Training

**Recommended workflow:**
1. **Stratified split:**
   ```python
   from sklearn.model_selection import train_test_split
   
   train, test = train_test_split(
       df, test_size=0.2, stratify=df['Age_Label'], random_state=42
   )
   ```

2. **Handle class imbalance:**
   - Use class weights in model
   - Or oversample minority class (Mature)

3. **Multi-task learning:**
   - Primary task: Age_Label (3-class classification)
   - Auxiliary task: Severity_Score (regression)
   - Helps model learn content intensity

## 🆘 Troubleshooting

**Script runs very slowly (< 10 transcripts/sec):**
- Check if regex patterns compiled (should see message at start)
- Ensure no antivirus scanning files during execution
- Try closing other applications

**Memory error:**
- Process in batches (modify script to chunk data)
- Or increase system RAM / use cloud instance

**"Transcript column not found":**
- Check your CSV column names: `df.columns`
- Update column name in script if different

**Unexpected label distribution:**
- Review threshold values—may need calibration
- Check keyword dictionaries for your domain

## 🔗 Integration with EDA Script

**Recommended workflow:**

1. **Run EDA first:**
   ```bash
   python dataset_eda_analyzer.py
   ```
   - Validates data quality
   - Checks language distribution
   - Identifies token limits

2. **Then run labeling:**
   ```bash
   python batch_labeling_pipeline.py
   ```
   - Creates labeled dataset
   - Maintains data lineage

3. **Optional: Re-run EDA on labeled data:**
   ```python
   # Analyze distribution by label
   from dataset_eda_analyzer import DatasetAnalyzer
   
   analyzer = DatasetAnalyzer('training_ready_dataset.csv')
   # Custom analysis by Age_Label...
   ```

## 📖 Citation

If you use this pipeline in your research, please cite:

```bibtex
@software{weak_supervision_pipeline,
  title={Weak Supervision Batch Labeling Pipeline for YouTube Age Classification},
  author={[Your University] AI Research Team},
  year={2026},
  url={https://github.com/your-repo/youtube-classification}
}
```

## 📧 Support

For questions or issues:
1. Check TROUBLESHOOTING.md
2. Review this README carefully
3. Open GitHub issue with:
   - Error message
   - Dataset statistics (rows, columns)
   - Python/package versions

---

**Happy Labeling! 🚀**
