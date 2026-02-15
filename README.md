# 📊 Production-Grade EDA Analyzer for NLP Datasets

A comprehensive Exploratory Data Analysis tool designed specifically for NLP research projects, with a focus on YouTube video classification datasets. Built for the university AI research team working on age bracket classification.

## 🎯 Purpose

This script performs rigorous data quality validation before model training, specifically tailored for:
- Deep Learning NLP models (BERT, Longformer, etc.)
- Large-scale text classification tasks
- YouTube video datasets with transcripts and metadata
- Memory-constrained environments (standard laptops)

## ✨ Key Features

### 1. Structural Health Checks
- Total rows, columns, and memory usage
- Precise missing value detection (count + percentage per column)
- Exact duplicate detection (full rows + ID columns)
- Progress indicators for long-running operations

### 2. NLP-Specific Analysis
- Automatic text vs. numeric column detection
- Hidden missing values detection (empty strings, whitespace, "None", etc.)
- **Token limit analysis** (critical for model selection)
  - Word count statistics (mean, median, min, max)
  - 90th, 95th, and 99th percentile analysis
  - Percentage of texts exceeding 500 words
  - Percentage exceeding 512 tokens (BERT limit)
- Recommendations for BERT vs. Longformer architecture

### 3. Language Distribution Analysis
- Lightweight language detection on random samples
- Confirms percentage of English vs. other languages
- Early warning for multilingual content

### 4. Comprehensive Outputs
- Beautifully formatted terminal output
- Text report (`dataset_health_report.txt`) for documentation
- Publication-quality visualizations:
  - Missing values bar chart
  - Text length distribution histograms

## 🚀 Quick Start

### Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd <your-repo-directory>
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Usage

1. **Update the file path** in the script:
```python
# In dataset_eda_analyzer.py, line ~750
CSV_FILE_PATH = "path/to/your/youtube_data.csv"
```

2. **Run the analysis**:
```bash
python dataset_eda_analyzer.py
```

3. **View results**:
   - Check the terminal output for immediate insights
   - Open `eda_outputs/dataset_health_report.txt` for the full report
   - Review visualization PNGs in `eda_outputs/`

## 📋 Requirements

- Python 3.8+
- pandas >= 2.0.0
- numpy >= 1.24.0
- matplotlib >= 3.7.0
- seaborn >= 0.12.0
- tqdm >= 4.65.0
- langdetect >= 1.0.9 (optional, for language detection)

## 📊 Output Files

### 1. Terminal Output
Real-time progress and key metrics displayed in a clean, formatted structure.

### 2. Text Report (`dataset_health_report.txt`)
Comprehensive analysis including:
- Structural health summary
- Missing value breakdown
- Hidden missing value detection
- Token limit analysis per text column
- Language distribution
- **Actionable recommendations for model training**

### 3. Visualizations
- `missing_values_analysis.png` - Bar chart showing missing data distribution
- `text_length_distribution.png` - Histograms showing word count distribution with key thresholds

## 🔍 Interpreting Results

### Token Limit Analysis (CRITICAL)

**Why it matters**: BERT models have a 512-token limit. Texts exceeding this must be truncated or require alternative architectures.

**What to look for**:
```
Texts > 512 tokens (est.): 2,543 (15.89%)
⚠️ WARNING: 15.9% exceed BERT's 512 token limit!
💡 RECOMMENDATION: Consider Longformer or truncation strategy
```

**Decision guide**:
- **< 5% exceed 512 tokens**: Standard BERT is fine, truncate rare long texts
- **5-15% exceed 512 tokens**: Consider head-tail truncation or sliding window
- **> 15% exceed 512 tokens**: Strongly consider Longformer or hierarchical models

### Missing Values

**What to look for**:
```
Missing Values Analysis:
  • transcript: 1,234 (7.71%)
  • description: 456 (2.85%)
```

**Action items**:
- **< 5% missing**: Safe to remove rows or impute
- **5-20% missing**: Investigate pattern (random vs. systematic?)
- **> 20% missing**: Column may not be usable, or requires special handling

### Hidden Missing Values

**Why it matters**: Values like `""`, `"None"`, or `" "` look present but contain no information.

```
Hidden Missing Values:
  • transcript: 234 hidden missing (1.46%)
```

**Action**: Treat these as NaN and handle accordingly.

### Language Distribution

```
Language Distribution:
  • EN: 94 (94.0%)
  • ES: 4 (4.0%)
  • FR: 2 (2.0%)

✅ Dataset is predominantly English (94.0%)
```

**Decision guide**:
- **> 95% English**: Proceed with English models
- **90-95% English**: Consider filtering or flagging non-English
- **< 90% English**: Must use multilingual models (mBERT, XLM-RoBERTa)

## 🏗️ Architecture

### Object-Oriented Design
```
DatasetAnalyzer
├── load_data()              # CSV loading with error handling
├── analyze_structure()      # Basic health metrics
├── analyze_hidden_missing() # Text-specific missing values
├── analyze_text_statistics()# NLP metrics & token analysis
├── analyze_language_distribution() # Language detection
├── generate_visualizations() # Plot creation
├── generate_text_report()   # Report generation
└── run_complete_analysis()  # Orchestration
```

### Design Principles
- **Modular**: Each analysis is an independent method
- **Type-safe**: Full type hints throughout
- **Documented**: Comprehensive docstrings
- **Error-resilient**: Graceful degradation if columns missing
- **Memory-efficient**: Chunked processing where needed
- **Progress-aware**: tqdm integration for long operations

## 🎓 Best Practices for Research

### Before Training
1. **Review the full report** - Don't just glance at terminal output
2. **Check visualizations** - Patterns often invisible in numbers
3. **Address data quality issues** - Clean before training
4. **Document decisions** - Save the report with your training logs

### Model Selection
Use the token analysis to inform architecture:
```python
# Example decision logic
if pct_exceeds_512 < 5:
    model = "bert-base-uncased"  # Standard BERT
elif pct_exceeds_512 < 15:
    model = "bert-base-uncased"  # With truncation strategy
else:
    model = "allenai/longformer-base-4096"  # Handle long texts
```

### Publication-Ready
All outputs are publication-quality:
- Visualizations: 300 DPI, ready for papers
- Text report: Can be included in supplementary materials
- Metrics: Comprehensive for reproducibility section

## 🤝 Contributing

This script is maintained by the research team. For improvements:
1. Fork the repository
2. Create a feature branch
3. Add tests if applicable
4. Submit a pull request

## 📚 References

- [BERT Token Limits](https://huggingface.co/docs/transformers/model_doc/bert)
- [Longformer Paper](https://arxiv.org/abs/2004.05150)
- [NLP Data Cleaning Best Practices](https://neptune.ai/blog/nlp-data-cleaning)

## 📝 License

[Your License Here]

## 👥 Team

Developed by [Your University] AI Research Team for the YouTube Age Classification Project.

---

**Questions or Issues?** Open an issue on GitHub or contact the team lead.
