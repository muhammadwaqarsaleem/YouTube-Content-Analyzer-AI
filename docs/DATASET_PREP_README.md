# 📊 Dataset Preparation Pipeline Documentation

Complete guide for the hierarchical dataset preparation system for long-document classification.

## 🎯 Overview

This script solves the **critical challenge** of processing YouTube transcripts that average ~10,000 words, far exceeding RoBERTa's 512-token limit. Instead of simple truncation (which loses important information), we implement **hierarchical chunking** with sliding windows.

## 🔑 Key Innovation: Hierarchical Chunking

### The Problem

**Standard approach (truncation):**
```python
# ❌ BAD: Loses 95% of a 10,000-word transcript
tokenizer(transcript, max_length=512, truncation=True)
# Only uses first ~340 words!
```

**Our approach (hierarchical):**
```python
# ✅ GOOD: Captures up to 10,000 words across 20 chunks
# Each chunk: 512 tokens with 256-token overlap
# Output shape: (20, 512) per transcript
```

### How It Works

```
Original Transcript (10,000 words / ~15,000 tokens)
         ↓
┌─────────────────────────────────────────────────┐
│  Sliding Window Chunking                        │
├─────────────────────────────────────────────────┤
│  Chunk 1:  [0:512]     tokens                   │
│  Chunk 2:  [256:768]   tokens (256 overlap)     │
│  Chunk 3:  [512:1024]  tokens (256 overlap)     │
│  ...                                             │
│  Chunk 20: [4864:5376] tokens                   │
└─────────────────────────────────────────────────┘
         ↓
Final Tensor Shape: (20, 512)
- 20 chunks maximum (caps at ~10K words)
- 512 tokens per chunk (RoBERTa limit)
- 50% overlap for context preservation
```

### Why 50% Overlap (Stride = 256)?

**Context preservation example:**
```
Chunk 1: "...the violence in this game is extremely graphic and..."
Chunk 2: "...graphic and disturbing with blood everywhere and..."
         ^^^^^^^^^^^^^ Overlap preserves sentence context
```

Without overlap, you might split: "graphic and" | "disturbing" → loses meaning!

## 🚀 Quick Start

### Prerequisites

```bash
pip install torch transformers scikit-learn pandas numpy tqdm
```

**Package versions (tested):**
- torch >= 2.0.0
- transformers >= 4.30.0
- scikit-learn >= 1.3.0

### Basic Usage

```bash
# 1. Ensure you have your labeled dataset
training_ready_dataset.csv  # From batch_labeling_pipeline.py

# 2. Run preparation
python dataset_preparation.py

# 3. Outputs created in data/processed_tensors/
#    - train_dataset.pt
#    - val_dataset.pt
#    - test_dataset.pt
#    - metadata.json
```

### Expected Runtime

| Dataset Size | Processing Time | RAM Usage |
|--------------|-----------------|-----------|
| 1,000 samples | ~3-5 minutes | ~2 GB |
| 6,200 samples | ~15-25 minutes | ~4 GB |
| 10,000 samples | ~30-40 minutes | ~6 GB |

**Note:** First run downloads RoBERTa tokenizer (~500MB) from Hugging Face.

## 📋 Configuration Options

### Chunking Configuration

```python
ChunkingConfig(
    chunk_size=512,      # Max tokens per chunk (don't change - RoBERTa limit)
    max_chunks=20,       # Adjust based on avg document length
    stride=256,          # Overlap amount (256 = 50% overlap)
    model_name="roberta-base"  # Or "roberta-large"
)
```

**When to adjust `max_chunks`:**

| Avg Transcript Length | Recommended max_chunks |
|----------------------|------------------------|
| ~2,000 words | 10 |
| ~5,000 words | 15 |
| ~10,000 words | 20 (default) |
| ~20,000 words | 30-40 |

**Formula:** `max_chunks ≈ (avg_words × 1.5) / stride`

### Split Configuration

```python
SplitConfig(
    train_size=0.8,      # 80% training
    val_size=0.1,        # 10% validation
    test_size=0.1,       # 10% test
    random_state=42,     # For reproducibility
    stratify=True        # Maintains class balance
)
```

## 🔬 Understanding the Output

### Tensor Shapes Explained

**Single sample structure:**
```python
input_ids:       torch.Size([20, 512])  # 20 chunks × 512 tokens
attention_mask:  torch.Size([20, 512])  # Which tokens are padding
label:           torch.Size([])         # Single integer (0, 1, or 2)
```

**Batch structure (e.g., batch_size=8):**
```python
input_ids:       torch.Size([8, 20, 512])  # 8 samples, 20 chunks, 512 tokens
attention_mask:  torch.Size([8, 20, 512])
labels:          torch.Size([8])           # 8 labels
```

### Understanding Attention Masks

```python
# Example chunk with padding
input_ids = [
    101,    # [CLS] token
    2023,   # "This"
    2003,   # "is"
    # ... 400 more tokens ...
    102,    # [SEP] token
    1,      # [PAD]
    1,      # [PAD]
    # ... 108 more padding tokens ...
]

attention_mask = [
    1,      # Attend to [CLS]
    1,      # Attend to "This"
    1,      # Attend to "is"
    # ... 400 more 1s ...
    1,      # Attend to [SEP]
    0,      # Ignore padding
    0,      # Ignore padding
    # ... 108 more 0s ...
]
```

### Label Mapping

```python
{
    'General': 0,  # Family-friendly content
    'Teen': 1,     # Teen-oriented content
    'Mature': 2    # Adult content
}
```

## 📊 Output Files

### 1. train_dataset.pt

```python
# Load in training script
train_data = torch.load('data/processed_tensors/train_dataset.pt')

# Contains:
train_data['input_ids']       # Tensor(num_train, 20, 512)
train_data['attention_mask']  # Tensor(num_train, 20, 512)
train_data['labels']          # Tensor(num_train)
```

### 2. metadata.json

```json
{
  "label_map": {"General": 0, "Teen": 1, "Mature": 2},
  "chunk_size": 512,
  "max_chunks": 20,
  "stride": 256,
  "model_name": "roberta-base",
  "train_size": 4960,
  "val_size": 620,
  "test_size": 620,
  "created_at": "2026-02-15T10:30:00"
}
```

## 🔧 Advanced Usage

### Custom Processing

```python
from dataset_preparation import HierarchicalDatasetPreparator, ChunkingConfig

# Custom configuration for very long documents
custom_config = ChunkingConfig(
    chunk_size=512,
    max_chunks=30,    # Handle up to ~15K words
    stride=384,       # 25% overlap (faster processing)
    model_name="roberta-base"
)

preparator = HierarchicalDatasetPreparator(
    csv_path="my_dataset.csv",
    output_dir="custom_output",
    chunking_config=custom_config
)

preparator.run_pipeline()
```

### Process Subset for Testing

```python
# Quick test on 100 samples
import pandas as pd

df = pd.read_csv('training_ready_dataset.csv')
df_sample = df.sample(100, random_state=42)
df_sample.to_csv('sample_100.csv', index=False)

# Run on sample
preparator = HierarchicalDatasetPreparator(
    csv_path="sample_100.csv",
    output_dir="data/test_output"
)
preparator.run_pipeline()
```

### Inspect Chunking Results

```python
import torch

# Load train data
data = torch.load('data/processed_tensors/train_dataset.pt')
input_ids = data['input_ids']

# Check a single sample
sample_idx = 0
sample_chunks = input_ids[sample_idx]  # Shape: (20, 512)

# Count non-empty chunks
non_empty = (sample_chunks != 1).any(dim=1).sum()  # 1 = pad_token_id
print(f"Sample {sample_idx} uses {non_empty} chunks")

# Decode first chunk
from transformers import RobertaTokenizer
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
first_chunk_text = tokenizer.decode(sample_chunks[0])
print(f"First chunk: {first_chunk_text[:200]}...")
```

## 🐛 Troubleshooting

### Common Issues

#### 1. "File not found: training_ready_dataset.csv"

**Solution:**
```bash
# Check current directory
ls -l

# Copy dataset to correct location
cp /path/to/training_ready_dataset.csv .

# Or update path in script (line ~684)
INPUT_CSV_PATH = "/full/path/to/training_ready_dataset.csv"
```

#### 2. Out of Memory (OOM) Error

**Problem:** Dataset too large for RAM

**Solutions:**

**Option A: Reduce batch processing**
```python
# Modify process_transcripts() to process in mini-batches
# Instead of loading all at once, process 1000 at a time
```

**Option B: Reduce max_chunks**
```python
chunking_config = ChunkingConfig(
    max_chunks=10,  # Instead of 20
    # Processes only first ~5K words
)
```

**Option C: Use cloud instance**
```bash
# Recommended: 8GB+ RAM
# AWS EC2 t3.large or Google Colab Pro
```

#### 3. "UnicodeDecodeError" During CSV Load

**Solution:**
```python
# In load_data() method, add encoding
self.df = pd.read_csv(self.csv_path, low_memory=False, encoding='utf-8')

# If that fails, try:
self.df = pd.read_csv(self.csv_path, low_memory=False, encoding='latin-1')
```

#### 4. Tokenizer Download Hangs

**Solution:**
```bash
# Pre-download tokenizer
python -c "from transformers import RobertaTokenizer; RobertaTokenizer.from_pretrained('roberta-base')"

# Or use cached version
export TRANSFORMERS_CACHE=/path/to/cache
```

#### 5. Labels Not Mapping Correctly

**Problem:** CSV has different label names (e.g., "G", "T", "M" instead of "General", "Teen", "Mature")

**Solution:**
```python
# Update label_map in __init__ (line ~64)
self.label_map = {'G': 0, 'T': 1, 'M': 2}
```

#### 6. Progress Bar Not Showing

**Solution:**
```bash
# Install tqdm
pip install tqdm

# If in notebook, use notebook version
pip install ipywidgets
```

### Performance Optimization

**Slow processing?**

1. **Check CPU cores:**
```python
import os
os.environ['OMP_NUM_THREADS'] = '4'  # Use 4 cores
```

2. **Use GPU for tokenization (if available):**
```python
# Tokenization is CPU-bound, but can parallelize
# Add to process_transcripts():
# Use multiprocessing Pool for tokenizing
```

3. **Reduce overlap:**
```python
ChunkingConfig(stride=384)  # 25% overlap instead of 50%
# Faster but slightly less context preservation
```

## 📈 Quality Checks

### Verify Preparation Success

```python
import torch
import json

# 1. Check files exist
from pathlib import Path
assert (Path('data/processed_tensors') / 'train_dataset.pt').exists()
assert (Path('data/processed_tensors') / 'val_dataset.pt').exists()
assert (Path('data/processed_tensors') / 'test_dataset.pt').exists()

# 2. Load and inspect
train = torch.load('data/processed_tensors/train_dataset.pt')
print(f"Train input_ids shape: {train['input_ids'].shape}")
print(f"Train labels shape: {train['labels'].shape}")

# 3. Check label distribution
labels = train['labels']
for i in range(3):
    count = (labels == i).sum()
    pct = (count / len(labels)) * 100
    print(f"Label {i}: {count} ({pct:.1f}%)")

# 4. Check for NaN or inf
assert not torch.isnan(train['input_ids']).any()
assert not torch.isinf(train['input_ids']).any()

# 5. Verify chunk usage
non_padding_chunks = (train['input_ids'] != 1).any(dim=2).sum(dim=1)
print(f"Avg chunks used: {non_padding_chunks.float().mean():.1f}")
print(f"Max chunks used: {non_padding_chunks.max()}")

print("\n✅ All quality checks passed!")
```

### Expected Output

```
Train input_ids shape: torch.Size([4960, 20, 512])
Train labels shape: torch.Size([4960])
Label 0: 3076 (62.0%)
Label 1: 1323 (26.7%)
Label 2: 561 (11.3%)
Avg chunks used: 12.3
Max chunks used: 20

✅ All quality checks passed!
```

## 🔬 Understanding Stratification

**Without stratification:**
```
Train:  60% General, 30% Teen, 10% Mature
Val:    70% General, 20% Teen, 10% Mature  ❌ Imbalanced!
Test:   50% General, 35% Teen, 15% Mature  ❌ Imbalanced!
```

**With stratification:**
```
Train:  62% General, 27% Teen, 11% Mature
Val:    62% General, 27% Teen, 11% Mature  ✅ Balanced!
Test:   62% General, 27% Teen, 11% Mature  ✅ Balanced!
```

Why this matters:
- Model sees representative distribution during training
- Validation metrics are reliable
- Test results generalize better

## 📚 Integration with Next Steps

### Loading in Training Script (train_model.py)

```python
import torch
from torch.utils.data import DataLoader, TensorDataset

# Load processed data
train_data = torch.load('data/processed_tensors/train_dataset.pt')
val_data = torch.load('data/processed_tensors/val_dataset.pt')

# Create TensorDatasets
train_dataset = TensorDataset(
    train_data['input_ids'],
    train_data['attention_mask'],
    train_data['labels']
)

# Create DataLoaders
train_loader = DataLoader(
    train_dataset,
    batch_size=4,  # Small batch for hierarchical model
    shuffle=True,
    num_workers=2
)

# Iterate
for batch_input_ids, batch_masks, batch_labels in train_loader:
    # batch_input_ids shape: (4, 20, 512)
    # batch_masks shape: (4, 20, 512)
    # batch_labels shape: (4,)
    
    # Pass to hierarchical model...
    outputs = model(batch_input_ids, batch_masks)
    loss = criterion(outputs, batch_labels)
    # ...
```

### Expected Model Input

Your HAN model should expect:
```python
def forward(self, input_ids, attention_mask):
    # input_ids: (batch_size, max_chunks, chunk_size)
    # attention_mask: (batch_size, max_chunks, chunk_size)
    
    batch_size, num_chunks, seq_len = input_ids.shape
    
    # Reshape for processing: (batch_size × num_chunks, seq_len)
    input_ids = input_ids.view(-1, seq_len)
    attention_mask = attention_mask.view(-1, seq_len)
    
    # Process through RoBERTa...
```

## 🎓 Research Best Practices

### For Publication

**Dataset section should report:**

1. **Preprocessing details:**
   - "Transcripts processed using hierarchical chunking with 512-token chunks and 256-token stride"
   - "Maximum 20 chunks per document (~10,000 words)"

2. **Split methodology:**
   - "Stratified 80/10/10 train/validation/test split (seed=42)"
   - "Train: 4,960 samples, Val: 620 samples, Test: 620 samples"

3. **Tokenization:**
   - "RoBERTa-base tokenizer (vocab size: 50,265)"
   - "Average 12.3 chunks per document"

### Reproducibility Checklist

- [ ] Document exact package versions: `pip freeze > requirements_freeze.txt`
- [ ] Set random seed in script (already done: random_state=42)
- [ ] Save metadata.json with preparation
- [ ] Note any data filtering or cleaning
- [ ] Report chunk usage statistics

## 🚨 Important Notes

### Memory Considerations

**Storage requirements:**
- Input CSV: ~500 MB
- Tokenizer cache: ~500 MB
- Processed tensors: ~400-600 MB total
- Peak RAM usage: ~4-6 GB

### GPU Requirements (for next step)

This preparation runs on **CPU only**. The processed tensors will be loaded on GPU during training.

**Recommended GPU memory** (for training):
- Minimum: 8 GB (batch_size=2)
- Recommended: 16 GB (batch_size=4-8)
- Optimal: 24+ GB (batch_size=16+)

### Data Versioning

```bash
# Save preparation config for reproducibility
cp dataset_preparation.py dataset_preparation_v1.0.py

# Version your processed data
mv data/processed_tensors data/processed_tensors_v1.0

# Document changes
echo "v1.0: Initial preparation with stride=256, max_chunks=20" > CHANGELOG.txt
```

---

## 📞 Support

**Issues or questions?**

1. Check this documentation thoroughly
2. Verify your input CSV has required columns
3. Review error messages carefully
4. Check GitHub issues (if applicable)
5. Contact research team lead

---

**Prepared for:** University AI Research Team  
**Project:** YouTube Age Classification using Hierarchical Attention Networks  
**Last Updated:** February 2026
