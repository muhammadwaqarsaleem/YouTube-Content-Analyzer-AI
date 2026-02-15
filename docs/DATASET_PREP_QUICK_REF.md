# 🚀 Dataset Preparation Quick Reference

## Installation

```bash
pip install torch transformers scikit-learn pandas numpy tqdm
```

## Basic Usage

```bash
# 1. Prepare input
training_ready_dataset.csv  # Must have 'transcript' and 'Age_Label' columns

# 2. Run preparation
python dataset_preparation.py

# 3. Verify output
python verify_datasets.py

# 4. Output location
data/processed_tensors/
├── train_dataset.pt
├── val_dataset.pt
├── test_dataset.pt
└── metadata.json
```

## Key Parameters

```python
# In dataset_preparation.py (lines ~684-695)

chunk_size = 512       # Don't change (RoBERTa limit)
max_chunks = 20        # Adjust for avg document length
stride = 256           # Overlap (256 = 50% overlap)
train_size = 0.8       # 80% training
val_size = 0.1         # 10% validation
test_size = 0.1        # 10% test
```

## Output Tensor Shapes

```python
# Single sample
input_ids:       (20, 512)   # 20 chunks, 512 tokens each
attention_mask:  (20, 512)   # Attention mask
label:           ()          # Single integer (0/1/2)

# Batch (e.g., batch_size=4)
input_ids:       (4, 20, 512)
attention_mask:  (4, 20, 512)
labels:          (4,)
```

## Label Mapping

```
General → 0
Teen    → 1
Mature  → 2
```

## Loading in Training Script

```python
import torch
from torch.utils.data import DataLoader, TensorDataset

# Load
train_data = torch.load('data/processed_tensors/train_dataset.pt')

# Create dataset
train_dataset = TensorDataset(
    train_data['input_ids'],
    train_data['attention_mask'],
    train_data['labels']
)

# Create loader
train_loader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True
)
```

## Expected Performance

| Samples | Time | RAM |
|---------|------|-----|
| 1,000 | ~3-5 min | ~2 GB |
| 6,200 | ~15-25 min | ~4 GB |
| 10,000 | ~30-40 min | ~6 GB |

## Common Adjustments

### Longer Documents (~20K words)
```python
ChunkingConfig(max_chunks=30)
```

### Faster Processing (Less Overlap)
```python
ChunkingConfig(stride=384)  # 25% overlap
```

### Different Split Ratio
```python
SplitConfig(
    train_size=0.7,
    val_size=0.15,
    test_size=0.15
)
```

## Verification Checks

```python
import torch

# Load train data
data = torch.load('data/processed_tensors/train_dataset.pt')

# Check shapes
print(data['input_ids'].shape)      # Should be (N, 20, 512)
print(data['attention_mask'].shape) # Should be (N, 20, 512)
print(data['labels'].shape)         # Should be (N,)

# Check label distribution
labels = data['labels']
for i in range(3):
    count = (labels == i).sum()
    print(f"Label {i}: {count} ({count/len(labels)*100:.1f}%)")

# Check chunk usage
non_padding = (data['input_ids'] != 1).any(dim=2).sum(dim=1)
print(f"Avg chunks used: {non_padding.float().mean():.1f}")
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| OOM error | Reduce `max_chunks` to 10-15 |
| Slow processing | Normal for large datasets; use progress bar |
| File not found | Check CSV path in script line ~684 |
| Wrong label names | Update `label_map` at line ~64 |
| Download hangs | Pre-download tokenizer (see docs) |

## Quick Inspection

```python
from transformers import RobertaTokenizer

# Load tokenizer
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

# Load data
data = torch.load('data/processed_tensors/train_dataset.pt')

# Decode first chunk of first sample
first_chunk = data['input_ids'][0, 0]  # First sample, first chunk
text = tokenizer.decode(first_chunk)
print(text[:200])  # First 200 chars
```

## Memory Requirements

**During Preparation:**
- Input CSV: ~500 MB
- Peak RAM: 4-6 GB
- Disk space: ~1 GB

**For Training (Next Step):**
- GPU RAM: 8+ GB recommended
- CPU RAM: 8+ GB
- Batch size: 2-4 for 8GB GPU

## File Sizes

```
train_dataset.pt:   ~300-400 MB
val_dataset.pt:     ~40-50 MB
test_dataset.pt:    ~40-50 MB
metadata.json:      ~1 KB
Total:              ~400-500 MB
```

## Integration Checklist

Before moving to training:

- [ ] All .pt files created
- [ ] verify_datasets.py passes
- [ ] Label distribution balanced (60-70% General)
- [ ] Avg chunks used: 10-15 (reasonable)
- [ ] No NaN or inf values
- [ ] Tensor shapes correct: (N, 20, 512)

## Next Steps

```bash
# 1. Verify datasets
python verify_datasets.py

# 2. Create model architecture (next script)
# hierarchical_roberta.py

# 3. Train model (next next script)
# train_model.py
```

## Research Reporting

**For your paper, include:**

- Preprocessing: "Hierarchical chunking (512 tokens/chunk, 256 stride)"
- Max chunks: 20
- Coverage: "~10,000 words per document"
- Split: "80/10/10 stratified split"
- Samples: "Train: 4,960, Val: 620, Test: 620"
- Tokenizer: "RoBERTa-base (50,265 vocab)"

## Pro Tips

1. **First run:** Test on 100 samples first
   ```python
   df = df.sample(100)  # In load_data()
   ```

2. **Save time:** Pre-download tokenizer
   ```bash
   python -c "from transformers import RobertaTokenizer; RobertaTokenizer.from_pretrained('roberta-base')"
   ```

3. **Check progress:** Monitor with `nvidia-smi` (if using GPU)
   
4. **Version control:** Save processed data with version tag
   ```bash
   cp -r data/processed_tensors data/processed_tensors_v1.0
   ```

5. **Reproducibility:** Set `PYTHONHASHSEED=0` for full reproducibility
   ```bash
   export PYTHONHASHSEED=0
   python dataset_preparation.py
   ```

---

**Status:** ✅ Dataset Preparation (1/4 Complete)  
**Next:** hierarchical_roberta.py (Model Architecture)
