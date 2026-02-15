# 🚀 Training Quick Reference

## Installation

```bash
pip install torch transformers tensorboard scikit-learn tqdm
```

## Basic Usage

```bash
# 1. Ensure data is prepared
ls data/processed_tensors/
# Should see: train_dataset.pt, val_dataset.pt, test_dataset.pt

# 2. Run training
python train_model.py

# 3. Monitor with TensorBoard (separate terminal)
tensorboard --logdir=runs
# Open: http://localhost:6006
```

## Quick Configuration

```python
# In train_model.py main():

config = TrainingConfig(
    num_epochs=15,          # Total epochs
    batch_size=4,           # Physical batch (GPU memory limited)
    gradient_accumulation_steps=4,  # Effective batch = 16
    learning_rate=2e-5,     # Peak LR
    use_amp=True,           # Mixed precision (2x faster)
    early_stopping_patience=5
)
```

## Memory Settings

| GPU | Batch Size | Grad Acc | Effective | AMP |
|-----|------------|----------|-----------|-----|
| 8GB | 2 | 8 | 16 | Required |
| 12GB | 4 | 4 | 16 | Recommended |
| 16GB | 6-8 | 2 | 12-16 | Optional |
| 24GB | 12-16 | 1 | 12-16 | Not needed |

## Key Optimizations

```python
# 1. AMP (Automatic Mixed Precision)
use_amp = True  # 2x faster, 40% less memory

# 2. Gradient Accumulation
batch_size = 4
gradient_accumulation_steps = 4
# Effective batch = 16 (like training with bs=16)

# 3. Class Weights (for 60-27-11 split)
use_class_weights = True  # Auto-calculated

# 4. Warmup Scheduler
warmup_ratio = 0.1  # 10% warmup, then cosine decay

# 5. Early Stopping
early_stopping_patience = 5  # Stop if no improvement
```

## Output Files

```
checkpoints/
├── best_model.pt              # Best model (by val F1)
├── checkpoint_epoch_*.pt      # Regular checkpoints
├── training.log               # Detailed logs
├── training_history.json      # Metrics per epoch
└── test_results.json          # Final test metrics

runs/
└── run_<timestamp>/
    └── events.out.tfevents.*  # TensorBoard logs
```

## Common Commands

### Check GPU

```bash
nvidia-smi
# Watch GPU usage
watch -n 1 nvidia-smi
```

### View Logs

```bash
# Live training log
tail -f checkpoints/training.log

# Metrics history
cat checkpoints/training_history.json | python -m json.tool
```

### Resume from Checkpoint

```python
checkpoint = torch.load('checkpoints/best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])
```

## Troubleshooting

| Issue | Quick Fix |
|-------|-----------|
| OOM Error | Reduce `batch_size` to 2 or 1 |
| Loss = NaN | Lower `learning_rate` to 1e-6 |
| Not learning | Check `model.count_trainable_parameters() > 0` |
| Too slow | Enable `use_amp=True` |
| Overfitting | Increase `early_stopping_patience` |

## Expected Performance

```
Epoch  Train Loss  Val Loss  Val F1
  1       1.05       1.10     0.45
  3       0.65       0.80     0.67
  5       0.50       0.72     0.71  ← Usually best
  8       0.35       0.75     0.69  ← Starting to overfit
 10+      < 0.30     > 0.75   < 0.70 ← Early stopping
```

**Good signs:**
- Train loss decreases smoothly
- Val F1 increases to 0.70-0.75
- Early stopping triggers around epoch 8-12

**Warning signs:**
- Loss stuck or increasing
- Val F1 < 0.50 after 5 epochs
- Large train/val gap (> 0.3)

## Hyperparameter Tuning

### Learning Rate

```python
learning_rate = 2e-5   # Default (safe)
learning_rate = 1e-5   # Conservative (stable)
learning_rate = 5e-5   # Aggressive (faster but risky)
learning_rate = 1e-4   # For frozen base only
```

### Batch Size vs Gradient Accumulation

```python
# Option 1: Small batch, more accumulation
batch_size = 2
gradient_accumulation_steps = 8
# Effective = 16, slower but fits in 6GB GPU

# Option 2: Larger batch, less accumulation  
batch_size = 8
gradient_accumulation_steps = 2
# Effective = 16, faster but needs 16GB GPU
```

### Early Stopping

```python
patience = 3   # Aggressive (stops quickly)
patience = 5   # Balanced (recommended)
patience = 10  # Conservative (trains longer)
```

## Metrics Guide

**F1 Macro** (primary metric):
- Balanced across all classes
- Target: > 0.65 (good), > 0.70 (excellent)

**Accuracy**:
- Overall correctness
- Baseline: 60% ("always predict General")
- Target: > 70%

**F1 Weighted**:
- Weighted by class frequency
- Usually higher than F1 Macro

## Monitor These During Training

1. **Learning rate decay**
   - Should warmup then decrease
   - Check TensorBoard: `train/learning_rate`

2. **Loss convergence**
   - Train loss should decrease smoothly
   - Val loss should decrease then plateau

3. **GPU utilization**
   - Should be 85-100%
   - Check with `nvidia-smi`

4. **Gradient norms**
   - Should be clipped to 1.0
   - Spikes indicate instability

## Advanced Options

### Label Smoothing

```python
label_smoothing = 0.1  # Mild smoothing (prevents overconfidence)
# Converts hard labels [0,0,1] to soft [0.03, 0.03, 0.94]
```

### Custom Class Weights

```python
# If auto-calculation doesn't work
class_weights = [0.5, 1.0, 2.0]  # Custom weights
use_class_weights = True
```

### Different Scheduler

```python
scheduler_type = "linear"  # Linear decay (simpler)
# vs
scheduler_type = "linear_warmup_cosine"  # Cosine decay (better)
```

## Research Reporting

**For your paper:**

```
Training: AdamW optimizer (lr=2e-5, wd=0.01), batch size 16 
(4 physical × 4 accumulation), linear warmup (10%) + cosine 
decay, FP16 mixed precision. Class-weighted cross-entropy 
loss (inverse frequency). Early stopping with patience=5 on 
validation F1 macro. Hardware: NVIDIA RTX 3080 (10GB).

Results: Best model at epoch 11 (val_f1=0.73). Training time: 
2.5 hours. Final test metrics: accuracy=0.74, f1_macro=0.71.
```

## Next Steps After Training

```bash
# 1. Check best model
ls -lh checkpoints/best_model.pt

# 2. View TensorBoard
tensorboard --logdir=runs

# 3. Analyze metrics
cat checkpoints/training_history.json

# 4. Test on holdout set (automatic if test_loader provided)
cat checkpoints/test_results.json

# 5. Proceed to evaluation
python evaluate_and_explain.py
```

---

**Status:** ✅ Training (3/4 Complete)  
**Next:** Evaluation & Explainability
