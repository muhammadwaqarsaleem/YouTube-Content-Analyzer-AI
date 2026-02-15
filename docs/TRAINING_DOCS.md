# 🚀 Training Pipeline Documentation

Complete guide to the production-grade training system for Hierarchical RoBERTa.

## 🎯 Overview

This training pipeline implements **state-of-the-art optimization techniques** used in modern deep learning:
- **AMP (Automatic Mixed Precision)**: 2x faster + 40% memory savings
- **Gradient Accumulation**: Simulate large batches on small GPUs
- **Warmup Scheduling**: Stabilize early training
- **Class Weighting**: Handle imbalanced data
- **Early Stopping**: Prevent overfitting
- **Comprehensive Logging**: TensorBoard + Python logging

---

## 🔑 Key Optimizations Explained

### 1. Automatic Mixed Precision (AMP)

**What it does:**
```python
# Instead of full FP32 (32-bit floats)
# Use FP16 (16-bit) for most operations
# Keep FP32 for numerically sensitive ops

with autocast():  # Automatic casting
    outputs = model(input_ids, attention_mask)
    loss = criterion(outputs['logits'], labels)
```

**Benefits:**
- **Speed**: 2-3x faster on modern GPUs (Volta, Turing, Ampere)
- **Memory**: 40-50% reduction in VRAM usage
- **Accuracy**: No loss (automatic precision management)

**How it works:**
```
Normal (FP32):  Model uses 4 bytes per weight
Mixed (FP16):   Model uses 2 bytes per weight + smart casting
                ↓
                Saves 50% parameter memory
                Doubles tensor core throughput
```

**When to disable:**
- Older GPUs (pre-Volta, e.g., GTX 1080)
- Numerical instability (rare with modern PyTorch)

### 2. Gradient Accumulation

**The Problem:**
```python
# Ideal: batch_size = 32 for stable training
# Reality: 8GB GPU can only fit batch_size = 4
# Solution: Accumulate gradients over 8 steps → effective batch = 32
```

**Implementation:**
```python
gradient_accumulation_steps = 8

for batch in dataloader:
    loss = model(batch) / gradient_accumulation_steps
    loss.backward()  # Accumulate gradients
    
    if (step + 1) % gradient_accumulation_steps == 0:
        optimizer.step()  # Update once every 8 batches
        optimizer.zero_grad()
```

**Why this works:**
- Gradient averaging: `grad = Σ(grad_batch_i) / N`
- Mathematically equivalent to larger batch
- Only updates weights every N steps

**Trade-offs:**
- **Pro**: Can use effective batch_size > GPU memory allows
- **Pro**: More stable training (larger batch statistics)
- **Con**: Slower (no weight update for N-1 steps)

**Configuration:**
```python
batch_size = 4                    # Physical batch (fits in 8GB GPU)
gradient_accumulation_steps = 4   # Accumulate over 4 batches
effective_batch_size = 16         # Actual gradient averaging
```

### 3. Learning Rate Warmup + Cosine Decay

**Why warmup?**

Early in training, model weights are random. Large learning rates cause instability:
```
No warmup:    Loss explodes in first epoch → NaN
With warmup:  Gradual increase → Stable training
```

**Schedule:**
```
LR
 ↑
 │     ╱────╲                  Cosine decay
 │    ╱      ╲___
 │   ╱           ╲___
 │  ╱                ╲___
 │ ╱                     ╲___
 │╱__________________________|_____→ Steps
 0    Warmup (10%)            100%
```

**Implementation:**
```python
# Warmup: 0 → max_lr over 10% of training
# Cosine: max_lr → 0 following cosine curve

num_warmup_steps = total_steps * 0.1
scheduler = OneCycleLR(
    optimizer,
    max_lr=2e-5,
    total_steps=total_steps,
    pct_start=0.1,  # 10% warmup
    anneal_strategy='cos'
)
```

**Why cosine over linear/step?**
- **Smooth decay**: No sudden jumps
- **Research-proven**: Best for transformers (Loshchilov & Hutter 2017)
- **Fine-tuning friendly**: Gradual slowdown near end

### 4. Class Weighting for Imbalanced Data

**Your data distribution:**
```
General: 60% (~3,000 samples)
Teen:    27% (~1,350 samples)
Mature:  11% (~550 samples)
```

**Without weighting:**
```python
Model learns: "Always predict General" → 60% accuracy!
But misses all Teen and Mature samples (useless model)
```

**With weighting:**
```python
# Formula: weight_i = total / (num_classes × count_i)
weights = [
    6200 / (3 × 3000) ≈ 0.69,  # General (downweight majority)
    6200 / (3 × 1350) ≈ 1.53,  # Teen (balance)
    6200 / (3 × 550)  ≈ 3.76   # Mature (upweight minority)
]

criterion = nn.CrossEntropyLoss(weight=weights)
```

**Effect:**
```
Loss contribution:
  General:  0.69 × error   (penalize less)
  Teen:     1.53 × error   (normal)
  Mature:   3.76 × error   (penalize more)
  
Result: Model learns to classify all classes, not just majority
```

### 5. AdamW Optimizer

**Why AdamW over Adam?**

```python
# Adam: Weight decay tied to gradients (incorrect for transformers)
# AdamW: Decoupled weight decay (correct)

AdamW pseudocode:
    grad = compute_gradient(loss)
    m = β1 × m + (1-β1) × grad           # First moment
    v = β2 × v + (1-β2) × grad²          # Second moment
    θ = θ - lr × m / √v                  # Adam update
    θ = θ - lr × weight_decay × θ        # Separate decay ✓
```

**Parameter groups:**
```python
# Different learning rates for different parts
optimizer_params = [
    {'params': roberta_params, 'lr': 2e-5},      # Base model
    {'params': attention_params, 'lr': 2e-5},    # Attention
    {'params': classifier_params, 'lr': 4e-5}    # Task head (2x)
]
```

**Why higher LR for task head?**
- Base RoBERTa: Pre-trained, needs gentle fine-tuning
- Task head: Random init, needs faster learning

### 6. Gradient Clipping

**The problem:**
```
Gradient explosion:
  Batch 1: grad_norm = 2.3   ✓
  Batch 2: grad_norm = 456.7  ✗ (exploding!)
  Batch 3: NaN loss           ✗ (dead model)
```

**Solution:**
```python
# Clip gradients to maximum norm
max_grad_norm = 1.0

torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

# If total gradient norm > 1.0, scale down proportionally
# grad = grad × (1.0 / grad_norm)
```

**Why 1.0?**
- Empirically proven for transformers (Vaswani et al.)
- Prevents instability without hurting performance

### 7. Early Stopping

**Prevents overfitting:**
```
Epoch  Train Loss  Val Loss  Val F1
  1      1.050      1.100    0.45
  2      0.850      0.950    0.58
  3      0.650      0.800    0.67   ← Improving
  4      0.500      0.720    0.71   ← Best
  5      0.380      0.730    0.70   ← Val getting worse!
  6      0.280      0.750    0.69   ← Stop here
```

**Implementation:**
```python
patience = 5  # Wait 5 epochs for improvement
min_delta = 0.001  # Minimum meaningful improvement

if val_f1 > best_f1 + min_delta:
    best_f1 = val_f1
    epochs_without_improvement = 0
else:
    epochs_without_improvement += 1
    
if epochs_without_improvement >= patience:
    stop_training()  # Prevent further overfitting
```

### 8. TensorBoard Logging

**Real-time monitoring:**
```bash
# Start TensorBoard
tensorboard --logdir=runs

# View at: http://localhost:6006
```

**What's logged:**
- Training loss (every 10 batches)
- Learning rate schedule
- Validation metrics (every epoch)
- Gradient norms
- Model graph

**Why it matters:**
- Catch issues early (loss exploding, not learning)
- Compare different runs
- Debug hyperparameters

---

## 📊 Configuration Guide

### TrainingConfig Parameters

```python
@dataclass
class TrainingConfig:
    # Data paths
    data_dir: str = "data/processed_tensors"
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "runs"
    
    # Model
    num_classes: int = 3
    max_chunks: int = 20
    use_gradient_checkpointing: bool = True
    
    # Training
    num_epochs: int = 15                # Total epochs
    batch_size: int = 4                 # Physical batch size
    gradient_accumulation_steps: int = 4  # Effective batch = 16
    learning_rate: float = 2e-5         # Peak learning rate
    weight_decay: float = 0.01          # L2 regularization
    max_grad_norm: float = 1.0          # Gradient clipping
    
    # LR schedule
    warmup_ratio: float = 0.1           # 10% warmup
    scheduler_type: str = "linear_warmup_cosine"
    
    # Optimization
    use_amp: bool = True                # Mixed precision
    label_smoothing: float = 0.0        # 0.0 = none, 0.1 = mild
    
    # Early stopping
    early_stopping_patience: int = 5
    metric_for_best_model: str = "val_f1_macro"
    
    # Class weighting
    use_class_weights: bool = True
    
    # Hardware
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 2
    seed: int = 42
```

### When to Adjust Parameters

**batch_size:**
```python
# 8GB GPU
batch_size = 2-4

# 12GB GPU
batch_size = 4-6

# 16GB+ GPU
batch_size = 8-12

# Rule: Start small, increase until OOM, then decrease
```

**gradient_accumulation_steps:**
```python
# Target effective batch size: 16-32
effective_batch = batch_size × gradient_accumulation_steps

# 8GB GPU example:
batch_size = 4
gradient_accumulation_steps = 4
effective_batch = 16  ✓
```

**learning_rate:**
```python
# Fine-tuning entire model
learning_rate = 2e-5  # Conservative, safe

# Fine-tuning with frozen base
learning_rate = 1e-3  # More aggressive

# Struggling to converge
learning_rate = 1e-5  # Very conservative

# Unstable / NaN loss
learning_rate = 1e-6  # Debug mode
```

**num_epochs:**
```python
# Small dataset (< 5K samples)
num_epochs = 10-15

# Medium dataset (5-20K)
num_epochs = 5-10

# Large dataset (> 20K)
num_epochs = 3-5

# Use early stopping to find optimal
```

**early_stopping_patience:**
```python
# Fast experimentation
patience = 3

# Thorough training
patience = 5-7

# Very noisy validation
patience = 10
```

---

## 🚀 Quick Start

### Basic Training Run

```bash
python train_model.py
```

**Expects:**
- `data/processed_tensors/train_dataset.pt`
- `data/processed_tensors/val_dataset.pt`
- `data/processed_tensors/test_dataset.pt`

**Creates:**
- `checkpoints/best_model.pt`
- `checkpoints/checkpoint_epoch_*.pt`
- `checkpoints/training.log`
- `checkpoints/training_history.json`
- `runs/run_<timestamp>/` (TensorBoard logs)

### Custom Configuration

```python
# In train_model.py, modify main():

config = TrainingConfig(
    num_epochs=20,
    batch_size=8,
    learning_rate=1e-5,
    use_amp=False,  # Disable for debugging
    early_stopping_patience=7
)
```

### Resume from Checkpoint

```python
# Load checkpoint
checkpoint = torch.load('checkpoints/best_model.pt')

# Restore model
model.load_state_dict(checkpoint['model_state_dict'])

# Restore optimizer (for continued training)
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# Restore scheduler
scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
```

---

## 📈 Expected Training Behavior

### Healthy Training

```
Epoch 1:
  Train Loss: 1.05 → 0.95 (decreasing)
  Val Loss:   1.10
  Val F1:     0.45
  
Epoch 2:
  Train Loss: 0.90 → 0.82
  Val Loss:   0.95 (decreasing)
  Val F1:     0.58 (increasing)
  
Epoch 5:
  Train Loss: 0.55 → 0.50
  Val Loss:   0.72
  Val F1:     0.71 ← Best

Epoch 10:
  Train Loss: 0.30 (still decreasing)
  Val Loss:   0.75 (starting to increase)
  Val F1:     0.69 (starting to decrease)
  
→ Early stopping triggered!
```

**Key indicators:**
- ✅ Train loss steadily decreases
- ✅ Val loss decreases then plateaus
- ✅ Val F1 increases then plateaus
- ✅ Gap between train/val widens slowly (normal)

### Warning Signs

**1. Not Learning:**
```
Epoch 1: Loss = 1.05
Epoch 2: Loss = 1.04
Epoch 3: Loss = 1.05
```

**Fixes:**
- Increase learning rate (try 1e-4)
- Check if model is frozen: `model.count_trainable_parameters()`
- Verify data is loading correctly

**2. Exploding Loss:**
```
Epoch 1: Loss = 1.05
Epoch 2: Loss = 2.34
Epoch 3: Loss = NaN
```

**Fixes:**
- Lower learning rate (try 1e-6 for debugging)
- Check gradient clipping is enabled
- Verify class weights aren't extreme

**3. Overfitting:**
```
Epoch 5:  Train=0.50, Val=0.72, Gap=0.22
Epoch 10: Train=0.30, Val=0.75, Gap=0.45 (widening!)
```

**Fixes:**
- Increase dropout in model config
- Enable label smoothing: `label_smoothing=0.1`
- Reduce num_epochs / use early stopping

**4. Underfitting:**
```
Epoch 15: Train=0.80, Val=0.82 (both high)
```

**Fixes:**
- Train longer (increase num_epochs)
- Increase model capacity (use roberta-large)
- Lower regularization (reduce dropout)

---

## 💾 Memory Management

### GPU Memory Breakdown (batch_size=4)

| Component | FP32 | FP16 (AMP) |
|-----------|------|------------|
| Model parameters | 500 MB | 250 MB |
| Optimizer states | 1000 MB | 500 MB |
| Gradients | 500 MB | 250 MB |
| Activations | 1500 MB | 600 MB |
| **Total** | **3.5 GB** | **1.6 GB** |

### OOM Troubleshooting

**Error: CUDA Out of Memory**

**Solutions (in order):**

1. **Enable AMP** (if not already)
   ```python
   use_amp = True  # 40% memory reduction
   ```

2. **Reduce batch size**
   ```python
   batch_size = 2  # or even 1 for debugging
   ```

3. **Increase gradient accumulation**
   ```python
   gradient_accumulation_steps = 8  # Maintain effective batch
   ```

4. **Enable gradient checkpointing** (if not already)
   ```python
   use_gradient_checkpointing = True  # In model config
   ```

5. **Reduce max_chunks**
   ```python
   max_chunks = 15  # Instead of 20
   ```

6. **Clear cache between runs**
   ```python
   torch.cuda.empty_cache()
   ```

---

## 📊 Monitoring Training

### TensorBoard

```bash
# Start TensorBoard (in separate terminal)
tensorboard --logdir=runs --port=6006

# Open browser to:
http://localhost:6006
```

**Key plots to watch:**
- `train/loss` - Should decrease smoothly
- `train/learning_rate` - Warmup → plateau → decay
- `epoch/val_f1_macro` - Should increase then plateau
- `epoch/val_loss` - Should decrease then plateau

### Log Files

**`checkpoints/training.log`:**
- Detailed Python logging output
- Errors, warnings, info messages
- Full training history

**`checkpoints/training_history.json`:**
- Structured metrics per epoch
- Easy to parse for plotting
- Contains all tracked metrics

---

## 🧪 Validation Metrics

### Metrics Explained

**Accuracy:**
```python
accuracy = correct_predictions / total_predictions

# Your data: 60% General, 27% Teen, 11% Mature
# Random baseline: ~33%
# "Always General" baseline: 60%
# Good model: > 70%
```

**F1 Score (Macro):**
```python
# Average F1 across all classes (equal weight)
f1_macro = (f1_general + f1_teen + f1_mature) / 3

# Better than accuracy for imbalanced data
# Target: > 0.65
```

**F1 Score (Weighted):**
```python
# Weighted by class frequency
f1_weighted = (f1_general × 0.60 + f1_teen × 0.27 + f1_mature × 0.11)

# Usually higher than macro (majority class matters more)
```

**Precision:**
```python
# Of predicted positives, how many are correct?
precision = true_positives / (true_positives + false_positives)

# High precision = few false alarms
```

**Recall:**
```python
# Of actual positives, how many did we find?
recall = true_positives / (true_positives + false_negatives)

# High recall = few missed cases
```

### Which Metric to Optimize?

**For your use case (content moderation):**

Use **F1 Macro** because:
- Equal importance to all classes
- Catching Mature content is critical (even though only 11%)
- Don't want to miss Teen classification either
- Prevents "predict General for everything" problem

**Alternative: Custom metric**
```python
# Weight Mature class more heavily (safety critical)
custom_metric = 0.3 × f1_general + 0.3 × f1_teen + 0.4 × f1_mature
```

---

## 🐛 Common Issues

### 1. Import Error

```
ModuleNotFoundError: No module named 'hierarchical_roberta'
```

**Fix:**
```bash
# Ensure files are in same directory
ls
# Should see: train_model.py, hierarchical_roberta.py

# Or add to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### 2. Data Not Found

```
FileNotFoundError: data/processed_tensors/train_dataset.pt
```

**Fix:**
```bash
# Run dataset preparation first
python dataset_preparation.py

# Verify files exist
ls data/processed_tensors/
# Should see: train_dataset.pt, val_dataset.pt, test_dataset.pt
```

### 3. CUDA Out of Memory

```
RuntimeError: CUDA out of memory
```

**Fix:** See "Memory Management" section above.

### 4. Loss is NaN

```
Epoch 2: Loss = nan
```

**Causes & Fixes:**
- Learning rate too high → Reduce to 1e-6
- No gradient clipping → Enable: `max_grad_norm=1.0`
- Extreme class weights → Check weight calculation
- Mixed precision issue → Disable AMP temporarily

### 5. Model Not Improving

```
Epoch 15: Val F1 = 0.35 (no improvement from epoch 1)
```

**Checklist:**
- [ ] Are parameters trainable? `model.count_trainable_parameters() > 0`
- [ ] Is learning rate too low? Try 1e-4
- [ ] Is data correct? Check dataloader output
- [ ] Is loss decreasing? If not, see "Not Learning" above

---

## 📚 Best Practices

### 1. Start with Small Experiments

```python
# First run: Quick validation
config = TrainingConfig(
    num_epochs=2,
    batch_size=2,
    early_stopping_patience=1
)
# Just verify everything runs without errors
```

### 2. Monitor Early Epochs Closely

```python
# If first epoch doesn't show loss decrease:
# - Stop and debug immediately
# - Don't waste hours on broken setup
```

### 3. Use Version Control

```bash
git init
git add train_model.py hierarchical_roberta.py dataset_preparation.py
git commit -m "Initial training setup"

# Before each experiment
git tag experiment_lr2e5_bs4
```

### 4. Document Experiments

```python
# Create experiment log
experiments = {
    'exp001': {
        'config': 'lr=2e-5, bs=4, acc=4',
        'results': 'val_f1=0.71',
        'notes': 'Baseline run'
    },
    'exp002': {
        'config': 'lr=1e-5, bs=8, acc=2',
        'results': 'val_f1=0.69',
        'notes': 'Lower LR didn't help'
    }
}
```

### 5. Save Everything

```python
# Checkpoint includes:
# - Model weights
# - Optimizer state
# - Scheduler state
# - Metrics
# - Config

# Can fully resume training later
```

---

## 🎓 For Publication

**Training section should report:**

```
Training Configuration:
- Optimizer: AdamW (lr=2e-5, weight_decay=0.01)
- Batch size: 4 physical, 16 effective (gradient accumulation)
- Epochs: 15 with early stopping (patience=5)
- LR schedule: Linear warmup (10%) + cosine decay
- Mixed precision: FP16 (Automatic Mixed Precision)
- Class weighting: Inverse frequency [0.69, 1.53, 3.76]
- Gradient clipping: max_norm=1.0
- Hardware: NVIDIA RTX 3080 (10GB VRAM)

Training Time: ~2.5 hours (15 epochs)
Best Model: Epoch 11 (val_f1_macro=0.73)
```

---

**Status:** ✅ Training Pipeline (3/4 Complete)  
**Next:** evaluate_and_explain.py (Evaluation & XAI)
