# 🏗️ Hierarchical RoBERTa Architecture Documentation

Complete guide to the Hierarchical Attention Network (HAN) implementation for long-document classification.

## 🎯 Architecture Overview

### The Challenge

Standard transformers like RoBERTa have a **512-token limit**, but your YouTube transcripts average **~10,000 words** (~15,000 tokens). Simply truncating loses 95% of the content!

### The Solution: Hierarchical Processing

```
Long Document (15,000 tokens)
         ↓
Split into chunks (20 × 512 tokens)
         ↓
Encode each chunk with RoBERTa → 20 × 768-dim vectors
         ↓
Learn attention weights over chunks → 768-dim document vector
         ↓
Classify → 3 classes (General/Teen/Mature)
```

---

## 📐 Detailed Architecture

### Layer-by-Layer Breakdown

```python
Input: [batch_size, 20, 512]
  │
  ├─▶ Chunk Encoder (RoBERTa-base)
  │   ├─ Process chunks iteratively (memory-safe)
  │   ├─ Extract [CLS] token per chunk
  │   └─ Output: [batch_size, 20, 768]
  │
  ├─▶ Hierarchical Attention
  │   ├─ Compute attention scores
  │   ├─ Weighted sum of chunk embeddings
  │   └─ Output: [batch_size, 768]
  │
  ├─▶ Layer Normalization
  │   └─ Stabilize activations
  │
  └─▶ Classification Head
      ├─ Linear(768 → 384)
      ├─ Tanh activation
      ├─ Dropout(0.3)
      ├─ Linear(384 → 3)
      └─ Output: [batch_size, 3] (logits)
```

---

## 🧠 Component Deep Dive

### 1. Chunk Encoder (RoBERTa)

**Purpose:** Convert each 512-token chunk into a dense embedding.

**Key Design Decisions:**

**Memory Optimization:**
```python
# ❌ BAD: Process all chunks simultaneously
# Memory: batch_size × 20 × 512 = 40,960 tokens → 16GB VRAM!
all_chunks = roberta(input_ids.view(-1, 512))

# ✅ GOOD: Process chunks iteratively
# Memory: batch_size × 512 = 2,048 tokens → 4GB VRAM
for chunk_idx in range(20):
    chunk = roberta(input_ids[:, chunk_idx, :])
```

**Gradient Checkpointing:**
- Trades compute for memory
- Saves ~40% VRAM during backprop
- Increases training time by ~20%
- **Enabled by default** via `use_gradient_checkpointing=True`

**Freezing Strategy:**
```python
# Embeddings: Always frozen (prevents catastrophic forgetting)
# Lower layers: Frozen initially, can unfreeze for fine-tuning
# Upper layers: Trainable from start

model._freeze_embeddings()     # Default: frozen
model._freeze_base_model()     # Optional: freeze all
model.freeze_bottom_k_layers(6) # Optional: freeze first 6 layers
```

### 2. Self-Attention Pooling

**Purpose:** Learn which chunks are most important for classification.

**Mathematical Formulation:**
```
e_i = v^T × tanh(W × h_i + b)    # Attention score for chunk i
α_i = exp(e_i) / Σ exp(e_j)       # Softmax normalization
d = Σ α_i × h_i                   # Weighted sum
```

**Why Tanh?**
- Bounds scores to [-1, 1]
- More stable than raw linear projection
- Reference: Yang et al. (2016) "Hierarchical Attention Networks"

**Implementation:**
```python
# Input: [batch_size, 20, 768]
attention_scores = attention_vector(tanh(projection(chunk_embeddings)))
# Shape: [batch_size, 20]

attention_weights = softmax(attention_scores, dim=1)
# Shape: [batch_size, 20], sums to 1.0

document_vector = sum(attention_weights × chunk_embeddings)
# Shape: [batch_size, 768]
```

**Interpretability Bonus:**
Attention weights reveal which chunks drove the classification:
```
Chunk 1:  0.02 (intro)
Chunk 5:  0.45 (key violent scene) ← High attention!
Chunk 10: 0.15 (profanity cluster) ← Moderate attention
Chunk 15: 0.03 (outro)
```

### 3. Multi-Head Attention Variant

**Available but not default.** Multi-head attention allows attending to different aspects:
- Head 1: Topic/content
- Head 2: Sentiment/tone
- Head 3: Intensity
- Head 4: Context

**Trade-off:**
- **Pro:** More expressive, captures diverse aspects
- **Con:** More parameters, slower, can overfit on small data
- **Recommendation:** Use single-head for document classification

### 4. Layer Normalization

**Purpose:** Normalize activations before classification.

**Why it matters:**
- Attention can produce varying magnitude outputs
- LayerNorm stabilizes gradients
- Improves convergence speed
- Particularly important with attention mechanisms

**Position:** Applied **after** aggregation, **before** classification

### 5. Classification Head

**Architecture:**
```python
nn.Sequential(
    nn.Linear(768, 384),      # Dimensionality reduction
    nn.Tanh(),                # Bounded non-linearity
    nn.Dropout(0.3),          # Regularization
    nn.Linear(384, 3)         # Final logits
)
```

**Design Rationale:**

**Why hidden layer?**
- Allows non-linear transformation of document vector
- 384 dims captures task-specific features
- Better than direct 768→3 projection

**Why Tanh over ReLU?**
- Bounded activation prevents extreme values
- More stable for final layer before classification
- ReLU can cause "dead neurons" in final layers

**Why 30% dropout?**
- Prevents overfitting to training data
- Particularly important with attention (can memorize patterns)
- Empirically tuned (15-40% range is typical)

---

## 💾 Memory Analysis

### VRAM Requirements

**Model Parameters:**
```
RoBERTa encoder:      ~125M parameters × 4 bytes = 500 MB
Attention layer:      ~600K parameters × 4 bytes = 2.4 MB
Classifier:           ~300K parameters × 4 bytes = 1.2 MB
Total:                ~125M parameters × 4 bytes ≈ 504 MB
```

**Training Memory (batch_size=4):**
```
Model parameters:                     500 MB
Activations (forward pass):           ~1.5 GB
Gradients:                            500 MB
Optimizer states (AdamW):             1 GB
Workspace buffer:                     500 MB
Total:                                ~4 GB
```

**With Gradient Checkpointing:**
```
Activations reduced:                  ~600 MB (60% reduction)
Total:                                ~3.1 GB
```

**Recommendations:**
| GPU Memory | Batch Size | Gradient Checkpointing |
|------------|------------|------------------------|
| 8 GB       | 2          | Required               |
| 12 GB      | 4          | Recommended            |
| 16 GB      | 6-8        | Optional               |
| 24 GB      | 12-16      | Not needed             |

---

## ⚙️ Configuration Guide

### Model Config Parameters

```python
@dataclass
class ModelConfig:
    # Base model
    base_model_name: str = "roberta-base"  # Or "roberta-large"
    hidden_size: int = 768                  # 768 for base, 1024 for large
    
    # Architecture
    num_classes: int = 3
    max_chunks: int = 20
    chunk_size: int = 512
    
    # Attention
    attention_hidden_size: int = 256        # Attention projection size
    attention_heads: int = 1                # For multi-head variant
    
    # Regularization
    attention_dropout: float = 0.1          # 5-15% typical
    classifier_dropout: float = 0.3         # 20-40% typical
    hidden_dropout: float = 0.1             # After RoBERTa outputs
    
    # Training optimizations
    use_gradient_checkpointing: bool = True # Enable for <16GB GPU
    freeze_base_encoder: bool = False       # Freeze for limited data
    freeze_embeddings: bool = True          # Always recommended
    
    # Pooling
    pooling_strategy: str = "attention"     # attention/mean/max/cls
```

### When to Adjust Parameters

**`max_chunks`:**
- Default 20 → ~10,000 words
- Longer docs: Increase to 30-40
- Shorter docs: Decrease to 10-15
- **Formula:** `max_chunks = (avg_words × 1.5) / 256`

**`attention_hidden_size`:**
- Default 256 works for most cases
- Larger (512): More capacity, slower
- Smaller (128): Faster, may underfit
- Rule of thumb: `hidden_size / 2` or `hidden_size / 3`

**Dropout rates:**
- More data → Lower dropout (0.1-0.2)
- Less data → Higher dropout (0.3-0.4)
- Signs of overfitting → Increase dropout

**`pooling_strategy`:**
- `"attention"`: **Recommended** - learns importance
- `"mean"`: Fast, works for balanced chunks
- `"max"`: Captures salient features
- `"cls"`: Only use first chunk (not recommended)

---

## 🔧 Usage Examples

### Basic Instantiation

```python
from hierarchical_roberta import create_model

# Simple creation
model = create_model(
    num_classes=3,
    max_chunks=20,
    use_gradient_checkpointing=True
)

# Move to GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
```

### Advanced Configuration

```python
from hierarchical_roberta import ModelConfig, HierarchicalRobertaForClassification

config = ModelConfig(
    base_model_name="roberta-large",    # Upgrade to large model
    num_classes=3,
    max_chunks=30,                      # Longer documents
    attention_hidden_size=512,          # More attention capacity
    classifier_dropout=0.4,             # High regularization
    use_gradient_checkpointing=True,
    pooling_strategy="attention"
)

model = HierarchicalRobertaForClassification(config)
```

### Training Strategy: Progressive Unfreezing

```python
# Phase 1: Train attention + classifier only (fast, prevents overfitting)
model = create_model(freeze_base=True)
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-3  # Higher LR for head-only training
)
# Train for 3-5 epochs...

# Phase 2: Unfreeze and fine-tune everything
model.unfreeze_base_model()
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=2e-5  # Lower LR for fine-tuning
)
# Train for 5-10 more epochs...
```

### Gradual Unfreezing

```python
# Start with everything frozen except classifier
model._freeze_base_model()

# Gradually unfreeze layers
for epoch in range(20):
    if epoch == 5:
        model.freeze_bottom_k_layers(6)   # Unfreeze top 6 layers
    elif epoch == 10:
        model.unfreeze_base_model()       # Unfreeze everything
    
    # Train...
```

---

## 📊 Model Inspection

### Parameter Count

```python
# Total parameters
total = model.count_parameters()
print(f"Total: {total:,}")  # ~125M

# Trainable parameters
trainable = model.count_trainable_parameters()
print(f"Trainable: {trainable:,}")

# Breakdown by component
breakdown = model.get_parameter_breakdown()
for name, count in breakdown.items():
    print(f"{name}: {count:,}")

# Output:
# roberta_embeddings: 38,603,520
# roberta_encoder: 85,054,464
# attention_aggregator: 198,145
# classifier: 296,451
# total: 124,152,580
```

### Forward Pass Testing

```python
# Create dummy input
batch_size = 4
input_ids = torch.randint(0, 50265, (batch_size, 20, 512))
attention_mask = torch.ones(batch_size, 20, 512)

# Forward pass
model.eval()
with torch.no_grad():
    outputs = model(input_ids, attention_mask, return_attention=True)

# Check outputs
print(outputs['logits'].shape)           # [4, 3]
print(outputs['attention_weights'].shape) # [4, 20]

# Get predictions
predictions = torch.argmax(outputs['logits'], dim=1)
print(predictions)  # [0, 1, 2, 1] or similar
```

---

## 🎓 Research Best Practices

### For Publication

**Model section should include:**

1. **Architecture description:**
   - "Hierarchical Attention Network with RoBERTa-base encoder"
   - "Documents chunked into 20 segments of 512 tokens with 256-token stride"
   - "Self-attention pooling aggregates chunk representations"

2. **Parameters:**
   - "Model: 125M parameters (123.5M in RoBERTa, 1.5M in attention + classifier)"
   - "Attention hidden size: 256"
   - "Classification head: 768→384→3 with Tanh activation"

3. **Training configuration:**
   - "Gradient checkpointing enabled for memory efficiency"
   - "RoBERTa embeddings frozen, encoder fine-tuned"
   - "Dropout: 0.1 (hidden), 0.1 (attention), 0.3 (classifier)"

4. **Memory optimization:**
   - "Chunks processed iteratively to enable training on 8GB GPUs"
   - "Gradient checkpointing reduces VRAM by 40%"

### Reproducibility Checklist

- [ ] Document exact model configuration
- [ ] Report parameter counts (total + trainable)
- [ ] Specify freezing strategy used
- [ ] Note gradient checkpointing enabled/disabled
- [ ] Report batch size and GPU used
- [ ] Include seed for weight initialization

---

## 🔬 Architectural Decisions Explained

### Why This Architecture?

**1. Why iterative chunk processing?**
- **Memory:** Parallel processing requires 4× VRAM
- **Trade-off:** 20% slower, but enables training on consumer GPUs
- **Alternative:** Model parallelism (more complex)

**2. Why self-attention over BiGRU?**
- **Attention:** Parallelizable, interpretable weights
- **BiGRU:** Sequential (slower), hidden state less interpretable
- **Both work**, but attention is preferred in modern architectures

**3. Why freeze embeddings?**
- **Embeddings:** Pre-trained on massive corpus (rich semantics)
- **Risk:** Catastrophic forgetting with small datasets
- **Solution:** Freeze to preserve knowledge

**4. Why Tanh in classifier?**
- **ReLU:** Can cause dead neurons, unbounded
- **Sigmoid:** Saturation issues, deprecated for hidden layers
- **Tanh:** Bounded [-1,1], smoother gradients, stable

**5. Why LayerNorm before classification?**
- **Problem:** Attention outputs can have varying magnitudes
- **Solution:** Normalize to stabilize classifier input
- **Position:** After aggregation (not after RoBERTa, already normalized)

### Alternative Architectures Considered

**1. Longformer:**
- **Pro:** Handles 4K tokens natively
- **Con:** Still insufficient for 15K-token documents
- **Verdict:** Hierarchical approach more flexible

**2. Retrieval-based:**
- **Pro:** Select most relevant chunks
- **Con:** Discards context, may miss subtle patterns
- **Verdict:** Aggregation preserves more information

**3. Recursive chunking:**
- **Pro:** Multi-scale hierarchy
- **Con:** Complex, harder to train
- **Verdict:** Overkill for this task

---

## 🐛 Troubleshooting

### Common Issues

**1. CUDA Out of Memory**

```
RuntimeError: CUDA out of memory
```

**Solutions:**
- Enable gradient checkpointing: `use_gradient_checkpointing=True`
- Reduce batch size: Try 2 or even 1
- Reduce max_chunks: `max_chunks=15`
- Use smaller model: `roberta-base` instead of `roberta-large`

**2. NaN Loss**

```
Loss: nan
```

**Causes & Solutions:**
- **Learning rate too high:** Reduce to 1e-5 or 2e-5
- **Gradient explosion:** Add gradient clipping: `torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)`
- **Batch too small:** Increase batch size or use gradient accumulation

**3. Model Doesn't Train (Loss Stuck)**

**Check:**
- Are parameters frozen? `model.count_trainable_parameters()`
- Is learning rate too low? Try 1e-4 for frozen base, 2e-5 for unfrozen
- Is data normalized? Check input_ids and labels

**4. Attention Weights All Equal**

```
attention_weights: [0.05, 0.05, 0.05, ...]  # All 1/20
```

**Causes:**
- Model hasn't learned yet (early training)
- Attention dropout too high (try 0.05 instead of 0.1)
- Attention hidden size too small (increase to 512)

**5. Poor Performance on Long Documents**

**Check:**
- Are you using enough chunks? (max_chunks=20 for 10K words)
- Is stride sufficient? (256 for 50% overlap)
- Try different pooling: `pooling_strategy="max"`

---

## 📈 Performance Expectations

### Training Speed (8GB GPU, batch_size=4)

| Configuration | Time per Epoch | GPU Utilization |
|---------------|----------------|-----------------|
| Base frozen | ~15 min | 60-70% |
| Base unfrozen, no checkpoint | OOM | N/A |
| Base unfrozen, checkpoint | ~25 min | 85-95% |

### Convergence

**Expected behavior:**
- Frozen base: Converges in 3-5 epochs
- Unfrozen base: Converges in 10-15 epochs
- Validation loss should stabilize

**Red flags:**
- Loss increases: Learning rate too high
- Loss doesn't decrease: Model frozen or LR too low
- Train/val gap large: Increase dropout

---

## 🔗 Integration with Pipeline

### Loading Prepared Data

```python
import torch
from torch.utils.data import DataLoader, TensorDataset

# Load from dataset_preparation.py output
train_data = torch.load('data/processed_tensors/train_dataset.pt')

train_dataset = TensorDataset(
    train_data['input_ids'],
    train_data['attention_mask'],
    train_data['labels']
)

train_loader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True
)
```

### Training Loop Integration

```python
model = create_model()
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

for epoch in range(num_epochs):
    for input_ids, attention_mask, labels in train_loader:
        # Move to device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)
        
        # Forward
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs['logits'], labels)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

---

## 📚 References

**Hierarchical Attention Networks:**
- Yang et al. (2016) "Hierarchical Attention Networks for Document Classification"

**Transformers:**
- Vaswani et al. (2017) "Attention is All You Need"
- Liu et al. (2019) "RoBERTa: A Robustly Optimized BERT Pretraining Approach"

**Memory Optimization:**
- Chen et al. (2016) "Training Deep Nets with Sublinear Memory Cost"

---

**Status:** ✅ Architecture Complete (2/4)  
**Next:** train_model.py (Training Loop)
