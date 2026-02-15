# 🚀 Hierarchical RoBERTa Quick Reference

## Installation

```bash
pip install torch transformers
```

## Basic Usage

```python
from hierarchical_roberta import create_model
import torch

# Create model
model = create_model(
    num_classes=3,
    max_chunks=20,
    use_gradient_checkpointing=True
)

# Move to GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Forward pass
batch = next(iter(train_loader))
input_ids, attention_mask, labels = batch
input_ids = input_ids.to(device)
attention_mask = attention_mask.to(device)

outputs = model(input_ids, attention_mask)
logits = outputs['logits']  # Shape: [batch_size, 3]
```

## Model Configuration

```python
from hierarchical_roberta import ModelConfig, HierarchicalRobertaForClassification

config = ModelConfig(
    num_classes=3,           # General/Teen/Mature
    max_chunks=20,           # Max document length
    chunk_size=512,          # Don't change (RoBERTa limit)
    attention_hidden_size=256,
    classifier_dropout=0.3,
    use_gradient_checkpointing=True,
    pooling_strategy="attention"
)

model = HierarchicalRobertaForClassification(config)
```

## Memory Requirements

| GPU VRAM | Batch Size | Gradient Checkpoint |
|----------|------------|---------------------|
| 8 GB     | 2          | Required            |
| 12 GB    | 4          | Recommended         |
| 16 GB    | 6-8        | Optional            |
| 24 GB    | 12-16      | Not needed          |

## Training Strategies

### Strategy 1: Frozen Base (Fast)

```python
model = create_model(freeze_base=True)
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-3
)
# Train 3-5 epochs
```

### Strategy 2: Fine-tune All (Best Performance)

```python
model = create_model(freeze_base=False)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
# Train 10-15 epochs
```

### Strategy 3: Progressive Unfreezing

```python
# Phase 1: Frozen (epochs 0-5)
model = create_model(freeze_base=True)
# ... train ...

# Phase 2: Unfrozen (epochs 5-15)
model.unfreeze_base_model()
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
# ... continue training ...
```

## Model Inspection

```python
# Parameter counts
print(f"Total: {model.count_parameters():,}")
print(f"Trainable: {model.count_trainable_parameters():,}")

# Breakdown
breakdown = model.get_parameter_breakdown()
for name, count in breakdown.items():
    print(f"{name}: {count:,}")
```

## Attention Visualization

```python
# Get attention weights
model.eval()
with torch.no_grad():
    outputs = model(input_ids, attention_mask, return_attention=True)
    attention = outputs['attention_weights']  # [batch_size, max_chunks]

# Visualize for first sample
import matplotlib.pyplot as plt
plt.bar(range(20), attention[0].cpu())
plt.xlabel('Chunk Index')
plt.ylabel('Attention Weight')
plt.title('Chunk Importance')
plt.show()
```

## Common Parameters

### Dropout Rates

```python
# Less data / overfitting → higher dropout
config.classifier_dropout = 0.4

# More data / underfitting → lower dropout
config.classifier_dropout = 0.2
```

### Max Chunks

```python
# Shorter docs (~5K words)
config.max_chunks = 15

# Longer docs (~20K words)
config.max_chunks = 40
```

### Pooling Strategy

```python
config.pooling_strategy = "attention"  # Best (default)
config.pooling_strategy = "mean"       # Fast, works for balanced chunks
config.pooling_strategy = "max"        # Captures salient features
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| OOM Error | Reduce batch size, enable gradient checkpointing |
| NaN Loss | Lower learning rate (1e-5), add gradient clipping |
| Not learning | Check trainable params, increase learning rate |
| Attention all equal | Increase attention_hidden_size to 512 |

## Testing

```bash
# Run built-in tests
python hierarchical_roberta.py

# Expected output:
# ✅ Forward pass successful
# ✅ Freezing mechanisms work
# ✅ ALL TESTS PASSED
```

## Integration with Training

```python
# In train_model.py
from hierarchical_roberta import create_model

model = create_model(
    num_classes=3,
    max_chunks=20,
    use_gradient_checkpointing=True
)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

# Training loop
for epoch in range(num_epochs):
    for batch in train_loader:
        input_ids, masks, labels = batch
        outputs = model(input_ids, masks)
        loss = criterion(outputs['logits'], labels)
        loss.backward()
        optimizer.step()
```

## Key Shapes

```
Input:
  input_ids:       [batch_size, 20, 512]
  attention_mask:  [batch_size, 20, 512]

Intermediate:
  chunk_embeddings: [batch_size, 20, 768]
  attention_weights: [batch_size, 20]
  document_vector:  [batch_size, 768]

Output:
  logits:          [batch_size, 3]
```

## Model Summary

```python
# Total parameters: ~125M
#   RoBERTa: ~123.5M (98.8%)
#   Attention: ~0.2M (0.2%)
#   Classifier: ~0.3M (0.2%)
#   Head-only trainable: ~0.5M
```

## Best Practices

1. **Always use gradient checkpointing** for GPUs < 16GB
2. **Freeze embeddings** to prevent catastrophic forgetting
3. **Start with frozen base** for 3-5 epochs, then unfreeze
4. **Use lower LR** (2e-5) when unfreezing RoBERTa
5. **Monitor attention weights** to ensure model is learning
6. **Gradient clipping** (1.0) prevents training instability

## Research Reporting

**For your paper:**
- Architecture: "Hierarchical Attention Network with RoBERTa-base"
- Parameters: "125M (123.5M RoBERTa, 1.5M task-specific)"
- Chunking: "20 chunks × 512 tokens with 256-token stride"
- Pooling: "Self-attention pooling with 256-dim hidden layer"
- Regularization: "Dropout 0.1 (hidden), 0.3 (classifier)"

---

**Status:** ✅ Model Architecture (2/4 Complete)  
**Next:** train_model.py
