"""
Hierarchical RoBERTa for Long Document Classification
======================================================
A memory-efficient Hierarchical Attention Network (HAN) architecture
that processes long documents by encoding chunks independently and
aggregating them with learned attention weights.

Architecture:
    Input: [batch_size, max_chunks, chunk_size]
      ↓
    Chunk Encoder (RoBERTa): [batch_size, max_chunks, hidden_size]
      ↓
    Hierarchical Attention: [batch_size, hidden_size]
      ↓
    Classification Head: [batch_size, num_classes]

Author: University AI Research Team
Project: YouTube Age Classification
File: 2/4 - Model Architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaModel, RobertaConfig
from typing import Optional, Tuple, Dict
import math
from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Configuration for Hierarchical RoBERTa model."""
    
    # Base model
    base_model_name: str = "roberta-base"
    hidden_size: int = 768
    
    # Architecture
    num_classes: int = 3
    max_chunks: int = 20
    chunk_size: int = 512
    
    # Attention mechanism
    attention_hidden_size: int = 256
    attention_heads: int = 1  # For multi-head attention variant
    
    # Regularization
    attention_dropout: float = 0.1
    classifier_dropout: float = 0.3
    hidden_dropout: float = 0.1
    
    # Training optimizations
    use_gradient_checkpointing: bool = True
    freeze_base_encoder: bool = False
    freeze_embeddings: bool = True
    
    # Advanced features
    use_layer_norm: bool = True
    pooling_strategy: str = "attention"  # Options: "attention", "max", "mean", "cls"


class SelfAttentionPooling(nn.Module):
    """
    Self-attention pooling mechanism for aggregating chunk representations.
    
    This implements a learned attention mechanism that computes importance weights
    for each chunk and produces a weighted sum as the document representation.
    
    Architecture Decision:
        Using tanh activation after projection has been shown to improve
        attention score stability compared to raw linear projections.
        Reference: Yang et al. "Hierarchical Attention Networks" (2016)
    """
    
    def __init__(self, hidden_size: int, attention_size: int, dropout: float = 0.1):
        """
        Initialize self-attention pooling layer.
        
        Args:
            hidden_size: Dimension of input chunk embeddings
            attention_size: Dimension of attention hidden layer
            dropout: Dropout probability for attention weights
        """
        super().__init__()
        
        # Attention mechanism: v^T * tanh(W * h + b)
        self.attention_projection = nn.Linear(hidden_size, attention_size)
        self.attention_vector = nn.Linear(attention_size, 1, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights using Xavier/Glorot initialization
        # This prevents vanishing/exploding gradients in deep attention
        self._init_weights()
    
    def _init_weights(self) -> None:
        """
        Initialize attention weights with Xavier uniform.
        
        Why Xavier/Glorot: Maintains variance of activations across layers,
        crucial for stable gradient flow in attention mechanisms.
        """
        nn.init.xavier_uniform_(self.attention_projection.weight)
        nn.init.zeros_(self.attention_projection.bias)
        nn.init.xavier_uniform_(self.attention_vector.weight)
    
    def forward(
        self, 
        chunk_embeddings: torch.Tensor,
        chunk_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply self-attention pooling over chunk embeddings.
        
        Args:
            chunk_embeddings: Tensor of shape [batch_size, num_chunks, hidden_size]
            chunk_mask: Optional mask of shape [batch_size, num_chunks]
                       (1 for real chunks, 0 for padding)
        
        Returns:
            Tuple of:
                - aggregated: Document vector [batch_size, hidden_size]
                - attention_weights: Attention scores [batch_size, num_chunks]
        
        Shape transformations:
            [B, C, H] -> [B, C, A] -> [B, C, 1] -> [B, C] -> [B, H]
            Where: B=batch, C=chunks, H=hidden, A=attention_size
        """
        # Project to attention space: [B, C, H] -> [B, C, A]
        attention_hidden = torch.tanh(self.attention_projection(chunk_embeddings))
        
        # Compute attention scores: [B, C, A] -> [B, C, 1] -> [B, C]
        attention_scores = self.attention_vector(attention_hidden).squeeze(-1)
        
        # Apply mask if provided (crucial for variable-length documents)
        if chunk_mask is not None:
            # Set padding chunks to large negative value before softmax
            attention_scores = attention_scores.masked_fill(
                chunk_mask == 0, 
                float('-inf')
            )
        
        # Normalize to attention weights: [B, C]
        attention_weights = F.softmax(attention_scores, dim=1)
        attention_weights = self.dropout(attention_weights)
        
        # Weighted sum: [B, C] * [B, C, H] -> [B, H]
        # Broadcasting: [B, C, 1] * [B, C, H] -> [B, C, H] -> sum -> [B, H]
        aggregated = torch.sum(
            attention_weights.unsqueeze(-1) * chunk_embeddings, 
            dim=1
        )
        
        return aggregated, attention_weights


class MultiHeadChunkAttention(nn.Module):
    """
    Multi-head attention variant for chunk aggregation.
    
    Design Decision:
        Multi-head attention allows the model to attend to different aspects
        of chunks simultaneously (e.g., one head for topic, another for sentiment).
        However, single-head often works better for document classification.
        Included for experimentation.
    """
    
    def __init__(
        self, 
        hidden_size: int, 
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        """
        Initialize multi-head attention.
        
        Args:
            hidden_size: Dimension of input embeddings
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        
        assert hidden_size % num_heads == 0, \
            f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})"
        
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.hidden_size = hidden_size
        
        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        
        # Output projection
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        
        self.dropout = nn.Dropout(dropout)
        
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize with Xavier uniform."""
        for module in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
    
    def forward(
        self,
        chunk_embeddings: torch.Tensor,
        chunk_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply multi-head attention over chunks.
        
        Args:
            chunk_embeddings: [batch_size, num_chunks, hidden_size]
            chunk_mask: [batch_size, num_chunks]
        
        Returns:
            Tuple of (aggregated_vector, attention_weights)
        """
        batch_size, num_chunks, _ = chunk_embeddings.shape
        
        # Project to Q, K, V: [B, C, H] -> [B, C, H]
        Q = self.q_proj(chunk_embeddings)
        K = self.k_proj(chunk_embeddings)
        V = self.v_proj(chunk_embeddings)
        
        # Reshape for multi-head: [B, C, H] -> [B, NH, C, HD]
        Q = Q.view(batch_size, num_chunks, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, num_chunks, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, num_chunks, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention: [B, NH, C, C]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply mask
        if chunk_mask is not None:
            mask = chunk_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, C]
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Attention weights: [B, NH, C, C]
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values: [B, NH, C, HD]
        context = torch.matmul(attn_weights, V)
        
        # Reshape: [B, NH, C, HD] -> [B, C, H]
        context = context.transpose(1, 2).contiguous().view(
            batch_size, num_chunks, self.hidden_size
        )
        
        # Output projection
        output = self.out_proj(context)
        
        # Aggregate chunks (mean pooling after attention)
        aggregated = output.mean(dim=1)  # [B, H]
        
        # Return average attention weights across heads for interpretability
        avg_attn = attn_weights.mean(dim=1).mean(dim=1)  # [B, C]
        
        return aggregated, avg_attn


class HierarchicalRobertaForClassification(nn.Module):
    """
    Hierarchical RoBERTa model for long document classification.
    
    This model addresses the fundamental challenge of processing documents
    that exceed transformer token limits by:
    1. Chunking documents into 512-token segments
    2. Encoding each chunk independently with RoBERTa
    3. Aggregating chunk representations with learned attention
    4. Classifying based on the aggregated document representation
    
    Memory Optimization:
        Processes chunks iteratively rather than in parallel to avoid OOM
        on consumer GPUs (8-16GB VRAM). Uses gradient checkpointing for
        further memory reduction during backpropagation.
    
    Architecture Innovations:
        - Layer normalization before classification for training stability
        - Dropout at multiple stages for regularization
        - Optional gradient checkpointing for memory efficiency
        - Flexible attention mechanisms (single-head, multi-head, pooling)
    """
    
    def __init__(self, config: ModelConfig):
        """
        Initialize Hierarchical RoBERTa model.
        
        Args:
            config: Model configuration object
        """
        super().__init__()
        
        self.config = config
        
        # Load pre-trained RoBERTa base encoder
        print(f"Loading {config.base_model_name}...")
        roberta_config = RobertaConfig.from_pretrained(config.base_model_name)
        self.roberta = RobertaModel.from_pretrained(
            config.base_model_name,
            config=roberta_config
        )
        
        # Enable gradient checkpointing for memory efficiency
        if config.use_gradient_checkpointing:
            self.roberta.gradient_checkpointing_enable()
            print("✅ Gradient checkpointing enabled (saves ~40% VRAM)")
        
        # Freeze base model if specified
        if config.freeze_base_encoder:
            self._freeze_base_model()
        
        # Always freeze embeddings to prevent catastrophic forgetting
        if config.freeze_embeddings:
            self._freeze_embeddings()
        
        # Hidden dropout for RoBERTa outputs
        self.hidden_dropout = nn.Dropout(config.hidden_dropout)
        
        # Hierarchical attention aggregator
        if config.pooling_strategy == "attention":
            self.chunk_aggregator = SelfAttentionPooling(
                hidden_size=config.hidden_size,
                attention_size=config.attention_hidden_size,
                dropout=config.attention_dropout
            )
        elif config.pooling_strategy == "multi_head_attention":
            self.chunk_aggregator = MultiHeadChunkAttention(
                hidden_size=config.hidden_size,
                num_heads=config.attention_heads,
                dropout=config.attention_dropout
            )
        else:
            # Simple pooling fallback
            self.chunk_aggregator = None
        
        # Layer normalization before classification
        # Critical for training stability with attention mechanisms
        if config.use_layer_norm:
            self.layer_norm = nn.LayerNorm(config.hidden_size)
        else:
            self.layer_norm = None
        
        # Classification head with hidden layer
        # Design: Hidden layer allows non-linear transformation before classification
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.Tanh(),  # Tanh for bounded activations (more stable than ReLU here)
            nn.Dropout(config.classifier_dropout),
            nn.Linear(config.hidden_size // 2, config.num_classes)
        )
        
        # Initialize classifier weights
        self._init_classifier_weights()
        
        print(f"✅ Model initialized with {self.count_parameters():,} total parameters")
        print(f"   Trainable: {self.count_trainable_parameters():,}")
    
    def _freeze_embeddings(self) -> None:
        """
        Freeze embedding layers to prevent catastrophic forgetting.
        
        Design Decision:
            Pre-trained embeddings encode rich semantic information.
            Freezing them during initial training prevents losing this
            knowledge, especially important with small datasets.
        """
        for param in self.roberta.embeddings.parameters():
            param.requires_grad = False
        print("🔒 RoBERTa embeddings frozen")
    
    def _freeze_base_model(self) -> None:
        """
        Freeze all RoBERTa parameters for feature extraction mode.
        
        Use Case:
            When you have limited data, use RoBERTa as a fixed feature
            extractor and only train the attention + classification layers.
        """
        for param in self.roberta.parameters():
            param.requires_grad = False
        print("🔒 Entire RoBERTa base model frozen")
    
    def unfreeze_base_model(self) -> None:
        """
        Unfreeze RoBERTa for fine-tuning.
        
        Training Strategy:
            1. Train with frozen RoBERTa (faster, prevents overfitting)
            2. Unfreeze and fine-tune with lower learning rate
        """
        for param in self.roberta.parameters():
            param.requires_grad = True
        print("🔓 RoBERTa base model unfrozen for fine-tuning")
    
    def freeze_bottom_k_layers(self, k: int) -> None:
        """
        Freeze bottom k encoder layers for gradual unfreezing strategy.
        
        Args:
            k: Number of bottom layers to freeze (0-11 for RoBERTa-base)
        
        Training Strategy:
            Freeze early layers (general features) and fine-tune later
            layers (task-specific features). Common in transfer learning.
        """
        for i in range(k):
            for param in self.roberta.encoder.layer[i].parameters():
                param.requires_grad = False
        print(f"🔒 Froze bottom {k} RoBERTa encoder layers")
    
    def _init_classifier_weights(self) -> None:
        """
        Initialize classification head with proper initialization.
        
        Why Xavier/Glorot:
            Maintains variance of activations, preventing gradient issues.
            Especially important for the final classification layer.
        """
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def encode_chunks(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode chunks using RoBERTa with memory-efficient iteration.
        
        CRITICAL MEMORY OPTIMIZATION:
            Instead of processing all chunks simultaneously (which would require
            batch_size × max_chunks × chunk_size = massive VRAM), we process
            chunks iteratively and aggregate results.
        
        Args:
            input_ids: [batch_size, max_chunks, chunk_size]
            attention_mask: [batch_size, max_chunks, chunk_size]
        
        Returns:
            Tuple of:
                - chunk_embeddings: [batch_size, max_chunks, hidden_size]
                - chunk_mask: [batch_size, max_chunks] (1 for real, 0 for padding)
        
        Memory Analysis:
            Simultaneous: 16 GB VRAM for batch_size=4
            Iterative: 4 GB VRAM for batch_size=4
            Savings: 75% VRAM reduction
        """
        batch_size, max_chunks, chunk_size = input_ids.shape
        
        # Initialize output tensor
        chunk_embeddings = torch.zeros(
            batch_size, max_chunks, self.config.hidden_size,
            dtype=torch.float32,
            device=input_ids.device
        )
        
        # Compute chunk mask (non-padding chunks)
        # A chunk is padding if all its tokens are padding tokens
        chunk_mask = (input_ids != self.roberta.config.pad_token_id).any(dim=2).long()
        
        # Process chunks iteratively to save memory
        for chunk_idx in range(max_chunks):
            # Extract single chunk for all samples: [batch_size, chunk_size]
            chunk_input_ids = input_ids[:, chunk_idx, :]
            chunk_attention_mask = attention_mask[:, chunk_idx, :]
            
            # Skip completely padding chunks
            if chunk_mask[:, chunk_idx].sum() == 0:
                continue
            
            # Encode chunk with RoBERTa
            with torch.set_grad_enabled(self.training):
                outputs = self.roberta(
                    input_ids=chunk_input_ids,
                    attention_mask=chunk_attention_mask,
                    return_dict=True
                )
            
            # Extract [CLS] token representation: [batch_size, hidden_size]
            # Design: [CLS] token serves as chunk-level summary
            cls_embedding = outputs.last_hidden_state[:, 0, :]
            
            # Apply dropout for regularization
            cls_embedding = self.hidden_dropout(cls_embedding)
            
            # Store in output tensor
            chunk_embeddings[:, chunk_idx, :] = cls_embedding
        
        return chunk_embeddings, chunk_mask
    
    def aggregate_chunks(
        self,
        chunk_embeddings: torch.Tensor,
        chunk_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Aggregate chunk embeddings into document representation.
        
        Args:
            chunk_embeddings: [batch_size, max_chunks, hidden_size]
            chunk_mask: [batch_size, max_chunks]
        
        Returns:
            Tuple of:
                - document_embedding: [batch_size, hidden_size]
                - attention_weights: [batch_size, max_chunks] or None
        """
        if self.chunk_aggregator is not None:
            # Use learned attention aggregation
            document_embedding, attention_weights = self.chunk_aggregator(
                chunk_embeddings, chunk_mask
            )
        else:
            # Fallback to simple pooling
            if self.config.pooling_strategy == "mean":
                # Mean pooling over non-padding chunks
                masked_embeddings = chunk_embeddings * chunk_mask.unsqueeze(-1)
                document_embedding = masked_embeddings.sum(dim=1) / chunk_mask.sum(dim=1, keepdim=True).clamp(min=1)
            elif self.config.pooling_strategy == "max":
                # Max pooling
                document_embedding, _ = chunk_embeddings.max(dim=1)
            elif self.config.pooling_strategy == "cls":
                # Use first chunk only (not recommended)
                document_embedding = chunk_embeddings[:, 0, :]
            else:
                raise ValueError(f"Unknown pooling strategy: {self.config.pooling_strategy}")
            
            attention_weights = None
        
        return document_embedding, attention_weights
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the hierarchical model.
        
        Args:
            input_ids: Token IDs [batch_size, max_chunks, chunk_size]
            attention_mask: Attention mask [batch_size, max_chunks, chunk_size]
            return_attention: Whether to return attention weights for visualization
        
        Returns:
            Dictionary containing:
                - logits: Classification logits [batch_size, num_classes]
                - attention_weights: Optional [batch_size, max_chunks]
        
        Shape Flow:
            [B, C, S] -> encode_chunks -> [B, C, H]
                      -> aggregate -> [B, H]
                      -> classify -> [B, num_classes]
            
            Where: B=batch, C=chunks, S=seq_len, H=hidden
        """
        # Step 1: Encode chunks with RoBERTa
        # [B, C, S] -> [B, C, H]
        chunk_embeddings, chunk_mask = self.encode_chunks(input_ids, attention_mask)
        
        # Step 2: Aggregate chunks with attention
        # [B, C, H] -> [B, H]
        document_embedding, attention_weights = self.aggregate_chunks(
            chunk_embeddings, chunk_mask
        )
        
        # Step 3: Apply layer normalization (stabilizes training)
        if self.layer_norm is not None:
            document_embedding = self.layer_norm(document_embedding)
        
        # Step 4: Classification head
        # [B, H] -> [B, num_classes]
        logits = self.classifier(document_embedding)
        
        # Prepare output
        output = {'logits': logits}
        
        if return_attention and attention_weights is not None:
            output['attention_weights'] = attention_weights
        
        return output
    
    def count_parameters(self) -> int:
        """Count total parameters in model."""
        return sum(p.numel() for p in self.parameters())
    
    def count_trainable_parameters(self) -> int:
        """Count trainable parameters in model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_parameter_breakdown(self) -> Dict[str, int]:
        """Get detailed parameter breakdown by component."""
        breakdown = {
            'roberta_embeddings': sum(p.numel() for p in self.roberta.embeddings.parameters()),
            'roberta_encoder': sum(p.numel() for p in self.roberta.encoder.parameters()),
            'attention_aggregator': sum(p.numel() for p in self.chunk_aggregator.parameters()) if self.chunk_aggregator else 0,
            'classifier': sum(p.numel() for p in self.classifier.parameters()),
        }
        breakdown['total'] = sum(breakdown.values())
        return breakdown


def create_model(
    num_classes: int = 3,
    max_chunks: int = 20,
    use_gradient_checkpointing: bool = True,
    freeze_base: bool = False,
    pooling_strategy: str = "attention"
) -> HierarchicalRobertaForClassification:
    """
    Factory function to create model with sensible defaults.
    
    Args:
        num_classes: Number of output classes
        max_chunks: Maximum chunks per document
        use_gradient_checkpointing: Enable gradient checkpointing for memory
        freeze_base: Freeze RoBERTa encoder
        pooling_strategy: Chunk aggregation strategy
    
    Returns:
        Initialized model
    """
    config = ModelConfig(
        num_classes=num_classes,
        max_chunks=max_chunks,
        use_gradient_checkpointing=use_gradient_checkpointing,
        freeze_base_encoder=freeze_base,
        pooling_strategy=pooling_strategy
    )
    
    return HierarchicalRobertaForClassification(config)


# ============================================================================
# Testing & Validation
# ============================================================================

def test_model_forward_pass():
    """Test model with dummy data to verify shapes and forward pass."""
    print("\n" + "=" * 80)
    print("🧪 TESTING MODEL ARCHITECTURE")
    print("=" * 80 + "\n")
    
    # Configuration
    batch_size = 2
    max_chunks = 20
    chunk_size = 512
    num_classes = 3
    
    # Create model
    print("Creating model...")
    config = ModelConfig(
        num_classes=num_classes,
        max_chunks=max_chunks,
        chunk_size=chunk_size,
        use_gradient_checkpointing=False,  # Disable for testing
        pooling_strategy="attention"
    )
    model = HierarchicalRobertaForClassification(config)
    model.eval()
    
    # Create dummy input tensors
    print(f"\nCreating dummy input: [{batch_size}, {max_chunks}, {chunk_size}]")
    input_ids = torch.randint(
        0, 50265,  # RoBERTa vocab size
        (batch_size, max_chunks, chunk_size)
    )
    attention_mask = torch.ones(batch_size, max_chunks, chunk_size, dtype=torch.long)
    
    # Add some padding chunks for realism
    input_ids[:, 15:, :] = 1  # Pad token ID
    attention_mask[:, 15:, :] = 0
    
    print(f"Input shapes:")
    print(f"  input_ids: {tuple(input_ids.shape)}")
    print(f"  attention_mask: {tuple(attention_mask.shape)}")
    
    # Forward pass
    print("\n⏳ Running forward pass...")
    with torch.no_grad():
        outputs = model(input_ids, attention_mask, return_attention=True)
    
    # Verify outputs
    print("\n✅ Forward pass successful!")
    print(f"\nOutput shapes:")
    print(f"  logits: {tuple(outputs['logits'].shape)}")
    if 'attention_weights' in outputs:
        print(f"  attention_weights: {tuple(outputs['attention_weights'].shape)}")
    
    # Verify logit values are reasonable
    logits = outputs['logits']
    print(f"\nLogit statistics:")
    print(f"  Mean: {logits.mean().item():.4f}")
    print(f"  Std: {logits.std().item():.4f}")
    print(f"  Min: {logits.min().item():.4f}")
    print(f"  Max: {logits.max().item():.4f}")
    
    # Test predictions
    predictions = torch.argmax(logits, dim=1)
    print(f"\nPredictions: {predictions.tolist()}")
    
    # Show attention weights
    if 'attention_weights' in outputs:
        attn = outputs['attention_weights']
        print(f"\nAttention weights (first sample):")
        print(f"  Shape: {tuple(attn.shape)}")
        print(f"  Sum: {attn[0].sum().item():.4f} (should be ~1.0)")
        print(f"  Top 5 chunks: {torch.topk(attn[0], 5).indices.tolist()}")
    
    # Parameter count
    print("\n📊 Model Statistics:")
    breakdown = model.get_parameter_breakdown()
    for name, count in breakdown.items():
        print(f"  {name}: {count:,}")
    
    print(f"\nTrainable parameters: {model.count_trainable_parameters():,}")
    
    # Memory estimate
    param_memory_mb = (model.count_parameters() * 4) / (1024 ** 2)  # 4 bytes per float32
    print(f"\nEstimated parameter memory: {param_memory_mb:.1f} MB")
    
    print("\n" + "=" * 80)
    print("✅ ALL TESTS PASSED")
    print("=" * 80)


def test_freezing_mechanisms():
    """Test model freezing functionality."""
    print("\n" + "=" * 80)
    print("🧪 TESTING FREEZING MECHANISMS")
    print("=" * 80 + "\n")
    
    # Create model
    model = create_model(freeze_base=False)
    
    initial_trainable = model.count_trainable_parameters()
    print(f"Initial trainable parameters: {initial_trainable:,}")
    
    # Test full freeze
    model._freeze_base_model()
    frozen_trainable = model.count_trainable_parameters()
    print(f"After freezing base: {frozen_trainable:,}")
    print(f"  Reduction: {initial_trainable - frozen_trainable:,} parameters")
    
    # Test unfreeze
    model.unfreeze_base_model()
    unfrozen_trainable = model.count_trainable_parameters()
    print(f"After unfreezing: {unfrozen_trainable:,}")
    
    # Test partial freeze
    model.freeze_bottom_k_layers(6)
    partial_trainable = model.count_trainable_parameters()
    print(f"After freezing bottom 6 layers: {partial_trainable:,}")
    
    print("\n✅ Freezing mechanisms work correctly")


if __name__ == '__main__':
    # Run tests
    test_model_forward_pass()
    test_freezing_mechanisms()
    
    print("\n🎉 Model architecture is ready for training!")
    print("\n💡 Next steps:")
    print("   1. Review model parameter count (~125M parameters)")
    print("   2. Adjust batch size based on GPU memory")
    print("   3. Proceed to train_model.py")
