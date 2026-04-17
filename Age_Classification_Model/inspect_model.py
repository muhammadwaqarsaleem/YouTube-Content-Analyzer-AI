"""
Model Inspection & Debugging Utility
=====================================
Tools for analyzing the Hierarchical RoBERTa model architecture,
visualizing attention weights, and diagnosing training issues.

Usage:
    python inspect_model.py
"""

import torch
import torch.nn as nn
from hierarchical_roberta import create_model, ModelConfig
from typing import Dict, List, Tuple
import json


class ModelInspector:
    """Comprehensive model inspection and analysis tools."""
    
    def __init__(self, model: nn.Module):
        """
        Initialize inspector.
        
        Args:
            model: Hierarchical RoBERTa model to inspect
        """
        self.model = model
        self.device = next(model.parameters()).device
    
    def print_architecture_summary(self) -> None:
        """Print detailed architecture summary."""
        print("\n" + "=" * 80)
        print("🏗️  MODEL ARCHITECTURE SUMMARY")
        print("=" * 80)
        
        print("\n📊 Configuration:")
        config = self.model.config
        print(f"   Base Model: {config.base_model_name}")
        print(f"   Hidden Size: {config.hidden_size}")
        print(f"   Max Chunks: {config.max_chunks}")
        print(f"   Chunk Size: {config.chunk_size}")
        print(f"   Num Classes: {config.num_classes}")
        print(f"   Pooling Strategy: {config.pooling_strategy}")
        print(f"   Gradient Checkpointing: {config.use_gradient_checkpointing}")
        
        print("\n🔢 Parameters:")
        breakdown = self.model.get_parameter_breakdown()
        total = breakdown['total']
        
        for name, count in breakdown.items():
            if name != 'total':
                pct = (count / total) * 100
                print(f"   {name:<25}: {count:>12,} ({pct:>5.1f}%)")
        
        print(f"   {'─' * 25}   {'─' * 12}   {'─' * 6}")
        print(f"   {'Total':<25}: {total:>12,} (100.0%)")
        
        trainable = self.model.count_trainable_parameters()
        trainable_pct = (trainable / total) * 100
        print(f"\n   Trainable: {trainable:,} ({trainable_pct:.1f}%)")
        print(f"   Frozen: {total - trainable:,} ({100-trainable_pct:.1f}%)")
        
        # Memory estimate
        param_memory_mb = (total * 4) / (1024 ** 2)  # float32
        print(f"\n💾 Estimated Memory:")
        print(f"   Parameters (FP32): {param_memory_mb:.1f} MB")
        print(f"   Parameters (FP16): {param_memory_mb / 2:.1f} MB")
        
        print()
    
    def analyze_layer_freezing(self) -> None:
        """Analyze which layers are frozen."""
        print("\n" + "=" * 80)
        print("🔒 LAYER FREEZING ANALYSIS")
        print("=" * 80)
        
        # Check embeddings
        embed_frozen = not any(p.requires_grad for p in self.model.roberta.embeddings.parameters())
        print(f"\n{'Embeddings':<30}: {'🔒 Frozen' if embed_frozen else '🔓 Trainable'}")
        
        # Check encoder layers
        if hasattr(self.model.roberta, 'encoder'):
            print(f"\n{'RoBERTa Encoder Layers:':<30}")
            for i, layer in enumerate(self.model.roberta.encoder.layer):
                layer_frozen = not any(p.requires_grad for p in layer.parameters())
                status = '🔒 Frozen' if layer_frozen else '🔓 Trainable'
                print(f"   Layer {i:<2}: {status}")
        
        # Check attention aggregator
        if self.model.chunk_aggregator is not None:
            attn_frozen = not any(p.requires_grad for p in self.model.chunk_aggregator.parameters())
            print(f"\n{'Attention Aggregator':<30}: {'🔒 Frozen' if attn_frozen else '🔓 Trainable'}")
        
        # Check classifier
        classifier_frozen = not any(p.requires_grad for p in self.model.classifier.parameters())
        print(f"{'Classification Head':<30}: {'🔒 Frozen' if classifier_frozen else '🔓 Trainable'}")
        
        print()
    
    def test_forward_pass(
        self, 
        batch_size: int = 2,
        verbose: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Test forward pass with dummy data.
        
        Args:
            batch_size: Number of samples in test batch
            verbose: Print detailed shape information
        
        Returns:
            Dictionary of outputs
        """
        if verbose:
            print("\n" + "=" * 80)
            print("🧪 TESTING FORWARD PASS")
            print("=" * 80)
        
        # Create dummy inputs
        max_chunks = self.model.config.max_chunks
        chunk_size = self.model.config.chunk_size
        
        input_ids = torch.randint(
            0, 50265,  # RoBERTa vocab size
            (batch_size, max_chunks, chunk_size),
            device=self.device
        )
        attention_mask = torch.ones(
            batch_size, max_chunks, chunk_size,
            dtype=torch.long,
            device=self.device
        )
        
        # Add some padding for realism
        input_ids[:, 15:, :] = 1  # Pad last 5 chunks
        attention_mask[:, 15:, :] = 0
        
        if verbose:
            print(f"\n📥 Input Shapes:")
            print(f"   input_ids: {tuple(input_ids.shape)}")
            print(f"   attention_mask: {tuple(attention_mask.shape)}")
        
        # Forward pass
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask, return_attention=True)
        
        if verbose:
            print(f"\n📤 Output Shapes:")
            for key, value in outputs.items():
                if isinstance(value, torch.Tensor):
                    print(f"   {key}: {tuple(value.shape)}")
            
            # Analyze logits
            logits = outputs['logits']
            print(f"\n📊 Logit Statistics:")
            print(f"   Mean: {logits.mean():.4f}")
            print(f"   Std: {logits.std():.4f}")
            print(f"   Min: {logits.min():.4f}")
            print(f"   Max: {logits.max():.4f}")
            
            # Predictions
            predictions = torch.argmax(logits, dim=1)
            print(f"\n🎯 Predictions: {predictions.tolist()}")
            
            # Attention weights
            if 'attention_weights' in outputs:
                attn = outputs['attention_weights']
                print(f"\n⚖️  Attention Analysis:")
                print(f"   Shape: {tuple(attn.shape)}")
                print(f"   Sum (should be ~1.0): {attn[0].sum():.4f}")
                print(f"   Max weight: {attn[0].max():.4f}")
                print(f"   Min weight: {attn[0].min():.4f}")
                
                # Top attended chunks
                top_k = min(5, max_chunks)
                top_indices = torch.topk(attn[0], top_k).indices
                print(f"   Top {top_k} chunks: {top_indices.tolist()}")
        
        if verbose:
            print("\n✅ Forward pass completed successfully!")
            print()
        
        return outputs
    
    def estimate_training_memory(
        self, 
        batch_size: int = 4,
        use_mixed_precision: bool = False
    ) -> Dict[str, float]:
        """
        Estimate training memory requirements.
        
        Args:
            batch_size: Training batch size
            use_mixed_precision: Whether using FP16
        
        Returns:
            Dictionary with memory estimates in MB
        """
        print("\n" + "=" * 80)
        print("💾 TRAINING MEMORY ESTIMATION")
        print("=" * 80)
        
        total_params = self.model.count_parameters()
        bytes_per_param = 2 if use_mixed_precision else 4
        
        # Model parameters
        param_memory = (total_params * bytes_per_param) / (1024 ** 2)
        
        # Gradients (same size as parameters)
        gradient_memory = param_memory
        
        # Optimizer states (AdamW stores 2 states per parameter)
        optimizer_memory = param_memory * 2
        
        # Activations (rough estimate)
        # Each chunk creates ~768-dim activation
        max_chunks = self.model.config.max_chunks
        hidden_size = self.model.config.hidden_size
        activation_memory = (batch_size * max_chunks * hidden_size * 4) / (1024 ** 2)
        
        # Gradient checkpointing reduces activation memory by ~60%
        if self.model.config.use_gradient_checkpointing:
            activation_memory *= 0.4
        
        # Workspace buffer
        workspace_memory = 500  # MB (rough estimate)
        
        total = (param_memory + gradient_memory + optimizer_memory + 
                activation_memory + workspace_memory)
        
        precision_str = "FP16 (Mixed Precision)" if use_mixed_precision else "FP32"
        checkpoint_str = "Enabled" if self.model.config.use_gradient_checkpointing else "Disabled"
        
        print(f"\n⚙️  Configuration:")
        print(f"   Batch Size: {batch_size}")
        print(f"   Precision: {precision_str}")
        print(f"   Gradient Checkpointing: {checkpoint_str}")
        
        print(f"\n📊 Memory Breakdown:")
        print(f"   Parameters:            {param_memory:>8.1f} MB")
        print(f"   Gradients:             {gradient_memory:>8.1f} MB")
        print(f"   Optimizer States:      {optimizer_memory:>8.1f} MB")
        print(f"   Activations:           {activation_memory:>8.1f} MB")
        print(f"   Workspace:             {workspace_memory:>8.1f} MB")
        print(f"   {'─' * 30}  {'─' * 8}")
        print(f"   Total Estimated:       {total:>8.1f} MB ({total/1024:.2f} GB)")
        
        print(f"\n💡 GPU Recommendations:")
        if total < 4000:
            print(f"   ✅ 4-6 GB GPU sufficient")
        elif total < 8000:
            print(f"   ✅ 8 GB GPU recommended")
        elif total < 12000:
            print(f"   ⚠️  12 GB GPU recommended")
        else:
            print(f"   ⚠️  16+ GB GPU required")
        
        print()
        
        return {
            'parameters_mb': param_memory,
            'gradients_mb': gradient_memory,
            'optimizer_mb': optimizer_memory,
            'activations_mb': activation_memory,
            'workspace_mb': workspace_memory,
            'total_mb': total
        }
    
    def visualize_attention_pattern(
        self,
        sample_input_ids: torch.Tensor,
        sample_attention_mask: torch.Tensor,
        save_path: str = "attention_visualization.txt"
    ) -> None:
        """
        Visualize attention weights for a sample.
        
        Args:
            sample_input_ids: [max_chunks, chunk_size]
            sample_attention_mask: [max_chunks, chunk_size]
            save_path: Where to save visualization
        """
        print("\n" + "=" * 80)
        print("🎨 ATTENTION PATTERN VISUALIZATION")
        print("=" * 80)
        
        # Add batch dimension
        input_ids = sample_input_ids.unsqueeze(0).to(self.device)
        attention_mask = sample_attention_mask.unsqueeze(0).to(self.device)
        
        # Get attention weights
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask, return_attention=True)
        
        if 'attention_weights' not in outputs:
            print("⚠️  Model does not return attention weights")
            return
        
        attn_weights = outputs['attention_weights'][0].cpu().numpy()
        
        # Create visualization
        print("\n📊 Attention Distribution:")
        print("   (Bar width represents attention weight)")
        print()
        
        max_width = 50
        for chunk_idx, weight in enumerate(attn_weights):
            # Check if chunk is padding
            is_padding = (sample_input_ids[chunk_idx] == 1).all()
            
            if is_padding:
                continue  # Skip padding chunks
            
            # Create bar
            bar_width = int(weight * max_width)
            bar = '█' * bar_width
            
            print(f"   Chunk {chunk_idx:2d}: {bar} {weight:.4f}")
        
        print(f"\n   Sum: {attn_weights.sum():.4f} (should be ~1.0)")
        
        # Save to file
        with open(save_path, 'w') as f:
            f.write("Attention Weight Visualization\n")
            f.write("=" * 60 + "\n\n")
            
            for chunk_idx, weight in enumerate(attn_weights):
                is_padding = (sample_input_ids[chunk_idx] == 1).all()
                if not is_padding:
                    bar_width = int(weight * max_width)
                    bar = '█' * bar_width
                    f.write(f"Chunk {chunk_idx:2d}: {bar} {weight:.4f}\n")
        
        print(f"\n💾 Visualization saved to: {save_path}")
        print()
    
    def check_gradient_flow(self) -> None:
        """Check which parameters will receive gradients."""
        print("\n" + "=" * 80)
        print("🌊 GRADIENT FLOW ANALYSIS")
        print("=" * 80)
        
        print("\n📊 Trainable Parameter Groups:")
        
        groups = {
            'RoBERTa Embeddings': [],
            'RoBERTa Encoder': [],
            'Attention': [],
            'Classifier': []
        }
        
        # Categorize parameters
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if 'embeddings' in name:
                    groups['RoBERTa Embeddings'].append((name, param.numel()))
                elif 'encoder' in name:
                    groups['RoBERTa Encoder'].append((name, param.numel()))
                elif 'chunk_aggregator' in name or 'attention' in name:
                    groups['Attention'].append((name, param.numel()))
                elif 'classifier' in name:
                    groups['Classifier'].append((name, param.numel()))
        
        for group_name, params in groups.items():
            if params:
                total = sum(p[1] for p in params)
                print(f"\n   {group_name}:")
                print(f"      Total: {total:,} parameters")
                print(f"      Layers: {len(params)}")
                if len(params) <= 5:
                    for param_name, param_count in params:
                        print(f"         • {param_name}: {param_count:,}")
            else:
                print(f"\n   {group_name}: 🔒 All frozen")
        
        print()
    
    def export_config(self, filepath: str = "model_config.json") -> None:
        """Export model configuration to JSON."""
        config_dict = {
            'base_model_name': self.model.config.base_model_name,
            'hidden_size': self.model.config.hidden_size,
            'num_classes': self.model.config.num_classes,
            'max_chunks': self.model.config.max_chunks,
            'chunk_size': self.model.config.chunk_size,
            'attention_hidden_size': self.model.config.attention_hidden_size,
            'attention_heads': self.model.config.attention_heads,
            'attention_dropout': self.model.config.attention_dropout,
            'classifier_dropout': self.model.config.classifier_dropout,
            'hidden_dropout': self.model.config.hidden_dropout,
            'use_gradient_checkpointing': self.model.config.use_gradient_checkpointing,
            'pooling_strategy': self.model.config.pooling_strategy,
            'total_parameters': self.model.count_parameters(),
            'trainable_parameters': self.model.count_trainable_parameters()
        }
        
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        print(f"✅ Config exported to: {filepath}")


def main():
    """Run comprehensive model inspection."""
    print("\n" + "=" * 80)
    print("🔍 HIERARCHICAL ROBERTA MODEL INSPECTOR")
    print("=" * 80)
    
    # Create model
    print("\n📦 Creating model...")
    model = create_model(
        num_classes=3,
        max_chunks=20,
        use_gradient_checkpointing=True
    )
    
    # Initialize inspector
    inspector = ModelInspector(model)
    
    # Run inspections
    inspector.print_architecture_summary()
    inspector.analyze_layer_freezing()
    inspector.check_gradient_flow()
    inspector.test_forward_pass(batch_size=2, verbose=True)
    inspector.estimate_training_memory(batch_size=4, use_mixed_precision=False)
    
    # Export config
    inspector.export_config()
    
    print("\n" + "=" * 80)
    print("✅ INSPECTION COMPLETE")
    print("=" * 80)
    print("\n💡 Model is ready for training!")
    print()


if __name__ == '__main__':
    main()
