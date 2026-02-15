"""
Dataset Verification Script
============================
Validates the output of dataset_preparation.py to ensure data quality
and correct tensor shapes for the HAN model.

Usage: python verify_datasets.py
"""

import torch
import json
from pathlib import Path
from typing import Dict, Tuple
import numpy as np


class DatasetVerifier:
    """Verify processed datasets are correct and ready for training."""
    
    def __init__(self, data_dir: str = "data/processed_tensors"):
        """
        Initialize verifier.
        
        Args:
            data_dir: Directory containing processed datasets
        """
        self.data_dir = Path(data_dir)
        self.metadata = None
        
    def load_metadata(self) -> bool:
        """Load and verify metadata file."""
        metadata_path = self.data_dir / 'metadata.json'
        
        if not metadata_path.exists():
            print(f"❌ Error: metadata.json not found in {self.data_dir}")
            return False
        
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        print("✅ Metadata loaded successfully\n")
        return True
    
    def verify_file_existence(self) -> bool:
        """Check all required files exist."""
        print("=" * 80)
        print("📂 CHECKING FILE EXISTENCE")
        print("=" * 80)
        
        required_files = ['train_dataset.pt', 'val_dataset.pt', 'test_dataset.pt', 'metadata.json']
        all_exist = True
        
        for filename in required_files:
            filepath = self.data_dir / filename
            if filepath.exists():
                size_mb = filepath.stat().st_size / (1024 * 1024)
                print(f"✅ {filename:<25} ({size_mb:>6.1f} MB)")
            else:
                print(f"❌ {filename:<25} MISSING")
                all_exist = False
        
        print()
        return all_exist
    
    def verify_tensor_shapes(self) -> bool:
        """Verify tensor shapes match expected dimensions."""
        print("=" * 80)
        print("📏 VERIFYING TENSOR SHAPES")
        print("=" * 80)
        
        expected_chunk_size = self.metadata.get('chunk_size', 512)
        expected_max_chunks = self.metadata.get('max_chunks', 20)
        
        all_valid = True
        
        for split in ['train', 'val', 'test']:
            print(f"\n{split.upper()} Dataset:")
            print("-" * 80)
            
            # Load dataset
            data = torch.load(self.data_dir / f"{split}_dataset.pt")
            
            # Get tensors
            input_ids = data['input_ids']
            attention_mask = data['attention_mask']
            labels = data['labels']
            
            # Expected shapes
            num_samples = len(labels)
            expected_input_shape = (num_samples, expected_max_chunks, expected_chunk_size)
            expected_mask_shape = (num_samples, expected_max_chunks, expected_chunk_size)
            expected_label_shape = (num_samples,)
            
            # Verify shapes
            if input_ids.shape == expected_input_shape:
                print(f"✅ input_ids shape:      {tuple(input_ids.shape)}")
            else:
                print(f"❌ input_ids shape:      {tuple(input_ids.shape)} "
                      f"(expected {expected_input_shape})")
                all_valid = False
            
            if attention_mask.shape == expected_mask_shape:
                print(f"✅ attention_mask shape: {tuple(attention_mask.shape)}")
            else:
                print(f"❌ attention_mask shape: {tuple(attention_mask.shape)} "
                      f"(expected {expected_mask_shape})")
                all_valid = False
            
            if labels.shape == expected_label_shape:
                print(f"✅ labels shape:         {tuple(labels.shape)}")
            else:
                print(f"❌ labels shape:         {tuple(labels.shape)} "
                      f"(expected {expected_label_shape})")
                all_valid = False
            
            # Data type checks
            if input_ids.dtype == torch.long:
                print(f"✅ input_ids dtype:      {input_ids.dtype}")
            else:
                print(f"❌ input_ids dtype:      {input_ids.dtype} (expected torch.long)")
                all_valid = False
            
            if labels.dtype == torch.long:
                print(f"✅ labels dtype:         {labels.dtype}")
            else:
                print(f"❌ labels dtype:         {labels.dtype} (expected torch.long)")
                all_valid = False
        
        print()
        return all_valid
    
    def verify_data_quality(self) -> bool:
        """Check for NaN, inf, and valid value ranges."""
        print("=" * 80)
        print("🔍 CHECKING DATA QUALITY")
        print("=" * 80)
        
        all_valid = True
        
        for split in ['train', 'val', 'test']:
            print(f"\n{split.upper()} Dataset:")
            print("-" * 80)
            
            data = torch.load(self.data_dir / f"{split}_dataset.pt")
            input_ids = data['input_ids']
            attention_mask = data['attention_mask']
            labels = data['labels']
            
            # Check for NaN
            if not torch.isnan(input_ids).any():
                print("✅ No NaN values in input_ids")
            else:
                print(f"❌ Found {torch.isnan(input_ids).sum()} NaN values in input_ids")
                all_valid = False
            
            # Check for inf
            if not torch.isinf(input_ids).any():
                print("✅ No inf values in input_ids")
            else:
                print(f"❌ Found {torch.isinf(input_ids).sum()} inf values in input_ids")
                all_valid = False
            
            # Check attention mask values (should be 0 or 1)
            unique_mask_values = attention_mask.unique()
            if set(unique_mask_values.tolist()).issubset({0, 1}):
                print(f"✅ Attention mask contains only 0 and 1")
            else:
                print(f"❌ Attention mask contains invalid values: {unique_mask_values.tolist()}")
                all_valid = False
            
            # Check label range (should be 0, 1, or 2)
            unique_labels = labels.unique()
            if set(unique_labels.tolist()).issubset({0, 1, 2}):
                print(f"✅ Labels are valid: {unique_labels.tolist()}")
            else:
                print(f"❌ Found invalid labels: {unique_labels.tolist()}")
                all_valid = False
            
            # Check for all-padding chunks
            non_padding_per_sample = (input_ids != 1).any(dim=2).sum(dim=1)
            all_padding_samples = (non_padding_per_sample == 0).sum()
            
            if all_padding_samples == 0:
                print(f"✅ No completely empty samples")
            else:
                print(f"⚠️  Warning: {all_padding_samples} samples have all-padding chunks")
        
        print()
        return all_valid
    
    def verify_class_distribution(self) -> bool:
        """Verify stratification maintained class distribution."""
        print("=" * 80)
        print("📊 VERIFYING CLASS DISTRIBUTION")
        print("=" * 80)
        
        distributions = {}
        
        for split in ['train', 'val', 'test']:
            data = torch.load(self.data_dir / f"{split}_dataset.pt")
            labels = data['labels']
            
            dist = {}
            for label in [0, 1, 2]:
                count = (labels == label).sum().item()
                pct = (count / len(labels)) * 100
                dist[label] = (count, pct)
            
            distributions[split] = dist
        
        # Print distributions
        print("\nClass Distribution Comparison:")
        print("-" * 80)
        print(f"{'Split':<10} {'General (0)':<20} {'Teen (1)':<20} {'Mature (2)':<20}")
        print("-" * 80)
        
        for split in ['train', 'val', 'test']:
            dist = distributions[split]
            general_str = f"{dist[0][0]:>5,} ({dist[0][1]:>5.1f}%)"
            teen_str = f"{dist[1][0]:>5,} ({dist[1][1]:>5.1f}%)"
            mature_str = f"{dist[2][0]:>5,} ({dist[2][1]:>5.1f}%)"
            print(f"{split:<10} {general_str:<20} {teen_str:<20} {mature_str:<20}")
        
        # Check if distributions are similar (within 5% tolerance)
        print("\nStratification Check:")
        print("-" * 80)
        
        train_dist = distributions['train']
        val_dist = distributions['val']
        test_dist = distributions['test']
        
        stratification_valid = True
        
        for label in [0, 1, 2]:
            label_name = {0: 'General', 1: 'Teen', 2: 'Mature'}[label]
            train_pct = train_dist[label][1]
            val_pct = val_dist[label][1]
            test_pct = test_dist[label][1]
            
            val_diff = abs(train_pct - val_pct)
            test_diff = abs(train_pct - test_pct)
            
            if val_diff < 5.0 and test_diff < 5.0:
                print(f"✅ {label_name}: Train-Val diff={val_diff:.1f}%, Train-Test diff={test_diff:.1f}%")
            else:
                print(f"⚠️  {label_name}: Train-Val diff={val_diff:.1f}%, Train-Test diff={test_diff:.1f}%")
                if val_diff >= 5.0 or test_diff >= 5.0:
                    stratification_valid = False
        
        print()
        return stratification_valid
    
    def verify_chunk_usage(self) -> None:
        """Analyze how many chunks are actually used per sample."""
        print("=" * 80)
        print("📈 CHUNK USAGE STATISTICS")
        print("=" * 80)
        
        for split in ['train', 'val', 'test']:
            data = torch.load(self.data_dir / f"{split}_dataset.pt")
            input_ids = data['input_ids']
            
            # Count non-padding chunks per sample
            # Padding token ID is typically 1 for RoBERTa
            non_padding_chunks = (input_ids != 1).any(dim=2).sum(dim=1)
            
            print(f"\n{split.upper()} Dataset:")
            print("-" * 80)
            print(f"   Average chunks used: {non_padding_chunks.float().mean():.2f}")
            print(f"   Median chunks used:  {non_padding_chunks.float().median():.0f}")
            print(f"   Min chunks used:     {non_padding_chunks.min()}")
            print(f"   Max chunks used:     {non_padding_chunks.max()}")
            
            # Histogram
            print("\n   Chunk usage distribution:")
            unique_counts = non_padding_chunks.unique(sorted=True)
            for count in unique_counts[:10]:  # Show first 10
                num_samples = (non_padding_chunks == count).sum()
                pct = (num_samples / len(input_ids)) * 100
                bar = '█' * int(pct / 2)
                print(f"      {count:>2} chunks: {num_samples:>5,} samples ({pct:>5.1f}%) {bar}")
        
        print()
    
    def print_summary(self) -> None:
        """Print configuration summary."""
        print("=" * 80)
        print("⚙️  CONFIGURATION SUMMARY")
        print("=" * 80)
        
        print(f"\nModel: {self.metadata.get('model_name', 'N/A')}")
        print(f"Chunk Size: {self.metadata.get('chunk_size', 'N/A')} tokens")
        print(f"Max Chunks: {self.metadata.get('max_chunks', 'N/A')}")
        print(f"Stride: {self.metadata.get('stride', 'N/A')} tokens")
        
        print(f"\nDataset Sizes:")
        print(f"   Train: {self.metadata.get('train_size', 'N/A'):,} samples")
        print(f"   Val:   {self.metadata.get('val_size', 'N/A'):,} samples")
        print(f"   Test:  {self.metadata.get('test_size', 'N/A'):,} samples")
        
        print(f"\nCreated: {self.metadata.get('created_at', 'N/A')}")
        print()
    
    def run_verification(self) -> bool:
        """Run complete verification pipeline."""
        print("\n" + "=" * 80)
        print("🔍 DATASET VERIFICATION PIPELINE")
        print("=" * 80 + "\n")
        
        # Load metadata
        if not self.load_metadata():
            return False
        
        # Check files
        if not self.verify_file_existence():
            print("❌ File check failed. Ensure dataset_preparation.py completed successfully.")
            return False
        
        # Verify shapes
        if not self.verify_tensor_shapes():
            print("❌ Shape verification failed.")
            return False
        
        # Verify data quality
        if not self.verify_data_quality():
            print("❌ Data quality check failed.")
            return False
        
        # Verify stratification
        self.verify_class_distribution()
        
        # Chunk usage
        self.verify_chunk_usage()
        
        # Summary
        self.print_summary()
        
        print("=" * 80)
        print("✅ VERIFICATION COMPLETE")
        print("=" * 80)
        print("\n💡 Next Steps:")
        print("   1. Datasets are ready for model training")
        print("   2. Proceed with hierarchical_roberta.py (model definition)")
        print("   3. Then train_model.py (training loop)")
        print()
        
        return True


def main():
    """Main entry point."""
    verifier = DatasetVerifier(data_dir="data/processed_tensors")
    success = verifier.run_verification()
    
    return 0 if success else 1


if __name__ == '__main__':
    import sys
    sys.exit(main())
