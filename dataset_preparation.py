"""
Dataset Preparation for Hierarchical Attention Network (HAN)
=============================================================
Preprocessing pipeline for YouTube video age classification using RoBERTa.

This script implements hierarchical chunking to handle long transcripts (avg ~10,000 words)
that exceed BERT's 512-token limit. Creates train/val/test splits with proper tokenization.

Author: University AI Research Team
Project: YouTube Age Classification with Deep Learning
File: 1/4 - Dataset Preparation
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset
from transformers import RobertaTokenizer
from sklearn.model_selection import train_test_split
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings
from tqdm import tqdm
import json
from datetime import datetime

warnings.filterwarnings('ignore')


@dataclass
class ChunkingConfig:
    """Configuration for hierarchical chunking."""
    chunk_size: int = 512          # Max tokens per chunk (RoBERTa limit)
    max_chunks: int = 20           # Max chunks per transcript (~10K words)
    stride: int = 256              # Sliding window overlap (50% overlap)
    model_name: str = "roberta-base"


@dataclass
class SplitConfig:
    """Configuration for train/val/test split."""
    train_size: float = 0.8
    val_size: float = 0.1
    test_size: float = 0.1
    random_state: int = 42
    stratify: bool = True


@dataclass
class DatasetStatistics:
    """Statistics for a dataset split."""
    name: str
    total_samples: int
    class_distribution: Dict[str, int]
    class_percentages: Dict[str, float]
    avg_chunks_per_sample: float
    tensor_shape: Tuple[int, ...]
    total_tokens: int


class HierarchicalDatasetPreparator:
    """
    Prepares hierarchical datasets for long-document classification.
    
    Implements sliding window chunking to handle transcripts exceeding
    the 512-token limit of transformer models.
    """
    
    def __init__(self, 
                 csv_path: str,
                 output_dir: str = "data/processed_tensors",
                 chunking_config: Optional[ChunkingConfig] = None,
                 split_config: Optional[SplitConfig] = None):
        """
        Initialize the dataset preparator.
        
        Args:
            csv_path: Path to the labeled CSV file
            output_dir: Directory to save processed tensors
            chunking_config: Chunking parameters
            split_config: Split parameters
        """
        self.csv_path = Path(csv_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Configurations
        self.chunk_config = chunking_config or ChunkingConfig()
        self.split_config = split_config or SplitConfig()
        
        # Label mapping
        self.label_map = {'General': 0, 'Teen': 1, 'Mature': 2}
        self.reverse_label_map = {0: 'General', 1: 'Teen', 2: 'Mature'}
        
        # Data
        self.df: Optional[pd.DataFrame] = None
        self.tokenizer: Optional[RobertaTokenizer] = None
        
        # Datasets
        self.train_dataset: Optional[TensorDataset] = None
        self.val_dataset: Optional[TensorDataset] = None
        self.test_dataset: Optional[TensorDataset] = None
        
        # Statistics
        self.stats: Dict[str, DatasetStatistics] = {}
        
        print("🔧 Dataset Preparator Initialized")
        print(f"   Chunk size: {self.chunk_config.chunk_size} tokens")
        print(f"   Max chunks: {self.chunk_config.max_chunks}")
        print(f"   Stride (overlap): {self.chunk_config.stride} tokens")
        print(f"   Split: {self.split_config.train_size:.0%} train, "
              f"{self.split_config.val_size:.0%} val, "
              f"{self.split_config.test_size:.0%} test\n")
    
    def load_tokenizer(self) -> None:
        """Load RoBERTa tokenizer from Hugging Face."""
        print("=" * 80)
        print("📥 LOADING TOKENIZER")
        print("=" * 80)
        
        print(f"Loading {self.chunk_config.model_name} tokenizer...")
        self.tokenizer = RobertaTokenizer.from_pretrained(self.chunk_config.model_name)
        
        print(f"✅ Tokenizer loaded successfully")
        print(f"   Vocab size: {self.tokenizer.vocab_size:,}")
        print(f"   Special tokens: PAD={self.tokenizer.pad_token_id}, "
              f"CLS={self.tokenizer.cls_token_id}, "
              f"SEP={self.tokenizer.sep_token_id}\n")
    
    def load_data(self) -> bool:
        """
        Load CSV data and validate required columns.
        
        Returns:
            bool: True if successful
        """
        print("=" * 80)
        print("📂 LOADING DATASET")
        print("=" * 80)
        
        if not self.csv_path.exists():
            print(f"❌ Error: File not found at {self.csv_path}")
            return False
        
        try:
            file_size_mb = self.csv_path.stat().st_size / (1024 * 1024)
            print(f"📊 File: {self.csv_path.name} ({file_size_mb:.1f} MB)")
            print(f"⏳ Loading CSV...")
            
            self.df = pd.read_csv(self.csv_path, low_memory=False)
            print(f"✅ Loaded {len(self.df):,} samples\n")
            
            # Validate required columns
            required_cols = ['transcript', 'Age_Label']
            missing_cols = [col for col in required_cols if col not in self.df.columns]
            
            if missing_cols:
                print(f"❌ Error: Missing required columns: {missing_cols}")
                return False
            
            # Display label distribution
            print("📊 Original Label Distribution:")
            label_counts = self.df['Age_Label'].value_counts()
            for label, count in label_counts.items():
                pct = (count / len(self.df)) * 100
                print(f"   • {label}: {count:,} ({pct:.2f}%)")
            print()
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading data: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def prepare_labels(self) -> np.ndarray:
        """
        Map Age_Label strings to integers.
        
        Returns:
            np.ndarray: Integer-encoded labels
        """
        print("🏷️  Mapping labels to integers...")
        
        # Map labels
        labels = self.df['Age_Label'].map(self.label_map).values
        
        # Check for unmapped labels (NaN)
        if np.isnan(labels).any():
            unmapped = self.df[self.df['Age_Label'].map(self.label_map).isna()]['Age_Label'].unique()
            print(f"⚠️  Warning: Found unmapped labels: {unmapped}")
            print("   These will be dropped.")
            
            # Remove unmapped
            valid_mask = ~np.isnan(labels)
            self.df = self.df[valid_mask].reset_index(drop=True)
            labels = labels[valid_mask]
        
        print(f"✅ Label mapping complete:")
        for label_str, label_int in self.label_map.items():
            count = (labels == label_int).sum()
            print(f"   {label_str} → {label_int}: {count:,} samples")
        print()
        
        return labels.astype(np.int64)
    
    def hierarchical_chunk_transcript(self, transcript: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Chunk a long transcript into multiple 512-token segments using sliding window.
        
        This is the CRITICAL hierarchical preprocessing step that enables the HAN
        to process documents exceeding the 512-token transformer limit.
        
        Args:
            transcript: Raw text transcript
            
        Returns:
            Tuple of:
                - input_ids: Tensor of shape (max_chunks, chunk_size)
                - attention_mask: Tensor of shape (max_chunks, chunk_size)
        """
        # Tokenize full transcript (without truncation)
        encoded = self.tokenizer(
            transcript,
            add_special_tokens=True,
            truncation=False,  # Don't truncate - we'll chunk manually
            return_tensors=None  # Get lists, not tensors yet
        )
        
        input_ids = encoded['input_ids']
        attention_mask = encoded['attention_mask']
        
        # Initialize output tensors with padding
        chunk_input_ids = torch.full(
            (self.chunk_config.max_chunks, self.chunk_config.chunk_size),
            self.tokenizer.pad_token_id,
            dtype=torch.long
        )
        chunk_attention_mask = torch.zeros(
            (self.chunk_config.max_chunks, self.chunk_config.chunk_size),
            dtype=torch.long
        )
        
        # Sliding window chunking
        total_tokens = len(input_ids)
        chunk_idx = 0
        start_idx = 0
        
        while start_idx < total_tokens and chunk_idx < self.chunk_config.max_chunks:
            # Extract chunk
            end_idx = min(start_idx + self.chunk_config.chunk_size, total_tokens)
            chunk_tokens = input_ids[start_idx:end_idx]
            chunk_mask = attention_mask[start_idx:end_idx]
            
            # Store in tensor (will be padded if shorter than chunk_size)
            actual_length = len(chunk_tokens)
            chunk_input_ids[chunk_idx, :actual_length] = torch.tensor(chunk_tokens)
            chunk_attention_mask[chunk_idx, :actual_length] = torch.tensor(chunk_mask)
            
            # Move window forward with stride (overlap)
            start_idx += self.chunk_config.stride
            chunk_idx += 1
        
        return chunk_input_ids, chunk_attention_mask
    
    def process_transcripts(self, 
                           transcripts: pd.Series,
                           labels: np.ndarray,
                           split_name: str) -> TensorDataset:
        """
        Process a batch of transcripts into hierarchical tensor format.
        
        Args:
            transcripts: Series of transcript texts
            labels: Array of integer labels
            split_name: Name for progress bar (train/val/test)
            
        Returns:
            TensorDataset with (input_ids, attention_mask, labels)
        """
        print(f"⏳ Processing {split_name} transcripts with hierarchical chunking...")
        
        all_input_ids = []
        all_attention_masks = []
        chunk_counts = []
        
        # Process each transcript with progress bar
        for transcript in tqdm(transcripts, desc=f"Tokenizing {split_name}", unit="doc"):
            input_ids, attention_mask = self.hierarchical_chunk_transcript(str(transcript))
            
            all_input_ids.append(input_ids)
            all_attention_masks.append(attention_mask)
            
            # Count non-padding chunks
            non_empty_chunks = (input_ids != self.tokenizer.pad_token_id).any(dim=1).sum().item()
            chunk_counts.append(non_empty_chunks)
        
        # Stack into tensors
        input_ids_tensor = torch.stack(all_input_ids)      # (num_samples, max_chunks, chunk_size)
        attention_mask_tensor = torch.stack(all_attention_masks)
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        
        # Statistics
        avg_chunks = np.mean(chunk_counts)
        print(f"   ✅ Processed {len(transcripts):,} documents")
        print(f"   📊 Average chunks per document: {avg_chunks:.1f}")
        print(f"   📏 Tensor shape: {tuple(input_ids_tensor.shape)}")
        print()
        
        # Store statistics
        class_dist = {self.reverse_label_map[i]: (labels == i).sum() 
                     for i in range(len(self.label_map))}
        class_pct = {k: (v / len(labels)) * 100 for k, v in class_dist.items()}
        
        self.stats[split_name] = DatasetStatistics(
            name=split_name,
            total_samples=len(transcripts),
            class_distribution=class_dist,
            class_percentages=class_pct,
            avg_chunks_per_sample=avg_chunks,
            tensor_shape=tuple(input_ids_tensor.shape),
            total_tokens=input_ids_tensor.numel()
        )
        
        # Create TensorDataset
        dataset = TensorDataset(input_ids_tensor, attention_mask_tensor, labels_tensor)
        
        return dataset
    
    def create_splits(self) -> bool:
        """
        Create stratified train/val/test splits.
        
        Returns:
            bool: True if successful
        """
        print("=" * 80)
        print("✂️  CREATING STRATIFIED SPLITS")
        print("=" * 80)
        
        try:
            # Prepare labels
            labels = self.prepare_labels()
            transcripts = self.df['transcript']
            
            # Validate split ratios
            total_ratio = (self.split_config.train_size + 
                          self.split_config.val_size + 
                          self.split_config.test_size)
            if abs(total_ratio - 1.0) > 0.01:
                print(f"⚠️  Warning: Split ratios sum to {total_ratio:.2f}, not 1.0")
            
            # First split: Train vs (Val + Test)
            train_transcripts, temp_transcripts, train_labels, temp_labels = train_test_split(
                transcripts,
                labels,
                test_size=(self.split_config.val_size + self.split_config.test_size),
                random_state=self.split_config.random_state,
                stratify=labels if self.split_config.stratify else None
            )
            
            # Second split: Val vs Test
            val_ratio = self.split_config.val_size / (self.split_config.val_size + self.split_config.test_size)
            val_transcripts, test_transcripts, val_labels, test_labels = train_test_split(
                temp_transcripts,
                temp_labels,
                test_size=(1 - val_ratio),
                random_state=self.split_config.random_state,
                stratify=temp_labels if self.split_config.stratify else None
            )
            
            print(f"📊 Split Sizes:")
            print(f"   • Train: {len(train_transcripts):,} samples ({len(train_transcripts)/len(self.df)*100:.1f}%)")
            print(f"   • Val:   {len(val_transcripts):,} samples ({len(val_transcripts)/len(self.df)*100:.1f}%)")
            print(f"   • Test:  {len(test_transcripts):,} samples ({len(test_transcripts)/len(self.df)*100:.1f}%)")
            print()
            
            # Process each split
            print("=" * 80)
            print("🔄 HIERARCHICAL TOKENIZATION & CHUNKING")
            print("=" * 80)
            print(f"This process handles {len(self.df):,} transcripts averaging ~10,000 words.")
            print(f"Using sliding window with {self.chunk_config.stride}-token stride for context preservation.\n")
            
            self.train_dataset = self.process_transcripts(
                train_transcripts.reset_index(drop=True),
                train_labels,
                "train"
            )
            
            self.val_dataset = self.process_transcripts(
                val_transcripts.reset_index(drop=True),
                val_labels,
                "val"
            )
            
            self.test_dataset = self.process_transcripts(
                test_transcripts.reset_index(drop=True),
                test_labels,
                "test"
            )
            
            return True
            
        except Exception as e:
            print(f"❌ Error creating splits: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def save_datasets(self) -> bool:
        """
        Save processed datasets to disk.
        
        Returns:
            bool: True if successful
        """
        print("=" * 80)
        print("💾 SAVING PROCESSED DATASETS")
        print("=" * 80)
        
        try:
            # Save each split
            splits = [
                ('train', self.train_dataset),
                ('val', self.val_dataset),
                ('test', self.test_dataset)
            ]
            
            for split_name, dataset in splits:
                # Extract tensors
                input_ids = dataset.tensors[0]
                attention_mask = dataset.tensors[1]
                labels = dataset.tensors[2]
                
                # Save as PyTorch tensors
                save_path = self.output_dir / f"{split_name}_dataset.pt"
                torch.save({
                    'input_ids': input_ids,
                    'attention_mask': attention_mask,
                    'labels': labels
                }, save_path)
                
                file_size_mb = save_path.stat().st_size / (1024 * 1024)
                print(f"✅ Saved {split_name} dataset: {save_path.name} ({file_size_mb:.1f} MB)")
            
            # Save metadata
            metadata = {
                'label_map': self.label_map,
                'reverse_label_map': self.reverse_label_map,
                'chunk_size': self.chunk_config.chunk_size,
                'max_chunks': self.chunk_config.max_chunks,
                'stride': self.chunk_config.stride,
                'model_name': self.chunk_config.model_name,
                'train_size': len(self.train_dataset),
                'val_size': len(self.val_dataset),
                'test_size': len(self.test_dataset),
                'created_at': datetime.now().isoformat()
            }
            
            metadata_path = self.output_dir / 'metadata.json'
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"✅ Saved metadata: {metadata_path.name}")
            print(f"\n📁 All files saved to: {self.output_dir}/")
            print()
            
            return True
            
        except Exception as e:
            print(f"❌ Error saving datasets: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def print_summary(self) -> None:
        """Print comprehensive summary of processed datasets."""
        print("=" * 80)
        print("📊 DATASET PREPARATION SUMMARY")
        print("=" * 80)
        
        print(f"\n🔧 Configuration:")
        print(f"   Model: {self.chunk_config.model_name}")
        print(f"   Chunk Size: {self.chunk_config.chunk_size} tokens")
        print(f"   Max Chunks: {self.chunk_config.max_chunks}")
        print(f"   Stride: {self.chunk_config.stride} tokens")
        print(f"   Overlap: {((self.chunk_config.chunk_size - self.chunk_config.stride) / self.chunk_config.chunk_size * 100):.0f}%")
        
        print(f"\n📈 Dataset Splits:")
        print("-" * 80)
        print(f"{'Split':<8} {'Samples':<10} {'Shape':<25} {'Avg Chunks':<12}")
        print("-" * 80)
        
        for split_name in ['train', 'val', 'test']:
            if split_name in self.stats:
                stat = self.stats[split_name]
                shape_str = f"{stat.tensor_shape}"
                print(f"{split_name:<8} {stat.total_samples:<10,} {shape_str:<25} {stat.avg_chunks_per_sample:<12.1f}")
        
        print()
        print(f"\n🏷️  Class Distribution (Stratified):")
        print("-" * 80)
        
        for split_name in ['train', 'val', 'test']:
            if split_name in self.stats:
                stat = self.stats[split_name]
                print(f"\n{split_name.upper()}:")
                for label_name in ['General', 'Teen', 'Mature']:
                    count = stat.class_distribution.get(label_name, 0)
                    pct = stat.class_percentages.get(label_name, 0)
                    print(f"   • {label_name:<10}: {count:<6,} ({pct:>5.1f}%)")
        
        print(f"\n💾 Storage:")
        print("-" * 80)
        total_size_mb = sum(
            (self.output_dir / f"{split}_dataset.pt").stat().st_size 
            for split in ['train', 'val', 'test']
        ) / (1024 * 1024)
        print(f"   Total Size: {total_size_mb:.1f} MB")
        print(f"   Location: {self.output_dir}/")
        
        # Calculate total tokens
        total_tokens = sum(stat.total_tokens for stat in self.stats.values())
        print(f"\n🔢 Token Statistics:")
        print(f"   Total Tokens Processed: {total_tokens:,}")
        print(f"   Avg Tokens per Sample: {total_tokens / len(self.df):,.0f}")
        
        print()
        print("=" * 80)
        print("✅ DATASET PREPARATION COMPLETE")
        print("=" * 80)
        print(f"\n💡 Next Steps:")
        print(f"   1. Verify tensor shapes match HAN architecture requirements")
        print(f"   2. Load datasets in train_model.py using:")
        print(f"      train_data = torch.load('data/processed_tensors/train_dataset.pt')")
        print(f"   3. Create DataLoaders with appropriate batch size")
        print(f"   4. Proceed with hierarchical_roberta.py model definition")
        print()
    
    def run_pipeline(self) -> bool:
        """
        Execute the complete data preparation pipeline.
        
        Returns:
            bool: True if successful
        """
        print("\n")
        print("╔" + "=" * 78 + "╗")
        print("║" + " " * 78 + "║")
        print("║" + "   HIERARCHICAL DATASET PREPARATION PIPELINE".center(78) + "║")
        print("║" + "   YouTube Age Classification - Deep Learning".center(78) + "║")
        print("║" + " " * 78 + "║")
        print("╚" + "=" * 78 + "╝")
        print()
        
        # Step 1: Load tokenizer
        self.load_tokenizer()
        
        # Step 2: Load data
        if not self.load_data():
            return False
        
        # Step 3: Create splits with hierarchical chunking
        if not self.create_splits():
            return False
        
        # Step 4: Save datasets
        if not self.save_datasets():
            return False
        
        # Step 5: Print summary
        self.print_summary()
        
        return True


def main():
    """Main entry point for dataset preparation."""
    # ============================================================================
    # CONFIGURATION
    # ============================================================================
    INPUT_CSV_PATH = "training_ready_dataset.csv"
    OUTPUT_DIR = "data/processed_tensors"
    
    # Chunking configuration (tuned for ~10,000 word transcripts)
    chunking_config = ChunkingConfig(
        chunk_size=512,      # RoBERTa max sequence length
        max_chunks=20,       # 20 chunks × 512 tokens ≈ 10,000 words
        stride=256,          # 50% overlap for context preservation
        model_name="roberta-base"
    )
    
    # Split configuration
    split_config = SplitConfig(
        train_size=0.8,
        val_size=0.1,
        test_size=0.1,
        random_state=42,
        stratify=True
    )
    
    # ============================================================================
    # Run Pipeline
    # ============================================================================
    try:
        preparator = HierarchicalDatasetPreparator(
            csv_path=INPUT_CSV_PATH,
            output_dir=OUTPUT_DIR,
            chunking_config=chunking_config,
            split_config=split_config
        )
        
        success = preparator.run_pipeline()
        
        if success:
            print("=" * 80)
            print("✅ SUCCESS - Dataset preparation completed!")
            print("=" * 80)
            print(f"\n📂 Processed datasets ready at: {OUTPUT_DIR}/")
            print(f"   • train_dataset.pt")
            print(f"   • val_dataset.pt")
            print(f"   • test_dataset.pt")
            print(f"   • metadata.json")
            print()
            return 0
        else:
            print("=" * 80)
            print("❌ FAILED - Dataset preparation encountered errors")
            print("=" * 80)
            return 1
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user.")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    import sys
    sys.exit(main())
