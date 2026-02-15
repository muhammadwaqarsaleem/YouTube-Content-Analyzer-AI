"""
Production-Grade Exploratory Data Analysis Script for NLP Dataset
==================================================================
A comprehensive data quality analyzer for YouTube video classification datasets.
Designed for large CSV files (>500MB) with memory efficiency and NLP-specific metrics.

Author: Research Team
Purpose: Pre-training dataset validation for age bracket classification
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import warnings
from datetime import datetime
from tqdm import tqdm
import sys

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Optional: langdetect for language detection
try:
    from langdetect import detect, LangDetectException
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False
    print("⚠️  Warning: 'langdetect' not installed. Language detection will be skipped.")
    print("   Install with: pip install langdetect\n")


@dataclass
class DatasetMetrics:
    """Data class to store dataset analysis metrics."""
    total_rows: int
    total_columns: int
    memory_usage_mb: float
    missing_values: Dict[str, Tuple[int, float]]
    duplicate_rows: int
    duplicate_percentage: float
    text_columns: List[str]
    numeric_columns: List[str]
    hidden_missing: Dict[str, int]
    text_statistics: Dict[str, Dict[str, float]]
    language_stats: Optional[Dict[str, float]] = None


class DatasetAnalyzer:
    """
    Comprehensive EDA analyzer for NLP datasets with focus on text data quality.
    
    This class performs structural health checks, NLP viability analysis,
    and generates reports and visualizations for research documentation.
    """
    
    def __init__(self, csv_path: str, output_dir: str = "./eda_outputs"):
        """
        Initialize the DatasetAnalyzer.
        
        Args:
            csv_path: Path to the CSV file to analyze
            output_dir: Directory to save output files (reports, plots)
        """
        self.csv_path = Path(csv_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.df: Optional[pd.DataFrame] = None
        self.metrics: Optional[DatasetMetrics] = None
        
        # Common placeholders for "missing" text data
        self.missing_placeholders = ['none', 'null', 'n/a', 'na', 'nan', 'missing', '']
        
    def load_data(self) -> bool:
        """
        Load CSV data with progress indication and error handling.
        
        Returns:
            bool: True if successful, False otherwise
        """
        print("=" * 80)
        print("📂 LOADING DATASET")
        print("=" * 80)
        
        if not self.csv_path.exists():
            print(f"❌ Error: File not found at {self.csv_path}")
            return False
        
        file_size_mb = self.csv_path.stat().st_size / (1024 * 1024)
        print(f"📊 File size: {file_size_mb:.2f} MB")
        print(f"📍 Loading from: {self.csv_path}")
        
        try:
            # Use low_memory=False for large files with mixed types
            print("⏳ Reading CSV... (this may take a moment for large files)")
            self.df = pd.read_csv(self.csv_path, low_memory=False)
            print(f"✅ Successfully loaded {len(self.df):,} rows × {len(self.df.columns)} columns\n")
            return True
        except FileNotFoundError:
            print(f"❌ Error: File not found at {self.csv_path}")
            return False
        except pd.errors.EmptyDataError:
            print("❌ Error: CSV file is empty")
            return False
        except Exception as e:
            print(f"❌ Error loading CSV: {str(e)}")
            return False
    
    def analyze_structure(self) -> None:
        """Analyze basic structural health and memory usage."""
        print("=" * 80)
        print("🏗️  STRUCTURAL HEALTH CHECK")
        print("=" * 80)
        
        # Basic dimensions
        total_rows, total_cols = self.df.shape
        print(f"📏 Dimensions: {total_rows:,} rows × {total_cols} columns")
        
        # Memory usage
        memory_usage = self.df.memory_usage(deep=True).sum() / (1024 * 1024)
        print(f"💾 Memory Usage: {memory_usage:.2f} MB")
        
        # Column types
        print(f"\n📋 Column Types Distribution:")
        type_counts = self.df.dtypes.value_counts()
        for dtype, count in type_counts.items():
            print(f"   • {dtype}: {count} columns")
        
        # Missing values analysis
        print(f"\n🔍 Missing Values Analysis:")
        missing_data = {}
        total_cells = total_rows * total_cols
        
        for col in self.df.columns:
            missing_count = self.df[col].isna().sum()
            missing_pct = (missing_count / total_rows) * 100
            if missing_count > 0:
                missing_data[col] = (missing_count, missing_pct)
                print(f"   • {col}: {missing_count:,} ({missing_pct:.2f}%)")
        
        if not missing_data:
            print("   ✅ No missing values detected!")
        
        # Duplicate rows analysis
        print(f"\n🔄 Duplicate Analysis:")
        duplicate_count = self.df.duplicated().sum()
        duplicate_pct = (duplicate_count / total_rows) * 100
        print(f"   • Exact duplicate rows: {duplicate_count:,} ({duplicate_pct:.2f}%)")
        
        # Check for duplicate IDs (if exists)
        id_columns = [col for col in self.df.columns if 'id' in col.lower() or 'url' in col.lower()]
        if id_columns:
            for id_col in id_columns:
                if self.df[id_col].dtype == 'object' or self.df[id_col].dtype == 'string':
                    dup_ids = self.df[id_col].duplicated().sum()
                    if dup_ids > 0:
                        print(f"   • Duplicate {id_col}: {dup_ids:,}")
        
        # Store metrics
        text_cols = self._identify_text_columns()
        numeric_cols = self._identify_numeric_columns()
        
        self.metrics = DatasetMetrics(
            total_rows=total_rows,
            total_columns=total_cols,
            memory_usage_mb=memory_usage,
            missing_values=missing_data,
            duplicate_rows=duplicate_count,
            duplicate_percentage=duplicate_pct,
            text_columns=text_cols,
            numeric_columns=numeric_cols,
            hidden_missing={},
            text_statistics={}
        )
        
        print()
    
    def _identify_text_columns(self) -> List[str]:
        """Identify columns likely containing text data."""
        text_cols = []
        for col in self.df.columns:
            if self.df[col].dtype == 'object' or self.df[col].dtype == 'string':
                # Sample a few non-null values
                sample = self.df[col].dropna().head(5)
                if len(sample) > 0:
                    # Check if values look like text (not just categorical)
                    avg_length = sample.astype(str).str.len().mean()
                    if avg_length > 20:  # Arbitrary threshold for "text" vs "category"
                        text_cols.append(col)
        return text_cols
    
    def _identify_numeric_columns(self) -> List[str]:
        """Identify numeric columns."""
        return self.df.select_dtypes(include=[np.number]).columns.tolist()
    
    def analyze_hidden_missing(self) -> None:
        """Detect hidden missing values in text columns (empty strings, whitespace, placeholders)."""
        print("=" * 80)
        print("🕵️  HIDDEN MISSING VALUES DETECTION")
        print("=" * 80)
        
        hidden_missing = {}
        
        for col in tqdm(self.metrics.text_columns, desc="Checking text columns", unit="col"):
            hidden_count = 0
            
            # Check for empty strings and whitespace
            if self.df[col].dtype == 'object':
                # Empty strings
                hidden_count += (self.df[col] == '').sum()
                # Whitespace only
                hidden_count += self.df[col].str.strip().eq('').sum()
                
                # Check for common placeholders
                for placeholder in self.missing_placeholders:
                    hidden_count += self.df[col].str.lower().str.strip().eq(placeholder).sum()
            
            if hidden_count > 0:
                hidden_missing[col] = hidden_count
                pct = (hidden_count / len(self.df)) * 100
                print(f"   • {col}: {hidden_count:,} hidden missing ({pct:.2f}%)")
        
        if not hidden_missing:
            print("   ✅ No hidden missing values detected!")
        
        self.metrics.hidden_missing = hidden_missing
        print()
    
    def analyze_text_statistics(self) -> None:
        """
        Analyze text statistics for NLP viability, including word counts and token limits.
        This is crucial for determining model architecture (BERT vs Longformer).
        """
        print("=" * 80)
        print("📝 NLP TEXT STATISTICS & TOKEN LIMIT ANALYSIS")
        print("=" * 80)
        
        text_stats = {}
        
        for col in tqdm(self.metrics.text_columns, desc="Analyzing text columns", unit="col"):
            print(f"\n📄 Column: {col}")
            print("-" * 60)
            
            # Get non-null, non-empty text
            valid_text = self.df[col].dropna()
            valid_text = valid_text[valid_text.astype(str).str.strip() != '']
            
            if len(valid_text) == 0:
                print("   ⚠️  No valid text data in this column")
                continue
            
            # Calculate word counts
            print("   ⏳ Calculating word counts...")
            word_counts = valid_text.astype(str).str.split().str.len()
            
            # Basic statistics
            mean_words = word_counts.mean()
            median_words = word_counts.median()
            min_words = word_counts.min()
            max_words = word_counts.max()
            
            # Percentiles for token limit analysis
            p90 = word_counts.quantile(0.90)
            p95 = word_counts.quantile(0.95)
            p99 = word_counts.quantile(0.99)
            
            # Critical metric: percentage exceeding 500 words
            exceeds_500 = (word_counts > 500).sum()
            pct_exceeds_500 = (exceeds_500 / len(word_counts)) * 100
            
            # Critical metric: percentage exceeding 512 tokens (approximate)
            # Rule of thumb: 1 word ≈ 1.3 tokens for English
            approx_tokens = word_counts * 1.3
            exceeds_512_tokens = (approx_tokens > 512).sum()
            pct_exceeds_512 = (exceeds_512_tokens / len(approx_tokens)) * 100
            
            print(f"   📊 Word Count Statistics:")
            print(f"      • Mean: {mean_words:.1f} words")
            print(f"      • Median: {median_words:.1f} words")
            print(f"      • Min: {min_words:.0f} words")
            print(f"      • Max: {max_words:.0f} words")
            print(f"\n   📈 Percentile Analysis:")
            print(f"      • 90th percentile: {p90:.1f} words")
            print(f"      • 95th percentile: {p95:.1f} words")
            print(f"      • 99th percentile: {p99:.1f} words")
            print(f"\n   🎯 Token Limit Impact (CRITICAL for Model Selection):")
            print(f"      • Texts > 500 words: {exceeds_500:,} ({pct_exceeds_500:.2f}%)")
            print(f"      • Texts > 512 tokens (est.): {exceeds_512_tokens:,} ({pct_exceeds_512:.2f}%)")
            
            if pct_exceeds_512 > 10:
                print(f"      ⚠️  WARNING: {pct_exceeds_512:.1f}% exceed BERT's 512 token limit!")
                print(f"      💡 RECOMMENDATION: Consider Longformer or truncation strategy")
            else:
                print(f"      ✅ BERT-compatible: Only {pct_exceeds_512:.1f}% exceed 512 tokens")
            
            # Store statistics
            text_stats[col] = {
                'mean_words': mean_words,
                'median_words': median_words,
                'min_words': min_words,
                'max_words': max_words,
                'p90_words': p90,
                'p95_words': p95,
                'p99_words': p99,
                'exceeds_500_count': exceeds_500,
                'exceeds_500_pct': pct_exceeds_500,
                'exceeds_512_tokens_count': exceeds_512_tokens,
                'exceeds_512_tokens_pct': pct_exceeds_512
            }
        
        self.metrics.text_statistics = text_stats
        print()
    
    def analyze_language_distribution(self, sample_size: int = 100) -> None:
        """
        Detect language distribution in text columns using langdetect.
        
        Args:
            sample_size: Number of random samples to check for language
        """
        if not LANGDETECT_AVAILABLE:
            print("=" * 80)
            print("🌍 LANGUAGE DISTRIBUTION ANALYSIS")
            print("=" * 80)
            print("⚠️  Skipped: langdetect library not available\n")
            return
        
        print("=" * 80)
        print("🌍 LANGUAGE DISTRIBUTION ANALYSIS")
        print("=" * 80)
        
        # Find the primary text column (usually longest average length)
        if not self.metrics.text_columns:
            print("   ⚠️  No text columns found for language detection\n")
            return
        
        # Select primary text column (e.g., transcript, description)
        primary_col = self.metrics.text_columns[0]
        if len(self.metrics.text_columns) > 1:
            # Try to find transcript or description column
            for col in self.metrics.text_columns:
                if 'transcript' in col.lower() or 'description' in col.lower():
                    primary_col = col
                    break
        
        print(f"📄 Analyzing language in column: '{primary_col}'")
        
        # Get valid text samples
        valid_text = self.df[primary_col].dropna()
        valid_text = valid_text[valid_text.astype(str).str.strip() != '']
        
        if len(valid_text) == 0:
            print("   ⚠️  No valid text data for language detection\n")
            return
        
        # Sample random texts
        sample_size = min(sample_size, len(valid_text))
        samples = valid_text.sample(n=sample_size, random_state=42)
        
        print(f"🔍 Detecting language in {sample_size} random samples...")
        
        language_counts = {}
        detection_failures = 0
        
        for text in tqdm(samples, desc="Detecting languages", unit="text"):
            try:
                lang = detect(str(text))
                language_counts[lang] = language_counts.get(lang, 0) + 1
            except LangDetectException:
                detection_failures += 1
        
        # Calculate percentages
        print(f"\n📊 Language Distribution:")
        for lang, count in sorted(language_counts.items(), key=lambda x: x[1], reverse=True):
            pct = (count / sample_size) * 100
            print(f"   • {lang.upper()}: {count} ({pct:.1f}%)")
        
        if detection_failures > 0:
            print(f"   • Failed to detect: {detection_failures} ({(detection_failures/sample_size)*100:.1f}%)")
        
        # Store metrics
        english_count = language_counts.get('en', 0)
        english_pct = (english_count / sample_size) * 100
        
        if english_pct < 90:
            print(f"\n   ⚠️  WARNING: Only {english_pct:.1f}% of samples are English!")
            print(f"   💡 Consider filtering non-English data or using multilingual models")
        else:
            print(f"\n   ✅ Dataset is predominantly English ({english_pct:.1f}%)")
        
        self.metrics.language_stats = {
            'distribution': language_counts,
            'english_percentage': english_pct,
            'detection_failures': detection_failures
        }
        
        print()
    
    def generate_visualizations(self) -> None:
        """Generate and save visualization plots."""
        print("=" * 80)
        print("📊 GENERATING VISUALIZATIONS")
        print("=" * 80)
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
        
        # 1. Missing Values Heatmap
        print("   📈 Creating missing values heatmap...")
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Calculate missing values percentage for all columns
        missing_pct = (self.df.isnull().sum() / len(self.df)) * 100
        missing_pct = missing_pct[missing_pct > 0].sort_values(ascending=False)
        
        if len(missing_pct) > 0:
            sns.barplot(x=missing_pct.values, y=missing_pct.index, palette='Reds_r', ax=ax)
            ax.set_xlabel('Missing Values (%)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Column Name', fontsize=12, fontweight='bold')
            ax.set_title('Missing Values Distribution Across Columns', fontsize=14, fontweight='bold')
            
            # Add percentage labels
            for i, v in enumerate(missing_pct.values):
                ax.text(v + 0.5, i, f'{v:.1f}%', va='center', fontweight='bold')
            
            plt.tight_layout()
            missing_plot_path = self.output_dir / 'missing_values_analysis.png'
            plt.savefig(missing_plot_path, dpi=300, bbox_inches='tight')
            print(f"      ✅ Saved to: {missing_plot_path}")
        else:
            print("      ℹ️  Skipped: No missing values to visualize")
        
        plt.close()
        
        # 2. Text Length Distribution Histogram
        if self.metrics.text_statistics:
            print("   📈 Creating text length distribution plots...")
            
            # Create subplot for each text column
            n_text_cols = len(self.metrics.text_statistics)
            fig, axes = plt.subplots(n_text_cols, 1, figsize=(12, 5*n_text_cols))
            
            if n_text_cols == 1:
                axes = [axes]
            
            for idx, (col, stats) in enumerate(self.metrics.text_statistics.items()):
                # Get word counts for this column
                valid_text = self.df[col].dropna()
                valid_text = valid_text[valid_text.astype(str).str.strip() != '']
                word_counts = valid_text.astype(str).str.split().str.len()
                
                ax = axes[idx]
                ax.hist(word_counts, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
                ax.axvline(stats['mean_words'], color='red', linestyle='--', linewidth=2, label=f'Mean: {stats["mean_words"]:.0f}')
                ax.axvline(stats['median_words'], color='green', linestyle='--', linewidth=2, label=f'Median: {stats["median_words"]:.0f}')
                ax.axvline(500, color='orange', linestyle='--', linewidth=2, label='500 words threshold')
                
                ax.set_xlabel('Word Count', fontsize=11, fontweight='bold')
                ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
                ax.set_title(f'Text Length Distribution: {col}', fontsize=12, fontweight='bold')
                ax.legend(loc='upper right')
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            text_dist_path = self.output_dir / 'text_length_distribution.png'
            plt.savefig(text_dist_path, dpi=300, bbox_inches='tight')
            print(f"      ✅ Saved to: {text_dist_path}")
            
            plt.close()
        
        print()
    
    def generate_text_report(self) -> None:
        """Generate a comprehensive text report and save to file."""
        print("=" * 80)
        print("📄 GENERATING TEXT REPORT")
        print("=" * 80)
        
        report_path = self.output_dir / 'dataset_health_report.txt'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            # Header
            f.write("=" * 80 + "\n")
            f.write("DATASET HEALTH REPORT\n")
            f.write("NLP YouTube Video Classification Project\n")
            f.write("=" * 80 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Dataset: {self.csv_path.name}\n")
            f.write("=" * 80 + "\n\n")
            
            # Structural Health
            f.write("1. STRUCTURAL HEALTH\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total Rows: {self.metrics.total_rows:,}\n")
            f.write(f"Total Columns: {self.metrics.total_columns}\n")
            f.write(f"Memory Usage: {self.metrics.memory_usage_mb:.2f} MB\n")
            f.write(f"Duplicate Rows: {self.metrics.duplicate_rows:,} ({self.metrics.duplicate_percentage:.2f}%)\n")
            f.write(f"\nColumn Distribution:\n")
            f.write(f"  - Text Columns: {len(self.metrics.text_columns)}\n")
            f.write(f"  - Numeric Columns: {len(self.metrics.numeric_columns)}\n\n")
            
            # Missing Values
            f.write("2. MISSING VALUES\n")
            f.write("-" * 80 + "\n")
            if self.metrics.missing_values:
                for col, (count, pct) in sorted(self.metrics.missing_values.items(), 
                                                key=lambda x: x[1][1], reverse=True):
                    f.write(f"  • {col}: {count:,} ({pct:.2f}%)\n")
            else:
                f.write("  ✅ No missing values detected\n")
            f.write("\n")
            
            # Hidden Missing
            f.write("3. HIDDEN MISSING VALUES\n")
            f.write("-" * 80 + "\n")
            if self.metrics.hidden_missing:
                for col, count in sorted(self.metrics.hidden_missing.items(), 
                                        key=lambda x: x[1], reverse=True):
                    pct = (count / self.metrics.total_rows) * 100
                    f.write(f"  • {col}: {count:,} ({pct:.2f}%)\n")
            else:
                f.write("  ✅ No hidden missing values detected\n")
            f.write("\n")
            
            # Text Statistics
            f.write("4. NLP TEXT STATISTICS & TOKEN LIMITS\n")
            f.write("-" * 80 + "\n")
            for col, stats in self.metrics.text_statistics.items():
                f.write(f"\nColumn: {col}\n")
                f.write(f"  Word Count Statistics:\n")
                f.write(f"    - Mean: {stats['mean_words']:.1f} words\n")
                f.write(f"    - Median: {stats['median_words']:.1f} words\n")
                f.write(f"    - Min: {stats['min_words']:.0f} words\n")
                f.write(f"    - Max: {stats['max_words']:.0f} words\n")
                f.write(f"  Percentiles:\n")
                f.write(f"    - 90th: {stats['p90_words']:.1f} words\n")
                f.write(f"    - 95th: {stats['p95_words']:.1f} words\n")
                f.write(f"    - 99th: {stats['p99_words']:.1f} words\n")
                f.write(f"  Token Limit Analysis:\n")
                f.write(f"    - Exceeds 500 words: {stats['exceeds_500_count']:,} ({stats['exceeds_500_pct']:.2f}%)\n")
                f.write(f"    - Exceeds 512 tokens: {stats['exceeds_512_tokens_count']:,} ({stats['exceeds_512_tokens_pct']:.2f}%)\n")
                
                if stats['exceeds_512_tokens_pct'] > 10:
                    f.write(f"    ⚠️  WARNING: Consider Longformer architecture\n")
                else:
                    f.write(f"    ✅ BERT-compatible\n")
            f.write("\n")
            
            # Language Distribution
            if self.metrics.language_stats:
                f.write("5. LANGUAGE DISTRIBUTION\n")
                f.write("-" * 80 + "\n")
                for lang, count in sorted(self.metrics.language_stats['distribution'].items(), 
                                         key=lambda x: x[1], reverse=True):
                    pct = (count / sum(self.metrics.language_stats['distribution'].values())) * 100
                    f.write(f"  • {lang.upper()}: {count} ({pct:.1f}%)\n")
                f.write(f"\nEnglish Percentage: {self.metrics.language_stats['english_percentage']:.1f}%\n")
                
                if self.metrics.language_stats['english_percentage'] < 90:
                    f.write("  ⚠️  WARNING: Dataset contains significant non-English content\n")
                else:
                    f.write("  ✅ Dataset is predominantly English\n")
                f.write("\n")
            
            # Recommendations
            f.write("6. RECOMMENDATIONS FOR MODEL TRAINING\n")
            f.write("-" * 80 + "\n")
            
            # Token limit recommendations
            for col, stats in self.metrics.text_statistics.items():
                if stats['exceeds_512_tokens_pct'] > 10:
                    f.write(f"  • {col}: Consider Longformer or implement truncation strategy\n")
            
            # Missing values recommendations
            if self.metrics.missing_values:
                f.write(f"  • Address missing values before training (imputation or removal)\n")
            
            # Duplicate recommendations
            if self.metrics.duplicate_percentage > 1:
                f.write(f"  • Remove {self.metrics.duplicate_rows:,} duplicate rows\n")
            
            # Language recommendations
            if self.metrics.language_stats and self.metrics.language_stats['english_percentage'] < 90:
                f.write(f"  • Filter non-English videos or use multilingual models\n")
            
            f.write("\n")
            f.write("=" * 80 + "\n")
            f.write("END OF REPORT\n")
            f.write("=" * 80 + "\n")
        
        print(f"✅ Report saved to: {report_path}\n")
    
    def run_complete_analysis(self) -> bool:
        """
        Run the complete EDA pipeline.
        
        Returns:
            bool: True if analysis completed successfully
        """
        print("\n")
        print("╔" + "=" * 78 + "╗")
        print("║" + " " * 78 + "║")
        print("║" + "   PRODUCTION-GRADE EDA ANALYZER FOR NLP DATASETS".center(78) + "║")
        print("║" + "   YouTube Video Classification Project".center(78) + "║")
        print("║" + " " * 78 + "║")
        print("╚" + "=" * 78 + "╝")
        print()
        
        # Load data
        if not self.load_data():
            return False
        
        # Run analyses
        self.analyze_structure()
        self.analyze_hidden_missing()
        self.analyze_text_statistics()
        self.analyze_language_distribution(sample_size=100)
        
        # Generate outputs
        self.generate_visualizations()
        self.generate_text_report()
        
        # Final summary
        print("=" * 80)
        print("✅ ANALYSIS COMPLETE")
        print("=" * 80)
        print(f"📊 Outputs saved to: {self.output_dir}")
        print(f"   • Text Report: dataset_health_report.txt")
        print(f"   • Visualizations: *.png files")
        print()
        print("💡 Next Steps:")
        print("   1. Review the text report for data quality issues")
        print("   2. Examine visualizations for patterns and outliers")
        print("   3. Address missing values and duplicates")
        print("   4. Choose appropriate model architecture based on token analysis")
        print("   5. Filter or clean data as needed before training")
        print()
        
        return True


def main():
    """Main entry point for the script."""
    # ============================================================================
    # CONFIGURATION - Update this path to your CSV file
    # ============================================================================
    CSV_FILE_PATH = "youtube_data.csv"
    OUTPUT_DIR = "./eda_outputs"
    
    # ============================================================================
    # Run Analysis
    # ============================================================================
    try:
        analyzer = DatasetAnalyzer(csv_path=CSV_FILE_PATH, output_dir=OUTPUT_DIR)
        success = analyzer.run_complete_analysis()
        
        if success:
            sys.exit(0)
        else:
            print("❌ Analysis failed. Please check the error messages above.")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Analysis interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
