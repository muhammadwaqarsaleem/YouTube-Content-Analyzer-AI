"""
Example Usage Script for DatasetAnalyzer
=========================================
Demonstrates different ways to use the EDA analyzer programmatically.
"""

from dataset_eda_analyzer import DatasetAnalyzer
import sys

def example_1_basic_usage():
    """
    Example 1: Basic usage with default settings
    """
    print("\n" + "="*80)
    print("EXAMPLE 1: Basic Usage")
    print("="*80 + "\n")
    
    analyzer = DatasetAnalyzer(
        csv_path="youtube_data.csv",
        output_dir="./eda_outputs"
    )
    
    # Run complete analysis
    success = analyzer.run_complete_analysis()
    
    if success:
        print("✅ Analysis completed successfully!")
        # Access metrics programmatically
        print(f"\n📊 Quick Stats:")
        print(f"   Total Rows: {analyzer.metrics.total_rows:,}")
        print(f"   Text Columns: {len(analyzer.metrics.text_columns)}")
        print(f"   Duplicate Percentage: {analyzer.metrics.duplicate_percentage:.2f}%")
    
    return success


def example_2_custom_analysis():
    """
    Example 2: Run specific analyses only
    """
    print("\n" + "="*80)
    print("EXAMPLE 2: Custom Analysis Pipeline")
    print("="*80 + "\n")
    
    analyzer = DatasetAnalyzer(
        csv_path="youtube_data.csv",
        output_dir="./custom_eda_outputs"
    )
    
    # Load data
    if not analyzer.load_data():
        return False
    
    # Run only specific analyses
    print("Running structural analysis...")
    analyzer.analyze_structure()
    
    print("Running text statistics...")
    analyzer.analyze_text_statistics()
    
    # Skip language detection if not needed
    # analyzer.analyze_language_distribution(sample_size=50)
    
    # Generate only specific outputs
    print("Generating visualizations...")
    analyzer.generate_visualizations()
    
    print("\n✅ Custom analysis completed!")
    return True


def example_3_access_metrics():
    """
    Example 3: Access and use metrics programmatically
    """
    print("\n" + "="*80)
    print("EXAMPLE 3: Programmatic Metric Access")
    print("="*80 + "\n")
    
    analyzer = DatasetAnalyzer(csv_path="youtube_data.csv")
    
    if not analyzer.load_data():
        return False
    
    # Run analyses
    analyzer.analyze_structure()
    analyzer.analyze_text_statistics()
    
    # Access metrics for decision making
    metrics = analyzer.metrics
    
    print("📊 Making Model Architecture Decisions:")
    print("-" * 80)
    
    for col, stats in metrics.text_statistics.items():
        print(f"\nColumn: {col}")
        print(f"  Average length: {stats['mean_words']:.0f} words")
        print(f"  95th percentile: {stats['p95_words']:.0f} words")
        print(f"  Exceeds 512 tokens: {stats['exceeds_512_tokens_pct']:.1f}%")
        
        # Decision logic
        if stats['exceeds_512_tokens_pct'] < 5:
            print(f"  ✅ Decision: Use BERT with simple truncation")
        elif stats['exceeds_512_tokens_pct'] < 15:
            print(f"  ⚠️  Decision: Use BERT with head-tail truncation")
        else:
            print(f"  🔄 Decision: Use Longformer or hierarchical model")
    
    # Check if dataset needs cleaning
    print("\n📋 Data Cleaning Recommendations:")
    print("-" * 80)
    
    if metrics.duplicate_percentage > 1:
        print(f"  • Remove {metrics.duplicate_rows:,} duplicate rows")
    
    if metrics.missing_values:
        print(f"  • Address missing values in {len(metrics.missing_values)} columns")
    
    if not metrics.missing_values and metrics.duplicate_percentage < 1:
        print(f"  ✅ Dataset is clean! Ready for training.")
    
    return True


def example_4_different_sample_sizes():
    """
    Example 4: Adjust language detection sample size
    """
    print("\n" + "="*80)
    print("EXAMPLE 4: Custom Language Detection")
    print("="*80 + "\n")
    
    analyzer = DatasetAnalyzer(csv_path="youtube_data.csv")
    
    if not analyzer.load_data():
        return False
    
    analyzer.analyze_structure()
    
    # Use larger sample for more accurate language detection
    print("Running language detection on 200 samples...")
    analyzer.analyze_language_distribution(sample_size=200)
    
    if analyzer.metrics.language_stats:
        english_pct = analyzer.metrics.language_stats['english_percentage']
        print(f"\n📊 Result: {english_pct:.1f}% English content")
        
        if english_pct < 90:
            print("⚠️  Recommendation: Filter non-English or use mBERT")
        else:
            print("✅ Recommendation: Standard English BERT is appropriate")
    
    return True


def example_5_batch_analysis():
    """
    Example 5: Analyze multiple datasets in batch
    """
    print("\n" + "="*80)
    print("EXAMPLE 5: Batch Analysis of Multiple Datasets")
    print("="*80 + "\n")
    
    datasets = [
        "youtube_data_train.csv",
        "youtube_data_val.csv",
        "youtube_data_test.csv"
    ]
    
    results = {}
    
    for dataset in datasets:
        print(f"\n{'='*60}")
        print(f"Analyzing: {dataset}")
        print('='*60)
        
        analyzer = DatasetAnalyzer(
            csv_path=dataset,
            output_dir=f"./eda_outputs/{dataset.replace('.csv', '')}"
        )
        
        success = analyzer.run_complete_analysis()
        
        if success:
            results[dataset] = {
                'rows': analyzer.metrics.total_rows,
                'duplicates': analyzer.metrics.duplicate_percentage,
                'text_cols': len(analyzer.metrics.text_columns)
            }
    
    # Summary comparison
    print("\n" + "="*80)
    print("BATCH ANALYSIS SUMMARY")
    print("="*80)
    print(f"\n{'Dataset':<30} {'Rows':>12} {'Duplicates':>12} {'Text Cols':>12}")
    print("-" * 80)
    
    for dataset, stats in results.items():
        print(f"{dataset:<30} {stats['rows']:>12,} {stats['duplicates']:>11.2f}% {stats['text_cols']:>12}")
    
    return True


def main():
    """
    Main function to demonstrate different usage examples.
    Uncomment the example you want to run.
    """
    
    # Example 1: Basic complete analysis (recommended for first run)
    success = example_1_basic_usage()
    
    # Example 2: Custom analysis pipeline
    # success = example_2_custom_analysis()
    
    # Example 3: Access metrics programmatically for decision making
    # success = example_3_access_metrics()
    
    # Example 4: Custom language detection sample size
    # success = example_4_different_sample_sizes()
    
    # Example 5: Batch analysis of multiple datasets
    # success = example_5_batch_analysis()
    
    if success:
        print("\n" + "="*80)
        print("✅ ALL EXAMPLES COMPLETED SUCCESSFULLY")
        print("="*80)
        sys.exit(0)
    else:
        print("\n❌ Example failed. Check error messages above.")
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
