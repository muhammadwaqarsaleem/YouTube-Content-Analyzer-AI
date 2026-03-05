"""
Post-Labeling Quality Assurance Script
=======================================
Validates the output of the batch labeling pipeline and provides
quality metrics for research documentation.

Usage: python validate_labels.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import warnings

warnings.filterwarnings('ignore')


class LabelQualityValidator:
    """Validate and analyze the quality of weak supervision labels."""
    
    def __init__(self, labeled_csv_path: str):
        """
        Initialize validator.
        
        Args:
            labeled_csv_path: Path to the labeled dataset CSV
        """
        self.csv_path = Path(labeled_csv_path)
        self.df: pd.DataFrame = None
        
    def load_data(self) -> bool:
        """Load the labeled dataset."""
        if not self.csv_path.exists():
            print(f"❌ Error: File not found at {self.csv_path}")
            return False
        
        try:
            self.df = pd.read_csv(self.csv_path, low_memory=False)
            print(f"✅ Loaded {len(self.df):,} labeled samples\n")
            return True
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
    
    def validate_columns(self) -> bool:
        """Check that all expected columns are present."""
        required_columns = [
            'Age_Label', 'Severity_Score', 'Mature_Density',
            'Teen_Density', 'Transcript_Word_Count', 'Content_Flags'
        ]
        
        missing = [col for col in required_columns if col not in self.df.columns]
        
        if missing:
            print(f"❌ Missing columns: {', '.join(missing)}\n")
            return False
        
        print("✅ All expected columns present\n")
        return True
    
    def analyze_label_distribution(self) -> None:
        """Analyze and validate label distribution."""
        print("=" * 80)
        print("📊 LABEL DISTRIBUTION ANALYSIS")
        print("=" * 80)
        
        distribution = self.df['Age_Label'].value_counts()
        percentages = self.df['Age_Label'].value_counts(normalize=True) * 100
        
        print(f"\n{'Label':<15} {'Count':<12} {'Percentage':<12}")
        print("-" * 40)
        for label in ['General', 'Teen', 'Mature']:
            count = distribution.get(label, 0)
            pct = percentages.get(label, 0)
            print(f"{label:<15} {count:<12,} {pct:<11.2f}%")
        
        # Sanity checks
        print("\n🔍 Distribution Checks:")
        general_pct = percentages.get('General', 0)
        teen_pct = percentages.get('Teen', 0)
        mature_pct = percentages.get('Mature', 0)
        
        if general_pct < 40:
            print(f"   ⚠️  WARNING: General% is low ({general_pct:.1f}%) - Expected 50-70%")
        elif general_pct > 80:
            print(f"   ⚠️  WARNING: General% is high ({general_pct:.1f}%) - May be under-labeling")
        else:
            print(f"   ✅ General distribution looks reasonable ({general_pct:.1f}%)")
        
        if mature_pct > 20:
            print(f"   ⚠️  WARNING: Mature% is high ({mature_pct:.1f}%) - Expected 5-15%")
        elif mature_pct < 2:
            print(f"   ⚠️  WARNING: Mature% is low ({mature_pct:.1f}%) - May be under-detecting")
        else:
            print(f"   ✅ Mature distribution looks reasonable ({mature_pct:.1f}%)")
        
        print()
    
    def analyze_severity_scores(self) -> None:
        """Analyze severity score distribution and outliers."""
        print("=" * 80)
        print("🔢 SEVERITY SCORE ANALYSIS")
        print("=" * 80)
        
        print("\nOverall Statistics:")
        print(f"   • Mean: {self.df['Severity_Score'].mean():.2f}")
        print(f"   • Median: {self.df['Severity_Score'].median():.1f}")
        print(f"   • Std Dev: {self.df['Severity_Score'].std():.2f}")
        print(f"   • Max: {self.df['Severity_Score'].max()}")
        
        # Per-label analysis
        print("\nBy Label:")
        for label in ['General', 'Teen', 'Mature']:
            label_data = self.df[self.df['Age_Label'] == label]['Severity_Score']
            if len(label_data) > 0:
                print(f"   {label}:")
                print(f"      Mean: {label_data.mean():.2f}, Median: {label_data.median():.1f}")
        
        # High severity outliers
        high_severity = self.df[self.df['Severity_Score'] > 100]
        if len(high_severity) > 0:
            print(f"\n⚠️  High Severity Outliers (Score > 100): {len(high_severity)} samples")
            print("   These may need manual review:")
            top_5 = high_severity.nlargest(5, 'Severity_Score')
            for idx, row in top_5.iterrows():
                title = row.get('title', 'N/A')[:50]
                print(f"      • Score {row['Severity_Score']}: {title}...")
        
        print()
    
    def analyze_density_distributions(self) -> None:
        """Analyze keyword density distributions."""
        print("=" * 80)
        print("📈 KEYWORD DENSITY ANALYSIS")
        print("=" * 80)
        
        # Mature density
        print("\nMature Density (per 1,000 words):")
        mature_dens = self.df['Mature_Density']
        print(f"   • Mean: {mature_dens.mean():.3f}")
        print(f"   • Median: {mature_dens.median():.3f}")
        print(f"   • 90th percentile: {mature_dens.quantile(0.90):.3f}")
        print(f"   • 95th percentile: {mature_dens.quantile(0.95):.3f}")
        print(f"   • Max: {mature_dens.max():.3f}")
        
        # Teen density
        print("\nTeen Density (per 1,000 words):")
        teen_dens = self.df['Teen_Density']
        print(f"   • Mean: {teen_dens.mean():.3f}")
        print(f"   • Median: {teen_dens.median():.3f}")
        print(f"   • 90th percentile: {teen_dens.quantile(0.90):.3f}")
        print(f"   • 95th percentile: {teen_dens.quantile(0.95):.3f}")
        print(f"   • Max: {teen_dens.max():.3f}")
        
        # Check for potential threshold issues
        print("\n🔍 Threshold Validation:")
        
        # Mature samples with low density (potential false positives)
        mature_low_dens = self.df[
            (self.df['Age_Label'] == 'Mature') & 
            (self.df['Mature_Density'] < 0.5)
        ]
        if len(mature_low_dens) > 0:
            print(f"   ⚠️  {len(mature_low_dens)} Mature samples with density < 0.5")
            print(f"      (Likely zero-tolerance triggers)")
        
        # General samples with high teen density (edge cases)
        general_high_teen = self.df[
            (self.df['Age_Label'] == 'General') & 
            (self.df['Teen_Density'] > 2.0)
        ]
        if len(general_high_teen) > 0:
            print(f"   ⚠️  {len(general_high_teen)} General samples with Teen density > 2.0")
            print(f"      (Close to threshold, may need review)")
        
        print()
    
    def analyze_content_flags(self) -> None:
        """Analyze content flag patterns."""
        print("=" * 80)
        print("🚩 CONTENT FLAGS ANALYSIS")
        print("=" * 80)
        
        # Count flagged vs clean
        flagged = (self.df['Content_Flags'] != 'None').sum()
        clean = len(self.df) - flagged
        
        print(f"\nFlagging Summary:")
        print(f"   • Flagged: {flagged:,} ({(flagged/len(self.df))*100:.2f}%)")
        print(f"   • Clean: {clean:,} ({(clean/len(self.df))*100:.2f}%)")
        
        # Analyze flag types
        flag_types = {
            'Violence_Reference': 0,
            'Strong_Language': 0,
            'Substance_Reference': 0,
            'Sexual_Content': 0,
            'Gaming_Content': 0,
            'Mild_Language': 0
        }
        
        for flags in self.df['Content_Flags']:
            if flags != 'None':
                for flag_type in flag_types.keys():
                    if flag_type in str(flags):
                        flag_types[flag_type] += 1
        
        print("\nFlag Type Frequency:")
        for flag_type, count in sorted(flag_types.items(), key=lambda x: x[1], reverse=True):
            if count > 0:
                pct = (count / len(self.df)) * 100
                print(f"   • {flag_type}: {count:,} ({pct:.2f}%)")
        
        print()
    
    def spot_check_samples(self) -> None:
        """Provide sample rows for manual spot-checking."""
        print("=" * 80)
        print("🔍 SPOT-CHECK SAMPLES")
        print("=" * 80)
        
        print("\nRandom samples from each category for manual review:\n")
        
        for label in ['General', 'Teen', 'Mature']:
            label_samples = self.df[self.df['Age_Label'] == label]
            if len(label_samples) > 0:
                sample = label_samples.sample(min(3, len(label_samples)))
                print(f"{label} Samples:")
                print("-" * 80)
                for idx, row in sample.iterrows():
                    title = row.get('title', 'N/A')[:60]
                    severity = row['Severity_Score']
                    flags = str(row['Content_Flags'])[:80]
                    print(f"   Title: {title}")
                    print(f"   Severity: {severity}, Flags: {flags}")
                    print()
        
    def export_summary_report(self, output_path: str = "label_quality_report.txt") -> None:
        """Export a summary report for research documentation."""
        print("=" * 80)
        print("📄 EXPORTING QUALITY REPORT")
        print("=" * 80)
        
        with open(output_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("WEAK SUPERVISION LABEL QUALITY REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Dataset: {self.csv_path.name}\n")
            f.write(f"Total Samples: {len(self.df):,}\n\n")
            
            # Label distribution
            f.write("LABEL DISTRIBUTION:\n")
            f.write("-" * 80 + "\n")
            distribution = self.df['Age_Label'].value_counts()
            percentages = self.df['Age_Label'].value_counts(normalize=True) * 100
            for label in ['General', 'Teen', 'Mature']:
                count = distribution.get(label, 0)
                pct = percentages.get(label, 0)
                f.write(f"{label}: {count:,} ({pct:.2f}%)\n")
            
            # Severity statistics
            f.write("\nSEVERITY SCORE STATISTICS:\n")
            f.write("-" * 80 + "\n")
            f.write(f"Mean: {self.df['Severity_Score'].mean():.2f}\n")
            f.write(f"Median: {self.df['Severity_Score'].median():.1f}\n")
            f.write(f"Std Dev: {self.df['Severity_Score'].std():.2f}\n")
            f.write(f"Max: {self.df['Severity_Score'].max()}\n")
            
            # Density statistics
            f.write("\nKEYWORD DENSITY STATISTICS:\n")
            f.write("-" * 80 + "\n")
            f.write(f"Mature Density Mean: {self.df['Mature_Density'].mean():.3f}\n")
            f.write(f"Mature Density 95th percentile: {self.df['Mature_Density'].quantile(0.95):.3f}\n")
            f.write(f"Teen Density Mean: {self.df['Teen_Density'].mean():.3f}\n")
            f.write(f"Teen Density 95th percentile: {self.df['Teen_Density'].quantile(0.95):.3f}\n")
            
            # Content flags
            f.write("\nCONTENT FLAGS:\n")
            f.write("-" * 80 + "\n")
            flagged = (self.df['Content_Flags'] != 'None').sum()
            f.write(f"Flagged Samples: {flagged:,} ({(flagged/len(self.df))*100:.2f}%)\n")
            f.write(f"Clean Samples: {len(self.df) - flagged:,} ({((len(self.df)-flagged)/len(self.df))*100:.2f}%)\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("END OF REPORT\n")
            f.write("=" * 80 + "\n")
        
        print(f"✅ Quality report saved to: {output_path}\n")
    
    def run_validation(self) -> bool:
        """Run complete validation pipeline."""
        print("\n" + "=" * 80)
        print("🔍 WEAK SUPERVISION LABEL QUALITY VALIDATION")
        print("=" * 80 + "\n")
        
        if not self.load_data():
            return False
        
        if not self.validate_columns():
            return False
        
        self.analyze_label_distribution()
        self.analyze_severity_scores()
        self.analyze_density_distributions()
        self.analyze_content_flags()
        self.spot_check_samples()
        self.export_summary_report()
        
        print("=" * 80)
        print("✅ VALIDATION COMPLETE")
        print("=" * 80)
        print("\n💡 Recommendations:")
        print("   1. Review the spot-check samples manually")
        print("   2. Investigate any high-severity outliers")
        print("   3. Check samples near category boundaries")
        print("   4. Consider adjusting thresholds if needed")
        print("   5. Proceed with train/test split and model training\n")
        
        return True


def main():
    """Main entry point."""
    # Configuration
    LABELED_CSV_PATH = "training_ready_dataset.csv"
    
    validator = LabelQualityValidator(LABELED_CSV_PATH)
    success = validator.run_validation()
    
    return 0 if success else 1


if __name__ == '__main__':
    import sys
    sys.exit(main())
