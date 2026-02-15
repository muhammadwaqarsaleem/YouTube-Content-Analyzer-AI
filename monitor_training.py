"""
Real-Time Training Monitor
===========================
Utility to monitor training progress and diagnose issues.

Usage:
    python monitor_training.py
"""

import json
import torch
from pathlib import Path
from typing import Dict, List, Optional
import time
from datetime import datetime


class TrainingMonitor:
    """Monitor training progress in real-time."""
    
    def __init__(
        self,
        checkpoint_dir: str = "checkpoints",
        refresh_interval: int = 5
    ):
        """
        Initialize monitor.
        
        Args:
            checkpoint_dir: Directory containing training outputs
            refresh_interval: Seconds between updates
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.refresh_interval = refresh_interval
        self.last_modified = {}
    
    def check_files_exist(self) -> bool:
        """Check if training has started."""
        log_file = self.checkpoint_dir / 'training.log'
        history_file = self.checkpoint_dir / 'training_history.json'
        
        if not log_file.exists() and not history_file.exists():
            print("⚠️  Training not started yet.")
            print(f"   Waiting for files in: {self.checkpoint_dir}")
            return False
        
        return True
    
    def load_training_history(self) -> Optional[List[Dict]]:
        """Load training history JSON."""
        history_file = self.checkpoint_dir / 'training_history.json'
        
        if not history_file.exists():
            return None
        
        # Check if file was modified
        current_mtime = history_file.stat().st_mtime
        if history_file in self.last_modified:
            if current_mtime == self.last_modified[history_file]:
                return None  # No update
        
        self.last_modified[history_file] = current_mtime
        
        try:
            with open(history_file, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            return None  # File being written
    
    def get_gpu_info(self) -> Optional[Dict]:
        """Get GPU utilization info."""
        if not torch.cuda.is_available():
            return None
        
        try:
            device = torch.cuda.current_device()
            total_mem = torch.cuda.get_device_properties(device).total_memory / 1e9
            allocated = torch.cuda.memory_allocated(device) / 1e9
            reserved = torch.cuda.memory_reserved(device) / 1e9
            
            return {
                'name': torch.cuda.get_device_name(device),
                'total_gb': total_mem,
                'allocated_gb': allocated,
                'reserved_gb': reserved,
                'utilization_pct': (allocated / total_mem) * 100
            }
        except:
            return None
    
    def print_latest_metrics(self, history: List[Dict]) -> None:
        """Print latest training metrics."""
        if not history:
            print("No metrics yet...")
            return
        
        latest = history[-1]
        epoch = latest['epoch']
        
        print("\n" + "=" * 70)
        print(f"📊 LATEST METRICS (Epoch {epoch})")
        print("=" * 70)
        
        print(f"\nLosses:")
        print(f"  Train Loss:      {latest['train_loss']:.4f}")
        print(f"  Val Loss:        {latest['val_loss']:.4f}")
        print(f"  Loss Gap:        {abs(latest['train_loss'] - latest['val_loss']):.4f}")
        
        print(f"\nMetrics:")
        print(f"  Val Accuracy:    {latest['val_accuracy']:.4f} ({latest['val_accuracy']*100:.2f}%)")
        print(f"  Val F1 (Macro):  {latest['val_f1_macro']:.4f}")
        print(f"  Val F1 (Weighted): {latest['val_f1_weighted']:.4f}")
        print(f"  Val Precision:   {latest['val_precision']:.4f}")
        print(f"  Val Recall:      {latest['val_recall']:.4f}")
        
        print(f"\nTraining:")
        print(f"  Learning Rate:   {latest['learning_rate']:.2e}")
        
        # Health indicators
        print(f"\n🏥 Health Indicators:")
        
        # Check if improving
        if len(history) > 1:
            prev = history[-2]
            val_f1_change = latest['val_f1_macro'] - prev['val_f1_macro']
            
            if val_f1_change > 0.01:
                print(f"  ✅ Improving (F1 +{val_f1_change:.4f})")
            elif val_f1_change > -0.01:
                print(f"  ⚠️  Plateauing (F1 {val_f1_change:+.4f})")
            else:
                print(f"  ⚠️  Degrading (F1 {val_f1_change:+.4f})")
        
        # Check overfitting
        gap = abs(latest['train_loss'] - latest['val_loss'])
        if gap < 0.2:
            print(f"  ✅ Good generalization (gap={gap:.3f})")
        elif gap < 0.4:
            print(f"  ⚠️  Some overfitting (gap={gap:.3f})")
        else:
            print(f"  ❌ Significant overfitting (gap={gap:.3f})")
        
        # Check learning
        if latest['val_f1_macro'] > 0.65:
            print(f"  ✅ Good performance (F1={latest['val_f1_macro']:.3f})")
        elif latest['val_f1_macro'] > 0.50:
            print(f"  ⚠️  Moderate performance (F1={latest['val_f1_macro']:.3f})")
        else:
            print(f"  ❌ Poor performance (F1={latest['val_f1_macro']:.3f})")
    
    def print_progress_chart(self, history: List[Dict]) -> None:
        """Print ASCII chart of training progress."""
        if len(history) < 2:
            return
        
        print("\n📈 Training Progress:")
        print("-" * 70)
        
        # Get metrics
        epochs = [h['epoch'] for h in history]
        train_losses = [h['train_loss'] for h in history]
        val_losses = [h['val_loss'] for h in history]
        val_f1s = [h['val_f1_macro'] for h in history]
        
        # Print header
        print(f"{'Epoch':<6} {'Train Loss':<12} {'Val Loss':<12} {'Val F1':<12}")
        print("-" * 70)
        
        # Print rows
        for i, epoch in enumerate(epochs):
            train_loss = train_losses[i]
            val_loss = val_losses[i]
            val_f1 = val_f1s[i]
            
            # Create bar for F1
            bar_width = int(val_f1 * 50)
            bar = '█' * bar_width
            
            print(f"{epoch:<6} {train_loss:<12.4f} {val_loss:<12.4f} {val_f1:<12.4f} {bar}")
        
        print("-" * 70)
        
        # Best epoch
        best_idx = val_f1s.index(max(val_f1s))
        print(f"\n🏆 Best: Epoch {epochs[best_idx]} (F1={val_f1s[best_idx]:.4f})")
    
    def check_checkpoints(self) -> None:
        """Check saved checkpoints."""
        checkpoints = list(self.checkpoint_dir.glob('checkpoint_epoch_*.pt'))
        best_model = self.checkpoint_dir / 'best_model.pt'
        
        print("\n💾 Checkpoints:")
        print("-" * 70)
        
        if best_model.exists():
            size_mb = best_model.stat().st_size / (1024 ** 2)
            print(f"  🏆 best_model.pt ({size_mb:.1f} MB)")
        
        if checkpoints:
            print(f"  📁 {len(checkpoints)} epoch checkpoint(s)")
            total_size = sum(c.stat().st_size for c in checkpoints) / (1024 ** 2)
            print(f"  💾 Total size: {total_size:.1f} MB")
    
    def monitor_continuous(self) -> None:
        """Continuously monitor training."""
        print("\n" + "=" * 70)
        print("🔍 TRAINING MONITOR")
        print("=" * 70)
        print(f"Monitoring: {self.checkpoint_dir}")
        print(f"Refresh: Every {self.refresh_interval} seconds")
        print("Press Ctrl+C to stop")
        print("=" * 70)
        
        try:
            while True:
                # Clear screen (optional)
                # print("\033[H\033[J")  # Uncomment to clear screen each update
                
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print(f"\n⏰ Update: {timestamp}")
                
                # Check if files exist
                if not self.check_files_exist():
                    time.sleep(self.refresh_interval)
                    continue
                
                # Load history
                history = self.load_training_history()
                
                if history:
                    # Print latest metrics
                    self.print_latest_metrics(history)
                    
                    # Print progress chart
                    if len(history) >= 2:
                        self.print_progress_chart(history)
                    
                    # Check checkpoints
                    self.check_checkpoints()
                    
                    # GPU info
                    gpu_info = self.get_gpu_info()
                    if gpu_info:
                        print(f"\n🎮 GPU: {gpu_info['name']}")
                        print(f"  Memory: {gpu_info['allocated_gb']:.1f} / {gpu_info['total_gb']:.1f} GB "
                              f"({gpu_info['utilization_pct']:.1f}%)")
                
                print(f"\n{'─' * 70}")
                print(f"Next update in {self.refresh_interval}s...")
                
                time.sleep(self.refresh_interval)
                
        except KeyboardInterrupt:
            print("\n\n⏹️  Monitoring stopped.")
    
    def print_final_summary(self) -> None:
        """Print final training summary."""
        print("\n" + "=" * 70)
        print("📊 TRAINING SUMMARY")
        print("=" * 70)
        
        # Load history
        history = self.load_training_history()
        if not history:
            print("No training history found.")
            return
        
        # Overall stats
        total_epochs = len(history)
        final_metrics = history[-1]
        
        print(f"\nTotal Epochs: {total_epochs}")
        print(f"Final Train Loss: {final_metrics['train_loss']:.4f}")
        print(f"Final Val Loss: {final_metrics['val_loss']:.4f}")
        print(f"Final Val F1: {final_metrics['val_f1_macro']:.4f}")
        
        # Best metrics
        val_f1s = [h['val_f1_macro'] for h in history]
        best_f1 = max(val_f1s)
        best_epoch = val_f1s.index(best_f1) + 1
        
        print(f"\nBest Model:")
        print(f"  Epoch: {best_epoch}")
        print(f"  Val F1: {best_f1:.4f}")
        print(f"  Val Accuracy: {history[best_epoch-1]['val_accuracy']:.4f}")
        
        # Test results if available
        test_results_file = self.checkpoint_dir / 'test_results.json'
        if test_results_file.exists():
            with open(test_results_file, 'r') as f:
                test_results = json.load(f)
            
            print(f"\nTest Set Results:")
            print(f"  Accuracy: {test_results['accuracy']:.4f}")
            print(f"  F1 (Macro): {test_results['f1_macro']:.4f}")
            print(f"  Precision: {test_results['precision']:.4f}")
            print(f"  Recall: {test_results['recall']:.4f}")
        
        print("\n" + "=" * 70)


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Monitor training progress')
    parser.add_argument(
        '--checkpoint_dir',
        type=str,
        default='checkpoints',
        help='Directory containing checkpoints'
    )
    parser.add_argument(
        '--refresh',
        type=int,
        default=5,
        help='Refresh interval in seconds'
    )
    parser.add_argument(
        '--summary',
        action='store_true',
        help='Print summary only (no continuous monitoring)'
    )
    
    args = parser.parse_args()
    
    monitor = TrainingMonitor(
        checkpoint_dir=args.checkpoint_dir,
        refresh_interval=args.refresh
    )
    
    if args.summary:
        monitor.print_final_summary()
    else:
        monitor.monitor_continuous()


if __name__ == '__main__':
    main()
