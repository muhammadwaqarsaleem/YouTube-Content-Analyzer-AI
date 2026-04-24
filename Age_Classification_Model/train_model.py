"""
Production-Grade Training Script for Hierarchical RoBERTa
==========================================================
Optimized training pipeline with advanced PyTorch features:
- Automatic Mixed Precision (AMP) for memory & speed
- Gradient accumulation for effective large batch sizes
- Linear warmup + cosine decay scheduler
- Class-weighted loss for imbalanced data
- Early stopping with patience
- TensorBoard logging
- Comprehensive checkpointing

Author: University AI Research Team
Project: YouTube Age Classification
File: 3/4 - Training Loop
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter

import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime
from tqdm import tqdm
import json
import logging
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report

# Import our model
from hierarchical_roberta import (
    HierarchicalRobertaForClassification,
    ModelConfig,
    create_model
)


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class TrainingConfig:
    """
    Training configuration with all hyperparameters.
    
    Why dataclass? Type safety, easy serialization, clear documentation.
    """
    # Data paths
    data_dir: str = "data/processed_tensors"
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "runs"
    
    # Model configuration
    num_classes: int = 3
    max_chunks: int = 20
    use_gradient_checkpointing: bool = True
    
    # Training hyperparameters
    num_epochs: int = 15
    batch_size: int = 4  # Safe for 8GB GPU
    gradient_accumulation_steps: int = 4  # Effective batch = 16
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0  # Gradient clipping
    
    # Learning rate schedule
    warmup_ratio: float = 0.1  # 10% of steps for warmup
    scheduler_type: str = "linear_warmup_cosine"  # or "linear"
    
    # Regularization
    label_smoothing: float = 0.0  # 0.0 = no smoothing, 0.1 = mild
    
    # Mixed precision training
    use_amp: bool = True  # Automatic Mixed Precision
    amp_opt_level: str = "O1"  # O1 = mixed precision
    
    # Early stopping
    early_stopping_patience: int = 5
    early_stopping_min_delta: float = 0.001
    
    # Checkpointing
    save_total_limit: int = 3  # Keep only best 3 checkpoints
    metric_for_best_model: str = "val_f1_macro"  # or "val_loss"
    greater_is_better: bool = True  # True for F1, False for loss
    
    # Logging
    logging_steps: int = 10  # Log every N batches
    eval_steps: int = -1  # -1 = evaluate after each epoch
    log_level: str = "INFO"
    
    # Hardware
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 2  # DataLoader workers
    pin_memory: bool = True
    
    # Reproducibility
    seed: int = 42
    
    # Class weights for imbalanced data
    use_class_weights: bool = True
    # Calculated from your data: ~60% General, ~27% Teen, ~11% Mature
    # Will be computed dynamically from training data
    class_weights: Optional[List[float]] = None


@dataclass
class TrainingMetrics:
    """Container for training metrics."""
    epoch: int
    train_loss: float
    val_loss: float
    val_accuracy: float
    val_f1_macro: float
    val_f1_weighted: float
    val_precision: float
    val_recall: float
    learning_rate: float
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for logging."""
        return {
            'epoch': self.epoch,
            'train_loss': self.train_loss,
            'val_loss': self.val_loss,
            'val_accuracy': self.val_accuracy,
            'val_f1_macro': self.val_f1_macro,
            'val_f1_weighted': self.val_f1_weighted,
            'val_precision': self.val_precision,
            'val_recall': self.val_recall,
            'learning_rate': self.learning_rate
        }


# ============================================================================
# Trainer Class
# ============================================================================

class HierarchicalRobertaTrainer:
    """
    Production-grade trainer with advanced optimization techniques.
    
    Key Features:
        - AMP for 2x speed + 40% memory reduction
        - Gradient accumulation for effective large batches
        - Warmup scheduler for stable early training
        - Class weighting for imbalanced data
        - Early stopping to prevent overfitting
        - Comprehensive metrics tracking
        - TensorBoard integration
    """
    
    def __init__(
        self,
        config: TrainingConfig,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: Optional[DataLoader] = None
    ):
        """
        Initialize trainer.
        
        Args:
            config: Training configuration
            model: Hierarchical RoBERTa model
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Optional test data loader
        """
        self.config = config
        self.model = model.to(config.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        
        # Setup directories
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        
        # Calculate class weights if needed
        if config.use_class_weights and config.class_weights is None:
            self.class_weights = self._calculate_class_weights()
        else:
            self.class_weights = config.class_weights
        
        # Setup loss function with class weights
        self._setup_criterion()
        
        # Setup optimizer
        self._setup_optimizer()
        
        # Setup learning rate scheduler
        self._setup_scheduler()
        
        # Setup AMP scaler
        self.scaler = GradScaler() if config.use_amp else None
        
        # Metrics tracking
        self.best_metric = -float('inf') if config.greater_is_better else float('inf')
        self.epochs_without_improvement = 0
        self.global_step = 0
        self.metrics_history: List[TrainingMetrics] = []
        
        # TensorBoard
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_dir = Path(config.log_dir) / f"run_{timestamp}"
        self.writer = SummaryWriter(log_dir)
        
        self.logger.info("=" * 80)
        self.logger.info("🚀 TRAINER INITIALIZED")
        self.logger.info("=" * 80)
        self.logger.info(f"Device: {config.device}")
        self.logger.info(f"AMP Enabled: {config.use_amp}")
        self.logger.info(f"Gradient Accumulation: {config.gradient_accumulation_steps} steps")
        self.logger.info(f"Effective Batch Size: {config.batch_size * config.gradient_accumulation_steps}")
        self.logger.info(f"Class Weights: {self.class_weights}")
        self.logger.info(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
        self.logger.info(f"Trainable Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        self.logger.info("=" * 80 + "\n")
    
    def _setup_logging(self) -> None:
        """Setup Python logging."""
        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.checkpoint_dir / 'training.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _calculate_class_weights(self) -> torch.Tensor:
        """
        Calculate class weights for imbalanced dataset.
        
        Why: Your dataset has ~60% General, ~27% Teen, ~11% Mature.
        Without weighting, model will bias toward majority class.
        
        Formula: weight_i = total_samples / (num_classes × count_i)
        This gives more weight to minority classes.
        
        Returns:
            Class weights tensor
        """
        self.logger.info("Calculating class weights from training data...")
        
        # Extract labels from train_loader
        all_labels = []
        for _, _, labels in self.train_loader:
            all_labels.extend(labels.tolist())
        
        all_labels = np.array(all_labels)
        
        # Count samples per class
        class_counts = np.bincount(all_labels, minlength=self.config.num_classes)
        total_samples = len(all_labels)
        
        # Calculate weights: inverse frequency
        weights = total_samples / (self.config.num_classes * class_counts)
        
        # Normalize to sum to num_classes (optional, for numerical stability)
        weights = weights * self.config.num_classes / weights.sum()
        
        self.logger.info(f"Class distribution: {class_counts}")
        self.logger.info(f"Class weights: {weights}")
        
        return torch.FloatTensor(weights).to(self.config.device)
    
    def _setup_criterion(self) -> None:
        """
        Setup loss function with class weights and label smoothing.
        
        Why class weights? Handles imbalanced data (60-27-11 split).
        Why label smoothing? Prevents overconfident predictions, acts as regularization.
        """
        if self.config.use_class_weights:
            self.criterion = nn.CrossEntropyLoss(
                weight=self.class_weights,
                label_smoothing=self.config.label_smoothing
            )
        else:
            self.criterion = nn.CrossEntropyLoss(
                label_smoothing=self.config.label_smoothing
            )
        
        self.logger.info(f"Loss function: CrossEntropyLoss")
        self.logger.info(f"  - Class weights: {self.config.use_class_weights}")
        self.logger.info(f"  - Label smoothing: {self.config.label_smoothing}")
    
    def _setup_optimizer(self) -> None:
        """
        Setup AdamW optimizer with weight decay.
        
        Why AdamW? Better than Adam for transformers (decouples weight decay).
        Why weight decay? Regularization to prevent overfitting.
        
        Parameter groups: Different LR for base model vs task-specific layers.
        """
        # Separate parameters: base model vs task-specific
        no_decay = ['bias', 'LayerNorm.weight']
        
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if not any(nd in n for nd in no_decay) and 'roberta' in n],
                'lr': self.config.learning_rate,
                'weight_decay': self.config.weight_decay,
                'name': 'roberta_with_decay'
            },
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if any(nd in n for nd in no_decay) and 'roberta' in n],
                'lr': self.config.learning_rate,
                'weight_decay': 0.0,
                'name': 'roberta_no_decay'
            },
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if 'roberta' not in n],
                'lr': self.config.learning_rate * 2,  # Higher LR for task head
                'weight_decay': self.config.weight_decay,
                'name': 'task_head'
            }
        ]
        
        self.optimizer = optim.AdamW(optimizer_grouped_parameters)
        
        self.logger.info(f"Optimizer: AdamW")
        self.logger.info(f"  - Learning rate: {self.config.learning_rate}")
        self.logger.info(f"  - Weight decay: {self.config.weight_decay}")
        self.logger.info(f"  - Task head LR: {self.config.learning_rate * 2}")
    
    def _setup_scheduler(self) -> None:
        """
        Setup learning rate scheduler with warmup.
        
        Why warmup? Stabilizes training in early epochs by gradually increasing LR.
        Why cosine decay? Smoothly decreases LR, often better than step decay.
        
        Schedule:
            Epochs 0-1.5 (10%): Linear warmup from 0 to max_lr
            Epochs 1.5-15: Cosine decay from max_lr to 0
        """
        num_training_steps = len(self.train_loader) * self.config.num_epochs
        num_warmup_steps = int(num_training_steps * self.config.warmup_ratio)
        
        self.logger.info(f"LR Scheduler: {self.config.scheduler_type}")
        self.logger.info(f"  - Total steps: {num_training_steps}")
        self.logger.info(f"  - Warmup steps: {num_warmup_steps} ({self.config.warmup_ratio:.0%})")
        
        if self.config.scheduler_type == "linear_warmup_cosine":
            # Linear warmup + cosine decay
            self.scheduler = optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=[self.config.learning_rate, self.config.learning_rate, 
                       self.config.learning_rate * 2],
                total_steps=num_training_steps,
                pct_start=self.config.warmup_ratio,
                anneal_strategy='cos',
                div_factor=25.0,  # Initial LR = max_lr / 25
                final_div_factor=10000.0
            )
        else:
            # Simple linear warmup + linear decay
            def lr_lambda(current_step):
                if current_step < num_warmup_steps:
                    return float(current_step) / float(max(1, num_warmup_steps))
                return max(0.0, float(num_training_steps - current_step) / 
                          float(max(1, num_training_steps - num_warmup_steps)))
            
            self.scheduler = optim.lr_scheduler.LambdaLR(
                self.optimizer, 
                lr_lambda
            )
    
    def train_epoch(self, epoch: int) -> float:
        """
        Train for one epoch.
        
        Args:
            epoch: Current epoch number
        
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        # Progress bar
        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch}/{self.config.num_epochs} [Train]",
            dynamic_ncols=True
        )
        
        # Zero gradients before accumulation
        self.optimizer.zero_grad()
        
        for batch_idx, (input_ids, attention_mask, labels) in enumerate(pbar):
            # Move to device
            input_ids = input_ids.to(self.config.device)
            attention_mask = attention_mask.to(self.config.device)
            labels = labels.to(self.config.device)
            
            # Forward pass with AMP
            if self.config.use_amp:
                with autocast():
                    outputs = self.model(input_ids, attention_mask)
                    loss = self.criterion(outputs['logits'], labels)
                    # Normalize loss for gradient accumulation
                    loss = loss / self.config.gradient_accumulation_steps
            else:
                outputs = self.model(input_ids, attention_mask)
                loss = self.criterion(outputs['logits'], labels)
                loss = loss / self.config.gradient_accumulation_steps
            
            # Backward pass with AMP
            if self.config.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Gradient accumulation: update every N steps
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                # Gradient clipping (prevents exploding gradients)
                if self.config.use_amp:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config.max_grad_norm
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm
                    )
                    self.optimizer.step()
                
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1
                
                # Log to TensorBoard
                if self.global_step % self.config.logging_steps == 0:
                    current_lr = self.scheduler.get_last_lr()[0]
                    self.writer.add_scalar('train/loss', loss.item() * self.config.gradient_accumulation_steps, 
                                          self.global_step)
                    self.writer.add_scalar('train/learning_rate', current_lr, self.global_step)
            
            # Accumulate loss (denormalized)
            total_loss += loss.item() * self.config.gradient_accumulation_steps
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item() * self.config.gradient_accumulation_steps:.4f}',
                'lr': f'{self.scheduler.get_last_lr()[0]:.2e}'
            })
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    @torch.no_grad()
    def evaluate(self, data_loader: DataLoader, split_name: str = "val") -> Dict[str, float]:
        """
        Evaluate model on validation or test set.
        
        Args:
            data_loader: Data loader for evaluation
            split_name: Name of split (for logging)
        
        Returns:
            Dictionary of metrics
        """
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        num_batches = 0
        
        pbar = tqdm(
            data_loader,
            desc=f"Evaluating [{split_name}]",
            dynamic_ncols=True
        )
        
        for input_ids, attention_mask, labels in pbar:
            # Move to device
            input_ids = input_ids.to(self.config.device)
            attention_mask = attention_mask.to(self.config.device)
            labels = labels.to(self.config.device)
            
            # Forward pass (no AMP needed for evaluation)
            outputs = self.model(input_ids, attention_mask)
            loss = self.criterion(outputs['logits'], labels)
            
            # Accumulate
            total_loss += loss.item()
            num_batches += 1
            
            # Get predictions
            predictions = torch.argmax(outputs['logits'], dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        avg_loss = total_loss / num_batches
        accuracy = accuracy_score(all_labels, all_predictions)
        f1_macro = f1_score(all_labels, all_predictions, average='macro')
        f1_weighted = f1_score(all_labels, all_predictions, average='weighted')
        precision = precision_score(all_labels, all_predictions, average='macro')
        recall = recall_score(all_labels, all_predictions, average='macro')
        
        metrics = {
            'loss': avg_loss,
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'precision': precision,
            'recall': recall
        }
        
        return metrics
    
    def save_checkpoint(
        self,
        epoch: int,
        metrics: TrainingMetrics,
        is_best: bool = False
    ) -> None:
        """
        Save model checkpoint.
        
        Args:
            epoch: Current epoch
            metrics: Training metrics
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics.to_dict(),
            'config': self.config.__dict__,
            'best_metric': self.best_metric
        }
        
        if self.config.use_amp and self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"💾 Checkpoint saved: {checkpoint_path}")
        
        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            self.logger.info(f"🏆 Best model saved: {best_path}")
        
        # Clean old checkpoints (keep only best N)
        self._cleanup_checkpoints()
    
    def _cleanup_checkpoints(self) -> None:
        """Remove old checkpoints, keeping only the best N."""
        checkpoints = list(self.checkpoint_dir.glob('checkpoint_epoch_*.pt'))
        
        if len(checkpoints) > self.config.save_total_limit:
            # Sort by modification time
            checkpoints.sort(key=lambda x: x.stat().st_mtime)
            
            # Remove oldest
            for ckpt in checkpoints[:-self.config.save_total_limit]:
                ckpt.unlink()
                self.logger.debug(f"Removed old checkpoint: {ckpt}")
    
    def check_early_stopping(self, current_metric: float) -> bool:
        """
        Check if training should stop early.
        
        Args:
            current_metric: Current epoch's metric value
        
        Returns:
            True if should stop, False otherwise
        """
        improved = False
        
        if self.config.greater_is_better:
            if current_metric > self.best_metric + self.config.early_stopping_min_delta:
                improved = True
                self.best_metric = current_metric
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1
        else:
            if current_metric < self.best_metric - self.config.early_stopping_min_delta:
                improved = True
                self.best_metric = current_metric
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1
        
        if self.epochs_without_improvement >= self.config.early_stopping_patience:
            self.logger.info(f"\n⏹️  Early stopping triggered after {self.epochs_without_improvement} epochs without improvement")
            return True
        
        return False
    
    def train(self) -> List[TrainingMetrics]:
        """
        Main training loop.
        
        Returns:
            List of metrics for each epoch
        """
        self.logger.info("\n" + "=" * 80)
        self.logger.info("🚀 STARTING TRAINING")
        self.logger.info("=" * 80 + "\n")
        
        for epoch in range(1, self.config.num_epochs + 1):
            # Train one epoch
            train_loss = self.train_epoch(epoch)
            
            # Evaluate on validation set
            val_metrics = self.evaluate(self.val_loader, "val")
            
            # Create metrics object
            current_lr = self.scheduler.get_last_lr()[0]
            metrics = TrainingMetrics(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_metrics['loss'],
                val_accuracy=val_metrics['accuracy'],
                val_f1_macro=val_metrics['f1_macro'],
                val_f1_weighted=val_metrics['f1_weighted'],
                val_precision=val_metrics['precision'],
                val_recall=val_metrics['recall'],
                learning_rate=current_lr
            )
            
            self.metrics_history.append(metrics)
            
            # Log to TensorBoard
            self.writer.add_scalar('epoch/train_loss', train_loss, epoch)
            self.writer.add_scalar('epoch/val_loss', val_metrics['loss'], epoch)
            self.writer.add_scalar('epoch/val_accuracy', val_metrics['accuracy'], epoch)
            self.writer.add_scalar('epoch/val_f1_macro', val_metrics['f1_macro'], epoch)
            
            # Print epoch summary
            self.logger.info("\n" + "=" * 80)
            self.logger.info(f"📊 EPOCH {epoch}/{self.config.num_epochs} SUMMARY")
            self.logger.info("=" * 80)
            self.logger.info(f"Train Loss:      {train_loss:.4f}")
            self.logger.info(f"Val Loss:        {val_metrics['loss']:.4f}")
            self.logger.info(f"Val Accuracy:    {val_metrics['accuracy']:.4f} ({val_metrics['accuracy']*100:.2f}%)")
            self.logger.info(f"Val F1 (Macro):  {val_metrics['f1_macro']:.4f}")
            self.logger.info(f"Val F1 (Weighted): {val_metrics['f1_weighted']:.4f}")
            self.logger.info(f"Val Precision:   {val_metrics['precision']:.4f}")
            self.logger.info(f"Val Recall:      {val_metrics['recall']:.4f}")
            self.logger.info(f"Learning Rate:   {current_lr:.2e}")
            self.logger.info("=" * 80 + "\n")
            
            # Check if best model
            current_metric = val_metrics[self.config.metric_for_best_model.replace('val_', '')]
            is_best = False
            
            if self.config.greater_is_better:
                is_best = current_metric > self.best_metric
            else:
                is_best = current_metric < self.best_metric
            
            if is_best:
                self.best_metric = current_metric
                self.epochs_without_improvement = 0
            
            # Save checkpoint
            self.save_checkpoint(epoch, metrics, is_best)
            
            # Early stopping check
            if self.check_early_stopping(current_metric):
                break
        
        # Final evaluation on test set if available
        if self.test_loader is not None:
            self.logger.info("\n" + "=" * 80)
            self.logger.info("🧪 FINAL TEST SET EVALUATION")
            self.logger.info("=" * 80 + "\n")
            
            # Load best model
            best_checkpoint = torch.load(self.checkpoint_dir / 'best_model.pt')
            self.model.load_state_dict(best_checkpoint['model_state_dict'])
            
            test_metrics = self.evaluate(self.test_loader, "test")
            
            self.logger.info(f"Test Loss:       {test_metrics['loss']:.4f}")
            self.logger.info(f"Test Accuracy:   {test_metrics['accuracy']:.4f} ({test_metrics['accuracy']*100:.2f}%)")
            self.logger.info(f"Test F1 (Macro): {test_metrics['f1_macro']:.4f}")
            self.logger.info(f"Test F1 (Weighted): {test_metrics['f1_weighted']:.4f}")
            self.logger.info("=" * 80 + "\n")
            
            # Save test metrics
            test_results_path = self.checkpoint_dir / 'test_results.json'
            with open(test_results_path, 'w') as f:
                json.dump(test_metrics, f, indent=2)
        
        # Save training history
        history_path = self.checkpoint_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump([m.to_dict() for m in self.metrics_history], f, indent=2)
        
        self.logger.info("✅ Training completed!")
        self.logger.info(f"📁 Checkpoints saved to: {self.checkpoint_dir}")
        self.logger.info(f"📊 Best {self.config.metric_for_best_model}: {self.best_metric:.4f}\n")
        
        self.writer.close()
        
        return self.metrics_history


# ============================================================================
# Data Loading
# ============================================================================

def load_datasets(data_dir: str) -> Tuple[TensorDataset, TensorDataset, TensorDataset]:
    """
    Load pre-processed tensor datasets.
    
    Args:
        data_dir: Directory containing train/val/test_dataset.pt files
    
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    data_path = Path(data_dir)
    
    print("=" * 80)
    print("📂 LOADING DATASETS")
    print("=" * 80)
    
    # Load train data
    train_path = data_path / 'train_dataset.pt'
    print(f"Loading training data from: {train_path}")
    train_data = torch.load(train_path)
    train_dataset = TensorDataset(
        train_data['input_ids'],
        train_data['attention_mask'],
        train_data['labels']
    )
    print(f"✅ Train: {len(train_dataset):,} samples")
    
    # Load validation data
    val_path = data_path / 'val_dataset.pt'
    print(f"Loading validation data from: {val_path}")
    val_data = torch.load(val_path)
    val_dataset = TensorDataset(
        val_data['input_ids'],
        val_data['attention_mask'],
        val_data['labels']
    )
    print(f"✅ Val:   {len(val_dataset):,} samples")
    
    # Load test data
    test_path = data_path / 'test_dataset.pt'
    print(f"Loading test data from: {test_path}")
    test_data = torch.load(test_path)
    test_dataset = TensorDataset(
        test_data['input_ids'],
        test_data['attention_mask'],
        test_data['labels']
    )
    print(f"✅ Test:  {len(test_dataset):,} samples")
    print()
    
    return train_dataset, val_dataset, test_dataset


def create_dataloaders(
    train_dataset: TensorDataset,
    val_dataset: TensorDataset,
    test_dataset: TensorDataset,
    config: TrainingConfig
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create DataLoader instances.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Test dataset
        config: Training configuration
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=True  # Drop incomplete batch for stable training
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )
    
    print(f"📊 DataLoaders created:")
    print(f"   Batch size: {config.batch_size}")
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")
    print(f"   Test batches: {len(test_loader)}")
    print()
    
    return train_loader, val_loader, test_loader


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # Note: Some operations are non-deterministic on GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================================
# Main Training Script
# ============================================================================

def main():
    """Main training pipeline."""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 78 + "║")
    print("║" + "   HIERARCHICAL ROBERTA TRAINING PIPELINE".center(78) + "║")
    print("║" + "   YouTube Age Classification".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("╚" + "=" * 78 + "╝")
    print()
    
    # Configuration
    config = TrainingConfig(
        # Data
        data_dir="data/processed_tensors",
        checkpoint_dir="checkpoints",
        log_dir="runs",
        
        # Model
        num_classes=3,
        max_chunks=20,
        use_gradient_checkpointing=True,
        
        # Training
        num_epochs=15,
        batch_size=4,  # Safe for 8GB GPU
        gradient_accumulation_steps=4,  # Effective batch = 16
        learning_rate=2e-5,
        weight_decay=0.01,
        
        # Optimization
        use_amp=True,
        warmup_ratio=0.1,
        max_grad_norm=1.0,
        
        # Early stopping
        early_stopping_patience=5,
        metric_for_best_model="val_f1_macro",
        
        # Class weighting
        use_class_weights=True,
        
        # Logging
        logging_steps=10,
        
        # Hardware
        num_workers=2,
        seed=42
    )
    
    # Set seed for reproducibility
    set_seed(config.seed)
    
    # Check GPU availability
    if not torch.cuda.is_available():
        print("⚠️  WARNING: CUDA not available. Training will be very slow on CPU!")
        print("   Consider using Google Colab or a GPU instance.\n")
    else:
        print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB\n")
    
    # Load datasets
    train_dataset, val_dataset, test_dataset = load_datasets(config.data_dir)
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        train_dataset, val_dataset, test_dataset, config
    )
    
    # Create model
    print("=" * 80)
    print("🏗️  CREATING MODEL")
    print("=" * 80)
    
    model = create_model(
        num_classes=config.num_classes,
        max_chunks=config.max_chunks,
        use_gradient_checkpointing=config.use_gradient_checkpointing,
        freeze_base=False,  # Will fine-tune
        pooling_strategy="attention"
    )
    
    print(f"✅ Model created")
    print(f"   Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print()
    
    # Create trainer
    trainer = HierarchicalRobertaTrainer(
        config=config,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader
    )
    
    # Train
    metrics_history = trainer.train()
    
    print("\n🎉 Training pipeline completed successfully!")
    print(f"📁 Results saved to: {config.checkpoint_dir}")
    print()


if __name__ == '__main__':
    main()
