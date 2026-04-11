"""
YouTube Content Analyzer AI - Main Pipeline
=============================================
Single entry point that runs the entire pipeline in sequence:

  1. Validate Labels       (quality check on labeled CSV)
  2. Prepare Dataset       (tokenize + chunk + train/val/test split)
  3. Verify Datasets       (tensor shape & quality checks)
  4. Train Model           (Hierarchical RoBERTa fine-tuning)
  5. Evaluate & Explain    (metrics, plots, XAI attention report)

Usage:
    python main.py                    # run full pipeline
    python main.py --skip-to train    # skip to training (if tensors exist)
    python main.py --skip-to eval     # skip to evaluation (if model exists)
    python main.py --epochs 10        # override number of training epochs
    python main.py --batch-size 2     # smaller batch if GPU memory is tight

Author: University AI Research Team
Project: YouTube Age Classification with Deep Learning
"""

import sys
import time
import argparse
from pathlib import Path
from datetime import datetime


# ============================================================================
# Helpers
# ============================================================================

def elapsed(start: float) -> str:
    """Format elapsed time."""
    secs = time.time() - start
    if secs < 60:
        return f"{secs:.1f}s"
    mins = int(secs // 60)
    secs = secs % 60
    if mins < 60:
        return f"{mins}m {secs:.0f}s"
    hrs = int(mins // 60)
    mins = mins % 60
    return f"{hrs}h {mins}m {secs:.0f}s"


def banner(title: str, step: int, total: int) -> None:
    """Print a stage banner."""
    print()
    print("=" * 70)
    print(f"  STAGE {step}/{total}  |  {title}")
    print("=" * 70)
    print()


def success_msg(title: str, duration: str) -> None:
    print(f"\n  >> {title} completed in {duration}\n")


def fail_msg(title: str) -> None:
    print(f"\n  >> {title} FAILED - see errors above\n")


# ============================================================================
# Pipeline stages
# ============================================================================

def stage_validate_labels(csv_path: str) -> bool:
    """Stage 1: Validate label quality."""
    from validate_labels import LabelQualityValidator

    validator = LabelQualityValidator(csv_path)
    return validator.run_validation()


def stage_prepare_dataset(csv_path: str, output_dir: str) -> bool:
    """Stage 2: Tokenize, chunk, and split into tensors."""
    from dataset_preparation import (
        HierarchicalDatasetPreparator,
        ChunkingConfig,
        SplitConfig,
    )

    chunking_config = ChunkingConfig(
        chunk_size=512,
        max_chunks=20,
        stride=256,
        model_name="roberta-base",
    )
    split_config = SplitConfig(
        train_size=0.8,
        val_size=0.1,
        test_size=0.1,
        random_state=42,
        stratify=True,
    )

    preparator = HierarchicalDatasetPreparator(
        csv_path=csv_path,
        output_dir=output_dir,
        chunking_config=chunking_config,
        split_config=split_config,
    )
    return preparator.run_pipeline()


def stage_verify_datasets(data_dir: str) -> bool:
    """Stage 3: Verify tensor shapes and data quality."""
    from verify_datasets import DatasetVerifier

    verifier = DatasetVerifier(data_dir=data_dir)
    return verifier.run_verification()


def stage_train(data_dir: str, epochs: int, batch_size: int) -> bool:
    """Stage 4: Train the Hierarchical RoBERTa model."""
    import torch
    import numpy as np
    from train_model import (
        TrainingConfig,
        HierarchicalRobertaTrainer,
        load_datasets,
        create_dataloaders,
        set_seed,
    )
    from hierarchical_roberta import create_model

    config = TrainingConfig(
        data_dir=data_dir,
        checkpoint_dir="checkpoints",
        log_dir="runs",
        num_classes=3,
        max_chunks=20,
        use_gradient_checkpointing=True,
        num_epochs=epochs,
        batch_size=batch_size,
        gradient_accumulation_steps=max(1, 16 // batch_size),
        learning_rate=2e-5,
        weight_decay=0.01,
        use_amp=True,
        warmup_ratio=0.1,
        max_grad_norm=1.0,
        early_stopping_patience=5,
        metric_for_best_model="val_f1_macro",
        use_class_weights=True,
        logging_steps=10,
        num_workers=2,
        seed=42,
    )

    set_seed(config.seed)

    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  GPU detected: {gpu} ({mem:.1f} GB)")
    else:
        print("  WARNING: No GPU detected - training will be very slow!")

    print(f"  Epochs: {config.num_epochs}")
    print(f"  Batch size: {config.batch_size}  (effective: {config.batch_size * config.gradient_accumulation_steps})")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Mixed precision (AMP): {config.use_amp}")
    print()

    train_ds, val_ds, test_ds = load_datasets(config.data_dir)
    train_loader, val_loader, test_loader = create_dataloaders(
        train_ds, val_ds, test_ds, config
    )

    model = create_model(
        num_classes=config.num_classes,
        max_chunks=config.max_chunks,
        use_gradient_checkpointing=config.use_gradient_checkpointing,
        freeze_base=False,
        pooling_strategy="attention",
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model parameters: {total_params:,} total, {trainable:,} trainable")
    print()

    trainer = HierarchicalRobertaTrainer(
        config=config,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
    )
    trainer.train()
    return True


def stage_evaluate(csv_path: str) -> bool:
    """Stage 5: Evaluate model and generate XAI reports."""
    from evaluate_and_explain import run_full_evaluation, EvalConfig

    cfg = EvalConfig(
        checkpoint_path="checkpoints/best_model.pt",
        test_data_path="data/processed_tensors/test_dataset.pt",
        csv_path=csv_path,
    )
    run_full_evaluation(cfg)
    return True


# ============================================================================
# Main
# ============================================================================

STAGES = ["validate", "prepare", "verify", "train", "eval"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="YouTube Content Analyzer AI - Full Training Pipeline",
    )
    parser.add_argument(
        "--dataset",
        default="dataset/training_ready_dataset.csv",
        help="Path to the labeled CSV dataset (default: dataset/training_ready_dataset.csv)",
    )
    parser.add_argument(
        "--skip-to",
        choices=STAGES,
        default=None,
        help="Skip earlier stages and start from this stage",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=15,
        help="Number of training epochs (default: 15)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Training batch size (default: 4, use 2 if OOM)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    csv_path = args.dataset
    data_dir = "data/processed_tensors"

    # Determine which stages to run
    if args.skip_to:
        start_idx = STAGES.index(args.skip_to)
    else:
        start_idx = 0

    stages_to_run = STAGES[start_idx:]
    total = len(stages_to_run)

    # Title
    print()
    print("+" + "-" * 68 + "+")
    print("|" + " " * 68 + "|")
    print("|" + "YOUTUBE CONTENT ANALYZER AI".center(68) + "|")
    print("|" + "Hierarchical RoBERTa Training Pipeline".center(68) + "|")
    print("|" + " " * 68 + "|")
    print("+" + "-" * 68 + "+")
    print()
    print(f"  Started at  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Dataset     : {csv_path}")
    print(f"  Stages      : {', '.join(stages_to_run)}")
    print(f"  Epochs      : {args.epochs}")
    print(f"  Batch size  : {args.batch_size}")

    # Verify dataset exists
    if not Path(csv_path).exists():
        print(f"\n  ERROR: Dataset not found at '{csv_path}'")
        print("  Place your training_ready_dataset.csv in the dataset/ folder")
        return 1

    pipeline_start = time.time()
    step = 0

    # ── Stage 1: Validate Labels ──────────────────────────────────────────
    if "validate" in stages_to_run:
        step += 1
        banner("VALIDATE LABELS", step, total)
        t = time.time()
        try:
            ok = stage_validate_labels(csv_path)
            if not ok:
                fail_msg("Label validation")
                print("  Continuing anyway (validation is advisory)...\n")
            else:
                success_msg("Label validation", elapsed(t))
        except Exception as e:
            print(f"  Label validation error: {e}")
            print("  Continuing anyway...\n")

    # ── Stage 2: Prepare Dataset ──────────────────────────────────────────
    if "prepare" in stages_to_run:
        step += 1
        banner("PREPARE DATASET (Tokenize + Chunk + Split)", step, total)
        t = time.time()
        try:
            ok = stage_prepare_dataset(csv_path, data_dir)
            if not ok:
                fail_msg("Dataset preparation")
                return 1
            success_msg("Dataset preparation", elapsed(t))
        except Exception as e:
            print(f"\n  Dataset preparation error: {e}")
            import traceback; traceback.print_exc()
            return 1

    # ── Stage 3: Verify Datasets ──────────────────────────────────────────
    if "verify" in stages_to_run:
        step += 1
        banner("VERIFY DATASETS", step, total)
        t = time.time()
        try:
            ok = stage_verify_datasets(data_dir)
            if not ok:
                fail_msg("Dataset verification")
                return 1
            success_msg("Dataset verification", elapsed(t))
        except Exception as e:
            print(f"  Verification error: {e}")
            print("  Continuing anyway...\n")

    # ── Stage 4: Train Model ──────────────────────────────────────────────
    if "train" in stages_to_run:
        step += 1
        banner("TRAIN MODEL (Hierarchical RoBERTa)", step, total)
        t = time.time()
        try:
            ok = stage_train(data_dir, args.epochs, args.batch_size)
            if not ok:
                fail_msg("Model training")
                return 1
            success_msg("Model training", elapsed(t))
        except KeyboardInterrupt:
            print("\n\n  Training interrupted by user.")
            print("  Best checkpoint (if any) saved in checkpoints/")
            return 1
        except Exception as e:
            print(f"\n  Training error: {e}")
            import traceback; traceback.print_exc()
            return 1

    # ── Stage 5: Evaluate & Explain ───────────────────────────────────────
    if "eval" in stages_to_run:
        step += 1
        banner("EVALUATE & EXPLAIN (Metrics + XAI)", step, total)
        t = time.time()
        try:
            ok = stage_evaluate(csv_path)
            if not ok:
                fail_msg("Evaluation")
                return 1
            success_msg("Evaluation", elapsed(t))
        except Exception as e:
            print(f"\n  Evaluation error: {e}")
            import traceback; traceback.print_exc()
            return 1

    # ── Done ──────────────────────────────────────────────────────────────
    total_time = elapsed(pipeline_start)
    print()
    print("+" + "-" * 68 + "+")
    print("|" + " " * 68 + "|")
    print("|" + "PIPELINE COMPLETE".center(68) + "|")
    print("|" + f"Total time: {total_time}".center(68) + "|")
    print("|" + " " * 68 + "|")
    print("+" + "-" * 68 + "+")
    print()
    print("  Output locations:")
    print(f"    Tensors      : {data_dir}/")
    print(f"    Checkpoints  : checkpoints/")
    print(f"    Best model   : checkpoints/best_model.pt")
    print(f"    Eval plots   : results/plots/")
    print(f"    Eval reports : results/reports/")
    print(f"    XAI report   : results/reports/attention_report.html")
    print(f"    TensorBoard  : runs/  (view with: tensorboard --logdir runs)")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
