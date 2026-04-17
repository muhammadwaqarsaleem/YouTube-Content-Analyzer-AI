import torch
from transformers import (
    RobertaForSequenceClassification,
    TrainingArguments,
    EarlyStoppingCallback,
)
from config import (
    MODEL_NAME, NUM_LABELS, DRIVE_SAVE_DIR,
    EPOCHS, TRAIN_BATCH, EVAL_BATCH, GRAD_ACCUM,
    LEARNING_RATE, WEIGHT_DECAY, MAX_GRAD_NORM, WARMUP_STEPS,
)
from loss import make_weighted_trainer, EpochSummaryCallback, compute_metrics


def load_model(device):
    print(f"Downloading {MODEL_NAME}...")
    model = RobertaForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        ignore_mismatched_sizes=True,
    ).to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"✅ Model loaded.")
    print(f"   Trainable params: {trainable:,}  /  {total:,} total")
    return model


def get_training_args():
    return TrainingArguments(
        output_dir                  = f"{DRIVE_SAVE_DIR}/checkpoints",
        num_train_epochs            = EPOCHS,
        per_device_train_batch_size = TRAIN_BATCH,
        gradient_accumulation_steps = GRAD_ACCUM,
        per_device_eval_batch_size  = EVAL_BATCH,
        learning_rate               = LEARNING_RATE,
        lr_scheduler_type           = "linear",
        warmup_steps                = WARMUP_STEPS,
        weight_decay                = WEIGHT_DECAY,
        max_grad_norm               = MAX_GRAD_NORM,
        fp16 = torch.cuda.is_available(),
        eval_strategy               = "epoch",
        save_strategy               = "epoch",
        logging_strategy            = "epoch",
        load_best_model_at_end      = True,
        metric_for_best_model       = "accuracy",
        greater_is_better           = True,
        report_to                   = "none",
    )


def run_training(model, train_dataset, val_dataset, weights_tensor):
    WeightedTrainer = make_weighted_trainer(weights_tensor)
    trainer = WeightedTrainer(
        model           = model,
        args            = get_training_args(),
        train_dataset   = train_dataset,
        eval_dataset    = val_dataset,
        compute_metrics = compute_metrics,
        callbacks       = [
            EpochSummaryCallback(),
            EarlyStoppingCallback(early_stopping_patience=3),
        ],
    )
    print("🚀 Starting training...")
    trainer.train()
    print("\n✅ Training complete!")
    return trainer


def save_model(model, tokenizer):
    final_path = f"{DRIVE_SAVE_DIR}/final_model"
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    print(f" Model saved to: {final_path}")
    return final_path