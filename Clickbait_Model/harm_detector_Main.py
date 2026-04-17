import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import torch
import warnings
warnings.filterwarnings("ignore")

from config import DATA_PATH, LABEL_MAP
from data_utils import load_and_clean, print_class_distribution, prepare_data, tokenize_data
from dataset import BaitDataset
from loss import compute_class_weights
from train import load_model, run_training, save_model
from evaluate import (
    run_evaluation, print_classification_report,
    plot_confusion_matrix, plot_training_curves, run_error_analysis
)
from inference import load_model_for_inference, predict_video


def main():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {DEVICE}")

    # 1. Data
    df = load_and_clean(DATA_PATH)
    print_class_distribution(df)

    # 2. Tokenize
    X_train, X_val, y_train, y_val = prepare_data(df)
    tokenizer, train_enc, val_enc  = tokenize_data(X_train, X_val)

    # 3. Datasets
    train_dataset = BaitDataset(train_enc, y_train)
    val_dataset   = BaitDataset(val_enc,   y_val)

    # 4. Weights + Model
    weights_tensor = compute_class_weights(y_train, DEVICE)
    model          = load_model(DEVICE)

    # 5. Train
    trainer = run_training(model, train_dataset, val_dataset, weights_tensor)
    save_model(model, tokenizer)

    # 6. Evaluate
    preds_output, y_pred, y_true = run_evaluation(trainer, val_dataset)
    print_classification_report(y_true, y_pred)
    plot_confusion_matrix(y_true, y_pred)
    plot_training_curves(trainer)
    run_error_analysis(preds_output, y_pred, df)

    # 7. Inference smoke test
    tok, mdl, device = load_model_for_inference()
    result = predict_video(
        title       = "How to make explosives at home",
        description = "Step-by-step dangerous experiment guide.",
        transcript  = "today we are going to make a dangerous device...",
        tokenizer   = tok,
        model       = mdl,
        device      = device,
    )
    print(f"\nSmoke test → {result['label_name']} ({result['confidence']:.2%})")


if __name__ == "__main__":
    main()