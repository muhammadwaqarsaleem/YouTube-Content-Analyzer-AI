import os

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_PATH      = r"C:\Users\muham\OneDrive\Desktop\harm_detector\YouTube-Content-Analyzer-AI\harm_detector\data\final_augmented_dataset_v2.csv"
DRIVE_SAVE_DIR = r"C:\Users\muham\OneDrive\Desktop\harm_detector\YouTube-Content-Analyzer-AI\harm_detector\outputs"

# ── Model ──────────────────────────────────────────────────────────────────────
MODEL_NAME = "roberta-base"
NUM_LABELS = 7
MAX_LENGTH = 512

# ── Data Split ─────────────────────────────────────────────────────────────────
VAL_SIZE    = 0.15
RANDOM_SEED = 42

# ── Training ───────────────────────────────────────────────────────────────────
EPOCHS         = 4
TRAIN_BATCH    = 16
EVAL_BATCH     = 16
GRAD_ACCUM     = 1
LEARNING_RATE  = 2e-5
WEIGHT_DECAY   = 0.01
MAX_GRAD_NORM  = 1.0
WARMUP_STEPS   = 500
HARMLESS_BOOST = 1.2

# ── Label Map ──────────────────────────────────────────────────────────────────
LABEL_MAP = {
    0: "Harmless",
    1: "Info Harm",
    2: "Addiction",
    3: "Physical Harm",
    4: "Sexual",
    5: "Hate/Harass",
    6: "Clickbait",
}

os.makedirs(DRIVE_SAVE_DIR, exist_ok=True)