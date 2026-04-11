import pandas as pd
from sklearn.model_selection import train_test_split
from config import (
    DATA_PATH, VAL_SIZE, RANDOM_SEED,
    MAX_LENGTH, MODEL_NAME, LABEL_MAP
)
from transformers import RobertaTokenizer


def load_and_clean(path=DATA_PATH):
    df = pd.read_csv(path, on_bad_lines="skip", engine="python")
    print(f"   Raw rows loaded: {len(df)}")

    df = df[pd.to_numeric(df["label"], errors="coerce").notnull()]
    df["label"] = df["label"].astype(int)
    df = df.dropna(subset=["title"])

    for col in ["transcript", "description", "title"]:
        df[col] = df[col].fillna("")

    print(f"   Clean rows:      {len(df)}")
    return df


def print_class_distribution(df):
    dist = df["label"].value_counts().sort_index()
    print("\n📊 Class distribution:")
    print(f"   {'Class':<18} {'Count':>6}  {'%':>6}")
    print("   " + "-" * 34)
    for idx, count in dist.items():
        pct = count / len(df) * 100
        print(f"   {LABEL_MAP[idx]:<18} {count:>6}  {pct:>5.1f}%")
    print(f"   {'TOTAL':<18} {len(df):>6}")


def build_input_text(row):
    title = str(row["title"]).strip()
    desc  = str(row["description"]).strip()
    trans = str(row["transcript"]).strip()[:1000]
    return f"{title} </s> {desc} </s> {trans}"


def prepare_data(df):
    df["input_text"] = df.apply(build_input_text, axis=1)

    X = df["input_text"].tolist()
    y = df["label"].values

    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=VAL_SIZE,
        random_state=RANDOM_SEED,
        stratify=y,
    )
    print(f"   Train: {len(X_train)} samples")
    print(f"   Val:   {len(X_val)} samples")
    return X_train, X_val, y_train, y_val


def tokenize_data(X_train, X_val):
    print(f"Loading tokenizer: {MODEL_NAME}...")
    tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)

    print("Tokenizing...")
    train_enc = tokenizer(X_train, truncation=True, padding=True, max_length=MAX_LENGTH)
    val_enc   = tokenizer(X_val,   truncation=True, padding=True, max_length=MAX_LENGTH)
    print(" Tokenization complete.")
    return tokenizer, train_enc, val_enc