import torch
from transformers import AutoTokenizer, RobertaForSequenceClassification
from Clickbait_Model.config import LABEL_MAP, MAX_LENGTH, DRIVE_SAVE_DIR


def load_model_for_inference(model_path=None, device=None):
    if model_path is None:
        model_path = f"{DRIVE_SAVE_DIR}/final_model"
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(model_path)
    mdl = RobertaForSequenceClassification.from_pretrained(model_path).to(device)
    mdl.eval()
    return tok, mdl, device


@torch.no_grad()
def predict_video(title, description, transcript, tokenizer, model, device):
    trans  = str(transcript).strip()[:1000]
    text   = f"{title.strip()} </s> {description.strip()} </s> {trans}"
    inputs = tokenizer(text, return_tensors="pt", truncation=True,
                       padding=True, max_length=MAX_LENGTH).to(device)
    logits = model(**inputs).logits
    probs  = torch.softmax(logits, dim=-1).cpu().numpy()[0]
    pred   = int(probs.argmax())
    return {
        "label":         pred,
        "label_name":    LABEL_MAP[pred],
        "confidence":    float(probs[pred]),
        "probabilities": {LABEL_MAP[i]: float(p) for i, p in enumerate(probs)},
    }