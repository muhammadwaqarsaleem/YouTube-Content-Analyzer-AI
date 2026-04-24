import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from inference import load_model_for_inference, predict_video

# ── Model loaded once, reused across all calls ────────────────────────────────
_tokenizer = None
_model     = None
_device    = None

def _ensure_model_loaded():
    global _tokenizer, _model, _device
    if _model is None:
        print("[harm_detector] Loading model...")
        _tokenizer, _model, _device = load_model_for_inference()
        print("[harm_detector] Model ready.")


def run_harm_detector(title: str, description: str, transcript: str) -> dict:
    """
    Entry point called from project root.

    ── INPUTS ────────────────────────────────────────────────────────────────
    title        (str)  YouTube video title
                        e.g. "10 Foods That Are Slowly Killing You"

    description  (str)  YouTube video description text
                        e.g. "In this video we explore..."
                        Pass "" if unavailable.

    transcript   (str)  Full video transcript / captions
                        e.g. "hey guys welcome back today we are going to..."
                        Pass "" if unavailable — accuracy will drop.

    ── OUTPUT ────────────────────────────────────────────────────────────────
    Returns dict:
    {
        "label"         : int,   # 0–6
        "label_name"    : str,   # e.g. "Clickbait"
        "confidence"    : float, # e.g. 0.91
        "probabilities" : dict   # all 7 class scores, sums to 1.0
    }

    Label map:
        0 = Harmless
        1 = Info Harm
        2 = Addiction
        3 = Physical Harm
        4 = Sexual
        5 = Hate/Harass
        6 = Clickbait
    ── ────────────────────────────────────────────────────────────────────────
    """
    _ensure_model_loaded()
    return predict_video(
        title       = title,
        description = description,
        transcript  = transcript,
        tokenizer   = _tokenizer,
        model       = _model,
        device      = _device,
    )


# ── Run directly for testing ──────────────────────────────────────────────────
if __name__ == "__main__":
    result = run_harm_detector(
        title       = "How to make explosives at home",
        description = "Step-by-step dangerous experiment guide.",
        transcript  = "today we are going to make a dangerous device...",
    )
    print(f"\nLabel      : {result['label_name']} (class {result['label']})")
    print(f"Confidence : {result['confidence']:.2%}")
    print("\nAll probabilities:")
    for name, prob in sorted(result["probabilities"].items(), key=lambda x: -x[1]):
        bar = "█" * int(prob * 30)
        print(f"  {name:<15}  {prob:5.1%}  {bar}")