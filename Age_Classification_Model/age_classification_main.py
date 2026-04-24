"""
inference_api.py - Age Bracket Classification API

=========================================================
INSTRUCTIONS FOR PIPELINE INTEGRATION:
=========================================================
1. Import this class into your main master orchestrator script:

2. Initialize the model ONCE outside your main video loop 
   (This prevents reloading the heavy PyTorch weights for every video):
   age_model = AgeBracketModel(checkpoint_path="age_classification/checkpoints/best_model.pt")
   
3. Inside your loop, after fetching the transcript, pass the string to the model:
   result = age_model.predict_age(transcript_text=video_transcript)
   
   # The 'result' object contains:
   # -> result.predicted_label (e.g., "General", "Teen", "Mature")
   # -> result.confidence (e.g., 0.85)
   # -> result.top3_chunks (List of dictionaries with XAI text snippets)
=========================================================
"""

import sys
import os
from transformers import RobertaTokenizer
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import existing tools from this module
from evaluate_and_explain import load_model, EvalConfig, explain_prediction

class AgeBracketModel:
    """
    A unified wrapper class for the Age Classification module.
    Allows the main orchestrator to load the model into memory once
    and run fast inference on fetched transcripts.
    """
    def __init__(self, checkpoint_path=None):
        if checkpoint_path is None:
            checkpoint_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best_model.pt")
        print("🧠 [Age Bracket AI] Initializing model & tokenizer...")
        self.cfg = EvalConfig(checkpoint_path=checkpoint_path)
        self.tokenizer = RobertaTokenizer.from_pretrained(self.cfg.model_name)
        self.model = load_model(self.cfg)
        print("✅ [Age Bracket AI] Ready for inference!")

    def predict_age(self, transcript_text: str):
        """
        The main inference function.
        
        INPUT PARAMETER:
        - transcript_text (str): The full, combined transcript of the YouTube video.
        
        RETURNS:
        - A result object containing predicted_label, confidence, and top3_chunks (XAI).
        """
        if not transcript_text or len(transcript_text.strip()) == 0:
            print("⚠️ [Age Bracket AI] Empty transcript provided. Returning None.")
            return None

        # Run the XAI function to get predictions and attention weights
        result = explain_prediction(
            transcript_text=transcript_text,
            model=self.model,
            tokenizer=self.tokenizer,
            cfg=self.cfg
        )
        return result