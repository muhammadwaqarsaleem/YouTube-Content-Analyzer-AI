"""
predict.py - Live Inference Script
Tests the Hierarchical RoBERTa model on any wild YouTube video.
"""

import sys
from youtube_transcript_api import YouTubeTranscriptApi
from transformers import RobertaTokenizer
import torch

# Import your existing tools
from evaluate_and_explain import load_model, EvalConfig, explain_prediction

def get_youtube_transcript(video_url: str) -> str:
    """Extracts the transcript from a YouTube URL."""
    try:
        # Extract the video ID from the URL (handles standard and shortened links)
        if "v=" in video_url:
            video_id = video_url.split("v=")[1].split("&")[0]
        elif "youtu.be/" in video_url:
            video_id = video_url.split("youtu.be/")[1].split("?")[0]
        else:
            raise ValueError("Invalid YouTube URL format.")

        print(f"📥 Fetching transcript for Video ID: {video_id}...")
        
        # New API syntax
        ytt_api = YouTubeTranscriptApi()
        transcript_list = ytt_api.fetch(video_id)
        
        # --- THE FIX: NEW OBJECT ATTRIBUTE SYNTAX ---
        # Instead of entry['text'] (dictionary), we use entry.text (object property)
        full_transcript = " ".join([entry.text for entry in transcript_list])
        # --------------------------------------------
        
        print(f"✅ Transcript fetched! Length: {len(full_transcript.split())} words.")
        return full_transcript

    except Exception as e:
        print(f"❌ Error fetching transcript: {e}")
        print("Note: The video might not have English subtitles enabled.")
        sys.exit(1)

def main():
    print("\n" + "="*60)
    print("🤖 YOUTUBE AGE CLASSIFIER - LIVE INFERENCE")
    print("="*60 + "\n")

    # 1. Ask the user for a wild YouTube URL
    video_url = input("🔗 Enter a YouTube URL to test: ")
    
    # 2. Get the transcript
    transcript_text = get_youtube_transcript(video_url)

    # 3. Load configuration, tokenizer, and your best model
    print("\n🧠 Loading Model (this takes a second)...")
    cfg = EvalConfig(checkpoint_path="checkpoints/best_model.pt")
    tokenizer = RobertaTokenizer.from_pretrained(cfg.model_name)
    model = load_model(cfg)

    # 4. Run your exact XAI function from File 4!
    print("🔍 Analyzing content semantics...\n")
    result = explain_prediction(
        transcript_text=transcript_text,
        model=model,
        tokenizer=tokenizer,
        cfg=cfg
    )

    # 5. Print the results beautifully
    print("="*60)
    print(f"🎯 FINAL PREDICTION: {result.predicted_label.upper()}")
    print(f"📊 Confidence Score: {result.confidence:.2%}")
    print("="*60)
    
    print("\n🧐 WHY DID THE AI CHOOSE THIS?")
    print("Here are the top 2 chunks of text that triggered this decision:\n")
    
    for item in result.top3_chunks[:2]:
        print(f"🔹 Evidence #{item['rank']} (Weight: {item['weight']:.4f})")
        # Print just the first 300 characters of the chunk so it doesn't flood the terminal
        snippet = item['text'][:300].replace('\n', ' ') + "..."
        print(f"   \"{snippet}\"\n")

if __name__ == "__main__":
    main()