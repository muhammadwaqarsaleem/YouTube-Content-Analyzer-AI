# 🚀 YouTube Content Analyzer AI

A professional-grade, multi-modal AI platform designed to perform deep forensic and impact analysis on YouTube content. This system leverages state-of-the-art Transformer models, Computer Vision, and advanced linguistic heuristics to provide a 360-degree view of any video's safety, category, and potential viral impact.

---

## 🌟 Core Analysis Modules

### 1. 🔞 Age Classification (Hierarchical RoBERTa)
Utilizes a **Hierarchical RoBERTa** architecture to classify content into age-appropriate brackets: **General**, **Teen**, or **Mature**.
- **Mechanism:** Processes the entire video transcript using a sliding-window approach (512-token chunks).
- **Explainability (XAI):** Provides transparent "evidence" by highlighting the specific transcript segments that most heavily influenced the classification decision.

### 2. 🎣 Harm & Clickbait Detection
A specialized **RoBERTa 7-Class Classifier** trained to identify various forms of harmful or misleading content.
- **Detected Classes:** Clickbait, Information Harm, Physical Harm, Addiction, Sexual Content, and Hate/Harassment.
- **Inference:** Analyzes the synergy between the Title, Description, and Transcript to detect deceptive patterns.

### 3. 📊 YouTube Video Impact Model (v9)
An advanced scoring engine that quantifies a video's success across five key dimensions.
- **Dimensions:** Quality, Engagement, Sentiment, Reach, and Virality.
- **Content Quality Analysis:** Deep-dives into transcript linguistics, including Vocabulary Richness, Flesch-Kincaid Readability, Lexical Diversity (MTLD), and Semantic Consistency.
- **Score:** Outputs a 0–100 **Impact Score** with qualitative tiers (Minimal to Viral).

### 4. 🏷️ Category Classification (Multi-modal)
Predicts the most accurate YouTube category using a blend of metadata and text features.
- **Model:** Leverages **XGBoost/LightGBM** on top of TF-IDF weighted text representations and log-transformed engagement metrics.
- **Accuracy:** Cross-references YouTube's native Category IDs with AI-predicted categories for verification.

### 5. 🚨 Violence Detection (ResNet-50 Computer Vision)
A computer vision pipeline that scans the visual stream for graphic or violent content.
- **Model:** **ResNet-50** fine-tuned for violent scene recognition.
- **Multi-Tier Fetching:** Robust frame extraction logic that attempts high-quality streams via `yt-dlp`, falls back to YouTube Storyboard APIs, and finally to static thumbnails if necessary.
- **Timeline:** Provides a second-by-second timeline of detected violent events.

---

## 🛠️ Technology Stack

- **Backend:** Python 3.10+, Flask
- **Machine Learning:** PyTorch, TensorFlow, Scikit-learn, LightGBM
- **NLP:** HuggingFace Transformers (RoBERTa), Sentence-Transformers, TextBlob, VADER
- **Computer Vision:** OpenCV, Pillow, ResNet-50
- **Data Acquisition:** YouTube Data API v3, yt-dlp, youtube-transcript-api

---

## 🚀 Getting Started

### 1. Prerequisites
Ensure you have Python installed and a valid **YouTube Data API Key** from the [Google Cloud Console](https://console.cloud.google.com/).

### 2. Installation
Clone the repository and install the required dependencies:
```bash
git clone https://github.com/muhammadwaqarsaleem/YouTube-Content-Analyzer-AI.git
cd YouTube-Content-Analyzer-AI
pip install -r requirements.txt
```

### 3. Configuration
Create a `.env` file in the root directory and add your API key:
```env
YOUTUBE_API_KEY=your_api_key_here
```

---

## 💻 Usage

### Option A: Web Dashboard (Recommended)
Launch the interactive web interface for a premium, visual experience:
```bash
python app.py
```
Open your browser and navigate to `http://127.0.0.1:5000`.

### Option B: CLI Mode
Run the analysis directly from your terminal:
```bash
python global_analyzer.py
```
Simply paste a YouTube URL when prompted to see a detailed text-based report.

---

## 📄 Documentation & Safety
This tool is intended for content moderation, brand safety analysis, and educational research. The models are probabilistic and should be used as a supplementary tool for content evaluation.
