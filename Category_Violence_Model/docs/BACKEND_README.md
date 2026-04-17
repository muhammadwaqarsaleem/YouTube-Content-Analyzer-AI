# YouTube Video Analysis Backend

## 🎯 Overview

A comprehensive backend system for analyzing YouTube videos using AI-powered violence detection and category prediction. The system accepts YouTube video URLs and provides detailed analysis through a modern web dashboard.

## ✨ Features

### Core Capabilities
- **Violence Detection**: Analyzes ALL video frames using ResNet50-based CNN model trained on ~20,000 violence/non-violence frames
- **Category Prediction**: Multi-modal category prediction (Entertainment, Comedy, Tech, etc.) using thumbnail images + metadata
- **Dynamic Processing**: User simply provides YouTube URL - system handles everything else
- **Web Dashboard**: Modern, responsive UI with real-time progress tracking and interactive visualizations

### Technical Highlights
- **yt-dlp Integration**: Professional-grade YouTube media extraction
- **All Frames Processing**: Processes every frame for maximum accuracy (configurable)
- **Background Processing**: Async task processing with progress tracking
- **RESTful API**: FastAPI-based REST API with automatic documentation
- **Memory Efficient**: Batch processing to handle large videos without memory overflow

## 📁 Project Structure

```
Project69/
├── api/                        # FastAPI application
│   ├── main.py                # Main API server
│   └── routes.py              # API endpoints
├── services/                   # Business logic layer
│   ├── violence_service.py    # Violence detection service
│   ├── category_service.py    # Category prediction service
│   └── analysis_aggregator.py # Results aggregation
├── src/                        # Core modules
│   ├── youtube_extractor.py   # YouTube media extraction
│   ├── violence_preprocess.py # Frame preprocessing
│   ├── category_features.py   # Feature extraction
│   ├── model.py               # Model definitions
│   ├── preprocess.py          # Data preprocessing
│   ├── trainer.py             # Model training
│   └── evaluator.py           # Model evaluation
├── utils/                      # Utilities
│   ├── file_manager.py        # File operations
│   └── video_utils.py         # Video utilities
├── frontend/                   # Web dashboard
│   ├── index.html             # Main HTML
│   ├── styles.css             # Styles
│   └── dashboard.js           # Frontend logic
├── temp/                       # Temporary files (auto-created)
│   ├── videos/                # Downloaded videos
│   ├── frames/                # Extracted frames
│   ├── thumbnails/            # Thumbnails
│   └── results/               # Analysis results
├── models/                     # Saved models
│   ├── logistic_regression_model.pkl
│   └── prediction_example.py
├── features/                   # Feature encoders
│   ├── tfidf_vectorizer.pkl
│   ├── cat_vectorizer.pkl
│   ├── num_scaler.pkl
│   └── target_encoder.pkl
├── violence_detection_model_resnet.h5  # Violence model
├── requirements.txt            # Dependencies
└── start_server.py            # Startup script
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- FFmpeg (for video processing)
- 4GB+ RAM recommended
- GPU optional (CUDA support for faster processing)

### Installation

1. **Install Dependencies**
```bash
pip install -r requirements.txt
```

2. **Verify Models Exist**
Ensure these model files are present:
- `violence_detection_model_resnet.h5`
- `models/logistic_regression_model.pkl`

3. **Start Server**
```bash
python start_server.py
```

The server will start at: http://localhost:8000

### Access Points
- **Web Dashboard**: http://localhost:8000/
- **API Documentation**: http://localhost:8000/docs
- **Alternative Docs**: http://localhost:8000/redoc

## 📖 Usage Guide

### Web Dashboard (Recommended)

1. Open browser to http://localhost:8000/
2. Paste YouTube URL
3. Select "Extract all frames" option (recommended for best accuracy)
4. Click "Analyze Video"
5. Wait for processing (progress bar shows status)
6. View comprehensive results:
   - Video information
   - Content rating (SAFE/CAUTION/RESTRICTED)
   - Violence metrics and timeline
   - Category prediction with confidence scores
7. Download report as JSON

### API Usage

#### Single Video Analysis

```python
import requests

# Start analysis
response = requests.post('http://localhost:8000/analyze/video', json={
    'video_url': 'https://youtube.com/watch?v=VIDEO_ID',
    'extract_all_frames': True
})

analysis_id = response.json()['analysis_id']

# Poll for results
import time
while True:
    result = requests.get(f'http://localhost:8000/analysis/{analysis_id}')
    if result.json()['status'] == 'complete':
        data = result.json()['data']
        break
    time.sleep(2)

print(data)
```

#### Batch Analysis

```python
response = requests.post('http://localhost:8000/analyze/batch', json={
    'video_urls': [
        'https://youtube.com/watch?v=VIDEO1',
        'https://youtube.com/watch?v=VIDEO2'
    ]
})
```

#### Get Results

```python
response = requests.get(f'http://localhost:8000/analysis/{analysis_id}')
results = response.json()
```

## 🔧 Configuration

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/analyze/video` | POST | Analyze single video |
| `/analyze/batch` | POST | Analyze multiple videos |
| `/analysis/{id}` | GET | Get analysis results |
| `/analysis/{id}` | DELETE | Delete analysis |
| `/health` | GET | Health check |
| `/stats` | GET | System statistics |

### Request Parameters

**Video Analysis Request:**
```json
{
  "video_url": "https://youtube.com/watch?v=...",
  "extract_all_frames": true,
  "max_frames": null  // Optional limit
}
```

### Processing Options

- **extract_all_frames**: 
  - `true` (default): Processes ALL frames - highest accuracy
  - `false`: Only thumbnail - fastest (~5 seconds)

- **max_frames**: Limit maximum frames processed (useful for very long videos)

## 🎯 Model Details

### Violence Detection Model
- **Architecture**: ResNet50 with custom classification head
- **Input**: 224x224 RGB images
- **Output**: Binary classification (Violence/Non-Violence)
- **Training Data**: ~20,000 labeled frames
- **Accuracy**: High precision on test set
- **Processing**: Batch inference for efficiency

### Category Prediction Model
- **Algorithm**: Logistic Regression with multi-modal features
- **Features**:
  - Image features (thumbnail pixels)
  - Text features (TF-IDF from title, description, tags)
  - Numerical features (views, duration, engagement)
  - Categorical features (quality, platform)
- **Categories**: 16+ categories including Entertainment, Comedy, Tech, Science, News, Food, etc.
- **Multi-label**: Supports multiple category predictions

## 📊 Output Format

### Analysis Report Structure

```json
{
  "analysis_id": "analysis_abc123_1234567890",
  "timestamp": "2024-01-01T12:00:00",
  "video_id": "abc123",
  
  "summary": {
    "violence": {
      "is_violent": false,
      "violence_percentage": 5.2,
      "severity": "LOW"
    },
    "category": {
      "primary_category": "Entertainment",
      "confidence": 0.85,
      "is_multi_label": true
    },
    "overall_rating": "SAFE",
    "recommendation": "Content is suitable..."
  },
  
  "violence_analysis": {
    "is_violent": false,
    "violence_percentage": 5.2,
    "violent_frame_count": 52,
    "total_frames": 1000,
    "timeline": [
      {
        "frame_index": 0,
        "timestamp": 0.0,
        "is_violent": false,
        "confidence": 0.95
      }
    ],
    "violent_frame_timestamps": []
  },
  
  "category_prediction": {
    "primary_category": "Entertainment",
    "primary_probability": 0.85,
    "all_categories": [...],
    "is_multi_label": true
  },
  
  "metadata": {
    "title": "...",
    "channel": "...",
    "duration": 300,
    "view_count": 50000
  },
  
  "processing_info": {
    "processing_time_seconds": 45.2
  }
}
```

## ⚙️ Advanced Features

### Memory Management

For handling large videos efficiently:

```python
# Automatic batch processing
batch_size = 32  # Configurable
for i in range(0, len(frames), batch_size):
    batch = frames[i:i+batch_size]
    predictions = model.predict(batch)
    del batch  # Free memory
```

### Performance Optimization

- **GPU Acceleration**: TensorFlow automatically uses GPU if available
- **Batch Processing**: Configurable batch sizes
- **Async I/O**: Non-blocking file operations
- **Caching**: Results cached for 24 hours

### Cleanup Strategy

Temporary files are automatically cleaned:
- Videos: Removed after processing
- Frames: Removed after analysis
- Results: Kept for 24 hours, then auto-deleted

## 🐛 Troubleshooting

### Common Issues

**1. Slow Processing**
- Normal for long videos (e.g., 10-min video ≈ 2-5 minutes)
- Enable GPU for 5-10x speedup
- Reduce max_frames for faster results

**2. Memory Error**
- Decrease batch_size in code
- Use max_frames parameter
- Close other applications

**3. Download Failed**
- Check internet connection
- Verify YouTube URL is valid
- Some videos may be region-restricted

**4. Model Not Found**
- Ensure model files are in correct location
- Check file paths in error message
- Re-run training scripts if needed

### Logs & Debugging

Enable debug logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Check console output for detailed processing logs.

## 📈 Performance Benchmarks

Typical processing times (CPU):
- 1-minute video: ~30 seconds
- 5-minute video: ~2-3 minutes
- 10-minute video: ~5-7 minutes

With GPU (CUDA):
- 5-10x faster than CPU

## 🔒 Security & Privacy

- Videos are downloaded temporarily and deleted after processing
- No data is sent to external servers (except YouTube for download)
- Results stored locally in `temp/results/`
- CORS enabled for frontend access (configure for production)

## 🚀 Production Deployment

### Recommended Setup

1. **Use Gunicorn with Uvicorn workers**
```bash
pip install gunicorn
gunicorn -w 4 -k uvicorn.workers.UvicornWorker api.main:app
```

2. **Configure CORS properly**
Edit `api/main.py` to specify allowed origins:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],
    ...
)
```

3. **Add authentication**
Implement API key or JWT authentication

4. **Use Redis for caching**
Cache frequently analyzed videos

5. **Set up monitoring**
- Prometheus + Grafana for metrics
- Log aggregation with ELK stack

## 📝 License

This project is for educational and research purposes. Ensure compliance with YouTube's Terms of Service when deploying.

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Additional ML models
- More category types
- Real-time streaming support
- Mobile app integration

## 📧 Support

For issues or questions, check:
- API Documentation: http://localhost:8000/docs
- Console logs for detailed errors

---

**Built with ❤️ using FastAPI, TensorFlow, and yt-dlp**
