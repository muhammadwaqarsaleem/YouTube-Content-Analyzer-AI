# Quick Start Guide - YouTube Video Analysis Backend

## 🚀 Installation (5 minutes)

### Step 1: Install Dependencies
```bash
cd C:\Users\Syed Ali\Desktop\Project69
pip install -r requirements.txt
```

### Step 2: Verify Installation
```bash
python start_server.py
```

**Expected Output:**
```
Checking dependencies...
  ✓ fastapi
  ✓ uvicorn
  ✓ yt_dlp
  ✓ tensorflow
  ...
✓ All dependencies satisfied

Starting server on http://0.0.0.0:8000
```

## 🎯 Usage (3 Steps)

### Method 1: Web Dashboard (Easiest)

1. **Open Browser**: Go to http://localhost:8000/

2. **Paste URL**: Enter any YouTube video URL
   ```
   https://youtube.com/watch?v=VIDEO_ID
   ```

3. **Click Analyze**: Wait 1-5 minutes depending on video length

4. **View Results**: 
   - Content rating (SAFE/CAUTION/RESTRICTED)
   - Violence percentage and timeline
   - Category prediction with confidence scores

### Method 2: Python Script

```python
import requests

# Analyze video
response = requests.post(
    'http://localhost:8000/analyze/video',
    json={'video_url': 'YOUR_YOUTUBE_URL'}
)

analysis_id = response.json()['analysis_id']

# Get results (poll every 2 seconds)
import time
while True:
    result = requests.get(f'http://localhost:8000/analysis/{analysis_id}')
    if result.json()['status'] == 'complete':
        print(result.json()['data'])
        break
    time.sleep(2)
```

### Method 3: cURL (Command Line)

```bash
# Start analysis
curl -X POST http://localhost:8000/analyze/video \
  -H "Content-Type: application/json" \
  -d '{"video_url":"https://youtube.com/watch?v=VIDEO_ID"}'

# Get results
curl http://localhost:8000/analysis/ANALYSIS_ID
```

## 📊 What You Get

### Input
```
YouTube URL → System processes automatically
```

### Output
```json
{
  "violence_analysis": {
    "is_violent": true/false,
    "violence_percentage": 15.5,
    "violent_frame_count": 155,
    "timeline": [...]
  },
  "category_prediction": {
    "primary_category": "Entertainment",
    "confidence": 0.85,
    "all_categories": [...]
  },
  "overall_rating": "SAFE/CAUTION/RESTRICTED"
}
```

## ⚙️ Configuration Options

### Fast Processing (Thumbnail Only)
```json
{
  "video_url": "...",
  "extract_all_frames": false
}
```
**Time**: ~5 seconds  
**Accuracy**: Lower (thumbnail only)

### Maximum Accuracy (All Frames)
```json
{
  "video_url": "...",
  "extract_all_frames": true,
  "max_frames": null
}
```
**Time**: 1-5 minutes  
**Accuracy**: Highest (analyzes every frame)

### Balanced Approach
```json
{
  "video_url": "...",
  "extract_all_frames": true,
  "max_frames": 1000
}
```
**Time**: ~30 seconds  
**Accuracy**: Good (samples up to 1000 frames)

## 🔧 Common Commands

### Start Server
```bash
python start_server.py
```

### Start on Different Port
```bash
python start_server.py --port 8001
```

### Check API Health
```bash
curl http://localhost:8000/health
```

### View API Documentation
```
http://localhost:8000/docs
```

## 📁 File Locations

| Component | Location |
|-----------|----------|
| Violence Model | `violence_detection_model_resnet.h5` |
| Category Model | `models/logistic_regression_model.pkl` |
| Feature Encoders | `features/*.pkl` |
| Analysis Results | `temp/results/` |
| Downloaded Videos | `temp/videos/` (temporary) |
| Extracted Frames | `temp/frames/` (temporary) |

## ⏱️ Processing Times

| Video Length | CPU Time | GPU Time |
|--------------|----------|----------|
| 1 minute | ~30 sec | ~5 sec |
| 5 minutes | ~2-3 min | ~20 sec |
| 10 minutes | ~5-7 min | ~45 sec |

*Times are approximate and vary by hardware*

## 🐛 Quick Troubleshooting

### "Module not found"
```bash
pip install -r requirements.txt
```

### "Model not found"
Ensure these files exist:
- `violence_detection_model_resnet.h5`
- `models/logistic_regression_model.pkl`

### "Port already in use"
```bash
python start_server.py --port 8001
```

### "Video download failed"
- Check internet connection
- Verify YouTube URL is valid
- Some videos may be region-restricted

## 💡 Pro Tips

1. **Use Short Videos First**: Test with 1-2 minute videos to verify setup

2. **Enable GPU**: If you have NVIDIA GPU, install CUDA-enabled TensorFlow:
   ```bash
   pip uninstall tensorflow
   pip install tensorflow-gpu
   ```

3. **Batch Processing**: Analyze multiple videos at once:
   ```python
   POST /analyze/batch
   {"video_urls": ["url1", "url2", "url3"]}
   ```

4. **Download Reports**: Use the web dashboard's "Download Report" button to save JSON

5. **Reuse Analysis**: Results are cached for 24 hours at `/temp/results/`

## 📞 Need Help?

1. Check console logs for detailed errors
2. Visit API docs: http://localhost:8000/docs
3. Review BACKEND_README.md for full documentation

---

**Ready to analyze! 🎬**
