# Dynamic Framing Feature Guide

## Overview

The dynamic framing feature allows you to control how many frames per minute are extracted from videos for analysis. This gives you precise control over the trade-off between processing speed and detection accuracy.

## Features

### Frontend Slider Control

- **Range**: 1-60 frames per minute
- **Default**: 10 frames per minute
- **Real-time Feedback**: Shows estimated frame count as you adjust the slider
- **Visual Design**: Beautiful gradient slider with smooth interactions

### Backend Processing

- **Dynamic Calculation**: Automatically calculates optimal frame interval based on video duration
- **Even Distribution**: Spreads frames evenly throughout the video
- **Smart Limits**: Respects maximum frame constraints if specified

## How It Works

### Frame Calculation Formula

```
Total Frames = Frames Per Minute × Video Duration (in minutes)
Extraction Interval = Total Video Frames / Total Frames to Extract
```

### Example Calculations

**3-minute video at 10 fps:**
- Expected frames: 10 × 3 = 30 frames
- If video has 5400 total frames (30 fps × 180 seconds)
- Extraction interval: 5400 / 30 = every 180th frame

**5-minute video at 20 fps:**
- Expected frames: 20 × 5 = 100 frames
- If video has 9000 total frames (30 fps × 300 seconds)
- Extraction interval: 9000 / 100 = every 90th frame

## Usage

### Frontend Usage

1. Open the dashboard at `http://localhost:3000`
2. Paste a YouTube URL
3. Adjust the "Frames per Minute" slider:
   - Lower values (1-10): Faster processing, good for quick scans
   - Medium values (10-30): Balanced speed and accuracy (recommended)
   - Higher values (30-60): More accurate, slower processing
4. Click "Analyze Video"

### API Usage

```python
import requests

# Single video analysis with custom frames per minute
response = requests.post('http://localhost:8000/analyze/video', json={
    'video_url': 'https://youtube.com/watch?v=VIDEO_ID',
    'frames_per_minute': 15  # Extract 15 frames per minute
})

result = response.json()
print(f"Analysis ID: {result['analysis_id']}")
```

### Python Script Usage

```python
from src.youtube_extractor import YouTubeMediaExtractor

extractor = YouTubeMediaExtractor()

# Download video
video_path = extractor.download_video('YOUTUBE_URL')

# Extract frames with dynamic framing
frame_paths = extractor.extract_frames(
    video_path,
    frames_per_minute=20  # Extract 20 frames per minute
)

print(f"Extracted {len(frame_paths)} frames")
```

## Performance Guidelines

### Recommended Settings by Use Case

| Use Case | FPS Setting | Processing Time | Accuracy |
|----------|-------------|-----------------|----------|
| Quick scan | 5-10 | Fast (~30s) | Good |
| Standard analysis | 10-20 | Medium (~1-2 min) | Very Good |
| Detailed analysis | 20-30 | Slower (~2-5 min) | Excellent |
| Maximum accuracy | 30-60 | Slowest (5+ min) | Best |

### Processing Time Estimates

For a typical 3-minute video:

- **5 fps**: ~30 frames → ~30 seconds processing
- **10 fps**: ~60 frames → ~45 seconds processing
- **20 fps**: ~120 frames → ~1.5 minutes processing
- **30 fps**: ~180 frames → ~2.5 minutes processing
- **60 fps**: ~360 frames → ~5 minutes processing

*Note: Actual times may vary based on hardware and network speed*

## Technical Details

### Backend Implementation

#### API Endpoint (`api/main.py`)

```python
class VideoAnalysisRequest(BaseModel):
    video_url: str
    frames_per_minute: Optional[int] = 10  # Default: 10 fps
```

#### Extractor Method (`src/youtube_extractor.py`)

```python
def extract_frames(self, video_path, output_dir=None, frames_per_minute=None, max_frames=None):
    """
    Extract frames dynamically based on frames per minute
    
    Args:
        video_path: Path to video file
        output_dir: Output directory for frames
        frames_per_minute: Number of frames per minute (None for all)
        max_frames: Maximum frames limit
    """
    # Calculate video duration
    duration_minutes = total_frames / fps / 60
    
    # Calculate target frames
    total_frames_to_extract = int(frames_per_minute * duration_minutes)
    
    # Calculate extraction interval
    interval = max(1, int(total_frames / total_frames_to_extract))
    
    # Extract frames at calculated interval
    ...
```

### Frontend Implementation

#### HTML Slider (`frontend/index.html`)

```html
<input type="range" 
       id="framesPerMinute" 
       min="1" 
       max="60" 
       value="10" 
       step="1">
```

#### JavaScript Handler (`frontend/dashboard.js`)

```javascript
// Send to API
const payload = {
    video_url: videoUrl,
    frames_per_minute: parseInt(document.getElementById('framesPerMinute').value)
};
```

## Testing

### Run Test Script

```bash
# Test with existing video
python scripts/test_dynamic_framing.py

# Test with YouTube URL
python scripts/test_dynamic_framing.py --url https://youtube.com/watch?v=VIDEO_ID
```

### What the Test Verifies

1. Different FPS settings (5, 10, 20, 30)
2. Expected vs actual frame counts
3. Frame distribution across video duration
4. Processing time correlation

## Troubleshooting

### Issue: Too few frames extracted

**Solution**: Increase frames per minute setting or check video duration

### Issue: Processing too slow

**Solution**: Decrease frames per minute setting (try 5-10 for faster results)

### Issue: Inaccurate violence detection

**Solution**: Increase frames per minute to capture more detail (try 20-30)

### Issue: Memory errors with long videos

**Solution**: Use lower FPS setting (5-10) for very long videos (>10 minutes)

## Migration Notes

### Backward Compatibility

- Old API calls without `frames_per_minute` parameter will use default value (10)
- Existing code continues to work without modifications
- Can still use `max_frames` parameter for absolute limits

### Breaking Changes

- `extract_all_frames` boolean parameter deprecated in favor of `frames_per_minute`
- Setting `frames_per_minute=None` extracts all frames (backward compatible behavior)

## Best Practices

1. **Start with default (10 fps)**: Good balance for most videos
2. **Adjust based on results**: Increase if missing violence, decrease if too slow
3. **Consider video length**: Longer videos need lower FPS to keep processing manageable
4. **Test different settings**: Each video type may have optimal setting
5. **Monitor performance**: Watch processing time and adjust accordingly

## Future Enhancements

Potential improvements for future versions:

- [ ] Auto-recommend FPS based on video category
- [ ] Variable FPS within same video (more frames during action scenes)
- [ ] Real-time preview of frame positions
- [ ] Save preferred FPS settings per user
- [ ] Batch processing with different FPS per video

## Support

For issues or questions:
1. Check this guide first
2. Review test script output
3. Check backend logs for detailed error messages
4. Verify video properties (duration, FPS, format)

---

**Last Updated**: 2026-03-04  
**Version**: 1.0.0
