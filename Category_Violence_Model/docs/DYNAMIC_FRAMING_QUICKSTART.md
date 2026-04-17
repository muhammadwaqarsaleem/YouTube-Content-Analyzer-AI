# Dynamic Framing - Quick Start Guide

## 🎯 What You Can Do Now

You have a **slider control** in your frontend that lets you choose how many frames per minute to extract from videos. This gives you control over:
- ⚡ **Processing Speed** - Lower FPS = Faster results
- 🎯 **Detection Accuracy** - Higher FPS = More thorough analysis

## 🚀 How to Use It

### Step 1: Start Your Servers

```bash
# Terminal 1 - Start Backend API
python api/main.py

# Terminal 2 - Start Frontend (if not using FastAPI static files)
# Open frontend/index.html in your browser or use a simple server
python -m http.server 3000 --directory frontend
```

### Step 2: Use the Slider

1. Open your dashboard at `http://localhost:3000`
2. You'll see a new slider labeled **"Frames per Minute"**
3. Default is set to **10 frames per minute**
4. Slide left/right to adjust (range: 1-60)

### Step 3: Analyze Video

1. Paste a YouTube URL
2. Adjust the slider to your preferred FPS setting
3. Click "Analyze Video"
4. Watch the processing with real-time status updates

## 📊 Recommended Settings

| Setting | Best For | Processing Time |
|---------|----------|-----------------|
| **5-10 fps** | Quick scans, short clips | ~30-60 seconds |
| **10-20 fps** | Standard analysis (RECOMMENDED) | ~1-2 minutes |
| **20-30 fps** | Detailed violence detection | ~2-4 minutes |
| **30-60 fps** | Maximum accuracy, research | ~5+ minutes |

## 💡 Example Scenarios

### Scenario 1: Quick Test (Fast)
```
Slider Setting: 5 fps
Video: 3-minute clip
Expected Frames: ~15
Processing Time: ~30 seconds
```

### Scenario 2: Standard Analysis (Balanced)
```
Slider Setting: 15 fps
Video: 5-minute video
Expected Frames: ~75
Processing Time: ~1.5 minutes
```

### Scenario 3: Detailed Analysis (Thorough)
```
Slider Setting: 30 fps
Video: 10-minute video
Expected Frames: ~300
Processing Time: ~5 minutes
```

## 🔍 Testing the Feature

Run this test script to see dynamic framing in action:

```bash
# Test with a local video file
python scripts/test_dynamic_framing.py

# Test with a YouTube URL
python scripts/test_dynamic_framing.py --url https://youtube.com/watch?v=VIDEO_ID
```

The test will:
1. Extract frames at different FPS settings (5, 10, 20, 30)
2. Show you expected vs actual frame counts
3. Display processing statistics

## 📝 API Changes

If you're calling the API directly:

### Old Way (Deprecated)
```python
{
    "video_url": "...",
    "extract_all_frames": True,
    "max_frames": 100
}
```

### New Way (Dynamic Framing)
```python
{
    "video_url": "...",
    "frames_per_minute": 15  # Extract 15 frames per minute
}
```

## 🎨 Visual Features

The slider includes:
- ✅ **Real-time value display** - Shows current FPS setting
- ✅ **Estimated frames counter** - Calculates approximate frames for 3-min video
- ✅ **Beautiful gradient design** - Modern UI with smooth interactions
- ✅ **Hover effects** - Visual feedback when interacting
- ✅ **Responsive layout** - Works on mobile and desktop

## ⚙️ How It Works

### Backend Calculation

```python
# For a 3-minute video at 10 fps:
duration_minutes = 3
frames_per_minute = 10
total_frames_to_extract = 10 * 3 = 30 frames

# If video has 5400 total frames (at 30fps):
extraction_interval = 5400 / 30 = every 180th frame
```

This ensures frames are **evenly distributed** throughout the entire video duration.

## 🛠️ Troubleshooting

### Issue: Slider not showing up
**Fix**: Clear browser cache and refresh (Ctrl+F5)

### Issue: Getting errors when analyzing
**Fix**: Make sure backend API is running on port 8000

### Issue: Too few frames extracted
**Fix**: Increase the FPS slider value

### Issue: Processing too slow
**Fix**: Decrease the FPS slider value (try 5-10)

## 📈 Performance Tips

1. **Start with default (10 fps)** - Works well for most videos
2. **Adjust based on needs**:
   - Missing violence? → Increase FPS
   - Too slow? → Decrease FPS
3. **Long videos (>10 min)** - Use lower FPS (5-10) to keep processing manageable
4. **Short videos (<2 min)** - Can use higher FPS (20-30) for better accuracy

## 🎯 Next Steps

1. ✅ Try the slider with different settings
2. ✅ Compare processing times at different FPS values
3. ✅ Run the test script to understand frame extraction
4. ✅ Check `DYNAMIC_FRAMING_GUIDE.md` for detailed documentation

## 📞 Support

If you encounter issues:
1. Check browser console for JavaScript errors
2. Check backend terminal for Python errors
3. Verify both servers are running
4. Review the full guide: `DYNAMIC_FRAMING_GUIDE.md`

---

**Ready to test?** Just start your servers and slide away! 🚀
