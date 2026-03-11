import yt_dlp
import urllib.request
import json
import time

def search_video(query):
    ydl_opts = {'extract_flat': True, 'quiet': True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        result = ydl.extract_info(f"ytsearch1:{query}", download=False)
        if 'entries' in result and len(result['entries']) > 0:
            return result['entries'][0]['id']
    return None

def trigger_analysis(video_id):
    url = f"https://www.youtube.com/watch?v={video_id}"
    print(f"Triggering analysis for: {url}")
    req = urllib.request.Request(
        'http://127.0.0.1:8000/analyze/video', 
        data=json.dumps({'video_url': url, 'frames_per_minute': 15}).encode('utf-8'), 
        headers={'Content-Type': 'application/json'}
    )
    try:
        res = urllib.request.urlopen(req)
        print("Response:", res.read().decode('utf-8'))
    except Exception as e:
        print(f"Failed to trigger {url}: {e}")

if __name__ == "__main__":
    queries = [
        "UFC knockout short", 
        "GTA 5 shootout", 
        "MKBHD review short"
    ]
    
    for q in queries:
        vid = search_video(q)
        if vid:
            print(f"Found {vid} for query '{q}'")
            trigger_analysis(vid)
        else:
            print(f"Could not find video for {q}")
        time.sleep(1)
