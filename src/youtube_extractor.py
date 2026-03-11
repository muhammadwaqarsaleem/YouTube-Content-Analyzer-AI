"""
YouTube Media Extractor Module
Extracts video frames, thumbnails, and metadata from YouTube videos using yt-dlp
"""

import os
import cv2
import numpy as np
import json
from pathlib import Path
import yt_dlp
from urllib.parse import urlparse, parse_qs


class YouTubeMediaExtractor:
    """
    Extract media and metadata from YouTube videos
    """
    
    def __init__(self, output_dir='temp'):
        """
        Initialize the extractor
        
        Args:
            output_dir: Base directory for temporary files
        """
        self.output_dir = output_dir
        self.video_dir = os.path.join(output_dir, 'videos')
        self.frame_dir = os.path.join(output_dir, 'frames')
        self.thumbnail_dir = os.path.join(output_dir, 'thumbnails')
        self.results_dir = os.path.join(output_dir, 'results')
        
        # Create directories
        for dir_path in [self.video_dir, self.frame_dir, self.thumbnail_dir, self.results_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        # Check for cookie file (for age-restricted videos)
        self.cookie_file = 'cookies/youtube_cookies.txt'
        self.has_cookies = os.path.exists(self.cookie_file)
        
        if self.has_cookies:
            print(f"✓ Cookie file found: {self.cookie_file}")
        else:
            print(f"⚠️ No cookie file found. Age-restricted videos may fail.")
            print(f"   To add cookies, export from browser to: {os.path.abspath(self.cookie_file)}")
        
        self.ydl_opts = {
            # Use general 'best' instead of forcing mp4, as YouTube now often restricts mp4s
            'format': 'bestvideo+bestaudio/best',
            'outtmpl': os.path.join(self.video_dir, '%(id)s.%(ext)s'),
            'quiet': True,
            'no_warnings': False,  # Show warnings for debugging
            'extract_flat': False,
            'ignoreerrors': True,  # Continue on errors
            'retries': 3,  # Retry on failure
            'fragment-retries': 3,  # Retry fragments
            'continuedl': True,  # Resume downloads if interrupted
            # Enable remote JS challenge solvers (required for age-restricted content)
            'remote_components': {'ejs:github', 'ejs:npm'},
            # Add extractor args to handle YouTube challenges better
            'extractor_args': {
                'youtube': {
                    # web/mweb support cookies + PO tokens; android/ios don't support cookies
                    'player_client': ['web', 'mweb', 'web_creator'], 
                    'skip': ['hlsmanifest', 'dashmanifest'],
                    'pot_provider_url': 'http://127.0.0.1:4416',
                }
            },
            'no_color': True,
        }
        
        # Add cookies if available
        if self.has_cookies:
            self.ydl_opts['cookiefile'] = self.cookie_file
            print("✓ Cookies enabled for age-restricted content")
    
    def get_video_id(self, url):
        """
        Extract video ID from YouTube URL
        
        Args:
            url: YouTube video URL
            
        Returns:
            Video ID string
        """
        parsed = urlparse(url)
        if parsed.hostname in ('www.youtube.com', 'youtube.com'):
            if parsed.path == '/watch':
                return parse_qs(parsed.query).get('v', [None])[0]
            elif parsed.path.startswith('/embed/'):
                return parsed.path.split('/')[2]
        elif parsed.hostname in ('youtu.be', 'www.youtu.be'):
            return parsed.path[1:]
        return None
    
    def download_video(self, url, output_path=None):
        """
        Download video from YouTube URL
        
        Args:
            url: YouTube video URL
            output_path: Optional custom output path
            
        Returns:
            Path to downloaded video file
        """
        print(f"Downloading video from: {url}")
        
        video_id = self.get_video_id(url)
        if not video_id:
            raise ValueError(f"Invalid YouTube URL: {url}")
        
        if output_path:
            opts = self.ydl_opts.copy()
            opts['outtmpl'] = output_path
        else:
            opts = self.ydl_opts
        
        try:
            # Overwrite ignoreerrors for download so it raises exceptions properly
            opts['ignoreerrors'] = False
            
            with yt_dlp.YoutubeDL(opts) as ydl:
                print(f"Extracting info for {url}...")
                try:
                    info = ydl.extract_info(url, download=True)
                except Exception as e:
                    # If it fails with cookies, and we have cookies, try WITHOUT them as a fallback
                    if self.has_cookies and ("sign in" in str(e).lower() or "format" in str(e).lower()):
                        print(f"⚠️ Download failed with cookies. Retrying WITHOUT cookies...")
                        no_cookie_opts = opts.copy()
                        no_cookie_opts.pop('cookiefile', None)
                        with yt_dlp.YoutubeDL(no_cookie_opts) as ydl_no_cookie:
                            info = ydl_no_cookie.extract_info(url, download=True)
                    else:
                        raise
                
                if info is None:
                    raise Exception(f"yt-dlp returned None for URL: {url} (possibly restricted or unavailable)")
                    
                print("Preparing filename...")
                video_path = ydl.prepare_filename(info)
                
                # Handle potential format changes
                if not os.path.exists(video_path):
                    # Try alternative extensions
                    for ext in ['.mp4', '.mkv', '.webm', '.avi', '.3gp']:
                        alt_path = video_path.rsplit('.', 1)[0] + ext
                        if os.path.exists(alt_path):
                            video_path = alt_path
                            break
                
                print(f"✓ Video downloaded successfully: {video_path}")
                return video_path
                
        except Exception as e:
            print(f"Error downloading video: {str(e)}")
            raise
    
    def extract_thumbnail(self, video_path_or_url, output_path=None):
        """
        Extract thumbnail from video or URL
        
        Args:
            video_path_or_url: Path to video file or YouTube URL
            output_path: Optional output path for thumbnail
            
        Returns:
            Path to thumbnail image
        """
        print(f"Extracting thumbnail from: {video_path_or_url}")
        
        # If it's a URL, get thumbnail directly from metadata
        if video_path_or_url.startswith(('http://', 'https://')):
            try:
                with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
                    info = ydl.extract_info(video_path_or_url, download=False)
                    thumbnail_url = info.get('thumbnail')
                    
                    if thumbnail_url:
                        import requests
                        response = requests.get(thumbnail_url)
                        
                        if output_path is None:
                            video_id = self.get_video_id(video_path_or_url)
                            output_path = os.path.join(self.thumbnail_dir, f"{video_id}_thumbnail.jpg")
                        
                        os.makedirs(os.path.dirname(output_path), exist_ok=True)
                        
                        with open(output_path, 'wb') as f:
                            f.write(response.content)
                        
                        print(f"✓ Thumbnail extracted: {output_path}")
                        return output_path
                        
            except Exception as e:
                print(f"Error extracting thumbnail from URL: {str(e)}")
        
        # If it's a video file, extract frame as thumbnail
        if os.path.exists(video_path_or_url):
            cap = cv2.VideoCapture(video_path_or_url)
            
            if not cap.isOpened():
                raise IOError(f"Cannot open video: {video_path_or_url}")
            
            # Get first frame
            ret, frame = cap.read()
            cap.release()
            
            if not ret:
                raise ValueError("Could not read video frame")
            
            if output_path is None:
                video_id = Path(video_path_or_url).stem
                output_path = os.path.join(self.thumbnail_dir, f"{video_id}_thumbnail.jpg")
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cv2.imwrite(output_path, frame)
            
            print(f"✓ Thumbnail extracted: {output_path}")
            return output_path
        
        # GRACEFUL DEGRADATION: Create placeholder if video unavailable
        print(f"⚠️ Video file not found, creating placeholder thumbnail")
        video_id = Path(video_path_or_url).stem if Path(video_path_or_url).exists() else "unknown"
        placeholder_path = os.path.join(self.thumbnail_dir, f"{video_id}_placeholder.jpg")
        
        # Create a professional placeholder image
        import numpy as np
        placeholder = np.zeros((720, 1280, 3), dtype=np.uint8) + 50  # Dark gray background
        
        # Add gradient for visual appeal
        for i in range(720):
            placeholder[i, :] = [50 + int(30 * (i / 720)), 50 + int(30 * (i / 720)), 50 + int(30 * (i / 720))]
        
        # Add text
        cv2.putText(placeholder, "Thumbnail Unavailable", (320, 340), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(placeholder, "Video may be private or deleted", (280, 400), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2, cv2.LINE_AA)
        cv2.putText(placeholder, f"ID: {video_id}", (480, 480), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 2, cv2.LINE_AA)
        
        # Add border
        cv2.rectangle(placeholder, (10, 10), (1270, 710), (100, 100, 100), 3)
        
        os.makedirs(os.path.dirname(placeholder_path), exist_ok=True)
        cv2.imwrite(placeholder_path, placeholder)
        print(f"✓ Placeholder thumbnail created: {placeholder_path}")
        return placeholder_path
    
    def extract_frames(self, video_path, output_dir=None, frames_per_minute=None, max_frames=100):
        """
        Extract frames from video using a hybrid sampling strategy:
        1. Intro Priority: Sample the first 60 seconds at a higher rate.
        2. Body Sampling: Sample the remaining duration evenly to reach max_frames.
        
        Args:
            video_path: Path to video file
            output_dir: Output directory for frames
            frames_per_minute: Number of frames to extract per minute of video (used if max_frames not set)
            max_frames: Maximum total frames to extract (default: 100)
            
        Returns:
            List of frame paths
        """
        print(f"Extracting frames from: {video_path}")
        
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        if output_dir is None:
            video_id = Path(video_path).stem
            output_dir = os.path.join(self.frame_dir, video_id)
        
        os.makedirs(output_dir, exist_ok=True)
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_seconds = total_frames / fps
        
        # Calculate how many frames to extract
        if max_frames is None and frames_per_minute is not None:
            max_frames = int(frames_per_minute * (duration_seconds / 60))
            max_frames = max(10, min(max_frames, 500))  # Reasonable bounds
        
        # Determine sampling distribution
        # Phase 1: Intro (First 60 seconds)
        intro_duration = min(60, duration_seconds)
        intro_frames_count = int(intro_duration * fps)
        
        # Allocate 50% of quota to intro, 50% to body (if long enough)
        if duration_seconds > 90:  # If video is significantly longer than intro
            intro_quota = max_frames // 2
            body_quota = max_frames - intro_quota
        else:
            intro_quota = max_frames
            body_quota = 0

        # Calculate indices
        frame_indices = []
        
        # Intro indices (spread across first 60s)
        if intro_quota > 0 and intro_frames_count > 0:
            intro_interval = max(1, intro_frames_count // intro_quota)
            for i in range(0, intro_frames_count, intro_interval):
                if len(frame_indices) < intro_quota:
                    frame_indices.append(i)
        
        # Body indices (spread across the rest)
        if body_quota > 0:
            body_start_frame = intro_frames_count
            body_frames_remaining = total_frames - body_start_frame
            if body_frames_remaining > 0:
                body_interval = max(1, body_frames_remaining // body_quota)
                for i in range(body_start_frame, total_frames, body_interval):
                    if len(frame_indices) < max_frames:
                        frame_indices.append(i)

        print(f"Hybrid Sampling Strategy:")
        print(f"  Total Video Frames: {total_frames}")
        print(f"  Intro Duration: {intro_duration:.1f}s (Allocated: {len([i for i in frame_indices if i < intro_frames_count])} frames)")
        print(f"  Body Duration: {max(0, duration_seconds - intro_duration):.1f}s (Allocated: {len([i for i in frame_indices if i >= intro_frames_count])} frames)")
        print(f"  Total planned extractions: {len(frame_indices)}")
        
        frame_paths = []
        saved_count = 0
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue
                
            frame_path = os.path.join(output_dir, f"frame_{saved_count:06d}.jpg")
            cv2.imwrite(frame_path, frame)
            frame_paths.append(frame_path)
            saved_count += 1
            
            if saved_count % 50 == 0:
                print(f"  Extracted {saved_count} frames...")
        
        cap.release()
        print(f"[OK] Extracted {saved_count} frames to: {output_dir}")
        return frame_paths
    
    def get_metadata(self, url):
        """
        Extract metadata from YouTube video
        
        Args:
            url: YouTube video URL
            
        Returns:
            Dictionary with video metadata
        """
        print(f"Extracting metadata from: {url}")
        
        try:
            # Use self.ydl_opts to get cookies, PO tokens, and remote_components
            metadata_opts = {**self.ydl_opts, 'skip_download': True, 'quiet': True}
            with yt_dlp.YoutubeDL(metadata_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                
                metadata = {
                    'video_id': info.get('id', ''),
                    'title': info.get('title', ''),
                    'description': info.get('description', ''),
                    'channel': info.get('channel', ''),
                    'channel_id': info.get('channel_id', ''),
                    'upload_date': info.get('upload_date', ''),
                    'duration': info.get('duration', 0),  # in seconds
                    'view_count': info.get('view_count', 0),
                    'like_count': info.get('like_count', 0),
                    'comment_count': info.get('comment_count', 0),
                    'tags': info.get('tags', []),
                    'categories': info.get('categories', []),
                    'thumbnail_url': info.get('thumbnail', ''),
                    'webpage_url': info.get('webpage_url', ''),
                    'extractor': info.get('extractor', ''),
                    'fps': info.get('fps', 0),
                    'resolution': info.get('resolution', ''),
                    'width': info.get('width', 0),
                    'height': info.get('height', 0)
                }
                
                print(f"✓ Metadata extracted successfully")
                print(f"  Title: {metadata['title']}")
                print(f"  Channel: {metadata['channel']}")
                print(f"  Duration: {metadata['duration']} seconds")
                
                return metadata
                
        except Exception as e:
            print(f"Error extracting metadata: {str(e)}")
            raise
    
    def get_transcript(self, video_id, languages=None):
        """
        Get video transcript/subtitles using yt-dlp (supports cookies/age-restricted)
        
        Args:
            video_id: YouTube video ID
            languages: List of English language codes to try (default: ['en', 'en-US', 'en-GB'])
            
        Returns:
            Transcript text or None if not available
        """
        if languages is None:
            languages = ['en', 'en-US', 'en-GB', 'en.*']
            
        print(f"🔍 Searching for English transcript for {video_id} using yt-dlp...")
        
        url = f"https://www.youtube.com/watch?v={video_id}"
        
        # Specific options for subtitle extraction
        sub_opts = self.ydl_opts.copy()
        
        # Unique output filename to avoid conflicts during concurrent requests
        import uuid
        import glob
        unique_id = str(uuid.uuid4())[:8]
        outtmpl = os.path.join(self.results_dir, f"{video_id}_{unique_id}.%(ext)s")
        
        sub_opts.update({
            'skip_download': True,      # Don't download the video
            'writesubtitles': True,     # Do write subtitles
            'writeautomaticsub': True,  # Do write auto-generated subtitles
            'subtitleslangs': languages,
            'subtitlesformat': 'json3', # Get JSON3 format which is easy to parse
            'outtmpl': outtmpl,
            'quiet': True,
            'no_warnings': True,
            'extract_flat': False
        })
        
        try:
            with yt_dlp.YoutubeDL(sub_opts) as ydl:
                # This will download the subtitle files to results_dir
                info = ydl.extract_info(url, download=True)
                
                # Check if any json3 files were written
                search_pattern = os.path.join(self.results_dir, f"{video_id}_{unique_id}.*.json3")
                sub_files = glob.glob(search_pattern)
                
                if not sub_files:
                    print(f"  ⚠ No English transcript available for {video_id}")
                    return None
                
                # Prioritize 'en' over 'en-US' etc by sort (not strictly necessary but good)
                sub_files.sort()
                sub_file = sub_files[0]
                
                try:
                    with open(sub_file, 'r', encoding='utf-8') as f:
                        sub_data = json.load(f)
                        
                    # Parse json3 format (events -> segs -> utf8)
                    text_parts = []
                    if 'events' in sub_data:
                        for event in sub_data['events']:
                            if 'segs' in event:
                                for seg in event['segs']:
                                    if 'utf8' in seg and seg['utf8'].strip():
                                        text_parts.append(seg['utf8'])
                                        
                    text = ' '.join(text_parts).replace('\n', ' ')
                    
                    # Clean up multiple spaces
                    import re
                    text = re.sub(r'\s+', ' ', text).strip()
                    
                    lang_found = sub_file.split('.')[-2]
                    print(f"  ✓ English transcript fetched with yt-dlp ({lang_found}): {len(text)} chars")
                    
                    # Cleanup files
                    for f in sub_files:
                        try:
                            os.remove(f)
                        except:
                            pass
                            
                    return text if text else None
                    
                except Exception as e:
                    print(f"  Error parsing subtitle file: {str(e)}")
                    # Cleanup on errors too
                    for f in sub_files:
                        try: os.remove(f)
                        except: pass
                    return None
                    
        except Exception as e:
            # yt-dlp often raises exceptions if something goes wrong, but we just want to return None
            # so the fallback kicks in gracefully
            print(f"  ⚠ Could not fetch transcript with yt-dlp (may not exist or restricted)")
            return None
    
    def cleanup_temp_files(self, video_id=None):
        """
        Clean up temporary files
        
        Args:
            video_id: Specific video ID to clean, or None for all
        """
        import shutil
        
        if video_id:
            # Clean specific video files
            for dir_path in [self.video_dir, self.frame_dir, self.thumbnail_dir]:
                target_dir = os.path.join(dir_path, video_id) if dir_path != self.thumbnail_dir else dir_path
                if os.path.exists(target_dir):
                    shutil.rmtree(target_dir)
                    print(f"Cleaned temp files for video: {video_id}")
        else:
            # Clean all temp files
            for dir_path in [self.video_dir, self.frame_dir, self.thumbnail_dir, self.results_dir]:
                if os.path.exists(dir_path):
                    shutil.rmtree(dir_path)
                    os.makedirs(dir_path)
            print("Cleaned all temp files")
    
    def process_video(self, url, frames_per_minute=None, max_frames=None):
        """
        Complete video processing pipeline
        
        Args:
            url: YouTube video URL
            frames_per_minute: Frames to extract per minute (None for all)
            max_frames: Maximum frames to extract
            
        Returns:
            Dictionary with all extracted data
        """
        print("="*70)
        print("PROCESSING VIDEO")
        print("="*70)
        
        video_id = self.get_video_id(url)
        if not video_id:
            raise ValueError(f"Invalid YouTube URL")
        
        # Download video
        video_path = self.download_video(url)
        
        # Extract thumbnail
        thumbnail_path = self.extract_thumbnail(url)
        
        # Get metadata
        metadata = self.get_metadata(url)
        
        # Extract frames based on frames_per_minute setting
        if frames_per_minute is not None:
            frame_paths = self.extract_frames(
                video_path, 
                frames_per_minute=frames_per_minute,
                max_frames=max_frames
            )
        elif max_frames is not None:
            # Use max_frames directly if frames_per_minute not specified
            frame_paths = self.extract_frames(video_path, max_frames=max_frames)
        else:
            # Extract all frames (default behavior for backward compatibility)
            frame_paths = self.extract_frames(video_path)
        
        # Get transcript (optional)
        transcript = self.get_transcript(video_id)
        
        result = {
            'video_id': video_id,
            'video_path': video_path,
            'thumbnail_path': thumbnail_path,
            'frame_paths': frame_paths,
            'metadata': metadata,
            'transcript': transcript
        }
        
        print("\n" + "="*70)
        print("PROCESSING COMPLETE")
        print("="*70)
        
        return result


def main():
    """
    Test the YouTube extractor
    """
    extractor = YouTubeMediaExtractor()
    
    # Example usage
    test_url = input("Enter YouTube URL: ")
    
    try:
        # Default behavior extracts all frames if no limiting args are provided
        result = extractor.process_video(test_url, frames_per_minute=10, max_frames=50)
        
        print("\nExtraction Summary:")
        print(f"  Video ID: {result['video_id']}")
        print(f"  Title: {result['metadata']['title']}")
        print(f"  Thumbnail: {result['thumbnail_path']}")
        print(f"  Frames extracted: {len(result['frame_paths'])}")
        
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
