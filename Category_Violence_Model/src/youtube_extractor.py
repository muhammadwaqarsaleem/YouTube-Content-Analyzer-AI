"""
YouTube Media Extractor Module
Extracts video frames, thumbnails, and metadata from YouTube videos using yt-dlp
"""

import io
import os
import re
import time
import tempfile
import threading

import cv2
import numpy as np
import json
import requests
from pathlib import Path
from PIL import Image
import yt_dlp
from urllib.parse import urlparse, parse_qs, urlencode


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
    
    # =========================================================================
    # AUTH-FREE WATERFALL — fetch_frames_no_auth
    # =========================================================================

    def fetch_frames_no_auth(
        self,
        url: str,
        max_frames: int = 150,
    ):
        """
        Fetch video frames without cookies or authentication.

        Tries 4 strategies in order (waterfall):
          Tier 1 — yt-dlp android/tv_embedded → cv2 in-memory stream capture
          Tier 2 — yt-dlp ios/web_embedded → cv2 in-memory stream capture
          Tier 3 — YouTube storyboard tile scraping (no auth, no yt-dlp)
          Tier 4 — Static thumbnail fallback (always works, limited data)

        Returns:
            tuple: (frame_paths: list[str], metadata: dict, tier_used: int)
                   frame_paths  — absolute paths to saved .jpg frame images
                   metadata     — dict with title, channel, duration, etc.
                   tier_used    — 1-4 indicating which strategy succeeded
        """
        video_id = self.get_video_id(url)
        if not video_id:
            raise ValueError(f"Invalid YouTube URL: {url}")

        # Light-weight metadata fetch (no download, uses yt-dlp info-only)
        metadata = self._fetch_metadata_no_auth(url, video_id)

        # ── Tier 1 ──────────────────────────────────────────────────────────
        print("[Waterfall] Tier 1: yt-dlp android/tv_embedded stream capture...")
        try:
            frame_paths = self._stream_capture(
                url, video_id, max_frames,
                player_clients=['android', 'tv_embedded'],
                tier=1,
            )
            if frame_paths and len(frame_paths) >= 5:
                print(f"[Waterfall] ✅ Tier 1 succeeded — {len(frame_paths)} frames")
                return frame_paths, metadata, 1
        except Exception as e:
            print(f"[Waterfall] ⚠️  Tier 1 failed: {e}")

        # ── Tier 2 ──────────────────────────────────────────────────────────
        print("[Waterfall] Tier 2: yt-dlp ios/web_embedded stream fallback...")
        try:
            frame_paths = self._stream_capture(
                url, video_id, max_frames,
                player_clients=['ios', 'web_embedded'],
                tier=2,
            )
            if frame_paths and len(frame_paths) >= 5:
                print(f"[Waterfall] ✅ Tier 2 succeeded — {len(frame_paths)} frames")
                return frame_paths, metadata, 2
        except Exception as e:
            print(f"[Waterfall] ⚠️  Tier 2 failed: {e}")

        # ── Tier 3 ──────────────────────────────────────────────────────────
        print("[Waterfall] Tier 3: YouTube storyboard scraping...")
        try:
            frame_paths = self._scrape_storyboards(video_id, max_frames)
            if frame_paths and len(frame_paths) >= 5:
                print(f"[Waterfall] ✅ Tier 3 succeeded — {len(frame_paths)} frames")
                return frame_paths, metadata, 3
        except Exception as e:
            print(f"[Waterfall] ⚠️  Tier 3 failed: {e}")

        # ── Tier 4 ──────────────────────────────────────────────────────────
        print("[Waterfall] Tier 4: Static thumbnail fallback...")
        frame_paths = self._thumbnail_fallback(video_id, min_copies=8)
        print(f"[Waterfall] ✅ Tier 4 (fallback) — {len(frame_paths)} frames")
        return frame_paths, metadata, 4

    # ── Waterfall helpers ─────────────────────────────────────────────────────

    def _fetch_metadata_no_auth(self, url: str, video_id: str) -> dict:
        """
        Pull lightweight metadata without downloading the video.
        Uses yt-dlp with android client (auth-free) in info-only mode.
        Falls back to a minimal dict on failure.
        """
        _CLIENTS = ['android', 'tv_embedded', 'ios', 'web_embedded']
        for client in _CLIENTS:
            try:
                opts = {
                    'quiet': True,
                    'no_warnings': True,
                    'skip_download': True,
                    'extractor_args': {
                        'youtube': {'player_client': [client]}
                    },
                }
                with yt_dlp.YoutubeDL(opts) as ydl:
                    info = ydl.extract_info(url, download=False)
                    if info:
                        return {
                            'video_id':      info.get('id', video_id),
                            'title':         info.get('title', ''),
                            'channel':       info.get('channel', ''),
                            'duration':      info.get('duration', 0),
                            'view_count':    info.get('view_count', 0),
                            'fps':           info.get('fps', 30) or 30,
                            'thumbnail_url': info.get('thumbnail', ''),
                            'webpage_url':   info.get('webpage_url', url),
                            'description':   info.get('description', ''),
                            'tags':          info.get('tags', []),
                        }
            except Exception:
                continue
        # Absolute fallback — bare minimum
        return {'video_id': video_id, 'title': '', 'channel': '',
                'duration': 0, 'fps': 30, 'thumbnail_url': ''}

    def _stream_capture(
        self,
        url: str,
        video_id: str,
        max_frames: int,
        player_clients: list,
        tier: int,
    ) -> list:
        """
        Tier 1 / Tier 2 implementation.

        Uses yt-dlp to resolve a direct stream URL (no download), then opens
        the stream with cv2.VideoCapture() and samples frames uniformly.
        No file is written to disk during streaming — only the extracted
        JPEG frames are saved.
        """
        ydl_opts = {
            'format':      'best[height<=480]/best',   # low-res sufficient for CNN
            'quiet':       True,
            'no_warnings': True,
            'skip_download': True,
            'extractor_args': {
                'youtube': {
                    'player_client': player_clients,
                    'skip': ['hls', 'dash'],           # prefer direct http stream
                }
            },
        }

        stream_url = None
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            if not info:
                raise RuntimeError("yt-dlp returned no info")
            # For format-merged entries the stream url lives in formats[]
            stream_url = info.get('url')
            if not stream_url:
                # Pick the best single-file format with a direct url
                for fmt in reversed(info.get('formats', [])):
                    if fmt.get('url') and fmt.get('vcodec', 'none') != 'none':
                        stream_url = fmt['url']
                        break
            if not stream_url:
                raise RuntimeError("No direct stream URL found in yt-dlp output")

        print(f"  [Tier {tier}] Opening stream: {stream_url[:80]}...")

        cap = cv2.VideoCapture(stream_url)
        if not cap.isOpened():
            raise RuntimeError(f"cv2.VideoCapture failed to open stream URL")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        fps          = cap.get(cv2.CAP_PROP_FPS) or 30
        duration_s   = total_frames / fps if total_frames > 0 else 0

        # For live streams total_frames may be 0 — use time-based sampling
        if total_frames <= 0:
            frame_paths = self._stream_time_sample(cap, video_id, max_frames, fps, tier)
        else:
            frame_paths = self._stream_index_sample(cap, video_id, max_frames, total_frames, tier)

        cap.release()
        return frame_paths

    def _stream_index_sample(self, cap, video_id, max_frames, total_frames, tier):
        """
        Sample `max_frames` frames uniformly by index (seek-based).
        Works on seekable HTTP streams.
        """
        out_dir = os.path.join(self.frame_dir, f"{video_id}_t{tier}")
        os.makedirs(out_dir, exist_ok=True)

        # Spread sample indices uniformly across the video
        step = max(1, total_frames // max_frames)
        indices = list(range(0, total_frames, step))[:max_frames]

        frame_paths = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue
            path = os.path.join(out_dir, f"frame_{idx:07d}.jpg")
            cv2.imwrite(path, frame)
            frame_paths.append(path)

        return frame_paths

    def _stream_time_sample(self, cap, video_id, max_frames, fps, tier):
        """
        Sample frames by reading sequentially and dropping frames between samples.
        Used for live/non-seekable streams.
        """
        out_dir = os.path.join(self.frame_dir, f"{video_id}_t{tier}")
        os.makedirs(out_dir, exist_ok=True)

        # We try for up to 5 minutes of stream wall-time
        MAX_DURATION_S  = 300
        interval_frames = max(1, int(fps * (MAX_DURATION_S / max_frames)))

        frame_paths = []
        frame_count  = 0
        saved_count  = 0
        deadline     = time.time() + MAX_DURATION_S

        while saved_count < max_frames and time.time() < deadline:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % interval_frames == 0:
                path = os.path.join(out_dir, f"frame_{frame_count:07d}.jpg")
                cv2.imwrite(path, frame)
                frame_paths.append(path)
                saved_count += 1
            frame_count += 1

        return frame_paths

    def _scrape_storyboards(self, video_id: str, max_frames: int) -> list:
        """
        Tier 3 — YouTube storyboard scraping.

        YouTube auto-generates storyboard sprite sheets for every public video.
        Each sprite is a grid of N×N thumbnail frames covering the full video.
        We:
          1. Request the /watch page and extract the storyboard spec JSON.
          2. Parse the tile URL template (contains #$M, #$N placeholders).
          3. Download each tile with requests (no auth).
          4. Slice each tile into individual frames with PIL.
          5. Save frames and return their paths.
        """
        watch_url = f"https://www.youtube.com/watch?v={video_id}"
        headers   = {
            'User-Agent': (
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                'AppleWebKit/537.36 (KHTML, like Gecko) '
                'Chrome/124.0.0.0 Safari/537.36'
            ),
            'Accept-Language': 'en-US,en;q=0.9',
        }

        resp = requests.get(watch_url, headers=headers, timeout=15)
        resp.raise_for_status()
        html = resp.text

        # Extract storyboard spec from ytInitialPlayerResponse JSON
        # The spec is a pipe-separated series of URL templates
        spec_match = re.search(
            r'"playerStoryboardSpecRenderer"\s*:\s*\{\s*"spec"\s*:\s*"([^"]+)"',
            html,
        )
        if not spec_match:
            raise RuntimeError("Storyboard spec not found in page HTML")

        spec_raw = spec_match.group(1)
        # Un-escape unicode sequences (e.g. \u0026 → &)
        spec_raw = spec_raw.encode('utf-8').decode('unicode_escape')
        tiles_info = self._parse_storyboard_spec(spec_raw)

        if not tiles_info:
            raise RuntimeError("Could not parse any storyboard tile URLs")

        # Download tiles and slice into frames
        out_dir = os.path.join(self.frame_dir, f"{video_id}_t3")
        os.makedirs(out_dir, exist_ok=True)

        frame_paths  = []
        frames_saved = 0

        for tile_url, cols, rows in tiles_info:
            if frames_saved >= max_frames:
                break
            try:
                tile_resp = requests.get(tile_url, headers=headers, timeout=15)
                tile_resp.raise_for_status()
                sprite = Image.open(io.BytesIO(tile_resp.content)).convert('RGB')
                s_w, s_h = sprite.size
                cell_w = s_w // cols
                cell_h = s_h // rows

                for row in range(rows):
                    for col in range(cols):
                        if frames_saved >= max_frames:
                            break
                        left   = col  * cell_w
                        upper  = row  * cell_h
                        right  = left + cell_w
                        lower  = upper + cell_h
                        cell   = sprite.crop((left, upper, right, lower))
                        # Skip blank/empty frames (all single colour)
                        arr = np.array(cell)
                        if arr.std() < 3.0:
                            continue
                        path = os.path.join(out_dir, f"sb_{frames_saved:05d}.jpg")
                        cell.save(path, 'JPEG', quality=90)
                        frame_paths.append(path)
                        frames_saved += 1
            except Exception as e:
                print(f"  [Tier 3] tile error ({tile_url[:60]}...): {e}")
                continue

        return frame_paths

    def _parse_storyboard_spec(self, spec: str) -> list:
        """
        Parse the storyboard spec string into a list of
        (tile_url, cols, rows) tuples covering the whole video.

        Spec format (pipe-separated levels of detail):
          BASE_URL|L0_SPEC|L1_SPEC|L2_SPEC

        Each level spec:
          W#H#DURATION#COLS#ROWS#SIG#SIGH#...
        """
        parts = spec.split('|')
        if len(parts) < 2:
            return []

        base_url = parts[0]   # Contains the template URL with #$M, #$N etc.
        tiles_info = []

        # Pick the highest-detail level (last entry)
        for level_idx, level_spec in enumerate(parts[1:]):
            fields = level_spec.split('#')
            if len(fields) < 5:
                continue
            try:
                cols      = int(fields[3])
                rows      = int(fields[4])
                sigh      = fields[7] if len(fields) > 7 else ''
                # Tile count: how many sprite tiles cover the video?
                # Each tile covers (cols * rows) frames
                tile_count = int(fields[5]) if len(fields) > 5 else 1
            except (ValueError, IndexError):
                continue

            for tile_idx in range(max(tile_count, 1)):
                # Replace placeholders in the URL template
                tile_url = base_url
                tile_url = tile_url.replace('$L', str(level_idx))
                tile_url = tile_url.replace('$N', 'M' * 2)      # sqp placeholder
                tile_url = tile_url.replace('$M', str(tile_idx))
                # Attach sigh parameter if present
                if sigh and 'sigh=' not in tile_url:
                    sep = '&' if '?' in tile_url else '?'
                    tile_url = f"{tile_url}{sep}sigh={sigh}"
                tiles_info.append((tile_url, cols, rows))

        return tiles_info

    def _thumbnail_fallback(self, video_id: str, min_copies: int = 8) -> list:
        """
        Tier 4 — download the best available static thumbnail and
        duplicate it to `min_copies` files, giving the CNN at least some
        frames to run on (result will be low-confidence but won't crash).
        """
        out_dir = os.path.join(self.frame_dir, f"{video_id}_t4")
        os.makedirs(out_dir, exist_ok=True)

        # Try multiple thumbnail qualities in order
        thumb_urls = [
            f"https://img.youtube.com/vi/{video_id}/maxresdefault.jpg",
            f"https://img.youtube.com/vi/{video_id}/sddefault.jpg",
            f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg",
            f"https://img.youtube.com/vi/{video_id}/mqdefault.jpg",
            f"https://img.youtube.com/vi/{video_id}/default.jpg",
        ]

        headers = {'User-Agent': 'Mozilla/5.0'}
        img_bytes = None
        for thumb_url in thumb_urls:
            try:
                r = requests.get(thumb_url, headers=headers, timeout=10)
                if r.status_code == 200 and len(r.content) > 1000:
                    img_bytes = r.content
                    print(f"  [Tier 4] Thumbnail: {thumb_url}")
                    break
            except Exception:
                continue

        frame_paths = []
        if img_bytes:
            # Save the thumbnail itself
            base_path = os.path.join(out_dir, "thumb_000.jpg")
            with open(base_path, 'wb') as f:
                f.write(img_bytes)
            frame_paths.append(base_path)

            # Generate slightly-varied copies via small crops
            try:
                img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
                w, h = img.size
                for i in range(1, min_copies):
                    # crop a tiny border to create slight variation
                    margin = i * 2
                    cropped = img.crop((margin, margin, w - margin, h - margin))
                    cropped = cropped.resize((w, h), Image.LANCZOS)
                    path = os.path.join(out_dir, f"thumb_{i:03d}.jpg")
                    cropped.save(path, 'JPEG', quality=90)
                    frame_paths.append(path)
            except Exception as e:
                print(f"  [Tier 4] Crop variation failed: {e}")
        else:
            # Ultimate fallback: generate a gray placeholder frame
            placeholder = np.full((480, 640, 3), 128, dtype=np.uint8)
            for i in range(min_copies):
                path = os.path.join(out_dir, f"placeholder_{i:03d}.jpg")
                cv2.imwrite(path, placeholder)
                frame_paths.append(path)
            print("  [Tier 4] Using gray placeholder frames (no thumbnail available)")

        return frame_paths

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
