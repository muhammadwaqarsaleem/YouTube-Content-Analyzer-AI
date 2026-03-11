"""
Video Processing Utilities
Helper functions for video operations
"""

import cv2
import numpy as np
from pathlib import Path


def get_video_info(video_path):
    """
    Get basic information about a video file
    
    Args:
        video_path: Path to video file
        
    Returns:
        Dictionary with video information
    """
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps if fps > 0 else 0
    
    cap.release()
    
    return {
        'fps': round(fps, 2),
        'total_frames': total_frames,
        'width': width,
        'height': height,
        'duration_seconds': round(duration, 2),
        'resolution': f"{width}x{height}"
    }


def estimate_processing_time(video_path, processing_speed_fps=30):
    """
    Estimate how long it will take to process a video
    
    Args:
        video_path: Path to video file
        processing_speed_fps: Estimated frames per second processing speed
        
    Returns:
        Estimated processing time in seconds
    """
    info = get_video_info(video_path)
    total_frames = info['total_frames']
    
    estimated_time = total_frames / processing_speed_fps
    
    return round(estimated_time, 2)


def validate_video_file(video_path, min_duration=1, max_duration=7200):
    """
    Validate that a video file is suitable for processing
    
    Args:
        video_path: Path to video file
        min_duration: Minimum duration in seconds
        max_duration: Maximum duration in seconds
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not Path(video_path).exists():
        return False, f"Video file not found: {video_path}"
    
    try:
        info = get_video_info(video_path)
        
        if info['duration_seconds'] < min_duration:
            return False, f"Video too short: {info['duration_seconds']}s (min: {min_duration}s)"
        
        if info['duration_seconds'] > max_duration:
            return False, f"Video too long: {info['duration_seconds']}s (max: {max_duration}s)"
        
        if info['total_frames'] == 0:
            return False, "Video has no frames"
        
        return True, None
        
    except Exception as e:
        return False, f"Error validating video: {str(e)}"


def create_video_thumbnail(video_path, output_path=None, frame_number=0):
    """
    Create thumbnail from specific frame of video
    
    Args:
        video_path: Path to video file
        output_path: Output path for thumbnail
        frame_number: Frame number to extract
        
    Returns:
        Path to created thumbnail
    """
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")
    
    # Set to specific frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        raise ValueError(f"Could not read frame {frame_number}")
    
    if output_path is None:
        output_path = str(Path(video_path).with_name(f"{Path(video_path).stem}_thumb.jpg"))
    
    cv2.imwrite(output_path, frame)
    
    return output_path


def count_extractable_frames(video_path):
    """
    Count total number of extractable frames from video
    
    Args:
        video_path: Path to video file
        
    Returns:
        Number of frames
    """
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    return total_frames


def calculate_frame_timestamps(video_path, frame_indices=None):
    """
    Calculate timestamps for given frame indices
    
    Args:
        video_path: Path to video file
        frame_indices: List of frame indices (None for all frames)
        
    Returns:
        List of timestamps in seconds
    """
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    if frame_indices is None:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = range(total_frames)
    
    timestamps = [idx / fps for idx in frame_indices]
    
    cap.release()
    
    return timestamps


def format_timestamp(seconds):
    """
    Format timestamp in seconds to HH:MM:SS.mmm
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted timestamp string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:05.2f}"
    else:
        return f"{minutes:02d}:{secs:05.2f}"


def parse_timestamp(timestamp_str):
    """
    Parse timestamp string to seconds
    
    Args:
        timestamp_str: Timestamp in format HH:MM:SS or MM:SS
        
    Returns:
        Time in seconds
    """
    parts = timestamp_str.split(':')
    
    if len(parts) == 2:
        minutes, seconds = map(float, parts)
        return minutes * 60 + seconds
    elif len(parts) == 3:
        hours, minutes, seconds = map(float, parts)
        return hours * 3600 + minutes * 60 + seconds
    else:
        raise ValueError(f"Invalid timestamp format: {timestamp_str}")


def main():
    """
    Test video utilities
    """
    test_video = input("Enter video path (or press Enter to skip): ")
    
    if test_video and Path(test_video).exists():
        print("\nVideo Information:")
        info = get_video_info(test_video)
        for key, value in info.items():
            print(f"  {key}: {value}")
        
        print(f"\nEstimated processing time at 30 FPS: {estimate_processing_time(test_video)} seconds")
        
        is_valid, error = validate_video_file(test_video)
        print(f"\nValidation: {'✓ Valid' if is_valid else f'✗ Invalid - {error}'}")
    else:
        print("No test video provided")


if __name__ == "__main__":
    main()
