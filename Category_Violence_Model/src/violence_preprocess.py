"""
Violence Frame Preprocessing Module
Handles preprocessing of video frames for violence detection model
"""

import cv2
import numpy as np
from pathlib import Path


class ViolenceFramePreprocessor:
    """
    Preprocess frames for violence detection model
    """
    
    def __init__(self, img_size=(224, 224)):
        """
        Initialize the preprocessor
        
        Args:
            img_size: Target image dimensions (width, height)
        """
        self.img_size = img_size
    
    def load_frame(self, frame_path):
        """
        Load and preprocess a single frame
        
        Args:
            frame_path: Path to frame image file
            
        Returns:
            Preprocessed frame array ready for model prediction
        """
        # Load image
        img = cv2.imread(str(frame_path))
        
        if img is None:
            raise FileNotFoundError(f"Could not load image: {frame_path}")
        
        # Resize to model input size
        img_resized = cv2.resize(img, self.img_size)
        
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        
        # Normalize pixel values to [0, 1]
        img_normalized = img_rgb.astype('float32') / 255.0
        
        return img_normalized
    
    def preprocess_batch(self, frame_paths, batch_size=32, show_progress=True):
        """
        Load and preprocess multiple frames in batches
        
        Args:
            frame_paths: List of frame paths
            batch_size: Batch size for processing
            show_progress: Show progress indicator
            
        Returns:
            Tuple of (preprocessed_frames_array, frame_paths_list)
        """
        print(f"Preprocessing {len(frame_paths)} frames...")
        
        all_frames = []
        valid_paths = []
        
        for i, frame_path in enumerate(frame_paths):
            try:
                frame = self.load_frame(frame_path)
                all_frames.append(frame)
                valid_paths.append(frame_path)
                
                if show_progress and (i + 1) % 500 == 0:
                    print(f"  Processed {i + 1}/{len(frame_paths)} frames")
                    
            except Exception as e:
                print(f"Warning: Could not process {frame_path}: {e}")
                continue
        
        if len(all_frames) == 0:
            raise ValueError("No valid frames were processed")
        
        # Stack into array
        frames_array = np.array(all_frames)
        
        print(f"✓ Preprocessed {len(frames_array)} frames successfully")
        print(f"  Frame shape: {frames_array.shape}")
        
        return frames_array, valid_paths
    
    def preprocess_from_video(self, video_path, max_frames=None):
        """
        Extract and preprocess frames directly from video file
        
        Args:
            video_path: Path to video file
            max_frames: Maximum number of frames to extract
            
        Returns:
            Tuple of (frames_array, frame_timestamps)
        """
        print(f"Extracting and preprocessing frames from: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if max_frames:
            total_frames = min(total_frames, max_frames)
        
        frames = []
        timestamps = []
        
        frame_count = 0
        while frame_count < total_frames:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Preprocess frame
            img_resized = cv2.resize(frame, self.img_size)
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            img_normalized = img_rgb.astype('float32') / 255.0
            
            frames.append(img_normalized)
            timestamps.append(frame_count / fps)  # Timestamp in seconds
            
            frame_count += 1
            
            if frame_count % 500 == 0:
                print(f"  Processed {frame_count} frames...")
        
        cap.release()
        
        frames_array = np.array(frames)
        
        print(f"✓ Extracted and preprocessed {len(frames_array)} frames")
        
        return frames_array, timestamps


def main():
    """
    Test the preprocessor
    """
    preprocessor = ViolenceFramePreprocessor()
    
    # Example usage
    test_frame_path = input("Enter path to test frame (or press Enter to skip): ")
    
    if test_frame_path and Path(test_frame_path).exists():
        frame = preprocessor.load_frame(test_frame_path)
        print(f"Frame shape: {frame.shape}")
        print(f"Frame dtype: {frame.dtype}")
        print(f"Value range: [{frame.min():.3f}, {frame.max():.3f}]")
    else:
        print("No test frame provided")


if __name__ == "__main__":
    main()
