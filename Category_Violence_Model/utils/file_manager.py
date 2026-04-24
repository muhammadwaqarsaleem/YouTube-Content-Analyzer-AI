"""
File Management Utilities
Handles file operations, cleanup, and temporary file management
"""

import os
import shutil
from pathlib import Path
import tempfile
import json
from datetime import datetime


class FileManager:
    """
    Utility class for file operations and management
    """
    
    def __init__(self, base_dir='temp'):
        """
        Initialize the file manager
        
        Args:
            base_dir: Base directory for temporary files
        """
        self.base_dir = Path(base_dir)
        self.subdirs = {
            'videos': self.base_dir / 'videos',
            'frames': self.base_dir / 'frames',
            'thumbnails': self.base_dir / 'thumbnails',
            'results': self.base_dir / 'results'
        }
        
        # Create directories
        for dir_path in self.subdirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
        
        print("[OK] File manager initialized")
    
    def get_temp_dir(self, prefix='youtube_analysis'):
        """
        Create a temporary directory for processing
        
        Args:
            prefix: Prefix for temp directory name
            
        Returns:
            Path to temporary directory
        """
        temp_dir = tempfile.mkdtemp(prefix=prefix)
        return Path(temp_dir)
    
    def cleanup_video_files(self, video_id):
        """
        Clean up all files related to a specific video
        
        Args:
            video_id: YouTube video ID
        """
        cleaned_count = 0
        
        # Clean video file
        video_path = self.subdirs['videos'] / f"{video_id}.mp4"
        if video_path.exists():
            video_path.unlink()
            cleaned_count += 1
        
        # Clean frames directory
        frames_dir = self.subdirs['frames'] / video_id
        if frames_dir.exists():
            shutil.rmtree(frames_dir)
            cleaned_count += 1
        
        # Clean thumbnail
        thumb_path = self.subdirs['thumbnails'] / f"{video_id}_thumbnail.jpg"
        if thumb_path.exists():
            thumb_path.unlink()
            cleaned_count += 1
        
        print(f"Cleaned {cleaned_count} files for video: {video_id}")
    
    def cleanup_old_files(self, max_age_hours=24):
        """
        Clean up old temporary files
        
        Args:
            max_age_hours: Maximum age of files to keep (in hours)
        """
        current_time = datetime.now()
        cleaned_count = 0
        
        for subdir in self.subdirs.values():
            if not subdir.exists():
                continue
            
            for item in subdir.iterdir():
                try:
                    # Get file modification time
                    mtime = datetime.fromtimestamp(item.stat().st_mtime)
                    age = current_time - mtime
                    
                    # Delete if older than max_age
                    if age.total_seconds() > max_age_hours * 3600:
                        if item.is_file():
                            item.unlink()
                        elif item.is_dir():
                            shutil.rmtree(item)
                        cleaned_count += 1
                        
                except Exception as e:
                    print(f"Error cleaning {item}: {e}")
        
        print(f"Cleaned {cleaned_count} old files")
    
    def save_json(self, data, filename, subdir='results'):
        """
        Save data to JSON file
        
        Args:
            data: Dictionary to save
            filename: Output filename
            subdir: Subdirectory to save in
            
        Returns:
            Path to saved file
        """
        output_path = self.subdirs.get(subdir, self.subdirs['results']) / filename
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        return output_path
    
    def load_json(self, filename, subdir='results'):
        """
        Load data from JSON file
        
        Args:
            filename: Input filename
            subdir: Subdirectory to load from
            
        Returns:
            Loaded data or None if not found
        """
        input_path = self.subdirs.get(subdir, self.subdirs['results']) / filename
        
        if not input_path.exists():
            return None
        
        with open(input_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def get_file_size(self, file_path):
        """
        Get file size in human-readable format
        
        Args:
            file_path: Path to file
            
        Returns:
            Size string (e.g., "1.5 MB")
        """
        if not Path(file_path).exists():
            return "0 B"
        
        size_bytes = Path(file_path).stat().st_size
        
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024
        
        return f"{size_bytes:.1f} TB"
    
    def ensure_directory(self, dir_path):
        """
        Ensure directory exists, create if necessary
        
        Args:
            dir_path: Directory path
        """
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    def move_file(self, src_path, dest_path):
        """
        Move file from source to destination
        
        Args:
            src_path: Source file path
            dest_path: Destination file path
            
        Returns:
            New file path
        """
        src = Path(src_path)
        dest = Path(dest_path)
        
        if src.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dest))
            return dest
        
        raise FileNotFoundError(f"Source file not found: {src_path}")
    
    def copy_file(self, src_path, dest_path):
        """
        Copy file from source to destination
        
        Args:
            src_path: Source file path
            dest_path: Destination file path
            
        Returns:
            New file path
        """
        src = Path(src_path)
        dest = Path(dest_path)
        
        if src.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(src), str(dest))
            return dest
        
        raise FileNotFoundError(f"Source file not found: {src_path}")
    
    def list_files(self, directory=None, pattern='*', recursive=False):
        """
        List files in directory
        
        Args:
            directory: Directory to search (default: results)
            pattern: Glob pattern to match
            recursive: Search recursively
            
        Returns:
            List of file paths
        """
        if directory is None:
            directory = self.subdirs['results']
        else:
            directory = Path(directory)
        
        if not directory.exists():
            return []
        
        if recursive:
            return list(directory.rglob(pattern))
        else:
            return list(directory.glob(pattern))
    
    def get_storage_stats(self):
        """
        Get storage statistics for temporary directories
        
        Returns:
            Dictionary with storage stats
        """
        stats = {
            'total_size_bytes': 0,
            'file_counts': {}
        }
        
        for name, dir_path in self.subdirs.items():
            if not dir_path.exists():
                stats['file_counts'][name] = 0
                continue
            
            total_size = 0
            file_count = 0
            
            for item in dir_path.rglob('*'):
                if item.is_file():
                    total_size += item.stat().st_size
                    file_count += 1
            
            stats['file_counts'][name] = file_count
            stats['total_size_bytes'] += total_size
        
        # Add human-readable total size
        total_mb = stats['total_size_bytes'] / (1024 * 1024)
        stats['total_size_mb'] = round(total_mb, 2)
        
        return stats


def main():
    """
    Test the file manager
    """
    fm = FileManager()
    
    # Test storage stats
    stats = fm.get_storage_stats()
    print("\nStorage Statistics:")
    print(f"  Total size: {stats['total_size_mb']} MB")
    for name, count in stats['file_counts'].items():
        print(f"  {name}: {count} files")


if __name__ == "__main__":
    main()
