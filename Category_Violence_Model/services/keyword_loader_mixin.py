"""
Mixin for loading keywords from external files
"""

from pathlib import Path
from typing import List


class KeywordLoaderMixin:
    """Mixin to load keywords from wordlist files"""
    
    def __init__(self, *args, **kwargs):
        # Initialize mixin attributes
        self._keyword_cache = None
        self._keyword_file_path = None
        
        # Call parent class init with all arguments
        # This allows the base class (CategoryOverrideDetector) to initialize
        super().__init__(*args, **kwargs)
    
    def load_keywords_from_file(self, filepath: str) -> List[str]:
        """
        Load keywords from external file
        
        Args:
            filepath: Path to keyword file relative to project root
            
        Returns:
            List of keywords
        """
        keywords = []
        
        # Try multiple paths
        possible_paths = [
            Path(filepath),
            Path('wordlists') / filepath,
            Path(__file__).parent.parent / 'wordlists' / filepath,
            Path.cwd() / 'wordlists' / filepath,
        ]
        
        for path in possible_paths:
            if path.exists():
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        for line in f:
                            line = line.strip()
                            if line and not line.startswith('#'):
                                keywords.append(line.lower())
                    
                    if keywords:
                        print(f"Loaded {len(keywords)} keywords from {path}")
                        return keywords
                except Exception as e:
                    print(f"Error loading {path}: {e}")
        
        # Fallback to empty list
        print(f"Warning: Could not load keywords from {filepath}, using empty list")
        return []
    
    def get_cached_keywords(self, filepath: str) -> List[str]:
        """Get keywords with caching to avoid repeated file reads"""
        if self._keyword_cache is None or self._keyword_file_path != filepath:
            self._keyword_cache = self.load_keywords_from_file(filepath)
            self._keyword_file_path = filepath
        
        return self._keyword_cache
