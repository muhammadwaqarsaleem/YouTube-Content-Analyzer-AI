# Configuration Template for DatasetAnalyzer
# ==========================================
# Copy this file to 'config.py' and customize as needed

# File paths
CSV_FILE_PATH = "youtube_data.csv"
OUTPUT_DIR = "./eda_outputs"

# Analysis settings
LANGUAGE_SAMPLE_SIZE = 100  # Number of samples for language detection
RANDOM_SEED = 42  # For reproducibility

# Missing value placeholders
# Add any custom placeholders your dataset uses
MISSING_PLACEHOLDERS = [
    'none', 'null', 'n/a', 'na', 'nan', 'missing', '',
    'N/A', 'NULL', 'None', 'NONE',
    '-', '--', 'undefined', 'not available'
]

# Token analysis thresholds
WORD_COUNT_THRESHOLD = 500  # For long text detection
TOKEN_COUNT_THRESHOLD = 512  # BERT token limit

# Text column detection
MIN_TEXT_LENGTH = 20  # Minimum average length to be considered "text" vs categorical

# Visualization settings
VISUALIZATION_DPI = 300  # High quality for publications
FIGURE_SIZE = (12, 8)  # Default figure size

# Performance settings
CHUNK_SIZE = 10000  # For processing very large files in chunks (optional)
USE_PROGRESS_BAR = True  # Show tqdm progress bars

# Language detection
LANGDETECT_ENABLED = True  # Set to False to skip language detection

# Advanced options
DETECT_ENCODING = True  # Auto-detect CSV encoding
LOW_MEMORY = False  # Set to True for files > 1GB

# Column name patterns (case-insensitive)
# Customize these to match your dataset's column names
TEXT_COLUMN_PATTERNS = [
    'transcript', 'description', 'title', 'comment', 
    'text', 'content', 'caption', 'subtitle'
]

ID_COLUMN_PATTERNS = [
    'id', 'video_id', 'url', 'link', 'video_url'
]

LABEL_COLUMN_PATTERNS = [
    'label', 'class', 'category', 'age_bracket', 
    'rating', 'classification'
]

# Report customization
INCLUDE_COLUMN_LIST = True  # Include full column list in report
INCLUDE_SAMPLE_DATA = False  # Include sample rows in report (not recommended for sensitive data)
MAX_SAMPLE_ROWS = 5  # Number of sample rows if enabled

# Output formats
GENERATE_TEXT_REPORT = True
GENERATE_VISUALIZATIONS = True
GENERATE_JSON_METRICS = False  # Export metrics as JSON

# Percentiles to calculate for text length analysis
TEXT_LENGTH_PERCENTILES = [0.50, 0.75, 0.90, 0.95, 0.99]

# Duplicate detection
CHECK_EXACT_DUPLICATES = True
CHECK_ID_DUPLICATES = True
CHECK_NEAR_DUPLICATES = False  # Computationally expensive, use for small datasets

# Email notification (optional - requires smtplib configuration)
SEND_EMAIL_NOTIFICATION = False
EMAIL_RECIPIENT = "your-email@university.edu"
EMAIL_SUBJECT = "EDA Analysis Complete"
