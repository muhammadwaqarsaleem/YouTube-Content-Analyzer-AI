# Troubleshooting Guide

## Common Issues and Solutions

### 1. File Not Found Error

**Error:**
```
❌ Error: File not found at youtube_data.csv
```

**Solutions:**
- Check that your CSV file is in the same directory as the script
- Update `CSV_FILE_PATH` variable to the correct path
- Use absolute path: `CSV_FILE_PATH = "/full/path/to/youtube_data.csv"`

### 2. Memory Error on Large Files

**Error:**
```
MemoryError: Unable to allocate array
```

**Solutions:**
1. Close other applications to free up RAM
2. Use chunked processing (modify script to read in chunks)
3. For files > 1GB, consider:
   ```python
   # Read only necessary columns
   columns_to_use = ['transcript', 'title', 'age_label']
   df = pd.read_csv(csv_path, usecols=columns_to_use)
   ```
4. Use a machine with more RAM or cloud instance

### 3. langdetect Not Installed

**Warning:**
```
⚠️ Warning: 'langdetect' not installed. Language detection will be skipped.
```

**Solution:**
```bash
pip install langdetect
```

If installation fails:
```bash
# Try with user flag
pip install --user langdetect

# Or use conda
conda install -c conda-forge langdetect
```

**Alternative:** Language detection will be skipped automatically - not critical for analysis.

### 4. Encoding Issues

**Error:**
```
UnicodeDecodeError: 'utf-8' codec can't decode byte
```

**Solutions:**
1. Try different encodings:
   ```python
   df = pd.read_csv(csv_path, encoding='latin-1')
   # or
   df = pd.read_csv(csv_path, encoding='cp1252')
   ```

2. Auto-detect encoding:
   ```python
   import chardet
   
   with open(csv_path, 'rb') as f:
       result = chardet.detect(f.read(100000))
       encoding = result['encoding']
   
   df = pd.read_csv(csv_path, encoding=encoding)
   ```

3. Add to script before `pd.read_csv()`:
   ```python
   # Add encoding parameter
   self.df = pd.read_csv(self.csv_path, encoding='utf-8', errors='ignore')
   ```

### 5. Script Appears Frozen

**Issue:**
Script shows no progress for long time on large datasets.

**Solutions:**
1. **This is normal** for large files (>500MB) during text analysis
2. Progress bars (tqdm) should show - if not, ensure tqdm is installed:
   ```bash
   pip install tqdm
   ```
3. Be patient - text analysis can take 5-10 minutes for 16K+ rows
4. Check CPU usage in Task Manager - if high, script is working

### 6. Plots Not Generating

**Issue:**
No PNG files in output directory.

**Solutions:**
1. Check matplotlib installation:
   ```bash
   pip install matplotlib seaborn
   ```

2. If on remote server without display:
   ```python
   # Add at top of script
   import matplotlib
   matplotlib.use('Agg')  # Non-interactive backend
   ```

3. Check write permissions for output directory

### 7. Import Errors

**Error:**
```
ModuleNotFoundError: No module named 'pandas'
```

**Solution:**
```bash
# Install all dependencies at once
pip install -r requirements.txt

# Or individually
pip install pandas numpy matplotlib seaborn tqdm
```

**For conda users:**
```bash
conda install pandas numpy matplotlib seaborn tqdm
pip install langdetect  # langdetect not in conda default channel
```

### 8. "No Text Columns Found"

**Issue:**
Script reports no text columns detected.

**Causes:**
- All text columns are very short (< 20 chars average)
- Columns detected as numeric or datetime

**Solutions:**
1. Check your data manually:
   ```python
   import pandas as pd
   df = pd.read_csv('youtube_data.csv')
   print(df.dtypes)
   print(df.head())
   ```

2. Adjust text detection threshold in script:
   ```python
   # In _identify_text_columns() method, change:
   if avg_length > 20:  # Change this to lower value, e.g., 10
   ```

### 9. Language Detection Takes Too Long

**Issue:**
Language detection is slow on large samples.

**Solutions:**
1. Reduce sample size:
   ```python
   analyzer.analyze_language_distribution(sample_size=50)  # Instead of 100
   ```

2. Skip language detection entirely:
   ```python
   # Comment out the line in run_complete_analysis():
   # self.analyze_language_distribution(sample_size=100)
   ```

### 10. Duplicate Video IDs Not Detected

**Issue:**
Script doesn't find duplicate IDs even though they exist.

**Causes:**
- Column name doesn't contain 'id' or 'url'
- IDs are numeric and treated differently

**Solution:**
Manually specify ID columns in script:
```python
# In analyze_structure() method, replace:
id_columns = [col for col in self.df.columns if 'id' in col.lower() or 'url' in col.lower()]

# With your specific column name:
id_columns = ['video_id']  # Or whatever your ID column is called
```

### 11. Permission Denied Error

**Error:**
```
PermissionError: [Errno 13] Permission denied: 'eda_outputs'
```

**Solutions:**
1. Run with admin/sudo (not recommended)
2. Change output directory to your home folder:
   ```python
   OUTPUT_DIR = "~/eda_outputs"  # Unix/Mac
   OUTPUT_DIR = "C:/Users/YourName/eda_outputs"  # Windows
   ```
3. Check folder isn't open in another program

### 12. Mixed Type Warning

**Warning:**
```
DtypeWarning: Columns have mixed types
```

**Solution:**
Add `dtype` parameter (not critical, can be ignored):
```python
df = pd.read_csv(csv_path, dtype={'column_name': str})
```

Or suppress warning:
```python
df = pd.read_csv(csv_path, low_memory=False)
```

## Performance Optimization Tips

### For Very Large Files (> 1GB)

1. **Use chunking:**
```python
chunk_size = 10000
chunks = []
for chunk in pd.read_csv(csv_path, chunksize=chunk_size):
    # Process chunk
    chunks.append(chunk)
df = pd.concat(chunks)
```

2. **Reduce data types:**
```python
# Convert object columns to category
for col in df.select_dtypes(include=['object']).columns:
    if df[col].nunique() < 0.5 * len(df):
        df[col] = df[col].astype('category')
```

3. **Use sampling for initial exploration:**
```python
# Analyze random sample first
sample_df = df.sample(n=5000, random_state=42)
```

## Getting Help

If you encounter issues not covered here:

1. Check the error message carefully
2. Verify all dependencies are installed: `pip list`
3. Try with a small sample of data first
4. Check GitHub issues: [Your Repo URL]
5. Contact the team lead

## System Requirements

**Minimum:**
- 4GB RAM
- Python 3.8+
- 1GB free disk space

**Recommended for 500MB+ files:**
- 8GB+ RAM
- Python 3.9+
- SSD storage
- Multi-core processor

## Quick Health Check

Run this to verify your setup:

```python
# test_setup.py
import sys
print(f"Python version: {sys.version}")

try:
    import pandas as pd
    print(f"✅ pandas {pd.__version__}")
except:
    print("❌ pandas not found")

try:
    import numpy as np
    print(f"✅ numpy {np.__version__}")
except:
    print("❌ numpy not found")

try:
    import matplotlib
    print(f"✅ matplotlib {matplotlib.__version__}")
except:
    print("❌ matplotlib not found")

try:
    import seaborn as sns
    print(f"✅ seaborn {sns.__version__}")
except:
    print("❌ seaborn not found")

try:
    import tqdm
    print(f"✅ tqdm {tqdm.__version__}")
except:
    print("❌ tqdm not found")

try:
    import langdetect
    print(f"✅ langdetect installed")
except:
    print("⚠️  langdetect not found (optional)")

print("\n✅ Setup check complete!")
```

Save as `test_setup.py` and run: `python test_setup.py`
