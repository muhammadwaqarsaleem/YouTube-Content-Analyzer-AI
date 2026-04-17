# Scripts Directory

This directory contains all utility and helper scripts for the YouTube Video Analysis system.

## 📋 Script Descriptions

### **Data Processing Scripts**

| Script | Purpose |
|--------|---------|
| `data_exploration.py` | Explore and analyze the dataset, generate statistics and visualizations |
| `data_preprocessing.py` | Preprocess raw data, clean and prepare for training |
| `feature_engineering.py` | Extract and engineer features from raw data |
| `image_preprocessing.py` | Process and preprocess images for model training |

### **Model Training Scripts**

| Script | Purpose |
|--------|---------|
| `model_training.py` | Train machine learning models |
| `model_evaluation.py` | Evaluate model performance with metrics |
| `convert_model_format.py` | Convert models between different formats (Keras, TensorFlow, etc.) |
| `save_production_model.py` | Save trained models in production-ready format |

### **Testing & Validation Scripts**

| Script | Purpose |
|--------|---------|
| `test_system.py` | Comprehensive system tests - verify all components work together |
| `test_model_on_new_data.py` | Test models on new/unseen data for validation |
| `performance_summary.py` | Generate performance summaries and reports |
| `final_summary.py` | Create final project summary and results |

### **Demo & Utility Scripts**

| Script | Purpose |
|--------|---------|
| `demo.py` | Demo script showcasing system capabilities |
| `frontend_server.py` | Simple HTTP server for serving frontend files (port 3000) |

## 🚀 How to Run Scripts

All scripts should be run from the project root directory:

```bash
# From project root
cd c:\Users\Syed Ali\Desktop\Project69

# Run any script
python scripts/<script_name>.py

# Examples
python scripts/data_exploration.py
python scripts/model_evaluation.py
python scripts/test_system.py
```

## 📁 Project Structure

```
Project69/
├── scripts/              ← You are here
│   ├── data_exploration.py
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── image_preprocessing.py
│   ├── model_training.py
│   ├── model_evaluation.py
│   ├── convert_model_format.py
│   ├── save_production_model.py
│   ├── test_system.py
│   ├── test_model_on_new_data.py
│   ├── performance_summary.py
│   ├── final_summary.py
│   └── demo.py
├── start_server.py       ← Main server startup (in root)
├── frontend_server.py    ← Frontend server (in root)
├── api/                  ← API layer
├── services/             ← Business logic
├── src/                  ← Core modules
├── utils/                ← Utility functions
├── models/               ← Trained models
├── features/             ← Feature extractors
└── ...
```

## ⚠️ Important Notes

1. **Run from Root**: Always run scripts from the project root directory to ensure correct import paths
2. **Dependencies**: Make sure all dependencies are installed (`pip install -r requirements.txt`)
3. **Python Version**: Requires Python 3.8+
4. **Main Entry Points**: 
   - `start_server.py` (in root) - Start both frontend and backend
   - `frontend_server.py` (in root) - Start frontend only

## 🎯 Common Workflows

### Data Preparation Workflow
```bash
python scripts/data_exploration.py      # Step 1: Explore data
python scripts/data_preprocessing.py    # Step 2: Clean data
python scripts/feature_engineering.py   # Step 3: Extract features
```

### Model Development Workflow
```bash
python scripts/model_training.py        # Step 1: Train model
python scripts/model_evaluation.py      # Step 2: Evaluate model
python scripts/save_production_model.py # Step 3: Save for production
```

### Testing Workflow
```bash
python scripts/test_system.py           # Full system test
python scripts/test_model_on_new_data.py # Test on new samples
```

---

**For main application usage, refer to `README.md` in the project root.**
